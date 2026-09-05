#!/usr/bin/env python3
"""
Phantom Vast.ai Orchestrator — manage GPU instances from the command line.

Usage:
    python vast/orchestrator.py start      # rent an instance → setup → pipeline → update .env
    python vast/orchestrator.py resume     # start a stopped instance (VAST_INSTANCE_ID)
    python vast/orchestrator.py stop       # stop it (disk survives, storage still bills)
    python vast/orchestrator.py terminate  # destroy it (disk and models are gone)
    python vast/orchestrator.py status     # state, GPU, cost, address
    python vast/orchestrator.py offers     # what is rentable right now, and what start would take
    python vast/orchestrator.py logs [n]   # tail the pipeline log
    python vast/orchestrator.py run "cmd"  # run one command on the instance
    python vast/orchestrator.py push <local> [remote]
    python vast/orchestrator.py pull <remote> [local]

Why Vast rather than RunPod — docs/VAST_MIGRATION.md has the measurement. Short
version: RunPod has fifty datacenters and none in the UK, EU-FR-1 carries no
eligible GPU at all, and the operator is in West Africa. Vast has verified
4090s in the UK at $0.31/hr, cheaper than the Romanian card they replace.

Three differences from the RunPod orchestrator drive most of this file:

  - **SSH is real.** `runtype: ssh_direct` provisions port 22 on the instance
    itself, so `exec_command` works and SFTP works. The RunPod version needed
    ~150 lines of `invoke_shell` sentinel parsing because its proxy silently
    dropped `exec_command`, and it could not copy a file *off* a pod at all.
    Both are gone; `pull` exists because of it.

  - **There is no TLS proxy.** RunPod handed out `wss://…proxy.runpod.net`.
    Vast gives a random external port on a shared public IP. `controller.py`
    already speaks `ws://host:port/ws`, which is the trap: it would work, in
    cleartext, with the operator's face in it. So the instance generates a
    self-signed certificate, `startup.sh` prints its fingerprint, and this
    script pins that fingerprint into .env beside the URL.

  - **Offers are searched, not enumerated.** RunPod exposed a global GPU
    catalogue and every filter ran locally; Vast filters server-side on
    geolocation, uplink, reliability and compute capability — including the two
    things that actually decide this workload, distance and upstream bandwidth.
    `dlperf` is a measured score on every offer, so the hand-typed `_GPU_PERF`
    table is gone.

Selection:
  - VAST_GEOLOCATIONS bounds distance; everything else is a quality floor.
  - VAST_PREFERRED_HOST pins one host so the IP is stable and the disk stays
    warm. When it has nothing rentable the filtered search runs instead.
  - When nothing matches at the preferred standard, the search relaxes in
    bounded steps (_RELAXATION_LADDER) rather than waiting: CPU first because
    it costs less, then GPU, then both. **Every rung still holds 20fps**, which
    is the target the ladder exists to protect — it stops rather than
    continuing into hardware that cannot deliver.
  - Only when no rung matches does it wait, retrying every minute for
    VAST_GPU_WAIT seconds. Waiting is free: billing starts when an instance
    runs, not while you are looking for one.
  - VAST_RELAX=false pins the top rung, for a measurement session where
    numbers from two architectures are not a comparison.

Warm models:
  - A stopped instance keeps its disk, so `stop`/`resume` is the fast path and
    the models stay put. Nothing is baked into an image, which means **the disk
    is the only copy**: `terminate` re-downloads everything next time.
  - Storage bills while stopped, per host, and Vast hosts charge far more than
    RunPod did for it. VAST_DISK is therefore a cost setting, not a detail.

Reads from .env in the repo root.
"""

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

# Repo root is one level up from vast/
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

_API = "https://console.vast.ai/api/v0"

_POLL_INTERVAL = 3       # seconds between status polls
_READY_TIMEOUT = 600     # seconds to wait for an instance to reach 'running'
_SSH_TIMEOUT = 300       # seconds to wait for the SSH port to answer
_PIPELINE_TIMEOUT = 180  # seconds to wait for the pipeline to bind port 9000
_SSH_CMD_TIMEOUT = 1800  # seconds for any single remote command (pip is slow)

_PIPELINE_PORT = 9000

# startup.sh reports the API token on its own line so the orchestrator can
# read it out of the transcript. That line must never be printed.
_TOKEN_LINE_RE = re.compile(r"(API_TOKEN\s+)\S+")

# Remote paths. Vast images use /workspace like RunPod's did, which is why the
# pipeline's own path handling needed no change at all.
_REMOTE_PHANTOM_DIR = "/workspace/Phantom"
_REMOTE_VENV_PYTHON = "/workspace/venv/bin/python"
_REMOTE_STARTUP = "{}/vast/startup.sh".format(_REMOTE_PHANTOM_DIR)
_REMOTE_PIPELINE = "{}/pipeline.py".format(_REMOTE_PHANTOM_DIR)
_PIPELINE_LOG = "/workspace/phantom-pipeline.log"
_REMOTE_CERT = "/workspace/phantom-tls.pem"
_REMOTE_KEY = "/workspace/phantom-tls.key"

# ── Selection defaults ────────────────────────────────────────────────────────

# Country codes. The filter matches on the code even though the field's value
# is a full string like "United Kingdom, GB". Ordered west to east: this is a
# distance list, and distance is the whole reason for the migration.
_DEFAULT_GEOLOCATIONS = "GB,IE,FR,NL,BE,DE"

# `dlperf` is Vast's own measured deep-learning score, present on every offer.
# It replaces the hand-maintained _GPU_PERF table the RunPod orchestrator
# carried, and `tests/test_gpu_tier.py` with it.
#
# Measured across western Europe on 2026-09-03: RTX 6000 Ada 113, RTX 4090
# 97.2, RTX 5080 83.8, A100 SXM4 83.3, RTX 3090 44.5. A floor of 90 therefore
# reproduces the old `perf >= 85` tier almost exactly — 4090 and better —
# without anybody typing a ranking in.
#
# The reason for having a floor at all is unchanged and worth restating: auto
# discovery accepting an L4 because the 4090 was busy cost a whole measurement
# session, every number in it against an architecture nothing else was
# measured on.
_DEFAULT_MIN_DLPERF = 90.0

_DEFAULT_MIN_VRAM = 16          # GB
_DEFAULT_MAX_PRICE = 1.00       # $/hr
_DEFAULT_MIN_RELIABILITY = 0.98
_DEFAULT_MIN_INET_UP = 100.0    # MB/s
_DEFAULT_MIN_PORTS = 32

# The compositor is OpenCV on the CPU and is the one stage a faster GPU does
# not touch: ~10.3ms of a ~29ms frame on a 4090 host, and ~20ms on an L4 host
# with the same models. So the host's CPU decides a third of the frame.
#
# Measured across western Europe on 2026-09-04, eligible offers only: clock
# ranged 3.5 to 6.0 GHz — a 1.7x spread on latency-sensitive single-threaded
# work. Desktop silicon wins here (Ryzen 9 7900X at 5.7, Core i9-13900KF at
# 6.0) and server silicon loses (EPYC 7663 at 3.5, EPYC 7443 at 4.0), because
# EPYC and Xeon trade clock for core count and this workload wants the
# opposite.
#
# The floor is deliberately loose. A slow CPU still holds the 50ms deadline at
# 20fps, so clock is a preference rather than a requirement, and a tight floor
# would collapse availability — at 4.5 GHz only two offers in all of western
# Europe survived. `_rank` is what actually selects on it.
_DEFAULT_MIN_CPU_GHZ = 3.5

# `cpu_cores_effective` is this instance's SHARE of the machine, not the
# machine's total — a host splitting 64 cores across 8 GPUs offers 8. OpenCV's
# parallel_for will use them, but on 256x256 crops the benefit saturates early,
# so this is a floor against a starved allocation rather than a lever.
_DEFAULT_MIN_CPU_CORES = 8.0
_DEFAULT_DISK = 25              # GB
# Relaxation, bounded by the one number that is not negotiable.
#
# **20fps is a hard target**, so every rung here still has to fit a frame into
# 50ms. The frame is roughly 19ms of GPU work and 10.3ms of CPU work at the
# reference (RTX 4090, dlperf 97), and GPU work scales about inversely with
# dlperf:
#
#     dlperf 90 -> ~21ms GPU -> ~31ms frame   19ms spare
#     dlperf 65 -> ~28ms GPU -> ~39ms frame   11ms spare
#     dlperf 55 -> ~34ms GPU -> ~44ms frame    6ms spare
#     dlperf 45 -> ~41ms GPU -> ~51ms frame    MISSES 20fps
#
# So 60 is the floor with margin, and nothing below it may be rented — an
# RTX 3090 at 44.5 would not hold the target.
#
# The CPU rung stops at 2.5 GHz for a different and weaker reason: the 10.3ms
# compositor was measured on a host whose clock is unknown, so scaling it is an
# extrapolation, and a hard target should not rest on one. 2.5 keeps the
# server boxes at 2.0-2.4 GHz out.
#
# CPU is relaxed before GPU because it costs less: dropping the CPU rung is
# worth a few milliseconds, dropping the GPU rung about seven.
_RELAXATION_LADDER = (
    ("preferred",    90.0, 3.5),
    ("relaxed CPU",  90.0, 2.5),
    ("relaxed GPU",  60.0, 3.5),
    ("relaxed both", 60.0, 2.5),
)

_DEFAULT_GPU_WAIT = 300         # seconds
_GPU_RETRY_INTERVAL = 60        # seconds

# The image's PyTorch/ONNX support ceiling, as Vast reports it: compute
# capability x 100, so 900 is sm_90 (Hopper). Blackwell lists as 1200 and would
# schedule happily against a CUDA 12.1 image and then fail after the money
# started. Unlike RunPod this is a server-side filter on a field the API
# actually publishes, so there is no keyword table to keep current.
_DEFAULT_MAX_COMPUTE_CAP = 900

# Pipeline settings forwarded from the local .env into the instance, so `start`
# produces a configured pipeline rather than a default one. Every name here is
# read by pipeline/core.py as an argparse default.
#
# Deliberately a list rather than "forward everything": the instance is a
# different machine, and blanket-forwarding would send local paths and secrets
# that mean nothing there.
_FORWARDED_ENV = (
    # Model selection and realism
    "SWAPPER_MODEL", "ENHANCER_MODEL", "ENHANCER_WEIGHT", "ENHANCE_STRENGTH",
    "ALIGNED_SIZE", "RESTORE_SIZE", "RESTORE_MIN_FACE",
    "TEMPORAL_ALPHA", "COLOR_STRENGTH", "TEXTURE_STRENGTH",
    "MASK_FEATHER", "MASK_ERODE", "DIFFUSE_STRENGTH",
    "ENHANCE", "GRAIN", "OCCLUDER",
    # Guards and their calibration
    "GUARDS", "GUARD_OBSERVE", "GUARD_REPORT",
    # Inference speed levers. These matter more here than anywhere: the whole
    # point of them is to be A/B'd against a rented GPU, so a lever that cannot
    # reach the instance cannot be measured at all.
    "FP16", "CUDA_GRAPHS", "CUDA_STREAMS", "ASYNC_ENCODE", "TRT", "TRT_GPUS",
    # Measurement
    "DEBUG_FRAMES_DIR", "DEBUG_FRAMES_STRIDE", "DEBUG_FRAMES_LIMIT",
    "LOG_LEVEL",
    # Batch scratch location, so a long video does not fill the disk
    "PHANTOM_TEMP_DIR",
)


class BootTimer:
    """
    Phase-by-phase timing for a cold start.

    A stopwatch gives the total; this gives the breakdown, which is the
    difference between "cold start is slow" and "pip is 80% of it, so bake an
    image" — a decision the total alone cannot support.

    Phases nest: the coarse ones are measured here, and `startup.sh` reports its
    own inner phases as `PHASE <name> <seconds>` lines, which `absorb` picks up.
    Neither side has to agree on anything but that prefix.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.start = time.time()
        self.phases: List[Tuple[str, float]] = []
        self.disk_state = "unknown"
        self._current: Optional[str] = None
        self._t0 = self.start

    def phase(self, name: str) -> None:
        """Close the running phase, if any, and begin `name`."""
        now = time.time()
        if self._current:
            self.phases.append((self._current, now - self._t0))
        self._current = name
        self._t0 = now
        if name:
            print("\n[phase] {}".format(name))

    def absorb(self, transcript: str) -> None:
        """
        Take inner phase timings out of a remote script's output.

        Replaces the coarse phase they occurred inside, so the breakdown does
        not double-count: `remote-setup` is exactly the sum of what startup.sh
        reported, and showing both would make the total meaningless.
        """
        inner: List[Tuple[str, float]] = []
        for line in transcript.splitlines():
            line = line.strip()
            if line.startswith("PHASE "):
                parts = line.split()
                if len(parts) == 3 and parts[1] != "total":
                    try:
                        inner.append(("  " + parts[1], float(parts[2])))
                    except ValueError:
                        pass
            elif line.startswith("DISK "):
                self.disk_state = line.split()[-1]

        if not inner:
            return

        expanded: List[Tuple[str, float]] = []
        for name, seconds in self.phases:
            expanded.append((name, seconds))
            if name == "remote-setup":
                expanded.extend(inner)
        self.phases = expanded

    def report(self) -> None:
        """Print the breakdown, slowest phase named last so it is the parting thought."""
        self.phase("")
        total = time.time() - self.start

        print("\n" + "=" * 58)
        print("{} - {:.0f}s total, disk: {}".format(
            self.label, total, self.disk_state))
        print("=" * 58)

        for name, seconds in self.phases:
            share = (seconds / total * 100.0) if total > 0 else 0.0
            bar = "#" * int(round(share / 4))
            # Indented names are inner phases already counted in their parent.
            marker = " " if name.startswith("  ") else "*"
            print("{} {:<22} {:>6.1f}s  {:>5.1f}%  {}".format(
                marker, name.strip(), seconds, share, bar))

        print("-" * 58)
        print("{} {:<22} {:>6.1f}s".format("*", "total", total))
        outer = [p for p in self.phases if not p[0].startswith("  ")]
        if outer:
            worst = max(outer, key=lambda p: p[1])
            print("\nSlowest phase: {} ({:.0f}s, {:.0f}% of total)".format(
                worst[0], worst[1], worst[1] / total * 100.0 if total else 0))
        print("Rows marked * are exclusive; indented rows break down the row above.")


# ── Env helpers ───────────────────────────────────────────────────────────────

def _update_env_key(key: str, value: str) -> None:
    """Rewrite a single key=value line in .env, appending if not present."""
    if not _ENV_PATH.exists():
        print("WARNING: .env not found at {}, skipping update".format(_ENV_PATH))
        return

    text = _ENV_PATH.read_text()
    pattern = r"^{}=.*$".format(re.escape(key))
    replacement = "{}={}".format(key, value)

    if re.search(pattern, text, re.MULTILINE):
        # A lambda, so the value is inserted literally. A plain replacement
        # string is interpreted: a value containing a backslash escape raises
        # "invalid group reference" and takes the whole call down. That call
        # runs *after* an instance is rented and running, so the failure would
        # strand a billing machine with its address never written to .env.
        new_text = re.sub(pattern, lambda _m: replacement, text, flags=re.MULTILINE)
    else:
        new_text = text.rstrip() + "\n{}\n".format(replacement)

    _ENV_PATH.write_text(new_text)
    # The token is a credential; the fingerprint is not, but printing 64 hex
    # characters into a scrollback helps nobody either.
    shown = value if key not in ("PHANTOM_API_TOKEN", "PHANTOM_TLS_FINGERPRINT") else "<set>"
    print("  Updated .env  {}={}".format(key, shown))


def _env_flag(name: str) -> bool:
    """
    Read a boolean setting from the environment.

    Same spellings the pipeline accepts (`pipeline/core.py::_env_bool`), so a
    `.env` shared between the two does not mean two different things. Unset or
    unrecognised is False: every flag read through here turns something off
    that is on by default, and a typo must not be the thing that disables it.
    """
    raw = (os.getenv(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print("WARNING: {}={!r} is not a number; using {}".format(name, raw, default))
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        print("WARNING: {}={!r} is not a number; using {}".format(name, raw, default))
        return default


# ── Vast API ──────────────────────────────────────────────────────────────────

def _api_key(required: bool = True) -> str:
    key = os.getenv("VAST_API_KEY") or ""
    if not key and required:
        print("ERROR: VAST_API_KEY not set in .env")
        print("  Create one at https://cloud.vast.ai/manage-keys/")
        sys.exit(1)
    return key


def _request(method: str, path: str, auth: bool = True,
             raise_on_error: bool = False, **kwargs: Any) -> Any:
    """
    One call against the Vast REST API.

    `auth=False` is for `/bundles/` only, which answers without a key — so
    `offers` works before an account exists, which is the command someone runs
    to decide whether to open one. Every other endpoint needs the key and says
    so rather than returning an empty result that reads like "nothing
    available".
    """
    url = "{}{}".format(_API, path)
    headers = {"Content-Type": "application/json"}
    key = _api_key(required=auth)
    if key:
        headers["Authorization"] = "Bearer {}".format(key)
    try:
        resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
    except Exception as exc:
        print("ERROR: {} {} failed: {}".format(method, path, exc))
        sys.exit(1)

    if resp.status_code >= 400:
        if raise_on_error:
            raise VastAPIError("HTTP {}: {}".format(resp.status_code, resp.text[:400]))
        print("ERROR: {} {} -> HTTP {}: {}".format(method, path, resp.status_code, resp.text[:400]))
        sys.exit(1)

    try:
        payload = resp.json()
    except ValueError:
        return {}

    # Vast answers state changes with a SimpleBooleanSuccessResponse, so a
    # refusal can arrive as HTTP 200 carrying {"success": false, "msg": ...}.
    # Checking only the status code read that as success — which matters most
    # in `resume`, where the capacity fallback exists precisely to notice a
    # refusal, and would instead have gone on to boot an instance that never
    # started.
    if isinstance(payload, dict) and payload.get("success") is False:
        detail = payload.get("msg") or payload.get("error") or resp.text[:300]
        if raise_on_error:
            raise VastAPIError("{} {}: {}".format(method, path, detail))
        print("ERROR: {} {} refused: {}".format(method, path, detail))
        sys.exit(1)

    return payload


class VastAPIError(Exception):
    """An API call that the caller wants to inspect rather than die on."""


# Phrases Vast uses when a stopped instance's host has no GPU left for it.
# Matched as substrings because the wording carries the machine's own numbers.
#
# A stopped instance keeps its host — that is what keeps its disk — so
# resuming needs a GPU free on that specific machine. Someone else taking it
# while you rested is an ordinary outcome rather than a fault, and it is the
# one resume failure that a *new* instance actually fixes.
_CAPACITY_MARKERS = (
    "no gpu", "not enough", "unavailable", "insufficient",
    "no longer available", "capacity", "already rented", "cannot start",
)


def _is_capacity_error(message: str) -> bool:
    """Whether a resume failure means the host is full rather than broken."""
    low = message.lower()
    return any(marker in low for marker in _CAPACITY_MARKERS)


def _search_offers(
    geolocations: List[str],
    min_dlperf: float,
    min_vram: int,
    max_price: float,
    min_reliability: float,
    min_inet_up: float,
    min_ports: int,
    min_cpu_ghz: float,
    min_cpu_cores: float,
    max_compute_cap: int,
    verified_only: bool,
    disk: int,
    host_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ask Vast what is rentable, filtered server-side.

    This is the one endpoint that answers without a key, which is what lets
    `offers` run from a clean checkout with no account. A key is still sent
    when one is configured, but do not read that as validation: search does not
    reject a bad key, so the first thing that actually checks it is whatever
    call comes next.

    Two format traps, both of which return zero offers rather than an error, so
    neither announces itself:

      - `gpu_name` uses spaces ("RTX 4090"). The CLI converts underscores; the
        REST API does not. We do not filter on it at all — `dlperf` and
        `compute_cap` say what we actually mean, and a name list would go stale.
      - `geolocation` values are strings like "United Kingdom, GB", but the
        filter matches the country code, so {"in": ["GB"]} is correct.

    `disk` is a filter as well as a request: an offer with less free space than
    we intend to allocate cannot host the instance.
    """
    body: Dict[str, Any] = {
        "limit": 200,
        "type": "ondemand",
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "dlperf": {"gte": min_dlperf},
        "gpu_ram": {"gte": min_vram * 1024},
        "dph_total": {"lte": max_price},
        "reliability": {"gte": min_reliability},
        "inet_up": {"gte": min_inet_up},
        "direct_port_count": {"gte": min_ports},
        "cpu_ghz": {"gte": min_cpu_ghz},
        "cpu_cores_effective": {"gte": min_cpu_cores},
        "compute_cap": {"lte": max_compute_cap},
        "disk_space": {"gte": disk},
        # Every ONNX model here runs on CUDAExecutionProvider, so a card
        # without it is not a slower option, it is no option. AMD's MI300X
        # listed at $0.50/hr with 192GB on RunPod and passed every other
        # filter, which a cheapest-first search reaches straight for: cheap and
        # unusable is the worst combination on a rented GPU.
        "gpu_arch": {"eq": "nvidia"},
        "order": [["dph_total", "asc"]],
    }
    if geolocations:
        body["geolocation"] = {"in": geolocations}
    if verified_only:
        body["verified"] = {"eq": True}
    if host_id is not None:
        # A pinned host is exempt from the *preferences*, never from the
        # *target*. It was chosen deliberately, so a transient reliability dip
        # or a slightly slow day should not silently move the session to
        # another country — but "I picked this host once" is not evidence that
        # it can still do the job.
        #
        # It is not hypothetical. `dlperf` is measured per OFFER, not per GPU
        # model, and the live market carries RTX 4090s scoring 50 and 51
        # against a normal 97 — hosts letting out a throttled or heavily shared
        # card. Pinning by host id and skipping the speed floor is the same
        # mistake as filtering by `gpu_name`: it trusts an identity where only a
        # measurement will do, and would rent half a 4090 at full price while
        # missing 20fps.
        #
        # So the floors that decide whether the product works still apply, at
        # the ladder's bottom rung rather than the preferred standard: below
        # that, no amount of loyalty to a host makes the session deliverable.
        floor_dlperf = min(d for _, d, _ in _RELAXATION_LADDER)
        floor_ghz = min(g for _, _, g in _RELAXATION_LADDER)
        body = {
            "limit": 50, "type": "ondemand", "rentable": {"eq": True},
            "num_gpus": {"eq": 1}, "host_id": {"eq": host_id},
            # Cannot run the software at all:
            "compute_cap": {"lte": max_compute_cap},
            "gpu_arch": {"eq": "nvidia"},
            "gpu_ram": {"gte": min_vram * 1024},
            "disk_space": {"gte": disk},
            # Cannot hold 20fps:
            "dlperf": {"gte": floor_dlperf},
            "cpu_ghz": {"gte": floor_ghz},
            # Cannot be allowed to bill unbounded because it was pinned once:
            "dph_total": {"lte": max_price},
            "order": [["dph_total", "asc"]],
        }

    resp = _request("POST", "/bundles/", auth=False, json=body)
    offers = [o for o in (resp.get("offers") or []) if isinstance(o, dict)]
    return _rank(offers, geolocations)


def _country(offer: Dict[str, Any]) -> str:
    """
    The country code out of a geolocation string like "United Kingdom, GB".

    Vast returns the whole label; the filter matches the code. Reading the last
    comma-separated token gives back the code the caller asked for.
    """
    label = str(offer.get("geolocation") or "")
    return label.rsplit(",", 1)[-1].strip().upper()


def _cpu_band(offer: Dict[str, Any]) -> float:
    """
    CPU clock rounded to the nearest 0.5 GHz.

    Banded rather than raw, and this is the whole design of the ranking. Sorted
    on raw clock, a 5.6 GHz box at $0.90 would beat a 5.7 GHz box at $0.29 — a
    difference inside the noise of what this workload can feel, bought at 3x.
    Banding makes clock decide only when the gap is large, and hands everything
    else back to price.

    It does not remove the problem at the band edges: 5.7 and 5.8 fall either
    side of one, so clock decides between them for no good reason. That is
    accepted rather than fixed. Any bucketing has edges, a narrower band only
    moves them, and the differences that matter here are several bands wide —
    3.5 GHz against 5.7 is 1.6x, which no edge case can hide.
    """
    return round(float(offer.get("cpu_ghz") or 0.0) * 2.0) / 2.0


def _dlperf_band(offer: Dict[str, Any]) -> float:
    """
    GPU score rounded to a 25-point band, for the same reason clock is banded:
    a few points is not worth a price gap.

    Across the eligible set (91 to 113 on 2026-09-04) the GPU stages vary by
    roughly 4ms of frame time — comparable to what the CPU spread is worth, so
    it belongs in the ordering rather than being left to the floor alone.
    """
    return round(float(offer.get("dlperf") or 0.0) / 25.0) * 25.0


def _rank(offers: List[Dict[str, Any]], geolocations: List[str]) -> List[Dict[str, Any]]:
    """
    Order by country, then CPU clock band, then price.

    **Country first** because VAST_GEOLOCATIONS is a priority order and
    distance is the term this deployment cannot buy back. Sorting on price
    alone had a French host at $0.336 beating a British one at $0.350 — three
    cents to give back part of the round trip the migration exists to remove.

    **CPU clock second, and this is the part that is not obvious.** The
    compositor is OpenCV on the CPU — ~10.3ms of a ~29ms frame — and is the one
    stage a faster GPU does not touch. Clock across eligible offers ranges 3.5
    to 6.0 GHz, so the host CPU swings roughly 4-5ms of frame time: more than
    switching restoration models bought on the detect stage, and free to
    choose. It also decides whether 30fps is reachable later, since that is the
    rate where a slow compositor breaks the deadline outright.

    **GPU third.** The frame is roughly 19ms of GPU work and 10.3ms of CPU
    work, so the GPU is the larger half — but `VAST_MIN_DLPERF` has already
    guaranteed it is adequate, and there is no equivalent guarantee for the
    CPU, whose floor is deliberately loose. So CPU ordering is doing real work
    and GPU ordering is a refinement among cards that all already hold the
    deadline. Across the eligible set each is worth about 4ms.

    **Price last**, within all three bands.
    """
    rank = {code: i for i, code in enumerate(geolocations)}
    fallback = len(rank)
    return sorted(
        offers,
        key=lambda o: (rank.get(_country(o), fallback),
                       -_cpu_band(o),
                       -_dlperf_band(o),
                       o.get("dph_total") or 0.0),
    )


def _describe_offer(offer: Dict[str, Any]) -> str:
    return ("{gpu} - {loc} - ${price:.3f}/hr - dlperf {dl:.0f} - "
            "cpu {ghz:.1f}GHz x{cores:.0f} - "
            "up {up:.0f} MB/s - rel {rel:.3f} - {ports} ports").format(
        gpu=offer.get("gpu_name", "?"),
        loc=(offer.get("geolocation") or "?"),
        price=offer.get("dph_total") or 0.0,
        dl=offer.get("dlperf") or 0.0,
        ghz=offer.get("cpu_ghz") or 0.0,
        cores=offer.get("cpu_cores_effective") or 0.0,
        up=offer.get("inet_up") or 0.0,
        rel=offer.get("reliability2") or 0.0,
        ports=offer.get("direct_port_count") or 0,
    )


def _hourly_bandwidth_cost(offer: Dict[str, Any]) -> float:
    """
    What an hour of streaming costs on top of the GPU, on this host.

    RunPod did not bill bandwidth; Vast does, per host, and the rates vary by
    two orders of magnitude across otherwise similar offers. At the `optimal`
    preset the stream is ~600 KB/s each way, so ~2.16 GB/hour in each
    direction. Small — a few cents — but it is the kind of small that is worth
    printing beside a price rather than discovering on an invoice.
    """
    gb_each_way = 2.16
    up = offer.get("inet_up_cost") or 0.0
    down = offer.get("inet_down_cost") or 0.0
    return gb_each_way * (float(up) + float(down))


def _get_instance(instance_id: str) -> Dict[str, Any]:
    """Fetch one instance, or an empty dict if it is gone."""
    resp = _request("GET", "/instances/{}/".format(instance_id))
    inst = resp.get("instances")
    if isinstance(inst, dict):
        return inst
    if isinstance(inst, list) and inst:
        return inst[0]
    return {}


def _instance_status(instance: Dict[str, Any]) -> str:
    return str(instance.get("actual_status") or instance.get("cur_state") or "unknown")


def _mapped_port(instance: Dict[str, Any], internal: int) -> Optional[int]:
    """
    External port for an internal one, from the instance's port map.

    Shape, which is docker's rather than Vast's:
        "ports": {"9000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}

    Only present on running instances, so a None here usually means "not yet"
    rather than "not exposed".
    """
    ports = instance.get("ports") or {}
    entry = ports.get("{}/tcp".format(internal))
    if not entry:
        return None
    try:
        return int(entry[0]["HostPort"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _ws_address(instance: Dict[str, Any]) -> Optional[str]:
    """'ip:port' for the pipeline WebSocket, or None if not published yet."""
    ip = instance.get("public_ipaddr")
    port = _mapped_port(instance, _PIPELINE_PORT)
    if not ip or not port:
        return None
    return "{}:{}".format(str(ip).strip(), port)


def _ssh_targets(instance: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Every SSH endpoint worth trying, best first.

    Direct first — the mapping on the instance's own public IP is what
    `runtype: ssh_direct` buys, and it is the difference between `exec_command`
    working and the RunPod situation where it silently did nothing. The proxy
    (`ssh_host`) comes second.

    Both are returned rather than one being chosen, because a host can
    *advertise* a direct port it does not actually accept. That is not
    hypothetical: instance 49966815 published 81.2.141.194:40000, refused every
    connection to it, and deployed perfectly over ssh3.vast.ai. `start` happened
    to run before the direct mapping appeared and used the proxy; `run`, `logs`,
    `push` and `pull` ran after, took the direct address on faith and timed out
    after five minutes each. Reachability is measured, not read off a field —
    the same lesson dlperf already taught about trusting a published identity.
    """
    targets: List[Tuple[str, int]] = []

    ip = instance.get("public_ipaddr")
    port = _mapped_port(instance, 22)
    if ip and port:
        targets.append((str(ip).strip(), port))

    host = instance.get("ssh_host")
    sport = instance.get("ssh_port")
    if host and sport:
        try:
            proxy = (str(host), int(sport))
        except (TypeError, ValueError):
            proxy = None
        if proxy and proxy not in targets:
            targets.append(proxy)

    return targets


def _ssh_target(instance: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """The preferred SSH endpoint, or None. See `_ssh_targets`."""
    targets = _ssh_targets(instance)
    return targets[0] if targets else None


# ── Waiting ───────────────────────────────────────────────────────────────────

def _wait_for_running(instance_id: str) -> Dict[str, Any]:
    """Poll until the instance reports 'running', returning it."""
    print("Waiting for instance {} to run (up to {}s)...".format(instance_id, _READY_TIMEOUT))
    deadline = time.time() + _READY_TIMEOUT
    last = ""

    while time.time() < deadline:
        inst = _get_instance(instance_id)
        status = _instance_status(inst)
        if status != last:
            print("  [{:>4.0f}s] {}".format(time.time() - (deadline - _READY_TIMEOUT), status))
            last = status
        if status == "running":
            return inst
        # A host that cannot pull the image parks here rather than failing, and
        # waiting the full ten minutes to be told nothing is worse than saying so.
        if status in ("offline", "error"):
            print("ERROR: instance entered '{}'. Check the Vast console.".format(status))
            sys.exit(1)
        time.sleep(_POLL_INTERVAL)

    print("ERROR: instance not running after {}s (last status: {}).".format(_READY_TIMEOUT, last))
    sys.exit(1)


def _wait_for_tcp(host: str, port: int, timeout: int, label: str) -> None:
    """Wait until a TCP connect succeeds."""
    print("  Waiting for {} at {}:{}...".format(label, host, port))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            print("  {} is up.".format(label))
            return
        except (socket.error, OSError):
            time.sleep(_POLL_INTERVAL)
    print("ERROR: {} not reachable at {}:{} after {}s.".format(label, host, port, timeout))
    sys.exit(1)


def _wait_for_any_tcp(
    targets: List[Tuple[str, int]],
    timeout: int,
    label: str,
) -> Optional[Tuple[str, int]]:
    """
    Wait until one of `targets` accepts a connection, and return it.

    Each is tried in turn every round, rather than one being waited out before
    the next is touched. During boot none of them answer, and spending the whole
    timeout proving that about the first candidate is how a working proxy goes
    unnoticed behind an advertised direct port.

    Args:
        targets: (host, port) pairs, best first
        timeout: total seconds to keep trying, across all of them
        label: what to call this in the output

    Returns:
        The endpoint that answered, or None if the deadline passed
    """
    if not targets:
        return None
    print("  Waiting for {} at {}...".format(
        label, ", ".join("{}:{}".format(h, p) for h, p in targets)))
    deadline = time.time() + timeout
    while time.time() < deadline:
        for host, port in targets:
            try:
                sock = socket.create_connection((host, port), timeout=5)
                sock.close()
                print("  {} is up at {}:{}.".format(label, host, port))
                return (host, port)
            except (socket.error, OSError):
                continue
        time.sleep(_POLL_INTERVAL)
    return None


def _wait_for_pipeline(ws_address: str, context: Any, token: Optional[str]) -> None:
    """
    Poll the pipeline with a real WebSocket health check.

    `context` already pins the certificate, so by the time the token is sent
    the peer has been proven. A plain TCP connect would prove only that docker
    published a port, which it does before the pipeline has loaded a model.
    """
    scheme = "wss" if context is not None else "ws"
    ws_url = "{}://{}/ws".format(scheme, ws_address)

    print("\nWaiting for pipeline at {} (up to {}s)...".format(ws_url, _PIPELINE_TIMEOUT))
    deadline = time.time() + _PIPELINE_TIMEOUT

    while time.time() < deadline:
        try:
            from websockets.sync.client import connect
        except ImportError:
            print("  WARNING: websockets not installed; falling back to a TCP check.")
            host, port = ws_address.rsplit(":", 1)
            _wait_for_tcp(host, int(port), _PIPELINE_TIMEOUT, "pipeline port")
            print("  Port reachable (TCP only - health not verified).")
            return

        try:
            kwargs: Dict[str, Any] = {"open_timeout": 5, "close_timeout": 2}
            if context is not None:
                kwargs["ssl"] = context
            with connect(ws_url, **kwargs) as ws:
                hello: Dict[str, Any] = {"action": "health"}
                if token:
                    hello["token"] = token
                ws.send(json.dumps(hello))
                # The server broadcasts events to every client, so our reply may
                # arrive behind a STATUS_CHANGED or two.
                for _ in range(20):
                    reply = json.loads(ws.recv(timeout=5))
                    if reply.get("status") == "healthy":
                        print("  Pipeline is ready (healthy).")
                        return
                    if reply.get("action") == "health":
                        print("  Unexpected health response: {}".format(reply))
                        break
        except Exception as exc:
            # 1008 is the server refusing the credential, which no amount of
            # waiting fixes. Retrying it for three minutes and then reporting a
            # timeout describes a slow boot, which is the wrong problem.
            if "1008" in str(exc) or "unauthorized" in str(exc).lower():
                print("ERROR: the pipeline refused our token.")
                print("  startup.sh reported one and the pipeline was launched")
                print("  with it, so they have diverged - most likely an older")
                print("  pipeline process survived `pkill` and is still holding")
                print("  the port. Check with:")
                print("    python vast/orchestrator.py run \"pgrep -af pipeline.py\"")
                sys.exit(1)
            print("  Not ready: {}".format(exc))
            time.sleep(_POLL_INTERVAL)

    print("ERROR: pipeline not healthy after {}s.".format(_PIPELINE_TIMEOUT))
    print("  Read the log:  python vast/orchestrator.py logs")
    sys.exit(1)


def _pinned_context(host: str, port: int, expected: str, deadline: float) -> Any:
    """
    A TLS context that will complete a handshake with one certificate and no
    other, built by fetching that certificate and checking its fingerprint.

    The earlier version connected with verification off and checked the
    fingerprint *afterwards*, in `_boot`, one call later. Two things were wrong
    with that and only the second is obvious:

      - **The token went first.** `_wait_for_pipeline` sends
        `{"action": "health", "token": ...}` to prove the pipeline is up, so the
        credential reached a peer whose certificate had not been checked yet.
        Same shape as the server bug where a client joined the broadcast set
        before authenticating: the ordering *is* the property.
      - **A mismatch only warned.** It then wrote the fingerprint to `.env` and
        printed "Done".

    Loading the certificate as its own trust anchor makes the handshake do the
    enforcing, so nothing can be sent to the wrong peer even by mistake. It is
    the same approach `tools/pipeline_link.py` uses, for the same reason.

    Retries until `deadline`, because docker publishes the port before the
    pipeline binds it, so the first connections are refused as a matter of
    course rather than as a fault.
    """
    import ssl

    probe = ssl.create_default_context()
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE

    der = None
    last = ""
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=10) as raw:
                with probe.wrap_socket(raw, server_hostname=host) as tls:
                    der = tls.getpeercert(True)
            if der:
                break
        except Exception as exc:
            last = str(exc)
            time.sleep(_POLL_INTERVAL)

    if not der:
        print("ERROR: no TLS certificate from {}:{} ({}).".format(host, port, last))
        print("  The pipeline may not have started. Read the log:")
        print("    python vast/orchestrator.py logs")
        sys.exit(1)

    import hashlib
    actual = hashlib.sha256(der).hexdigest()
    if actual.lower() != expected.strip().lower():
        print("ERROR: certificate fingerprint mismatch at {}:{}".format(host, port))
        print("  expected {}...".format(expected[:16]))
        print("  got      {}...".format(actual[:16]))
        print("")
        print("  startup.sh generates the certificate once and reuses it, so")
        print("  this means something regenerated it after it was read, or")
        print("  something is answering on that address that is not the")
        print("  pipeline. Refusing to continue: writing this to .env would")
        print("  hand the desktop a pin it cannot satisfy, and the token has")
        print("  deliberately not been sent.")
        sys.exit(1)

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
    return context


# ── SSH ───────────────────────────────────────────────────────────────────────

def _require_paramiko() -> Any:
    """Lazy-import paramiko so a plain `offers` or `status` never needs it."""
    try:
        import paramiko
        return paramiko
    except ImportError:
        print("ERROR: paramiko not installed. Run: pip install paramiko")
        sys.exit(1)


def _load_ssh_key(key_path: str) -> object:
    """Load an SSH private key, trying ed25519, RSA, and ECDSA formats."""
    paramiko = _require_paramiko()
    path = os.path.expanduser(key_path)
    if not os.path.isfile(path):
        print("ERROR: SSH private key not found at {}".format(path))
        print("  VAST_SSH_KEY_PATH={} expands against the home directory of".format(key_path))
        print("  whichever shell runs this script.")
        if os.name == "nt":
            print("  A key generated inside WSL lives in the WSL home, not the Windows one:")
            print("    wsl -e cp ~/.ssh/id_ed25519 /mnt/c/Users/<you>/.ssh/id_ed25519")
        sys.exit(1)
    for key_class in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return key_class.from_private_key_file(path)
        except paramiko.ssh_exception.SSHException:
            continue
    print("ERROR: could not load SSH key from {}. Supported: ed25519, RSA, ECDSA.".format(path))
    sys.exit(1)


def _preflight_ssh_key() -> None:
    """
    Load the SSH key before anything starts billing.

    It is otherwise first reached after the instance exists, so an unreadable
    key surfaced only once a GPU was already running and left something to be
    stopped by hand. Same principle as the execution-provider check: fail
    before the money, not after.
    """
    key_path = os.getenv("VAST_SSH_KEY_PATH", "~/.ssh/id_ed25519")
    _load_ssh_key(key_path)
    print("SSH key OK: {}".format(os.path.expanduser(key_path)))


def _connect_ssh(instance: Dict[str, Any]) -> Any:
    """
    Open an SSH client to the instance, retrying while the container boots.

    Vast publishes the port before sshd is listening on it, so the first
    connections are refused as a matter of course rather than as a fault.
    """
    paramiko = _require_paramiko()
    targets = _ssh_targets(instance)
    if not targets:
        print("ERROR: instance exposes no SSH port yet.")
        sys.exit(1)
    reachable = _wait_for_any_tcp(targets, _SSH_TIMEOUT, "SSH")
    if reachable is None:
        print("ERROR: no SSH endpoint reachable after {}s. Tried: {}.".format(
            _SSH_TIMEOUT,
            ", ".join("{}:{}".format(h, p) for h, p in targets)))
        sys.exit(1)
    host, port = reachable

    key = _load_ssh_key(os.getenv("VAST_SSH_KEY_PATH", "~/.ssh/id_ed25519"))
    attempts = 12
    for attempt in range(1, attempts + 1):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            print("  Connecting via SSH as root@{}:{} (attempt {}/{})...".format(
                host, port, attempt, attempts))
            client.connect(hostname=host, port=port, username="root", pkey=key, timeout=30)
            return client
        except Exception as exc:
            client.close()
            if attempt == attempts:
                print("ERROR: SSH failed after {} attempts: {}".format(attempts, exc))
                sys.exit(1)
            print("  Not ready ({}) - retrying in 10s...".format(exc))
            time.sleep(10)
    raise AssertionError("unreachable")


def _console_safe() -> None:
    """
    Make stdout survive whatever the instance prints down the transcript.

    The remote half of a deploy is not ours to sanitise. pip draws progress with
    box-drawing characters, apt and git emit UTF-8 freely, and a Windows console
    defaults to cp1252 - so sys.stdout.write raises UnicodeEncodeError and takes
    the orchestrator down mid-install. That is an expensive way to lose a
    deploy: the instance carries on billing, the work is abandoned, and the
    traceback names an encoding rather than anything that actually went wrong.

    Reconfigure where that is supported, and fall back to per-line replacement
    in `_printable` where it is not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _printable(text: str) -> str:
    """
    `text`, guaranteed to survive an encode to the console's encoding.

    Belt to `_console_safe`'s braces: a stream that cannot be reconfigured -
    already wrapped, redirected by a caller, replaced in a test - still must not
    raise while echoing a line the instance wrote.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, "replace").decode(encoding, "replace")


def _redact(text: str, secrets: Optional[List[str]] = None) -> str:
    """
    Blank out anything that must not reach a terminal, a log or a screen share.

    Two credentials travel through `_ssh_run`:

      - `startup.sh` prints `API_TOKEN <hex>` for the orchestrator to read out
        of the transcript, and that transcript is streamed to stdout.
      - The launch command carries `PHANTOM_API_TOKEN` and a Vast API key —
        the **account-wide** one whenever VAST_SCOPED_API_KEY is unset.

    Both were printed in full. `_update_env_key` already masked them on the way
    into `.env`, which shows the intent; it just did not cover the louder path.

    Redaction is by value where the value is known, so a secret is hidden
    wherever it appears rather than only in the shape it was expected in. The
    token is also matched by pattern, because the first time it appears is the
    line we are reading it from and it is not yet known.
    """
    for secret in (secrets or []):
        if secret and len(secret) >= 8:
            text = text.replace(secret, "<redacted>")
    return _TOKEN_LINE_RE.sub(r"\1<redacted>", text)


def _ssh_run(client: Any, command: str, label: str, check: bool = True,
             secrets: Optional[List[str]] = None) -> str:
    """
    Run one command over SSH, streaming its output, and return the transcript.

    The **returned** transcript is unredacted — the caller parses the token out
    of it. Only what is printed is masked.

    This is `exec_command`, which is worth a note because the RunPod
    orchestrator could not use it: its proxy accepted the call and silently ran
    nothing, so that file carried ~150 lines of interactive-shell driving with
    a sentinel echo to recover the exit code. `ssh_direct` is a real sshd, so
    the exit status is just there.
    """
    print("\n[{}] $ {}".format(label, _redact(command, secrets)))
    stdin, stdout, stderr = client.exec_command(command, timeout=_SSH_CMD_TIMEOUT, get_pty=False)
    stdin.close()

    chunks: List[str] = []
    channel = stdout.channel
    channel.settimeout(60.0)
    for raw in iter(stdout.readline, ""):
        line = raw.rstrip("\n")
        chunks.append(raw)
        if line.strip():
            sys.stdout.write("  " + _printable(_redact(line, secrets)) + "\n")
            sys.stdout.flush()

    exit_code = channel.recv_exit_status()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if err:
        for line in err.splitlines():
            sys.stdout.write("  " + _printable(_redact(line, secrets)) + "\n")
        chunks.append(err)

    if check and exit_code != 0:
        print("ERROR: '{}' failed (exit {}).".format(label, exit_code))
        sys.exit(1)
    return "".join(chunks)


def _remote_env_exports() -> str:
    """Forwarded pipeline settings, as a shell prefix for the launch command."""
    parts = []
    for name in _FORWARDED_ENV:
        value = os.getenv(name)
        if value is not None and value != "":
            parts.append("export {}={};".format(name, _shell_quote(value)))
    return " ".join(parts)


def _shell_quote(value: str) -> str:
    """Single-quote a value for a POSIX shell."""
    return "'" + value.replace("'", "'\\''") + "'"


def _require_direct_reachable(instance: Dict[str, Any]) -> None:
    """
    Refuse a host whose published ports do not answer from here.

    The desktop reaches the pipeline at `public_ipaddr:<mapped 9000>`. There is
    no proxy in front of it - that is the point of the Vast migration, and the
    reason RunPod's `proxy.runpod.net` was dropped. So a host that publishes
    port mappings and then accepts nothing on them cannot serve this product,
    however good its GPU is.

    Host 596679 did exactly that: an excellent offer on every visible axis -
    RTX 4090, dlperf 97, 5.7GHz, and **98 open ports** in its own listing - on
    which 40000 and 40013 both timed out while ssh3.vast.ai answered instantly.
    The port count is a claim by the host, not a measurement, which is the same
    trap that dlperf-per-offer already documents.

    Checked through SSH, because SSH is the only port open before the pipeline
    starts. If the direct SSH address refuses while the proxy answers, the
    WebSocket port on the same IP will refuse too - and learning that after the
    deploy costs a full cold start to find out nothing. Twice, in this case.

    Silent when the host is fine, and when no direct address is published to
    test against.
    """
    targets = _ssh_targets(instance)
    if len(targets) < 2:
        return

    host, port = targets[0]
    # Retried rather than sampled once: sshd binding the direct port a moment
    # after the proxy is ready is a race, and calling that a dead host would be
    # a false alarm that costs a good rental.
    for _ in range(3):
        try:
            sock = socket.create_connection((host, port), timeout=8)
            sock.close()
            return
        except (socket.error, OSError):
            time.sleep(5)

    print("")
    print("ERROR: this host publishes ports it does not accept connections on.")
    print("  {}:{} refused, while the SSH proxy answered.".format(host, port))
    print("")
    print("  The desktop connects straight to {}:<mapped 9000>, with no".format(host))
    print("  proxy in front of it, so the pipeline would deploy correctly and")
    print("  then be unreachable. Stopping now rather than after the install.")
    print("")
    preferred = os.getenv("VAST_PREFERRED_HOST", "").strip()
    if preferred:
        print("  VAST_PREFERRED_HOST={} is pinning this host. Clear it, or".format(preferred))
        print("  set another id from `vast/orchestrator.py offers`.")
    else:
        print("  Re-run `start` to take the next offer, or pin a known-good")
        print("  host with VAST_PREFERRED_HOST.")
    sys.exit(1)


def _setup_and_start(instance: Dict[str, Any], timer: Optional[BootTimer] = None) -> Tuple[str, str]:
    """
    Clone the repo if missing, run startup.sh, launch the pipeline.

    Returns (tls_fingerprint, api_token).

    Every step is idempotent: the clone is skipped when the directory exists,
    startup.sh reuses an existing venv and the certificate is generated once
    and then reused, which is what keeps a `resume` from handing the desktop a
    fingerprint that no longer matches.
    """
    client = _connect_ssh(instance)
    # Before apt, pip and the weights: an unreachable host is worth five
    # seconds to detect and a full cold start to discover.
    _require_direct_reachable(instance)
    try:
        # Before anything else, because the clone below needs git and the base
        # image does not ship it. startup.sh installs what the *pipeline*
        # needs, but it arrives with the repo, so it cannot bootstrap the tools
        # used to fetch the repo. Idempotent, and a no-op on a warm disk.
        _ssh_run(
            client,
            "command -v git >/dev/null 2>&1 || { apt-get update -qq && "
            "apt-get install -y -qq git ca-certificates; }",
            "bootstrap",
        )

        repo_url = os.getenv("VAST_REPO_URL")
        if repo_url:
            _ssh_run(
                client,
                # Updated, not skipped. Skipping made every failed deploy
                # poison the next one: the clone that a crashed run left behind
                # was accepted as "present", so a fix pushed in between was
                # never fetched and the instance failed again identically. A
                # directory that is not a git checkout is a partial clone and
                # is replaced outright.
                "if [ -d {dir}/.git ]; then echo 'Repo present, updating.'; "
                "git -C {dir} fetch --prune origin && "
                "git -C {dir} reset --hard @{{u}}; "
                "else rm -rf {dir} && git clone --progress {url} {dir}; fi".format(
                    dir=_REMOTE_PHANTOM_DIR, url=repo_url),
                "git-clone",
            )
        else:
            _ssh_run(
                client,
                "[ -d {dir} ] || {{ echo 'ERROR: {dir} not found. "
                "Set VAST_REPO_URL in .env to auto-clone.'; exit 1; }}".format(
                    dir=_REMOTE_PHANTOM_DIR),
                "repo-check",
            )

        transcript = _ssh_run(client, "bash {}".format(_REMOTE_STARTUP), "startup")
        if timer is not None:
            timer.absorb(transcript)

        fingerprint = ""
        token = ""
        for line in transcript.splitlines():
            line = line.strip()
            if line.startswith("CERT_FINGERPRINT "):
                fingerprint = line.split()[-1]
            elif line.startswith("API_TOKEN "):
                token = line.split()[-1]

        if not fingerprint:
            print("ERROR: startup.sh did not report a certificate fingerprint.")
            print("  Without it the desktop cannot pin the connection, and an")
            print("  unpinned wss:// to a self-signed cert on a shared IP is")
            print("  not meaningfully better than cleartext. Refusing to continue.")
            sys.exit(1)

        if not token:
            # An empty token is not "no token" to the server — it is
            # authentication *disabled*, because that is how a local pipeline
            # stays usable without one. On a public IP that is an open socket
            # anyone who finds the port can start a stream on, and it would
            # look exactly like a working deploy.
            print("ERROR: startup.sh did not report an API token.")
            print("  An empty PHANTOM_API_TOKEN disables authentication rather")
            print("  than refusing connections, so this would deploy an open")
            print("  WebSocket on a public IP and report success.")
            sys.exit(1)

        _ssh_run(client, "pkill -f 'python.*pipeline.py' 2>/dev/null || true",
                 "kill-old-pipeline", check=False)

        # Sourced explicitly rather than inherited: startup.sh worked out the
        # cuDNN LD_LIBRARY_PATH in its own process, which is a child of this
        # one and cannot export back, and profile.d is read by login shells,
        # which exec_command does not open. Without this the pipeline starts
        # with no cuDNN on the loader path and ONNX quietly falls back to CPU —
        # on a GPU that is billing.
        launch = (
            "[ -f /etc/profile.d/cudnn.sh ] && . /etc/profile.d/cudnn.sh;"
            " {env}"
            " export PHANTOM_TLS_CERT={cert} PHANTOM_TLS_KEY={key};"
            " export PHANTOM_API_TOKEN={token};"
            " export VAST_API_KEY={vast_key} VAST_INSTANCE_ID={iid};"
            " export VAST_MAX_UPTIME={uptime} VAST_STOP_WARNING={warn};"
            " nohup {python} {pipeline} --execution-provider cuda"
            " > {log} 2>&1 &"
        ).format(
            env=_remote_env_exports(),
            cert=_REMOTE_CERT,
            key=_REMOTE_KEY,
            token=_shell_quote(token),
            vast_key=_shell_quote(os.getenv("VAST_SCOPED_API_KEY") or _api_key()),
            iid=_shell_quote(str(instance.get("id", ""))),
            uptime=_env_int("VAST_MAX_UPTIME", 120),
            warn=_env_int("VAST_STOP_WARNING", 5),
            python=_REMOTE_VENV_PYTHON,
            pipeline=_REMOTE_PIPELINE,
            log=_PIPELINE_LOG,
        )
        _ssh_run(client, launch, "pipeline-start",
                 secrets=[token, os.getenv("VAST_SCOPED_API_KEY") or "", _api_key()])
        print("\n  Pipeline started (log: {}).".format(_PIPELINE_LOG))
        print("  Read it with:  python vast/orchestrator.py logs")
        return fingerprint, token
    finally:
        client.close()


# ── Deploy ────────────────────────────────────────────────────────────────────

def _selection_settings() -> Dict[str, Any]:
    """Every knob `start` and `offers` both read, resolved once."""
    geo_raw = os.getenv("VAST_GEOLOCATIONS", _DEFAULT_GEOLOCATIONS)
    return {
        "geolocations": [g.strip().upper() for g in geo_raw.split(",") if g.strip()],
        "min_dlperf": _env_float("VAST_MIN_DLPERF", _DEFAULT_MIN_DLPERF),
        "min_vram": _env_int("VAST_MIN_VRAM", _DEFAULT_MIN_VRAM),
        "max_price": _env_float("VAST_MAX_PRICE", _DEFAULT_MAX_PRICE),
        "min_reliability": _env_float("VAST_MIN_RELIABILITY", _DEFAULT_MIN_RELIABILITY),
        "min_inet_up": _env_float("VAST_MIN_INET_UP", _DEFAULT_MIN_INET_UP),
        "min_ports": _env_int("VAST_MIN_PORTS", _DEFAULT_MIN_PORTS),
        "min_cpu_ghz": _env_float("VAST_MIN_CPU_GHZ", _DEFAULT_MIN_CPU_GHZ),
        "min_cpu_cores": _env_float("VAST_MIN_CPU_CORES", _DEFAULT_MIN_CPU_CORES),
        "max_compute_cap": _env_int("VAST_MAX_COMPUTE_CAP", _DEFAULT_MAX_COMPUTE_CAP),
        # Default OFF. Vast's "verified" is a datacenter badge, not a
        # measurement, and `reliability` already carries the measured signal.
        # Left on, it excluded the best box in the UK on every axis that
        # matters — $0.294 against $0.737, a 5.7GHz Ryzen against a 3.5GHz
        # EPYC, $1.70/month storage against $8.30 — for a reliability
        # difference of 0.993 against 0.997. The badge was costing 2.5x.
        "verified_only": _env_flag("VAST_VERIFIED_ONLY"),
        "disk": _env_int("VAST_DISK", _DEFAULT_DISK),
    }


def _find_offer() -> Dict[str, Any]:
    """
    Pick an offer, relaxing standards in bounded steps rather than waiting out
    a shortage — but never past the point where 20fps stops holding.

    Each pass walks `_RELAXATION_LADDER` from strictest to loosest and takes
    the first thing it finds, so a better box is always preferred when one
    exists and a worse one is only reached when nothing better does. Every rung
    still fits the frame inside 50ms; the ladder ends rather than continuing,
    because the target is the target.

    This replaces an unbounded `VAST_GPU_FALLBACK`, which dropped the speed
    floor to zero at the timeout and could therefore have rented a GTX 1080 at
    dlperf 3.4 — an instance that bills normally and misses the target by 10x.
    A bounded ladder is strictly better: it relaxes sooner, and it cannot
    relax into something that does not work.

    The preferred host is tried first on every pass, not just the first: it is
    the one whose disk is warm and whose address the desktop already has, so a
    pass that skipped it after one miss would skip the answer.
    """
    settings = _selection_settings()
    preferred_raw = (os.getenv("VAST_PREFERRED_HOST") or "").strip()
    preferred = int(preferred_raw) if preferred_raw.isdigit() else None
    wait = _env_int("VAST_GPU_WAIT", _DEFAULT_GPU_WAIT)

    # A measurement session wants one architecture or none: numbers from two
    # cards are not a comparison. Everything else wants a working session.
    relax = (os.getenv("VAST_RELAX") or "true").strip().lower() \
        not in ("0", "false", "no", "off")
    ladder = _RELAXATION_LADDER if relax else _RELAXATION_LADDER[:1]

    deadline = time.time() + wait
    announced = False

    while True:
        if preferred is not None:
            pinned = _search_offers(host_id=preferred, **settings)
            if pinned:
                offer = pinned[0]
                print("Preferred host {} has capacity.".format(preferred))
                # It cleared the delivery floors or it would not be here. But
                # it may still have degraded since it was chosen, and a warm
                # disk is not worth a card that is quietly half the one you
                # picked — so say so rather than let it pass as normal.
                if (offer.get("dlperf") or 0) < settings["min_dlperf"] or \
                        (offer.get("cpu_ghz") or 0) < settings["min_cpu_ghz"]:
                    print("  WARNING: it is below the preferred standard now -")
                    print("    {}".format(_describe_offer(offer)))
                    print("    wanted dlperf>={:.0f}, cpu>={:.1f}GHz".format(
                        settings["min_dlperf"], settings["min_cpu_ghz"]))
                    print("  Still inside the 20fps budget, so it is taken.")
                    print("  Clear VAST_PREFERRED_HOST to search instead.")
                return offer
            print("Preferred host {} has nothing rentable; searching.".format(preferred))

        for label, min_dlperf, min_cpu_ghz in ladder:
            rung = dict(settings)
            rung["min_dlperf"] = min(settings["min_dlperf"], min_dlperf) \
                if label != "preferred" else settings["min_dlperf"]
            rung["min_cpu_ghz"] = min(settings["min_cpu_ghz"], min_cpu_ghz) \
                if label != "preferred" else settings["min_cpu_ghz"]
            offers = _search_offers(**rung)
            if offers:
                if label != "preferred":
                    print("Nothing at the preferred standard; taking '{}'"
                          " (dlperf>={:.0f}, cpu>={:.1f}GHz).".format(
                              label, rung["min_dlperf"], rung["min_cpu_ghz"]))
                    print("  Still inside the 20fps budget - see"
                          " _RELAXATION_LADDER for the arithmetic.")
                return offers[0]

        if not announced:
            print("\nNothing matches at any rung. Waiting up to {}s, retrying"
                  " every {}s.".format(wait, _GPU_RETRY_INTERVAL))
            print("  Waiting is free - billing starts when an instance runs.")
            print("  Rungs tried: {}".format(
                ", ".join(name for name, _, _ in ladder)))
            announced = True

        if time.time() >= deadline:
            break
        time.sleep(min(_GPU_RETRY_INTERVAL, max(1.0, deadline - time.time())))

    print("\nERROR: no offer matched after {}s, at any rung.".format(wait))
    print("  See what is available:  python vast/orchestrator.py offers")
    print("  Widen VAST_GEOLOCATIONS, or raise VAST_MAX_PRICE. The speed floors")
    print("  are NOT the thing to lower by hand: the bottom rung is already the")
    print("  slowest hardware that holds 20fps.")
    sys.exit(1)


def _create_instance(offer: Dict[str, Any]) -> str:
    """Rent an offer and return the new instance id."""
    image = os.getenv("VAST_IMAGE") or "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
    disk = _env_int("VAST_DISK", _DEFAULT_DISK)

    # `env` carries environment variables and docker -p flags in one object;
    # the port entries are keys with "1" as their value, which is Vast's own
    # convention rather than a hack.
    env: Dict[str, str] = {
        "-p {p}:{p}".format(p=_PIPELINE_PORT): "1",
        "OPEN_BUTTON_PORT": str(_PIPELINE_PORT),
    }

    body = {
        "image": image,
        "label": "phantom",
        "disk": disk,
        # ssh_direct rather than ssh: the proxy variant is what made the RunPod
        # orchestrator so much larger than this one.
        "runtype": "ssh_direct",
        "env": env,
    }

    print("\nRenting: {}".format(_describe_offer(offer)))
    print("  image {} - {} GB disk".format(image, disk))
    bw = _hourly_bandwidth_cost(offer)
    print("  ${:.3f}/hr GPU + ~${:.3f}/hr bandwidth at the optimal preset".format(
        offer.get("dph_total") or 0.0, bw))
    print("  storage ${:.3f}/GB/month -> ~${:.2f}/month standing while stopped".format(
        offer.get("storage_cost") or 0.0, (offer.get("storage_cost") or 0.0) * disk))

    resp = _request("PUT", "/asks/{}/".format(offer["id"]), json=body)
    instance_id = resp.get("new_contract")
    if not instance_id:
        print("ERROR: create returned no contract id: {}".format(resp))
        sys.exit(1)
    print("  Instance {} created.".format(instance_id))
    # Written here, not at the end of a successful deploy. Billing starts
    # now, so this is the first moment the id can be lost - and every
    # failure after this line used to leave a running instance that `stop`
    # could not name. It happened twice in one session; both times the id
    # had to be read out of the scrollback by hand while the meter ran.
    _update_env_key("VAST_INSTANCE_ID", instance_id)
    return str(instance_id)


def _boot(instance_id: str, timer: Optional[BootTimer] = None) -> None:
    """Shared boot path: wait → setup → wait for the pipeline → update .env."""
    if timer:
        timer.phase("wait-for-running")
    instance = _wait_for_running(instance_id)

    if timer:
        timer.phase("remote-setup")
    fingerprint, token = _setup_and_start(instance, timer)

    # Re-read: the port map is only published on a running instance, and a
    # restart can hand back a different external port on the same host.
    instance = _get_instance(instance_id)
    ws_address = _ws_address(instance)
    if not ws_address:
        print("ERROR: instance publishes no mapping for port {}.".format(_PIPELINE_PORT))
        print("  Check that the instance was created with -p {p}:{p}.".format(p=_PIPELINE_PORT))
        sys.exit(1)

    if timer:
        timer.phase("pipeline-ready")

    # Pin first, then speak. Building the context proves the certificate before
    # a single byte of credential leaves this machine, and exits rather than
    # warning if it does not match.
    host, port_str = ws_address.rsplit(":", 1)
    context = _pinned_context(host, int(port_str), fingerprint,
                              time.time() + _PIPELINE_TIMEOUT)
    print("  Certificate matches the recorded fingerprint.")

    _wait_for_pipeline(ws_address, context, token)

    _update_env_key("PHANTOM_API_URL", "wss://{}/ws".format(ws_address))
    _update_env_key("PHANTOM_TLS_FINGERPRINT", fingerprint)
    if token:
        _update_env_key("PHANTOM_API_TOKEN", token)

    print("\nDone. Open the desktop:")
    print("  python desktop.py")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_start() -> None:
    """Rent a new instance and boot it."""
    _preflight_ssh_key()
    timer = BootTimer("Cold start")
    timer.phase("find-offer")
    offer = _find_offer()
    timer.phase("provision")
    instance_id = _create_instance(offer)
    _boot(instance_id, timer)
    timer.report()


def cmd_resume(instance_id: str) -> None:
    """
    Start a stopped instance, falling back to a new one if its host is full.

    A stopped instance keeps its disk but frees its GPU, so the host may have
    rented it out in the meantime. That is an ordinary outcome of resting
    rather than a fault, and the only thing that helps is a different host.

    The fallback is gated on the error actually being about capacity, and that
    gate is the point: falling back on *any* resume failure would rent a
    billing instance in response to a typo'd id or a rejected key.

    It costs more here than it did on RunPod, and the difference is worth
    stating. There, a new pod re-attached the network volume and came up warm.
    Here nothing is baked into an image and the disk does not survive, so this
    is a genuine cold start — venv, weights and all. That is the price of
    "stop/start only" in docs/VAST_MIGRATION.md, and it is why
    VAST_PREFERRED_HOST is worth setting.
    """
    _preflight_ssh_key()
    instance = _get_instance(instance_id)
    if not instance:
        print("ERROR: instance {} not found. It may have been destroyed.".format(instance_id))
        sys.exit(1)

    timer = BootTimer("Resume")
    timer.phase("resume")
    print("Starting instance {}...".format(instance_id))
    try:
        _request("PUT", "/instances/{}/".format(instance_id),
                 raise_on_error=True, json={"state": "running"})
    except VastAPIError as exc:
        if not _is_capacity_error(str(exc)):
            print("ERROR: resume failed: {}".format(exc))
            sys.exit(1)

        print("Resume failed: {}".format(exc))
        print("")
        print("  A stopped instance stays on the host it was rented from, and")
        print("  that host has no GPU free. Resume cannot move it; only a new")
        print("  rental can be placed somewhere with capacity.")
        print("")
        print("  Falling back to `start`. Nothing is baked into an image, so")
        print("  this is a COLD start: the venv and the model weights download")
        print("  again. Expect minutes, not seconds.")
        print("")
        print("  Instance {} is left stopped and still billing for its".format(instance_id))
        print("  disk. VAST_INSTANCE_ID is about to name the new one, so")
        print("  `terminate` will no longer reach it - destroy it at")
        print("  https://cloud.vast.ai/instances/ .")
        print("")
        cmd_start()
        return

    _boot(instance_id, timer)
    timer.report()


def cmd_stop(instance_id: str) -> None:
    """Stop the instance. The disk survives; storage keeps billing."""
    print("Stopping instance {}...".format(instance_id))
    _request("PUT", "/instances/{}/".format(instance_id), json={"state": "stopped"})
    disk = _env_int("VAST_DISK", _DEFAULT_DISK)
    print("Stopped. The disk survives, so the venv and models stay warm.")
    print("Storage still bills - roughly {} GB at this host's rate.".format(disk))
    print("To stop paying entirely:  python vast/orchestrator.py terminate")


def cmd_terminate(instance_id: str) -> None:
    """Destroy the instance. The disk goes with it."""
    print("This destroys instance {} and its disk.".format(instance_id))
    print("Nothing is baked into an image, so the venv, the model weights and")
    print("the repo all go, and the next `start` downloads them again.")
    confirm = input("Destroy it? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return
    _request("DELETE", "/instances/{}/".format(instance_id))
    # The id names nothing now. Left behind, it makes the next `start`
    # warn about an instance that cannot be looked up.
    _update_env_key("VAST_INSTANCE_ID", "")
    print("Instance destroyed. Billing has stopped.")


def cmd_status(instance_id: str) -> None:
    """Show instance state, GPU, cost, and the current address."""
    instance = _get_instance(instance_id)
    if not instance:
        print("Instance {} not found.".format(instance_id))
        return

    status = _instance_status(instance)
    print("Instance: {} ({})".format(instance.get("label") or "phantom", instance_id))
    print("Status:   {}".format(status))
    print("GPU:      {} x{}".format(instance.get("gpu_name", "?"), instance.get("num_gpus", 1)))
    print("Location: {}".format(instance.get("geolocation") or "?"))
    print("Cost:     ${:.4f}/hr".format(float(instance.get("dph_total") or 0.0)))
    up = instance.get("inet_up")
    if up:
        print("Uplink:   {:.0f} MB/s".format(float(up)))

    if status == "running":
        ws = _ws_address(instance)
        if ws:
            print("URL:      wss://{}/ws".format(ws))
        else:
            print("URL:      not published yet")
        # Every endpoint, not just the preferred one: a host can advertise a
        # direct port it refuses, and then the only line printed here is the
        # one that does not work.
        key_path = os.getenv("VAST_SSH_KEY_PATH", "~/.ssh/id_ed25519")
        for index, (shost, sport) in enumerate(_ssh_targets(instance)):
            label = "SSH:     " if index == 0 else "  or:    "
            print("{} ssh root@{} -p {} -i {}".format(label, shost, sport, key_path))
    else:
        print("URL:      not available (status: {})".format(status))


def _rejection_reasons(offer: Dict[str, Any], settings: Dict[str, Any]) -> List[str]:
    """
    Why this offer is not eligible, checking every floor the search applies.

    Every one, deliberately. An earlier version checked five of the eight and
    printed "compute cap" whenever none of those matched, which named the wrong
    reason with complete confidence — the failure this codebase refuses
    everywhere else. A floor with no message here is a floor nobody can
    diagnose, so the fallback says it does not know rather than guessing.
    """
    why: List[str] = []
    if (offer.get("dlperf") or 0) < settings["min_dlperf"]:
        why.append("dlperf {:.0f}".format(offer.get("dlperf") or 0))
    if (offer.get("dph_total") or 0) > settings["max_price"]:
        why.append("${:.2f}".format(offer.get("dph_total") or 0))
    if (offer.get("reliability2") or 0) < settings["min_reliability"]:
        why.append("rel {:.3f}".format(offer.get("reliability2") or 0))
    if (offer.get("inet_up") or 0) < settings["min_inet_up"]:
        why.append("up {:.0f}MB/s".format(offer.get("inet_up") or 0))
    if (offer.get("direct_port_count") or 0) < settings["min_ports"]:
        why.append("{} ports".format(offer.get("direct_port_count") or 0))
    if (offer.get("cpu_ghz") or 0) < settings["min_cpu_ghz"]:
        why.append("cpu {:.1f}GHz".format(offer.get("cpu_ghz") or 0))
    if (offer.get("cpu_cores_effective") or 0) < settings["min_cpu_cores"]:
        why.append("{:.0f} cpu cores".format(offer.get("cpu_cores_effective") or 0))
    if (offer.get("compute_cap") or 0) > settings["max_compute_cap"]:
        why.append("sm_{}".format(int((offer.get("compute_cap") or 0) / 10)))
    if (offer.get("gpu_ram") or 0) < settings["min_vram"] * 1024:
        why.append("{:.0f}GB VRAM".format((offer.get("gpu_ram") or 0) / 1024))
    if (offer.get("disk_space") or 0) < settings["disk"]:
        why.append("{:.0f}GB disk".format(offer.get("disk_space") or 0))
    if settings["verified_only"] and offer.get("verification") != "verified":
        why.append(str(offer.get("verification")))
    return why or ["refused, reason not among the floors this lists"]


def cmd_offers() -> None:
    """
    Show what is rentable, and what `start` would take.

    The equivalent of the RunPod `gpus` and `datacenters` commands, collapsed
    into one because on Vast they are the same question: an offer *is* a
    machine in a place at a price.

    Offers that fail only the quality floors are listed too, marked, because
    "why is there no instance" has to be answerable from this command.
    """
    settings = _selection_settings()
    preferred_raw = (os.getenv("VAST_PREFERRED_HOST") or "").strip()
    preferred = int(preferred_raw) if preferred_raw.isdigit() else None

    print("Searching Vast: {} - dlperf>={:.0f} - <=${:.2f}/hr - rel>={:.2f}"
          " - up>={:.0f} MB/s - cpu>={:.1f}GHz\n".format(
        ",".join(settings["geolocations"]) or "anywhere",
        settings["min_dlperf"], settings["max_price"],
        settings["min_reliability"], settings["min_inet_up"],
        settings["min_cpu_ghz"]))

    eligible = _search_offers(**settings)

    # The same geography with every quality floor dropped, so the listing can
    # say what the floors are actually costing.
    wide = dict(settings)
    wide.update({"min_dlperf": 0.0, "min_reliability": 0.0, "min_inet_up": 0.0,
                 "min_ports": 0, "verified_only": False, "max_price": 999.0,
                 "min_cpu_ghz": 0.0, "min_cpu_cores": 0.0})
    everything = _search_offers(**wide)

    # Compared on `machine_id`, not `id`. An offer's `id` is stable for a given
    # query but **not across queries**: the same machine came back as 43933077
    # from the filtered search and 43933078 from the wide one, so differencing
    # on it listed an eligible offer as refused and then could not say why.
    # `machine_id` is the host and does not move.
    #
    # The corollary matters more than the listing: an `id` is only valid for
    # the search that produced it, so `_find_offer` has to hand its own offer
    # dict straight to `_create_instance`. An id cached from an earlier query
    # would rent something else, or nothing.
    eligible_machines = {o.get("machine_id") for o in eligible}

    if eligible:
        print("Eligible - `start` takes the first of these:")
        header = "  {:<3} {:<14} {:<17} {:>7} {:>6} {:>5} {:>5} {:>7} {:>10}"
        print(header.format("", "GPU", "location", "$/hr", "dlperf",
                            "GHz", "cores", "$/mo st", "host"))
        for i, o in enumerate(eligible[:20]):
            mark = ">" if i == 0 else ""
            if preferred is not None and o.get("host_id") == preferred:
                mark = "P"
            print(header.format(
                mark, str(o.get("gpu_name"))[:14], str(o.get("geolocation"))[:17],
                "{:.3f}".format(o.get("dph_total") or 0.0),
                "{:.0f}".format(o.get("dlperf") or 0.0),
                "{:.1f}".format(o.get("cpu_ghz") or 0.0),
                "{:.0f}".format(o.get("cpu_cores_effective") or 0.0),
                "{:.1f}".format((o.get("storage_cost") or 0.0) * settings["disk"]),
                "{} {:.3f}".format(
                    str(o.get("verification") or "?")[:6],
                    o.get("reliability2") or 0.0),
            ))
        print("\n  > is what `start` takes; P marks VAST_PREFERRED_HOST.")
        print("  Ordered by country, then CPU band, then GPU band, then price.")
        print("  GHz is the compositor's speed - the one stage a faster GPU")
        print("  cannot help with. 'host' is Vast's badge plus MEASURED")
        print("  reliability; the badge is not filtered on, the number is.")
        print("  '$/mo st' is standing storage for {} GB while stopped.".format(settings["disk"]))
    else:
        print("Nothing is eligible right now. `start` would wait {}s and then {}.".format(
            _env_int("VAST_GPU_WAIT", _DEFAULT_GPU_WAIT),
            "relax through the ladder"
            if (os.getenv("VAST_RELAX") or "true").strip().lower()
            not in ("0", "false", "no", "off") else "fail"))

    rejected = [o for o in everything if o.get("machine_id") not in eligible_machines]
    if rejected:
        rejected.sort(key=lambda o: -(o.get("dlperf") or 0.0))
        print("\nIn range geographically, refused by a floor (top 12):")
        for o in rejected[:12]:
            why = _rejection_reasons(o, settings)
            print("    {:<16} {:<22} {:>8}  - {}".format(
                str(o.get("gpu_name"))[:16], str(o.get("geolocation"))[:22],
                "{:.3f}".format(o.get("dph_total") or 0.0),
                ", ".join(why)))

    print("\n{} eligible, {} refused, in {}.".format(
        len(eligible), len(rejected), ",".join(settings["geolocations"]) or "anywhere"))


def cmd_logs(instance_id: str, lines: int = 120) -> bool:
    """Print the tail of the pipeline log."""
    instance = _get_instance(instance_id)
    if not instance or _instance_status(instance) != "running":
        print("ERROR: instance is not running.")
        return False
    client = _connect_ssh(instance)
    try:
        _ssh_run(client, "tail -n {} {} 2>&1 || echo 'no log at {}'".format(
            lines, _PIPELINE_LOG, _PIPELINE_LOG), "pipeline-log", check=False)
        return True
    finally:
        client.close()


def cmd_run(instance_id: str, command: str) -> bool:
    """
    Run one command on the instance.

    The venv goes on PATH with `export` rather than as a `VAR=x cmd` prefix,
    which binds only to the first word of a line — so the second half of any
    `&&` chain ran under /usr/bin/python. That was a real bug on the RunPod
    side and is worth not reintroducing.
    """
    instance = _get_instance(instance_id)
    if not instance or _instance_status(instance) != "running":
        print("ERROR: instance is not running.")
        return False
    client = _connect_ssh(instance)
    try:
        wrapped = "export PATH=/workspace/venv/bin:$PATH && cd {} && {}".format(
            _REMOTE_PHANTOM_DIR, command)
        _ssh_run(client, wrapped, "run", check=False)
        return True
    finally:
        client.close()


def cmd_push(instance_id: str, local: str, remote: Optional[str] = None) -> bool:
    """Copy a local file to the instance over SFTP."""
    if not os.path.isfile(local):
        print("ERROR: {} not found.".format(local))
        return False
    instance = _get_instance(instance_id)
    if not instance or _instance_status(instance) != "running":
        print("ERROR: instance is not running.")
        return False

    target = remote or "/workspace/{}".format(os.path.basename(local))
    client = _connect_ssh(instance)
    try:
        sftp = client.open_sftp()
        size = os.path.getsize(local)
        print("  {} -> {} ({:.1f} MB)".format(local, target, size / 1e6))
        sftp.put(local, target)
        sftp.close()
        print("  Done.")
        return True
    except Exception as exc:
        print("ERROR: {}: {}".format(type(exc).__name__, exc))
        return False
    finally:
        client.close()


def cmd_pull(instance_id: str, remote: str, local: Optional[str] = None) -> bool:
    """
    Copy a file off the instance.

    This did not exist on RunPod and could not: its SSH proxy carries no SFTP,
    so a 45 KB montage of comparison frames could not be brought home and
    visual review was impossible rather than merely awkward. `ssh_direct` is a
    real sshd, so this is four lines.
    """
    instance = _get_instance(instance_id)
    if not instance or _instance_status(instance) != "running":
        print("ERROR: instance is not running.")
        return False

    target = local or os.path.basename(remote)
    client = _connect_ssh(instance)
    try:
        sftp = client.open_sftp()
        print("  {} -> {}".format(remote, target))
        sftp.get(remote, target)
        sftp.close()
        print("  Done ({:.1f} KB).".format(os.path.getsize(target) / 1e3))
        return True
    except Exception as exc:
        print("ERROR: {}: {}".format(type(exc).__name__, exc))
        return False
    finally:
        client.close()


def _warn_existing_instance(instance_id: str) -> None:
    """
    Confirm before `start` replaces the instance recorded in .env.

    VAST_INSTANCE_ID is the only place the existing instance's id is kept, so
    losing it puts that instance beyond the reach of stop, terminate and
    status — and if it is running it goes on billing with nothing left to
    report it.
    """
    try:
        instance = _get_instance(instance_id)
        status = _instance_status(instance) if instance else "not found"
    except SystemExit:
        status = "unknown"

    print("WARNING: VAST_INSTANCE_ID is already set ({}), status: {}.".format(
        instance_id, status))
    if status == "running":
        print("  That instance is RUNNING and billing right now.")
        print("  Stop it first:  python vast/orchestrator.py stop")
    print("  'start' rents a NEW instance and overwrites VAST_INSTANCE_ID,")
    print("  after which stop/terminate/status can no longer reach {}.".format(instance_id))
    print("  To boot the existing one instead: python vast/orchestrator.py resume")

    answer = input("\nProceed with a new instance? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted. Nothing was created and .env is unchanged.")
        sys.exit(0)

    # Repeated after the prompt on purpose: once .env is rewritten this line is
    # the only remaining copy of the old id.
    print("\nReplacing instance id. Previous: {} ({}).".format(instance_id, status))
    print("Recover it from https://cloud.vast.ai/instances/ if it needs stopping.\n")


def main() -> None:
    """Parse args and dispatch."""
    _console_safe()
    parser = argparse.ArgumentParser(
        description="Phantom Vast.ai Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  start        Rent an instance, set it up, start the pipeline, update .env
  resume       Start the stopped instance named by VAST_INSTANCE_ID
  stop         Stop it (disk survives, storage keeps billing)
  terminate    Destroy it (disk and models go too)
  status       State, GPU, cost, address
  offers       What is rentable now, and why anything eligible was refused
  logs [n]     Tail the pipeline log
  run "cmd"    Run one command on the instance
  push <local> [remote]
  pull <remote> [local]
        """,
    )
    parser.add_argument("command", choices=[
        "start", "resume", "stop", "terminate", "status", "offers",
        "logs", "run", "push", "pull"])
    parser.add_argument("paths", nargs="*", help="push/pull: <src> [dst]")
    args = parser.parse_args()

    instance_id = os.getenv("VAST_INSTANCE_ID") or None

    if args.command == "offers":
        cmd_offers()
        return
    if args.command == "start":
        if instance_id:
            _warn_existing_instance(instance_id)
        cmd_start()
        return

    if not instance_id:
        print("ERROR: VAST_INSTANCE_ID not set in .env")
        print("  Rent one first:  python vast/orchestrator.py start")
        sys.exit(1)

    if args.command == "run":
        if not args.paths:
            print("ERROR: run needs a command")
            print('  python vast/orchestrator.py run "nvidia-smi"')
            sys.exit(1)
        sys.exit(0 if cmd_run(instance_id, " ".join(args.paths)) else 1)
    elif args.command == "logs":
        count = int(args.paths[0]) if args.paths and args.paths[0].isdigit() else 120
        sys.exit(0 if cmd_logs(instance_id, count) else 1)
    elif args.command == "push":
        if not args.paths:
            print("ERROR: push needs a local file")
            print("  python vast/orchestrator.py push clip.mp4")
            sys.exit(1)
        remote = args.paths[1] if len(args.paths) > 1 else None
        sys.exit(0 if cmd_push(instance_id, args.paths[0], remote) else 1)
    elif args.command == "pull":
        if not args.paths:
            print("ERROR: pull needs a remote path")
            print("  python vast/orchestrator.py pull /workspace/report.json")
            sys.exit(1)
        local = args.paths[1] if len(args.paths) > 1 else None
        sys.exit(0 if cmd_pull(instance_id, args.paths[0], local) else 1)
    elif args.command == "resume":
        cmd_resume(instance_id)
    elif args.command == "stop":
        cmd_stop(instance_id)
    elif args.command == "terminate":
        cmd_terminate(instance_id)
    elif args.command == "status":
        cmd_status(instance_id)


if __name__ == "__main__":
    main()
