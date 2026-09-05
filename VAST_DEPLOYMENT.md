# Phantom — Vast.ai Deployment

One-time setup, the day-to-day loop, and a reference for every setting
`vast/orchestrator.py` reads.

Why Vast rather than RunPod is a measurement, not a preference:
[docs/VAST_MIGRATION.md](docs/VAST_MIGRATION.md). Short version — RunPod has
fifty datacenters and none in the UK, EU-FR-1 carries no eligible GPU at all,
and the operator is in West Africa. The felt problem was never compute.

## What does NOT belong on an instance

**OBS Studio and VB-Audio Cable belong on the operator machine.** The instance
is headless: it receives JPEG frames, swaps, and sends them back. It has no
virtual camera, no conferencing app and no audio path at all — audio is never
uploaded to it. Installing either driver on a rented GPU does nothing.

See [docs/SETUP_CHECKLIST.md](docs/SETUP_CHECKLIST.md).

## Where to start

Before opening an account, ask what is available:

```bash
pip install -r requirements-orchestrator.txt
python vast/orchestrator.py offers
```

Search needs no API key, so this works from a clean checkout. It prints what
`start` would rent, and — for everything it refuses — which floor it failed.
If that listing is empty, no amount of configuration helps and the answer is to
widen `VAST_GEOLOCATIONS`.

## The shape of the whole thing

    offers        what is rentable, and why anything was refused
      |
    start         rent -> ssh -> startup.sh -> pipeline -> write .env
      |
    (use it)      python desktop.py
      |
    stop          disk survives, GPU freed, storage keeps billing
      |
    resume        same host, warm disk, back in ~a minute
      |
    terminate     disk destroyed, billing ends completely

**`stop` and `resume` are the loop. `terminate` is leaving it.** That
distinction matters more here than it did on RunPod, where a network volume
outlived the pod and a fresh pod came up warm. Here **the instance disk is the
only copy** of the venv and the model weights, because nothing is baked into an
image. Destroying an instance means the next `start` downloads everything again.

The cost of staying in the loop is storage. Vast bills it per host, while the
instance exists, stopped or not — see `VAST_DISK` below.

> **Doing this for the first time?**
> [docs/VAST_MANUAL_SETUP.md](docs/VAST_MANUAL_SETUP.md) is the same ground as
> a tick-list, and it leads with the billing setting that protects the only
> copy of your model weights. This file is the reference; that one is the run.

## Part 1 — The account

### 1a — API key → `VAST_API_KEY`

<https://cloud.vast.ai/manage-keys/> → create a key. Paste it into `.env`.

### 1b — A scoped key → `VAST_SCOPED_API_KEY`

The instance needs a key of its own, so it can stop itself when
`VAST_MAX_UPTIME` expires. Leaving this empty forwards `VAST_API_KEY`, which
works and is worse: an account-wide key sitting on a rented machine can destroy
every other instance you own.

Vast supports scoped keys, so use one:

```bash
cat > perms.json <<'EOF'
{"api": {"instance_read": {}, "instance_write": {}}}
EOF
vastai create api-key --name phantom-instance --permission_file perms.json
```

This closes a risk `docs/ACCEPTED_RISKS.md` carried for the whole RunPod era.

### 1c — SSH key → `VAST_SSH_KEY_PATH`

Every deploy goes over SSH — the instance is a stock CUDA image that
`vast/startup.sh` builds on, so there is no image-only path.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

**Must be unencrypted**: the orchestrator loads it without a passphrase prompt.
Register the *public* half at <https://cloud.vast.ai/manage-keys/>.

On Windows, a key generated inside WSL lives in the WSL home, not the Windows
one:

```bash
wsl -e cp ~/.ssh/id_ed25519 /mnt/c/Users/<you>/.ssh/id_ed25519
```

### 1d — Pick a host → `VAST_PREFERRED_HOST`

Run `offers`, pick a row, and put its host id here. A pinned host keeps its IP
and its disk, so `resume` is warm and the desktop's address stops moving. The
filtered search still runs whenever that host has nothing rentable, so pinning
costs nothing when it is busy.

Weigh four things per host, all shown by `offers`:

- **`$/hr`** — the GPU
- **`GHz`** — the host CPU. See below; this is the column people skip and it
  matters more than the GPU column does
- **`up MB/s`** — the host's uplink. The app needs ~0.6, so anything here is
  plenty; a low number is a signal about the host, not a capacity limit
- **`$/mo st`** — standing storage while stopped, which varies by more than 3x
  across otherwise similar offers

### Why the CPU column matters more than it looks

The compositor is OpenCV on the CPU, and it is **the one stage a faster GPU
does not touch**: ~10.3ms of a ~29ms frame on a 4090 host, and ~20ms on an L4
host running identical models. On a fast card it is over a third of the frame.

Measured across western Europe on 2026-09-04, eligible offers ranged **3.5 to
6.0 GHz** — a 1.7x spread on latency-sensitive, lightly-threaded work. That is
roughly **4-5ms of frame time decided by which host you take**, which is more
than switching restoration models bought on the detect stage, and it is free to
choose.

**Desktop silicon wins; server silicon loses.** Ryzen 9 7900X at 5.7 GHz and
Core i9-13900KF at 6.0 beat EPYC 7663 at 3.5 and the H100 hosts at 2.0-2.4.
EPYC and Xeon trade clock for core count, and this workload wants the opposite.

So `offers` orders by **country, then CPU clock band, then price** — not by GPU
speed, because `VAST_MIN_DLPERF` has already removed everything below a 4090
and further GPU headroom goes unused at 20fps.

The clock is *banded* to 0.5 GHz rather than sorted raw, so a trivial
difference cannot override a large price gap. Band edges still decide the odd
pair unfairly; that is accepted, because the differences that matter here are
several bands wide.

At 20fps none of this is load-bearing — a 3.5 GHz host still holds the 50ms
deadline. It decides whether 30fps is reachable later, and it costs nothing to
prefer the better box today.

## Part 2 — `.env`

Copy `.env.example` to `.env` and fill in `VAST_API_KEY`. Everything else has a
working default. `VAST_INSTANCE_ID`, `PHANTOM_API_URL`,
`PHANTOM_TLS_FINGERPRINT` and `PHANTOM_API_TOKEN` are **written by the
orchestrator** — do not hand-edit them.

## Part 3 — The first run

```bash
python vast/orchestrator.py start
python desktop.py
```

`start` prints a phase breakdown. A first run is dominated by pip and the model
downloads; a resume skips both.

## Part 4 — The loop

```bash
python vast/orchestrator.py stop      # done for now
python vast/orchestrator.py resume    # back to it
```

### When `resume` cannot get a GPU

A stopped instance stays on its host — that is what keeps its disk — so
resuming needs a GPU free on that specific machine. If someone took it,
`resume` falls back to `start` automatically.

That fallback is **gated on the error actually being about capacity**. Falling
back on any resume failure would rent a billing instance in response to a
typo'd id or a rejected key.

Note what it costs here: a new host means a new disk, so it is a genuine cold
start. The old instance is left stopped and still billing for storage, and
`VAST_INSTANCE_ID` now names the new one — so destroy the old one by hand at
<https://cloud.vast.ai/instances/>.

### What resting costs

GPU billing stops. Storage does not. At `VAST_DISK=25` and a typical
$0.20–0.40/GB/month that is **$5–10 a month** to keep the models warm. If a
gap is going to be weeks rather than days, `terminate` and pay the cold start
later.

## Command reference

| Command | What it does |
|---|---|
| `offers` | What is rentable, what `start` would take, why anything was refused |
| `start` | Rent, set up, launch the pipeline, write `.env` |
| `resume` | Start the stopped instance; falls back to `start` if its host is full |
| `stop` | Stop it — disk survives, storage keeps billing |
| `terminate` | Destroy it — disk and models go too |
| `status` | State, GPU, location, cost, uplink, address |
| `logs [n]` | Tail the pipeline log |
| `run "cmd"` | Run one command on the instance, inside the venv |
| `push <local> [remote]` | Copy a file to the instance |
| `pull <remote> [local]` | Copy a file back |

`pull` did not exist on RunPod and could not: its SSH proxy carried no SFTP, so
a montage of comparison frames could not be brought home and visual review was
impossible rather than merely awkward. `runtype: ssh_direct` is a real sshd.

## Networking, and why it is `wss://`

RunPod terminated TLS at its proxy and handed out a stable hostname. Vast maps
port 9000 to a **random external port on a shared public IP** and terminates
nothing.

`desktop/controller.py` already speaks `ws://host:port/ws`, which is what makes
cleartext the dangerous option here rather than the broken one — it would work,
with the operator's face in it.

So:

- `vast/startup.sh` generates a self-signed certificate on the instance, once,
  and prints the SHA-256 of its DER encoding.
- The orchestrator writes that into `.env` as `PHANTOM_TLS_FINGERPRINT`.
- The desktop and the measurement tools pin it. Hostname and CA verification
  are off because there is no name to check and no CA that can vouch for an IP
  that moves with the host — the key itself is the identity.
- A random `PHANTOM_API_TOKEN` is generated with it, and the server drops any
  client that does not present it in the first frame.

A fingerprint mismatch is refused. If the instance was genuinely rebuilt, re-run
`start`; it rewrites both values.

## Configuring the run before it starts

Anything in the `# Pipeline settings` block of `.env` is copied into the
instance environment by the orchestrator, so `start` produces a configured
pipeline rather than a default one. The list is `_FORWARDED_ENV` in
`vast/orchestrator.py` — deliberately a list rather than "forward everything",
because the instance is a different machine and blanket-forwarding would send
local paths and secrets that mean nothing there.

`exec_command` opens no login shell, so those values are exported as part of
the launch command itself rather than left to `.bashrc`.

### Changing settings on an instance that already exists

`.env` reaches an instance only at launch. On a running one, use:

```bash
python tools/realism.py --host <ip> --port <port> enhancer_model=codeformer
python tools/realism.py --host <ip> --port <port> --show
python tools/stats.py   --host <ip> --port <port>
```

Take the host and port from `status`. All three read `PHANTOM_TLS_FINGERPRINT`
and `PHANTOM_API_TOKEN` out of `.env` automatically, so no extra flags are
needed against a live instance.

## `.env` reference — orchestrator settings

| Setting | Default | Meaning |
|---|---|---|
| `VAST_API_KEY` | — | Account key. Required for everything but `offers` |
| `VAST_SCOPED_API_KEY` | — | Key given to the instance so it can stop itself. Scope it |
| `VAST_INSTANCE_ID` | — | Written by `start`; names the instance the other commands act on |
| `VAST_GEOLOCATIONS` | `GB,IE,FR,NL,BE,DE` | Country codes, in priority order. The setting this migration exists for |
| `VAST_PREFERRED_HOST` | — | `host_id` to pin, for a stable IP and a warm disk |
| `VAST_MIN_DLPERF` | `90` | Vast's measured DL score. 90 is "4090 or better" |
| `VAST_MIN_VRAM` | `16` | GB |
| `VAST_MAX_PRICE` | `1.00` | $/hr, GPU only |
| `VAST_MIN_RELIABILITY` | `0.98` | Host reliability, 0–1 |
| `VAST_MIN_INET_UP` | `100` | MB/s. The app needs ~0.6; this is a proxy for a real connection |
| `VAST_MIN_PORTS` | `32` | `direct_port_count` — a host with too few cannot publish 9000 |
| `VAST_MIN_CPU_GHZ` | `3.5` | Host CPU clock. **The compositor is CPU-bound** — see below |
| `VAST_MIN_CPU_CORES` | `8` | Effective cores, i.e. this instance's *share* of the machine |
| `VAST_MAX_COMPUTE_CAP` | `900` | sm_90. Blackwell reports 1200 and would fail after billing started |
| `VAST_VERIFIED_ONLY` | `true` | Unverified hosts are self-reported |
| `VAST_GPU_WAIT` | `300` | Seconds to retry the search. Waiting is free |
| `VAST_RELAX` | `true` | Relax the speed floors in bounded steps when nothing matches. Every step still holds 20fps |
| `VAST_IMAGE` | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` | Stock base; startup.sh does the rest |
| `VAST_DISK` | `25` | GB. **A cost setting** — billed while stopped |
| `VAST_MAX_UPTIME` | `120` | Minutes before auto-stop. 0 disables |
| `VAST_STOP_WARNING` | `5` | Minutes of warning before that |
| `VAST_SSH_KEY_PATH` | `~/.ssh/id_ed25519` | Must be unencrypted |
| `VAST_REPO_URL` | — | Cloned on first deploy; include a token for a private repo |

## Auto-stop

The pipeline runs its own timer and **stops** the instance at
`VAST_MAX_UPTIME`, so an abandoned session is capped even with no desktop
connected. It broadcasts `auto_stop_warning` `VAST_STOP_WARNING` minutes
earlier; the desktop offers to extend, which sends `keep_alive`.

Stop rather than destroy, deliberately: the disk and the models survive, so the
next session resumes warm. It does not end storage billing — that is what
`terminate` is for.

## Troubleshooting

**`offers` lists nothing.** Widen `VAST_GEOLOCATIONS`, or lower
`VAST_MIN_DLPERF`. The refused list names the floor each offer failed.

**The pipeline never becomes healthy.** `python vast/orchestrator.py logs`
tails `/workspace/phantom-pipeline.log`, which is the only view of a pipeline
running under `nohup`. That works now — on RunPod it could not, because only
port 9000 was open and the proxy carried neither SFTP nor `exec_command`.

**Certificate fingerprint mismatch.** Either the instance was rebuilt — re-run
`start` — or something is in the middle of the connection. Do not disable the
pin.

**Everything runs but slowly, on CPU.** `python tools/stats.py` exits non-zero
when a requested accelerator is not the one loaded. ONNX Runtime falls back to
CPU silently; `pipeline/services/execution.py` raises rather than allowing it,
and `vast/startup.sh` fails if cuDNN cannot load. A pod on CPU bills a full GPU
hour and produces unusable output while appearing to work.
