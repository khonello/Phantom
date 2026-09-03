# Migrating from RunPod to Vast.ai

Research gathered before any code was changed. Everything with a number in it
was measured against the live API on 2026-09-03, not recalled.

## 1. Why — the measurement that settles it

The felt problem is latency, not compute. `gpen_bfr_256` put the frame at ~27ms
against a 50ms deadline and **the stutter went away while the lag did not**. The
remaining term is the ~350ms round trip to EU-RO-1 in Romania, against an
operator in West Africa.

The obvious first answer — move to a nearer RunPod datacenter — is dead.
RunPod's GraphQL *does* support per-datacenter stock (`lowestPrice(input:
{dataCenterId})`, contradicting the note in CLAUDE.md), and asking it gives:

| RunPod datacenter | eligible GPUs in stock |
|---|---|
| EU-FR-1 (France) | **none** |
| EU-NL-1 (Netherlands) | L40S only, `Low`, $0.79/hr |
| EU-CZ-1 (Czechia) | RTX 4090 `Low`, RTX 3090 `Low` |
| EU-RO-1 (Romania) | RTX 4090 `Medium`, L4 `Low` |

RunPod has **50 datacenters and none in the UK**. EU-RO-1 is not an accident —
it is the only European datacenter with non-`Low` 4090 stock.

Vast, same day, filtered to `compute_cap <= 900`, `verified`, `reliability >=
0.98`, `inet_up >= 100 MB/s`, `direct_port_count >= 32`, single GPU:

| GPU | Location | $/hr | uplink | reliability | ports | volume |
|---|---|---|---|---|---|---|
| **RTX 4090** | **United Kingdom** | **0.311** | 885 MB/s | 0.982 | 199 | 515 GB |
| RTX 4090 | France | 0.336 | 876 MB/s | 0.997 | 249 | 734 GB |
| RTX 4090 | Germany | 0.348 | 255 MB/s | 0.981 | 62 | 1499 GB |
| RTX 4090 | France | 0.361 | 779 MB/s | 0.998 | 99 | 490 GB |
| RTX 4090D | Netherlands | 0.401 | 14928 MB/s | 0.998 | 99 | 3218 GB |
| RTX 3090 | Belgium | 0.188 | 890 MB/s | 0.999 | 249 | 268 GB |
| A10 | Netherlands | 0.243 | 1678 MB/s | 1.000 | 249 | 1050 GB |

The lead candidate (`host_id=135666`, UK, verified, `static_ip=True`) is
**cheaper than the Romanian 4090 at $0.34/hr** and roughly 2000km closer, on a
7 Gbps uplink against an application that needs 4.8 Mbps.

The `docs/PRODUCTION.md` claim that Vast means "slight instability" was written
2026-03-10, before any of these filters were applied. `reliability >= 0.98` and
`verified` are the answer to it, and they still leave six 4090s in western
Europe.

**What is still unmeasured: the actual RTT.** Everything above is a proxy for
it. Rent the UK host for ten minutes and measure before believing any of this.

## 2. What Vast gives us that RunPod does not

Four of these close problems already recorded in this repo.

- **`runtype: "ssh_direct"` is real SSH.** Port 22 provisioned on the instance,
  no proxy. That kills two recorded gotchas at once: `exec_command` works
  (CLAUDE.md records RunPod's proxy "silently drops commands ... must use
  `invoke_shell()`"), and **SFTP works**, so the missing
  `orchestrator.py pull` becomes trivial. "Nothing can be copied off the pod"
  stops being true, and visual review of `--debug-frames` output becomes
  routine instead of impossible.
- **Scoped API keys.** `create_api_key(name=..., permission_file=...)` with a
  permission set like `{"api": {"instance_read": {}, "instance_write": {}}}`.
  The instance needs a key only to stop itself at `MAX_UPTIME`; today it carries
  an account-wide `RUNPOD_API_KEY` that could terminate every other pod, which
  is an open risk in `docs/ACCEPTED_RISKS.md`. This closes it.
- **`dlperf` is a real measured performance score on every offer.** It replaces
  the hand-maintained `_GPU_PERF` table and `tests/test_gpu_tier.py`, which pin
  a ranking somebody typed in.
- **The search API filters on the thing that is actually our bottleneck.**
  `inet_up`, `inet_down`, `geolocation`, `reliability`, `direct_port_count`,
  `static_ip`, `compute_cap`, `cuda_max_good` are all server-side filters. The
  RunPod orchestrator had to fetch a global catalogue and filter locally, and
  could not filter on network quality at all.

## 3. What Vast takes away — the hard parts

### 3.1 No TLS proxy. This is the significant one.

RunPod gives `wss://{pod_id}-9000.proxy.runpod.net/ws` — a stable TLS hostname.
Vast gives a **random external port on a shared public IP**, discovered from the
instance object's `ports` map:

    "public_ipaddr": "192.0.2.45",
    "ports": {"9000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}

`desktop/controller.py` already handles `ws://host:port/ws`, so this *works*
with no desktop change — and that is the trap. It would move the operator's
face from encrypted-but-unauthenticated to **cleartext and unauthenticated**,
over the public internet. `docs/ACCEPTED_RISKS.md` accepts the missing auth
partly because "the proxy URL is pod-specific and unguessable"; neither half of
that survives a bare IP and port.

So TLS has to be solved as part of the migration rather than after it. Three
options, in order of preference:

1. **Self-signed cert generated per instance, fingerprint pinned in `.env`.**
   The orchestrator already writes `PHANTOM_API_URL`; it writes the fingerprint
   beside it. No external dependency, no added hop, no added latency. Pair it
   with a token in the first frame and the unauthenticated-WebSocket risk
   closes too.
2. Cloudflare Tunnel — gives a stable TLS hostname free, but routes through
   Cloudflare's edge, which is a hop added to the one number this migration
   exists to reduce. Rejected unless measurement says otherwise.
3. Plain `ws://` plus a token — fixes control, not confidentiality. Not
   acceptable for video of a person's face.

### 3.2 Volumes are machine-locked

> "A volume is physically tied to the machine it was created on ... can only be
> attached to instances running on the same physical machine."

There is no equivalent of a datacenter-local network volume that any new pod can
mount. The warm-volume model has to be replaced. Two viable shapes:

- **Stop/start one long-lived instance.** A stopped Vast instance keeps its
  disk and bills storage only, which is the same shape as a stopped RunPod pod.
  Models stay warm. The risk is that stopping frees the GPU, so a busy host may
  not let you start again.
- **Bake the weights into a Docker image.** Removes the volume question
  entirely and makes any host a warm host. Costs an image build and a pull on
  first use per host.

Recommendation: both. Stop/start as the fast path, a baked image as the fallback
that makes host loss cheap instead of fatal.

### 3.3 Storage is billed while stopped, and it is not cheap

`storage_cost` is per-host. The UK candidate charges **$0.40/GB/month** against
RunPod's ~$0.07. Today's config allocates 40 GB (`RUNPOD_CONTAINER_DISK=20` +
`RUNPOD_VOLUME_DISK=20`), which would be **$16/month standing** on that host.
The French candidate is $0.20 and one German host $0.13.

Actionable: keep the disk at what the venv and weights actually need (~25 GB),
and treat `storage_cost` as a selection criterion, not a detail.

### 3.4 Bandwidth is metered

RunPod does not bill it; Vast does, per host. At the `optimal` preset the stream
is ~600 KB/s each way, so ~2.16 GB/hour each way. On the UK candidate
($0.0039/GB up, $0.0026/GB down) that is **~$0.014/hr**, about 4% on top of the
GPU. Real but negligible. Worth recomputing for `production`, and worth
avoiding the one candidate at $0.0078/GB.

### 3.5 Host quality varies, and the filters are the mitigation

Not a reason to avoid Vast — a reason the search query is load-bearing. The
minimum bar this project needs: `verified`, `reliability >= 0.98`,
`inet_up >= 100`, `direct_port_count >= 32`, `compute_cap <= 900`,
`geolocation in [GB, IE, FR, NL, BE, DE]`.

## 4. Migration surface

`grep -ril runpod` outside the venvs hits **47 files**. By coupling depth:

### Rewrite

| File | refs | Note |
|---|---|---|
| `runpod/orchestrator.py` | 168 | 2158 lines, becomes `vast/orchestrator.py` |
| `.env.example` | 37 | 20 `RUNPOD_*` vars become `VAST_*` |
| `tests/test_wiring.py` | 26 | asserts `RUNPOD_DEPLOYMENT.md` stays in step with code |
| `pipeline/api/server.py` | 16 | auto-stop calls `runpod.stop_pod()` |
| `runpod/startup.sh` | 13 | mostly generic Linux setup; paths and messages |
| `tests/test_gpu_tier.py` | 10 | pins `_GPU_PERF` tier; `dlperf` replaces it |
| `requirements-orchestrator.txt` | 3 | `runpod>=1.6.0` becomes `vastai` |

### Rename/comment only — no logic change

`pipeline/services/{face_detection,face_swapping,masking,enhancement,execution,
templates}.py`, `pipeline/io/ffmpeg.py`, `pipeline/services/onnx_session.py`,
`pipeline/core.py`, `pipeline/processing/pipeline.py`. All of these key off
**`/workspace`**, which is the convention on Vast images too. They say "RunPod
Network Volume" in comments and constants like `_RUNPOD_CACHE`; the paths
themselves are already right.

### Docs

`RUNPOD_DEPLOYMENT.md`, `runpod/TROUBLESHOOTING.md`, `docs/PRODUCTION.md`,
`docs/ACCEPTED_RISKS.md`, `docs/SETUP_CHECKLIST.md`, `docs/LOCAL_GPU_SETUP.md`,
`CLAUDE.md`, `README.md`, and the per-datacenter-filtering claim that this
research falsified.

## 5. API reference gathered

Search is **unauthenticated**; everything else needs a key.

    POST   https://console.vast.ai/api/v0/bundles/       # search offers
    PUT    https://console.vast.ai/api/v0/asks/{id}/     # create instance
    GET    https://console.vast.ai/api/v0/instances/     # list
    GET    https://console.vast.ai/api/v0/instances/{id}/
    PUT    https://console.vast.ai/api/v0/instances/{id}/    # stop/start
    DELETE https://console.vast.ai/api/v0/instances/{id}/    # destroy

Format traps found by probing, both of which silently return zero offers:

- `gpu_name` uses **spaces**: `"RTX 4090"`, not `"RTX_4090"` (the CLI converts;
  the raw API does not).
- `geolocation` values are full strings like `"United Kingdom, GB"`, but the
  filter matches on the **country code** — `{"in": ["GB"]}` works.

Create body:

    {
      "image": "...", "label": "phantom", "disk": 25,
      "runtype": "ssh_direct",
      "env": {"-p 9000:9000": "1", "PHANTOM_...": "..."},
      "onstart": "bash /workspace/phantom/vast/startup.sh"
    }

SDK: `pip install vastai` gives both CLI and `from vastai import VastAI`, with
`search_offers`, `create_instance`, `show_instance`, `start_instance`,
`stop_instance`, `destroy_instance`, `ssh_url`, `copy`, `create_api_key`.

Note `env` carries **both** environment variables and docker `-p` port flags,
and that under `ssh_direct` the image's entrypoint is replaced — `onstart` is
where the pipeline gets launched.

## 6. Decisions taken

1. **TLS: self-signed certificate, fingerprint pinned.** The instance generates
   a cert at startup and the orchestrator writes its fingerprint into `.env`
   beside `PHANTOM_API_URL`; the desktop pins it. No third party in the path,
   so nothing is added to the number this migration exists to reduce. A token
   in the first frame goes in at the same time, which closes the
   unauthenticated-WebSocket risk rather than deepening it.
2. **Warm models: stop/start only.** No baked image for now. A stopped instance
   keeps its disk, which is the same shape as a stopped RunPod pod, so the
   resume path carries over directly. The accepted risk is that stopping frees
   the GPU and a busy host may refuse to start it again — mitigated by (3)
   rather than by an image build.
3. **Host policy: pin preferred, search as fallback.** `VAST_PREFERRED_HOST`
   names the chosen UK host, for a stable IP and a warm disk. When it is
   unavailable the filtered search runs, which is the same shape as the
   existing bounded wait in `_try_deploy_pass` — and the reasoning that put it
   there holds here too: billing starts when an instance runs, not while you
   are waiting for one.
4. **Auto-stop: stop, not destroy.** Matches today's behaviour and keeps the
   models warm. It keeps paying storage, which is why §3.3's disk sizing is not
   a detail: at 25 GB on the lead candidate that is ~$10/month standing.

Consequence worth stating plainly: (2) and (4) together mean **the disk is the
only copy of the warm state**, and destroying the instance re-downloads
everything. That is the trade taken for not maintaining an image build.
