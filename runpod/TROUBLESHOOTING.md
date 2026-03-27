# RunPod Orchestrator — Troubleshooting & Lessons Learned

This documents every issue encountered while building the orchestrator and how each was resolved.
Refer to this before debugging RunPod integration issues.

---

## 1. GPU Names: Display Name vs API ID

**Problem**: `runpod.create_pod(gpu_type_id=...)` expects the GPU **ID** (e.g. `NVIDIA GeForce RTX 4090`), not the **display name** (e.g. `RTX 4090`). Passing the display name returns: `No GPU found with the specified ID`.

**What we expected**: The display name shown in the RunPod dashboard and returned by `gpuTypes { displayName }` would work as the `gpu_type_id`.

**What actually happens**: The `gpu_type_id` parameter requires the `id` field from the GraphQL `gpuTypes` query, which is a longer string like `NVIDIA GeForce RTX 4090` or `NVIDIA RTX A4500`.

**Solution**: Query `gpuTypes { id displayName }` via GraphQL, build a `displayName → id` mapping, and pass the `id` to `create_pod()`. The `.env` file uses display names (shorter, human-readable) and the orchestrator resolves them to IDs at deploy time.

**Key mapping examples**:
| Display Name (in .env) | API ID (for create_pod) |
|------------------------|------------------------|
| RTX 4090               | NVIDIA GeForce RTX 4090 |
| RTX A4500              | NVIDIA RTX A4500        |

Run `python runpod/orchestrator.py gpus` to see all display names and their IDs.

---

## 2. GraphQL API: What Works and What Doesn't

**Introspection is blocked**: RunPod's GraphQL API returns 400 for `__type` introspection queries. You cannot discover the schema dynamically.

**Per-datacenter GPU filtering doesn't exist**: There is no `gpuTypes(input: { dataCenterId: ... })` parameter. The query `gpuTypes { displayName }` returns all GPUs globally — there is no server-side datacenter filtering.

**What works**:
```graphql
# List all GPU types (id + display name)
query { gpuTypes { id displayName } }

# List all datacenters
query { dataCenters { id name location } }

# Get pod details including SSH info (machine.podHostId)
query Pod($podId: String!) {
  pod(input: { podId: $podId }) {
    id machineId
    machine { podHostId gpuDisplayName }
    runtime { ports { ip isIpPublic privatePort publicPort } }
  }
}
```

**What returns 400**:
```graphql
# Schema introspection
query { __type(name: "GpuType") { fields { name } } }

# Datacenter-filtered GPU query
query { gpuTypes(input: { dataCenterId: "EU-RO-1" }) { displayName } }
```

---

## 3. SSH Access: The `podHostId` Field

**Problem**: SSH authentication failed with `AuthenticationException: transport shut down or saw EOF`.

**Root cause**: The SSH username for RunPod's proxy is `{pod_id}-{numeric_id}@ssh.runpod.io`, but we were using `{pod_id}-{machineId}` where `machineId` is an alphanumeric string (e.g. `bfarqx9slmpp`) — not the numeric ID the SSH proxy expects (e.g. `64411247`).

**Where the correct SSH username lives**:
- **NOT** `pod["machineId"]` — this returns `bfarqx9slmpp` (internal machine identifier)
- **NOT** `pod["machine"]["podHostId"]` via `runpod.get_pod()` — the Python SDK doesn't return this field
- **YES** `pod.machine.podHostId` via **direct GraphQL query** — returns `{pod_id}-{numeric_id}` (e.g. `5bwt4ynkuk28ve-64411247`)

**Solution**: Query GraphQL directly for `machine { podHostId }` and use the full value as the SSH username:
```
ssh {podHostId}@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Important**: The `runpod` Python SDK's `get_pod()` returns a subset of fields. It does NOT include `machine.podHostId`. You must use a direct GraphQL query to get it.

---

## 4. Pod Stuck on Pending / Slow Scheduling

**Problem**: Pod created successfully via API but stays in pending state and never gets assigned ports. Dashboard deploys for the same GPU work instantly.

**Causes found**:

### 4a. `support_public_ip=True` constrains scheduling
Setting `support_public_ip=True` forces RunPod to schedule only on nodes with spare public IPs — much more constrained than the dashboard default. Pods can sit in pending for a very long time or never schedule.

**Solution**: Set `support_public_ip=True` only for SSH mode (needs direct port for SSH), `False` for Docker mode (uses proxy URLs). Even with SSH mode, the pod may be slower to schedule than dashboard deploys.

**Current behavior**: SSH mode uses `support_public_ip=True` because SSH port mapping requires a public IP assignment. Docker mode uses `False`.

### 4b. Requesting both `volume_in_gb` and `network_volume_id`
Passing both parameters may conflict. If you have a network volume, don't also request a local volume.

**Solution**: Pass `network_volume_id` OR `volume_in_gb`, never both.

### 4c. Exposing port 8888/http triggers JupyterLab
Exposing `8888/http` causes RunPod to initialize JupyterLab, which adds startup time and shows a "JupyterLab initializing — this is taking longer than expected" warning.

**Solution**: Only expose `9000/tcp` (the pipeline WebSocket port). We don't need JupyterLab.

---

## 5. Docker Image: `runtime` Tag Doesn't Exist

**Problem**: Pod logs show `manifest unknown` error when pulling the image.

**Root cause**: We changed the image from `devel` to `runtime` in commit `a00bf23` to reduce container disk size, but RunPod does not publish a `runtime` variant of `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-*-ubuntu22.04`.

**Available tags for CUDA 12.4 + Python 3.11 + Ubuntu 22.04**:
- `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (7.1 GB) — **only option**
- No `runtime`, `cudnn-runtime`, or `base` variant exists

**Newer tag format** (v0.7.0+) doesn't use devel/runtime:
- `runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2204` (8.0 GB)

**Solution**: Use `devel`. Despite the name, it's actually the smallest option (7.1 GB). The "devel" label means it includes CUDA compiler toolchains we don't need, but there's no lighter alternative. This is also the image RunPod uses by default for RTX A4500 in the dashboard.

---

## 6. WebSocket URL: Proxy vs Direct IP

**Problem**: Without a public IP, there's no `ip:port` for the WebSocket connection.

**RunPod provides two connection methods**:

| Method | Format | When available |
|--------|--------|----------------|
| Proxy  | `{pod_id}-9000.proxy.runpod.net` (port 443, wss://) | Always, once pod is RUNNING |
| Direct | `{public_ip}:{public_port}` (any port, ws://) | Only with `support_public_ip=True` AND port exposed |

**Solution**: Use proxy URLs by default (`wss://{pod_id}-9000.proxy.runpod.net/ws`). The `_wait_for_pipeline` function handles both formats — if the address contains `:` with a numeric port suffix, it connects directly; otherwise it connects to port 443 (proxy HTTPS).

---

## 7. `start` vs `resume` — Command Separation

**Problem**: The original `start` command tried to resume an existing pod if `RUNPOD_POD_ID` was set, making it impossible to deploy a fresh pod without first clearing the ID.

**Solution**: Split into two commands:
- `start` — always deploys a **new** pod, writes new ID to `.env`
- `resume` — resumes the pod in `RUNPOD_POD_ID`, requires it to be set

---

## 8. Exception Handling: Never Swallow Errors

**Rule**: Every `except` block must print the error. Silent exception handling hides the root cause and makes debugging impossible.

**Examples of silent handlers that caused problems**:
- GraphQL query returning 400 was caught and returned `[]` with a vague "query may have failed" message — hid the fact that the field names were wrong
- Socket connection failures during polling were silently retried — hid network issues

**Pattern to follow**:
```python
except Exception as exc:
    print("ERROR: description of what failed: {}".format(exc))
```

---

## 9. `runpod.get_pod()` vs GraphQL Direct Query

The `runpod` Python SDK's `get_pod()` returns a **subset** of pod fields. Key fields it does and does NOT include:

**Included by `get_pod()`**:
- `id`, `machineId`, `desiredStatus`, `name`
- `machine.gpuDisplayName`
- `runtime.ports[].ip`, `runtime.ports[].isIpPublic`, `runtime.ports[].privatePort`, `runtime.ports[].publicPort`
- `costPerHr`, `uptimeSeconds`, `gpuCount`, `vcpuCount`, `memoryInGb`

**NOT included by `get_pod()` — requires direct GraphQL**:
- `machine.podHostId` (needed for SSH proxy username)

When you need a field not in the SDK response, query GraphQL directly at `https://api.runpod.io/graphql` with `Authorization: Bearer {api_key}`.

---

## Quick Reference: Working .env Configuration

```env
# Datacenters with paired network volumes (tried in order)
RUNPOD_DATACENTERS=EU-RO-1:z8now7p5ts

# GPU auto-discovery: min VRAM and max hourly price
RUNPOD_MIN_VRAM=16
RUNPOD_MAX_PRICE=1.00
# Or manual override (optional):
# RUNPOD_GPU_TYPES=RTX 4090,RTX 3090

# Image must be devel — runtime tag doesn't exist
RUNPOD_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Auto-stop: stop pod after N minutes (0 = disabled), warning M minutes before
RUNPOD_MAX_UPTIME=120
RUNPOD_STOP_WARNING=5

# SSH key must be uploaded to RunPod dashboard → Settings → SSH Public Keys
RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519
```

## Quick Reference: Commands

```bash
python runpod/orchestrator.py start        # deploy fresh pod
python runpod/orchestrator.py resume       # resume stopped pod
python runpod/orchestrator.py stop         # pause pod
python runpod/orchestrator.py terminate    # delete pod
python runpod/orchestrator.py status       # show pod info
python runpod/orchestrator.py gpus         # list GPUs with VRAM, pricing, eligibility
python runpod/orchestrator.py datacenters  # list datacenters
```
