# Vast.ai Orchestrator — Troubleshooting & Lessons Learned

Every entry here was hit, not anticipated. The RunPod version of this file was
13 KB of GraphQL trivia that stopped applying the day the provider changed;
this one is deliberately short, and anything that goes stale should be deleted
rather than left to mislead.

## 1. The search API fails silently on two format mistakes

Both return `HTTP 200` with **zero offers**, which reads as "nothing is
available" rather than "your query is wrong". This cost the first hour.

**`gpu_name` uses spaces, not underscores.** The CLI accepts `RTX_4090` and
converts it; the REST API does not.

```json
{"gpu_name": {"in": ["RTX 4090"]}}     // correct
{"gpu_name": {"in": ["RTX_4090"]}}     // 200 OK, zero offers
```

The orchestrator does not filter on `gpu_name` at all — `dlperf` and
`compute_cap` say what is actually meant, and a name list goes stale.

**`geolocation` values are labels, but the filter matches country codes.** An
offer's field reads `"United Kingdom, GB"`; the filter takes `GB`.

```json
{"geolocation": {"in": ["GB", "FR"]}}  // correct
```

`_country()` exists to read the code back out of the label, because ranking on
the whole string would bucket every offer as unknown and silently turn the
country priority back into price ordering.

## 2. An offer `id` is not stable across queries

The same physical machine came back as `43933077` from the filtered search and
`43933078` from a wider one, seconds apart.

Two consequences:

- **Comparing two result sets must use `machine_id`.** Differencing on `id`
  reported an eligible offer as refused, and then could not say which floor it
  had failed — because it had failed none.
- **An `id` is only valid for the search that produced it.** `_find_offer`
  hands its own offer dict straight to `_create_instance` for this reason. An
  id cached from an earlier query rents something else, or nothing.

## 3. Search needs no API key; everything else does

`POST /api/v0/bundles/` answers unauthenticated. That is why
`vast/orchestrator.py offers` works from a clean checkout with no account —
which is the command someone runs to decide whether to open one.

Every other endpoint needs `Authorization: Bearer`. `_request(auth=False)` is
for `/bundles/` alone; elsewhere a missing key must be an error rather than an
empty result that reads like "nothing available".

## 4. Volumes are locked to one machine

> "A volume is physically tied to the machine it was created on ... can only be
> attached to instances running on the same physical machine."

There is no equivalent of a datacenter-local network volume. This is why the
deployment is stop/start on one instance rather than "rent anywhere, mount the
models": **the instance disk is the only copy**, and `terminate` destroys it.

Storage also bills while the instance is stopped, per host, at rates up to
$0.40/GB/month — several times what RunPod charged. `VAST_DISK` is a cost
setting.

## 5. There is no TLS proxy, and the failure mode is that it works

Vast maps port 9000 to a random external port on a shared public IP and
terminates nothing. `desktop/controller.py` already speaks `ws://host:port/ws`.

So the naive port produces a **working** connection carrying the operator's
face in cleartext. Nothing errors, nothing warns, and the swap looks correct.

The instance therefore generates a self-signed certificate and the orchestrator
pins its fingerprint. Two details that are easy to get wrong:

- **Fingerprint the DER form, not the PEM.** Python's
  `getpeercert(binary_form=True)` returns DER, so `openssl x509 -outform DER |
  sha256sum` is what matches. Hashing the PEM produces a pin that never
  matches — and a pin that always fails gets disabled, which is worse than
  never having had one.
- **Generate the certificate once.** A cert regenerated per boot hands the
  desktop a fingerprint that stops matching after any restart, which reads as
  an attack rather than a restart.

For the async tools, the certificate is loaded as its own trust anchor
(`load_verify_locations(cadata=...)`) rather than compared after connecting.
Reaching the peer certificate through `websockets` means a different private
attribute on each client implementation, so the check silently becomes a no-op
on a version bump.

## 5b. cuDNN, and why a warning is not enough

*(Carried over from the RunPod file. It is about ONNX Runtime, not the
provider, so it still applies.)*

`onnxruntime-gpu` needs `libcudnn.so.9`, which most base images do not carry,
and **ONNX Runtime does not error when a provider cannot initialise — it
silently uses CPU.** Every model that decides how the output looks is ONNX, so
that fallback is seconds per frame on a GPU that is billing.

Three things must stay hard failures. Do not downgrade any of them to a
warning:

- `pipeline/services/execution.py::verify` raises `ExecutionProviderError`.
- `vast/startup.sh` exits non-zero if cuDNN still will not load after install.
- `tools/stats.py` exits non-zero when the requested accelerator is not the one
  loaded.

ONNX Runtime already emits a warning, and that warning is exactly what let this
ship broken once. The value is that it halts.

The related trap: `insightface` depends on the CPU `onnxruntime`, and both
wheels write the same `onnxruntime/` directory, so the CPU one lands last and
shadows the GPU build. The repair keys off `get_available_providers()` rather
than off the uninstall, because removing the CPU wheel deletes files that were
the GPU build's own.

## 6. `exec_command` and SFTP both work — use them

`runtype: "ssh_direct"` provisions port 22 on the instance itself. This is the
single largest simplification against RunPod, whose proxy accepted
`exec_command` and silently ran nothing, forcing ~150 lines of interactive
shell driving with a sentinel echo to recover the exit code.

It also means files can be copied **off** the instance. On RunPod they could
not, so a montage of comparison frames could not be brought home and visual
review was impossible rather than merely awkward. That is what `pull` is.

Use `ssh_direct`, not `ssh` — the latter is an alias for `ssh_proxy` and gives
back the RunPod situation.

## 7. Ports are published before anything listens on them

An instance reports `running` and exposes its port map before sshd is up and
well before the pipeline has bound 9000. So:

- The SSH connect retries rather than failing on the first refusal.
- Readiness is a real WebSocket health check, not a TCP connect. A TCP connect
  proves docker published a port, which it does before a single model has
  loaded.

## 8. Environment variables do not reach an SSH session

Vast's own docs warn that variables set at instance creation are visible to the
`onstart` script but **not** inside an SSH or tmux session by default. Since
every deploy here goes over SSH, `vast/startup.sh` sources `/etc/environment`
explicitly.

The pipeline launch does not rely on that at all: `exec_command` opens no login
shell, so the forwarded settings are exported as part of the launch command
itself. A setting that decides what a paid session measures should not depend
on a file existing.

## 9. `export PATH=...`, not `PATH=... cmd`

A `VAR=value command` prefix binds only to the first word of a line, so the
second half of any `&&` chain ran under `/usr/bin/python` instead of the venv.
`cmd_run` uses `export PATH=... && cd ... && <command>`.

## Quick reference — a working `.env`

```bash
VAST_API_KEY=<from https://cloud.vast.ai/manage-keys/>
VAST_SCOPED_API_KEY=<instance_read + instance_write only>
VAST_GEOLOCATIONS=GB,IE,FR,NL,BE,DE
VAST_PREFERRED_HOST=<host_id from `offers`>
VAST_MIN_DLPERF=90
VAST_DISK=25
VAST_SSH_KEY_PATH=~/.ssh/id_ed25519
VAST_REPO_URL=https://github.com/khonello/Phantom.git
```

`VAST_INSTANCE_ID`, `PHANTOM_API_URL`, `PHANTOM_TLS_FINGERPRINT` and
`PHANTOM_API_TOKEN` are written by the orchestrator. Do not hand-edit them.
