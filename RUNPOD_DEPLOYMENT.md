# Phantom — RunPod Deployment

The pipeline runs on a rented RunPod GPU; the desktop connects to it over
WebSocket. `runpod/orchestrator.py` owns the whole lifecycle from your machine —
provisioning, setup, launching the pipeline, and writing the connection URL back
into `.env`.

## Where to start

**`.env` is the entire input.** There is no config file, no CLI flag and no
saved state anywhere else: the orchestrator reads `.env`, and writes two keys
back into it. So the whole question is whether `.env` is filled in yet.

Check:

```bash
grep -E '^RUNPOD_(API_KEY|DATACENTERS|POD_ID)=' .env
```

| What you see | Where you are | Go to |
|---|---|---|
| No `.env` at all | fresh install | **Part 1** — the dashboard, which mints the values |
| Keys present but blank | `.env.example` copied | **Part 2** — fill them in |
| All filled, `RUNPOD_POD_ID` empty | set up, never deployed | **Part 3** — `start` |
| All filled, `RUNPOD_POD_ID` set | **the steady state** | **Part 4** — `status`, then `resume` |

Parts 1–3 happen once, ever. **Part 4 is the loop you live in**, and it is two
commands to enter and one to leave:

```bash
python runpod/orchestrator.py resume     # ... work ...
python desktop.py
python runpod/orchestrator.py stop
```

Part 5 is the handful of reasons to leave that loop — a second region, a bigger
volume, discarding the pod. Those are the only times the dashboard reopens.

For what to *do* with the session once it is up — the measurement runbook, guard
calibration, what to set before spending money — see
[docs/PENDING_WORK.md](docs/PENDING_WORK.md).

---

## The shape of the whole thing

There is a setup road walked once, a **loop you live in**, and a small set of
exits from that loop. Almost all of your time is one arc of the loop.

```
   ONCE, EVER                     ┌─────────────────────────────────────┐
   ══════════                     │   Part 5 — exits, rare              │
                                  │                                     │
  ┌────────────────┐              │  new datacenter → new volume        │
  │ Part 1         │              │  grow the volume                    │
  │ DASHBOARD      │  YOU         │  terminate / switch deploy mode     │
  │  API key       │  manual      │       (dashboard + hand-edit .env)  │
  │  SSH key       │              └──────────────┬──────────────────────┘
  │  volume + DC   │                             │ rejoins at `start`
  └───────┬────────┘                             │
          │ values                               │
          ▼                                      ▼
  ┌────────────────┐                    ┌──────────────────┐
  │ Part 2         │  YOU               │ Part 3           │
  │ .env           │  manual, once ───► │ start            │
  │  paste 3 values│                    │  (fresh pod)     │
  └────────────────┘                    └────────┬─────────┘
                                                 │ writes RUNPOD_POD_ID
   THE LOOP                                      │ and PHANTOM_API_URL
   ════════                                      ▼
                          ┌──────────────────────────────────────┐
                          │                                      │
                          │   ┌────────┐   work    ┌────────┐    │
                          │   │ resume │ ────────► │  stop  │    │
                          │   └────────┘           └────┬───┘    │
                          │        ▲                    │        │
                          │        └────────────────────┘        │
                          │            ▲                         │
                          │            │                         │
                          │      YOU ARE HERE,                   │
                          │      ~99% of the time:               │
                          │      pod stopped, volume warm,       │
                          │      nothing billing but storage     │
                          └──────────────────────────────────────┘
                               Part 4 — no dashboard, no .env edits
```

**Where you actually sit, most of the time: stopped.** Between `stop` and the
next `resume` the pod exists but is not running, the network volume holds the
venv, the models and the repo, and the only thing billing is storage — around
$2.10/month at 30 GB. That is the resting state, and getting back to work from
it is one command.

### The loop, and what it does not touch

```
resume  ──►  desktop.py  ──►  ...work...  ──►  stop  ──►  (resting)  ──►  resume
```

Around that loop: **the dashboard is never opened, and `.env` is never edited by
hand.** The orchestrator rewrites `PHANTOM_API_URL` on every boot because the
pod's address changes, and that is the only `.env` write in the steady state.

You leave the loop for exactly three reasons, all in Part 5: you want a second
region, you have outgrown the volume, or you are discarding the pod.

### Who runs what

| Script | Runs where | When | You type it? |
|---|---|---|---|
| `runpod/orchestrator.py` | your machine | every session | **yes** — the only one |
| `desktop.py` | your machine | after `resume` | **yes** |
| `runpod/startup.sh` | on the pod | called by the orchestrator | no |
| `runpod/prewarm.py` | on the pod | called by `startup.sh` | no |
| `pipeline.py` | on the pod | launched by the orchestrator | no |

Inside a `start` or `resume`, in order, all automatic: pick GPU and datacenter →
create or resume the pod → wait for SSH → `git clone` if first time →
`startup.sh` (ffmpeg, `git pull`, venv, pip sync, cuDNN, model download,
pre-warm) → launch `pipeline.py` under nohup → health-check port 9000 → write
`.env` → print the cold-start table.

You never SSH in as part of normal operation — only to read a log when something
has gone wrong.

### When `.env` is touched, and by whom

This is the whole answer to "do I edit `.env`?" — mostly no, and never in the
loop.

| Key | Written by | When |
|---|---|---|
| `RUNPOD_API_KEY` | **you, by hand** | once, Part 2 |
| `RUNPOD_SSH_KEY_PATH` | **you, by hand** | once, Part 2 |
| `RUNPOD_IMAGE` | ships correct | only to switch deploy mode |
| `RUNPOD_DATACENTERS` | **you, by hand** | Part 2, then only to add a region |
| `RUNPOD_VOLUME_DISK` | **you, by hand** | only when growing the volume |
| `RUNPOD_POD_ID` | *the orchestrator* | every `start` |
| `PHANTOM_API_URL` | *the orchestrator* | every `start` and `resume` |
| `SWAPPER_MODEL`, `GUARD_*`, `DEBUG_FRAMES_*` | **you, by hand** | before a run you want configured differently |

The last row is the only hand-editing that recurs, and it is optional — those
are the settings forwarded into the pod so `start` produces a configured
pipeline. See [Configuring the run](#configuring-the-run-before-it-starts).

### So on a genuinely fresh start: dashboard or `.env`?

**The dashboard, then `.env`** — and it cannot be the other way round. Three of
the four required values are *minted* by RunPod: the API key, the volume ID and
the datacenter that volume was created in. Nothing local can invent them, so
`.env` is downstream of the dashboard by definition. `.env` is where they come
to rest; the dashboard is where they come from.

---

## Part 1 — The dashboard

Three values live on RunPod and cannot be derived from anything local. This is
the only manual work, and it is done once. Step 1d is here to say explicitly
that the fourth thing you might expect to collect is not collected by hand.

Sign up at [runpod.io](https://www.runpod.io) and add billing first — pods
cannot be created on a zero balance, and the failure surfaces as "no GPUs
available", which reads like a capacity problem.

### 1a — API key → `RUNPOD_API_KEY`

**Settings → API Keys → Create API Key.** Copy it now; RunPod shows the secret
once.

This key is also forwarded into the pod so the pipeline can stop itself when
`RUNPOD_MAX_UPTIME` expires. It is not scoped — see [Security](#security).

### 1b — SSH public key → `RUNPOD_SSH_KEY_PATH`

Only for `ssh` mode, which is the default.

```bash
ssh-keygen -t ed25519 -C "you@email.com"      # skip if you already have one
cat ~/.ssh/id_ed25519.pub                      # copy this
```

**Settings → SSH Public Keys → Add SSH Key**, paste, save.

What goes into `.env` is the path to the **private** half, and it must be
unencrypted — the orchestrator loads it with no passphrase prompt, so a
protected key fails at the connect step rather than asking.

Docker mode never SSHes, but register the key anyway: it is how you get a shell
on a pod when something goes wrong.

### 1c — Network volume → `RUNPOD_DATACENTERS`

**Storage → Network Volumes → New Network Volume.**

| Field | Value |
|---|---|
| Name | `phantom-workspace` |
| Size | **30 GB** |
| Datacenter | pick one and note it, e.g. `EU-RO-1` |

30 GB is the working figure: the venv is ~6–8 GB, models ~2 GB, and batch
scratch defaults onto this volume too. Volumes can be grown in place but **never
shrunk**, and they bill at $0.07/GB/month **whether or not a pod is running** —
about $2.10/month at 30 GB. Going to 100 GB "just in case" is $7/month standing
for idle capacity.

The volume is what makes the second session fast: the venv, the models and the
repo all live on it, so `start` skips the expensive steps once it is warm.

Copy the **volume ID** from the storage list and pair it with its datacenter:

    RUNPOD_DATACENTERS=EU-RO-1:z8now7p5ts

The datacenter and the volume travel together because volumes are
datacenter-local — a pod can only be placed where its volume lives. One entry is
all you need now; **Part 5a** covers adding a second region and why it drags a
second volume along with it.

### 1d — Pod ID → `RUNPOD_POD_ID`

**Leave this blank.** It is the one dashboard-shaped value you should *not*
collect by hand: `start` creates the pod and writes the ID back into `.env`
itself. Only paste one in if you are adopting a pod that already exists.

---

## Part 2 — `.env`

```bash
cp .env.example .env
pip install -r requirements-orchestrator.txt
```

Paste in what Part 1 produced. Four keys must be set; everything else has a
working default and `.env.example` documents each one inline.

```env
RUNPOD_API_KEY=rp_xxxxxxxxxxxxxxxxxxxx          # 1a
RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519           # 1b — ssh mode only
RUNPOD_DATACENTERS=EU-RO-1:z8now7p5ts           # 1c
RUNPOD_IMAGE=runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUNPOD_POD_ID=                                  # 1d — written by `start`
PHANTOM_API_URL=                                # written by `start` / `resume`
```

`RUNPOD_IMAGE` ships correct in `.env.example` and must stay in step with the
`Dockerfile` — CI asserts they match. Do not "upgrade" it casually; the
`runtime` tag it looks like it should have does not exist for `runpod/pytorch`,
and only `devel` is published.

Leave `RUNPOD_GPU_TYPES` empty unless you have a reason to pin. Empty means the
orchestrator picks the GPU for you; see the [`.env` reference](#env-reference).

The two commented keys are the loop closing: the dashboard gives you three
values, and the orchestrator fills in the rest. Nothing else is ever edited by
hand.

---

## Part 3 — The first run

```bash
python runpod/orchestrator.py start
```

That is the whole of it — there is no manual dry-run step first. If SSH, the
volume mount or the GPU are broken, `start` fails at that exact phase and names
it, which is the same information a hand-run would have given you, without the
dashboard round trip.

Expect **five to eight minutes** on a first run against an empty volume, most of
it `pip install`. Watch for three things in the output:

- `Created pod: <id> (<GPU> ... )` — provisioning found capacity, and
  `RUNPOD_POD_ID` is now written
- `Execution provider: CUDAExecutionProvider confirmed` — if instead it exits
  with `ExecutionProviderError`, that is deliberate; see
  [Troubleshooting](#troubleshooting)
- the cold-start phase table at the end — printed, not saved, so copy it

Then:

```bash
python desktop.py
```

The desktop reads `PHANTOM_API_URL`, which `start` has just rewritten.

**When you are done, `stop`.** The auto-stop timer is a backstop, not a
workflow.

```bash
python runpod/orchestrator.py stop
```

---

## Part 4 — The loop

**This is where you live.** `.env` is complete and stays that way, and the
dashboard is finished with. Every session is the same four commands.

```bash
python runpod/orchestrator.py status     # what exists, and is it billing?
python runpod/orchestrator.py resume     # boot the pod named in RUNPOD_POD_ID
python desktop.py                        # ... work ...
python runpod/orchestrator.py stop       # back to resting
```

`status` first is worth the three seconds — it answers which command you want,
and catches the expensive case where you left one running:

| `status` says | Meaning | Do |
|---|---|---|
| `EXITED` / `STOPPED` | **the resting state.** Pod intact, no GPU billing | `resume` |
| `RUNNING` | live and billing right now | connect the desktop, or `stop` |
| `not found` | terminated; only the volume survives | `start` — see Part 5 |

### `resume`, not `start`

They are not interchangeable, and the loop runs on `resume`.

| | `resume` | `start` |
|---|---|---|
| Pod | the existing one | a brand new one |
| Container disk | kept, so `apt-get` does not re-run | fresh |
| `RUNPOD_POD_ID` | unchanged | **overwritten** |
| The old pod | is the one you booted | orphaned, and still billing if it was up |

`start` overwrites the only record of the previous pod's ID, after which `stop`
and `terminate` cannot reach it. The orchestrator now checks its status, says so
plainly and prints the outgoing ID before proceeding — but the cheap habit is to
reach for `resume` and keep `start` for the deliberate cases in Part 5.

### What resting costs

Stopped, you pay for the network volume only: $0.07/GB/month, about $2.10 at
30 GB, billed whether or not any pod exists. GPU billing stops with the pod.

That is the whole reason the volume exists. It holds the venv, the models and
the repo, so `resume` skips the expensive setup and the loop stays cheap to
re-enter. Deleting the volume to save $2 would put five to eight minutes of
`pip install` back onto every session.

If you forget to `stop`, `RUNPOD_MAX_UPTIME` stops the pod for you after 120
minutes — see [Auto-stop](#auto-stop). Treat it as a backstop, not the plan.

---

## Part 5 — Leaving the loop

Three reasons, all uncommon. Each one is dashboard work plus a hand-edit to
`.env`, and each rejoins the loop at a `start`.

### 5a — Adding a datacenter (and why it needs its own volume)

Do this when `start` has actually failed to find a GPU in your region — not
before. It costs money from the moment the volume exists.

**A datacenter cannot be added on its own.** RunPod network volumes are
datacenter-local: a volume in `EU-RO-1` cannot be mounted by a pod anywhere
else, and there is no replication between regions. So a fallback region with no
volume of its own would deploy a pod with no persistent storage — every model
re-downloaded, the venv rebuilt from scratch, and everything lost again at
`stop`. Silently, while appearing to work.

That is why `RUNPOD_DATACENTERS` pairs them in one variable instead of having a
separate list of regions and volumes. The pairing makes the invariant
unstateable-wrong: you cannot name a region without naming the volume that makes
it usable.

```env
RUNPOD_DATACENTERS=EU-RO-1:z8now7p5ts,US-KS-2:newvolumeid
```

The procedure:

1. `python runpod/orchestrator.py datacenters` — pick one that is not yours
2. **Storage → New Network Volume** in that datacenter, same size as the first.
   Copy the volume ID
3. Append `DC:vol` to `RUNPOD_DATACENTERS` **by hand**
4. `start` — the orchestrator tries every GPU in the first pair, then falls
   through to the second

**Order is priority order**, and it decides which volume you land on. The
existing warm pair stays first for normal work; putting a new, empty pair first
is how you would deliberately measure an empty-volume cold start.

The cost is the point: **the second volume bills continuously from creation,
used or not.** Regional redundancy you cannot yet exercise is resilience bought
early — which is why [docs/PENDING_WORK.md](docs/PENDING_WORK.md) §1.4 defers
this until the pipeline is proven on a GPU or `EU-RO-1` has actually run dry.

### 5b — Growing the volume

Volumes grow in place and **never shrink**, so this is one-way.

1. **Storage → your volume → Edit →** new size
2. Update `RUNPOD_VOLUME_DISK` in `.env` so a future fresh deploy matches

No `start` needed — a running pod picks up the new size. Batch video is what
forces this: extracted frames are ~4 MB each at 1080p, so a five-minute clip
wants ~36 GB of scratch, and scratch defaults onto this volume.

### 5c — Terminating, or switching deploy mode

`terminate` deletes the pod; the network volume survives, so the next `start` is
warm. This is the only routine use of `start` — the pod is gone, so there is
nothing to `resume`.

Switching between `ssh` and `docker` needs the same thing: the mode is baked
into how the pod was created, so `terminate` then `start`, rather than `resume`.
See [Deploy modes](#deploy-modes).

---

## Command reference

Every command is `python runpod/orchestrator.py <command>`.

| Command | Does | Costs | Writes `.env` |
|---|---|---|---|
| `status` | state, GPU, cost/hr, uptime, URL | — | — |
| `start` | **new** pod → setup → pipeline; prints the cold-start table | starts GPU billing | `RUNPOD_POD_ID`, `PHANTOM_API_URL` |
| `resume` | boots the pod in `RUNPOD_POD_ID`, same setup path | starts GPU billing | `PHANTOM_API_URL` |
| `stop` | pauses it; volume and container disk survive | ends GPU billing | — |
| `terminate` | deletes the pod; the network volume survives | ends GPU billing | — |
| `gpus` | every GPU with VRAM, price, and whether it is eligible | — | — |
| `datacenters` | every datacenter and its ID | — | — |

`stop` ends a session; `terminate` is for discarding the pod itself. Neither
touches the network volume, so the venv and models are never at risk either way.
The volume keeps billing regardless — that is storage, not GPU.

`gpus` and `datacenters` are for when provisioning fails, or when you are
choosing a second region.

---

## Configuring the run *before* it starts

The orchestrator forwards a fixed list of pipeline settings from your local
`.env` into the pod's environment at creation, so `start` produces a
**configured** pipeline rather than a default one. Without this, changing what a
session runs means SSHing in and restarting the pipeline by hand — which is most
of a session.

```env
SWAPPER_MODEL=hyperswap_1a_256
GUARD_OBSERVE=true
GUARD_REPORT=/workspace/guards.json
DEBUG_FRAMES_DIR=/workspace/clip
DEBUG_FRAMES_STRIDE=3
```

The forwarded list is `_FORWARDED_ENV` in `orchestrator.py` — model selection,
the realism knobs, the guard settings, the debug-frame settings, `LOG_LEVEL` and
`PHANTOM_TEMP_DIR`. It is a list rather than "forward everything" on purpose: the
pod is a different machine, and blanket forwarding would send local paths and
secrets that mean nothing there.

`tests/test_wiring.py` asserts it in both directions — every forwarded name is
read by the pipeline, and no pipeline setting is stranded on the local machine.

---

## What `start` does

### SSH mode (`RUNPOD_DEPLOY_MODE=ssh`)

```
provision       resolve GPU candidates, then create the pod
                ├─ RUNPOD_GPU_TYPES set?  use those names, in order
                └─ otherwise             auto-discover by RUNPOD_MIN_VRAM /
                                         RUNPOD_MAX_PRICE, cheapest first
                try every GPU in datacenter 1, then fall through to
                datacenter 2 with its own volume, and so on
                writes RUNPOD_POD_ID to .env

wait-for-ssh    poll until RunPod assigns ports, then TCP-poll 22 on
                {podHostId}@ssh.runpod.io

remote-setup    paramiko interactive shell
                ├─ git clone           first deploy only
                ├─ runpod/startup.sh   apt/ffmpeg, git pull, venv, dependency
                │                      sync, cuDNN 9, GFPGAN, model pre-warm
                ├─ pkill any old pipeline
                └─ nohup pipeline.py --execution-provider cuda
                        > /workspace/phantom-pipeline.log 2>&1 &

pipeline-ready  WebSocket health check against
                wss://{pod_id}-9000.proxy.runpod.net/ws
                writes PHANTOM_API_URL to .env
```

Every step is idempotent. `git clone` only runs when `/workspace/Phantom` is
missing, `startup.sh` reuses the venv when it exists, and **it runs `git pull
--ff-only` on every start** — so a code change is deployed by pushing it and
running `start` or `resume`, with no manual SSH step.

`startup.sh` re-runs `pip install` only when `requirements-pipeline-gpu.txt`
differs from the snapshot it stored on the volume last time.

### Docker mode (`RUNPOD_DEPLOY_MODE=docker`)

```
provision          same GPU and datacenter search, custom image,
                   no SSH and no public IP
wait-for-container poll until port 9000 is assigned; the pipeline
                   auto-starts from the image CMD
pipeline-ready     same health check, same .env update
```

No SSH step and no `support_public_ip`, which is why docker mode schedules more
freely — public IPs materially constrain which machines RunPod will place a pod
on. Model weights still live on the network volume by design, so the image does
not carry them.

### The cold-start table

`start` and `resume` print a phase breakdown at the end. It is printed, not
written to a file, so save it.

```
Cold start (ssh) — 421s total, volume: empty
* provision                74.0s   17.6%
* wait-for-ssh             38.0s    9.0%
* remote-setup            268.0s   63.7%
    apt-get                18.0s    4.3%
    pip-install           171.0s   40.6%
    cudnn                  22.0s    5.2%
    gfpgan-download        31.0s    7.4%
    model-load             10.0s    2.4%
* pipeline-ready           41.0s    9.7%
```

Rows marked `*` are exclusive; indented rows break down the row above.
`startup.sh` emits `PHASE <name> <seconds>` lines that the orchestrator parses
out of the SSH transcript, so the two need agree on nothing else. The `volume:`
label records warm or empty, so two runs stay distinguishable later.

This table is what settles whether to bake a Docker image — see
[docs/PENDING_WORK.md](docs/PENDING_WORK.md) §3.4.

---

## Networking

Both connections go through RunPod's proxy, so no ports are opened to the
internet and no IP is copied by hand.

| | Address |
|---|---|
| WebSocket | `wss://{pod_id}-9000.proxy.runpod.net/ws` (port 443) |
| SSH | `{podHostId}@ssh.runpod.io` port 22 |

Only `9000/tcp` is exposed on the pod. 8888 is deliberately left out — exposing
it triggers a slow JupyterLab init on RunPod's base images.

`podHostId` comes from a direct GraphQL query. The RunPod SDK's `get_pod()` does
not return it, which is one of several gotchas recorded in
[runpod/TROUBLESHOOTING.md](runpod/TROUBLESHOOTING.md).

---

## `.env` reference

Read by the orchestrator. `.env.example` carries the same list with commentary.

| Variable | Default | What it does |
|---|---|---|
| `RUNPOD_API_KEY` | — | **Required.** Also forwarded to the pod so it can stop itself |
| `RUNPOD_POD_ID` | — | Written by `start`; read by `resume` / `stop` / `terminate` / `status` |
| `RUNPOD_DEPLOY_MODE` | `ssh` | `ssh` or `docker` |
| `RUNPOD_DATACENTERS` | — | `DC:vol,DC2:vol2` — priority order, each paired with its own volume |
| `RUNPOD_DATACENTER_ID` | — | Legacy single-datacenter form, with `RUNPOD_NETWORK_VOLUME_ID` |
| `RUNPOD_NETWORK_VOLUME_ID` | — | Legacy; ignored when `RUNPOD_DATACENTERS` is set |
| `RUNPOD_GPU_TYPES` | *unset* | Exact display names, in order. **Setting this disables auto-discovery** |
| `RUNPOD_MIN_VRAM` | `16` | Auto-discovery floor, GB |
| `RUNPOD_MAX_PRICE` | `1.00` | Auto-discovery ceiling, $/hr |
| `RUNPOD_IMAGE` | — | **Required.** Must stay in step with the `Dockerfile` |
| `RUNPOD_CONTAINER_DISK` | `20` | GB, ephemeral, lost on stop |
| `RUNPOD_VOLUME_DISK` | `20` | GB, only used when no network volume is attached |
| `RUNPOD_SSH_KEY_PATH` | `~/.ssh/id_ed25519` | ssh mode; must be unencrypted |
| `RUNPOD_REPO_URL` | — | Cloned on first deploy. Embed a token for a private repo |
| `RUNPOD_MAX_UPTIME` | `120` | Minutes before the pod stops itself. `0` disables |
| `RUNPOD_STOP_WARNING` | `5` | Minutes of warning before that |

Leave `RUNPOD_GPU_TYPES` unset unless you have a reason to pin. Auto-discovery
queries RunPod, filters by VRAM and price, drops GPUs whose compute capability
exceeds what the image's PyTorch and ONNX builds support — Blackwell `sm_120` on
an `sm_90` image, say — and tries the cheapest first. A pinned list gets none of
that and goes stale as RunPod's fleet changes.

---

## Auto-stop

`RUNPOD_MAX_UPTIME` (default 120 minutes) is forwarded into the pod at creation.
The pipeline's own background timer calls `runpod.stop_pod()` when it expires, so
**an abandoned session is capped even with no desktop connected** — which is the
case that matters, since a forgotten pod is the expensive one.

Five minutes before, it emits `auto_stop_warning`. The desktop shows a dialog
with an **Extend** button, which sends `keep_alive`.

This is a backstop, not a workflow. Run `stop` when you are done.

---

## Deploy modes

| | ssh | docker |
|---|---|---|
| Use for | development | stable releases |
| Code change | push, then `start` / `resume` — `startup.sh` pulls | `docker build && docker push`, then redeploy |
| Boot | slower; setup runs on the pod | faster; everything baked |
| Debugging | full shell, live logs, `git pull` to test instantly | pod logs on the dashboard |
| Scheduling | constrained by `support_public_ip` | freer |

Switching is one line in `.env` plus the matching `RUNPOD_IMAGE`. The mode is
baked into how the pod is created, so `terminate` and `start` fresh rather than
`resume`. The volume carries over.

```bash
docker build -t <your-registry>/phantom-pipeline:latest .
docker push <your-registry>/phantom-pipeline:latest
```

Rebuild when pipeline code, dependencies or system packages change — not for
runtime config, which goes through `.env` and the WebSocket API.

**Keep both modes alive.** CI builds the image on every push for exactly this
reason: docker mode once shipped unbuildable and would have run every model on
CPU, and nothing noticed because nobody was using it.

---

## Troubleshooting

Deeper RunPod-specific detail, and the history behind each workaround, is in
[runpod/TROUBLESHOOTING.md](runpod/TROUBLESHOOTING.md).

**`start` says the pod was not found.** `RUNPOD_POD_ID` points at a deleted pod.
`start` deploys a new one regardless; no action needed.

**No GPU capacity across datacenters.** Raise `RUNPOD_MAX_PRICE`, lower
`RUNPOD_MIN_VRAM`, or add a second datacenter — which needs its own volume there.
`python runpod/orchestrator.py gpus` shows what is eligible and why.

**SSH timeout.** The proxy accepts TCP before the container is ready. The
orchestrator already retries 12 times at 10-second intervals; a failure past that
usually means the pod itself is unhealthy. Retry `start`.

**Pipeline never binds 9000 (ssh mode).**

```bash
ssh <podHostId>@ssh.runpod.io
tail -f /workspace/phantom-pipeline.log
```

The pipeline runs under `nohup`, not tmux — there is no session to attach to, and
the log is the whole picture.

**The pipeline exited with `ExecutionProviderError`.** Working as designed. ONNX
Runtime does not error when CUDA fails to initialise, it silently uses CPU — and
since the swapper, CodeFormer and XSeg are all ONNX, that is seconds per frame on
a pod billing a full GPU hour. The check refuses to start instead. See
`runpod/TROUBLESHOOTING.md` §5b. **Do not downgrade it to a warning**;
`--execution-provider cpu` is the supported way to run without an accelerator.

**Desktop says "disconnected — reconnecting…".** `PHANTOM_API_URL` is stale.
`start` and `resume` both rewrite it; `status` prints the current one.

---

## Security

- Proxy URLs are pod-specific — treat them as secrets and do not share them
- The WebSocket server has no authentication; it relies on RunPod's isolation
- An SSH key registered on RunPod grants root on the pod
- `RUNPOD_API_KEY` is forwarded into the pod so it can stop itself. A pod that
  can stop itself can also stop your others — the key is not scoped
- Production needs the WebSocket behind an authenticated reverse proxy
