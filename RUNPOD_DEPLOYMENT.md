# Phantom — RunPod Deployment

The pipeline runs on a rented RunPod GPU; the desktop connects to it over
WebSocket. `runpod/orchestrator.py` owns the whole lifecycle from your machine —
provisioning, setup, launching the pipeline, and writing the connection URL back
into `.env`.

## What does NOT belong on a pod

OBS Studio and a virtual audio cable are **operator-machine** software. The pod
is headless: it receives JPEG frames over a WebSocket, swaps the face, and sends
them back. It has no virtual camera, no conferencing app, and no audio path at
all — audio is never uploaded to the pipeline, and neither `pyvirtualcam` nor
`sounddevice` is imported by it or listed in its requirements.

Installing either on a pod does nothing. See
[docs/SETUP_CHECKLIST.md](docs/SETUP_CHECKLIST.md) for which machine needs what.

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
| No `.env`, but the account already has a volume | **same account, new machine** | **Part 1**, reusing rather than creating — see below |
| Keys present but blank | `.env.example` copied | **Part 2** — fill them in |
| All filled, `RUNPOD_POD_ID` empty | set up, never deployed | **Part 3** — `start` |
| All filled, `RUNPOD_POD_ID` set | **the steady state** | **Part 4** — `status`, then `resume` |

**On a new machine with an account you already set up**, Part 1 still applies,
but two of its four steps are lookups rather than acts. Nothing about the
account is recreated:

| Part 1 step | On a new machine |
|---|---|
| 1a API key | **mint a new one** — RunPod shows a secret once, so the old one is unrecoverable |
| 1b SSH key | **check first** — the public half is likely registered already; only the *private* half is missing, so copy it across or add a second key |
| 1c Volume | **copy the existing ID** out of Storage. Do **not** create a second one — it is a second monthly bill, in a datacenter your models are not on |
| 1d Pod ID | leave blank, exactly as on a first run |

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

### 1b — SSH key → `RUNPOD_SSH_KEY_PATH`

Only for `ssh` mode, the default. There are two halves — a public one RunPod
keeps, and a private one that stays on your machine — and **the public half is
often already done**, because it is registered once per *account* and never
again. Check before doing anything:

```bash
curl -s https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pubKey } }"}'
```

- **A key comes back** → registration is done, permanently. Skip to step 3, and
  only make sure the *private* half is on this machine.
- **`"pubKey": null` or empty** → do all three steps.

**1 — Generate the pair.** Skip if `~/.ssh/id_ed25519` already exists.

```bash
ssh-keygen -t ed25519 -C "you@email.com"       # accept the default path
                                               # leave the passphrase EMPTY
```

Two files: `~/.ssh/id_ed25519` (private, never leaves your machine) and
`~/.ssh/id_ed25519.pub` (public). The passphrase must be empty — the
orchestrator loads the key with no prompt, so a protected key fails at the
connect step rather than asking.

**2 — Register the public half.** `cat ~/.ssh/id_ed25519.pub`, then RunPod →
**Settings → SSH Public Keys → Add SSH Key**, paste, save. RunPod writes the
registered keys into a pod's `authorized_keys` **when the pod is created**, so
this must precede the first `start`. It is a one-time act you will not remember
doing — hence the check above.

**3 — Point `.env` at the private half.**

```env
RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519
```

#### What goes wrong

- **`~` is expanded against whichever shell runs the orchestrator**, via
  `os.path.expanduser` — not the shell you generated the key in. Generating in
  WSL and running from PowerShell is the usual way to get this wrong: the key
  sits in `/home/<you>/.ssh/` while the orchestrator looks in
  `C:\Users\<you>\.ssh\`. Copy it across, keeping the WSL copy:

  ```bash
  wsl -e cp ~/.ssh/id_ed25519 /mnt/c/Users/<you>/.ssh/id_ed25519
  wsl -e cp ~/.ssh/id_ed25519.pub /mnt/c/Users/<you>/.ssh/id_ed25519.pub
  ```

  Use `cp` through WSL, not a PowerShell redirect — a redirect can add CRLF or a
  BOM and paramiko then rejects the key. An absolute path in
  `RUNPOD_SSH_KEY_PATH` sidesteps `~` entirely. No `icacls` fix is needed;
  paramiko does not enforce OpenSSH's permission checks.

- **The key never belongs in the repo.** The repo is `git clone`d onto the pod
  at `/workspace/Phantom`, which is the network volume and survives
  `terminate` — a committed key would be left on persistent rented storage
  after the pod is gone.

- **A pod's SSH panel is a different thing.** It appears only once a pod exists,
  which is after all of Part 1, so it is never part of setup:

  | | Where | When |
  |---|---|---|
  | Registering the public key | Settings → SSH Public Keys | once, **before** the first `start` |
  | The `ssh …` connect line | the pod's page → Connect → SSH | only **after** a pod exists |

  You need neither for a normal session: `status` prints the connect line, and
  the orchestrator resolves the address itself.

`start` and `resume` load the key before creating or resuming anything, so a
missing or unreadable key stops you while nothing is billing. Docker mode never
SSHes and skips the check, but register a key anyway — it is how you get a shell
when something goes wrong.

### 1c — Network volume → `RUNPOD_DATACENTERS`

**Storage → Network Volumes.** If a volume is already listed, this step is a
copy of its ID — skip to the bottom of this section. Create a second one only
to add a *region* (Part 5a), never to set up a second machine.

Otherwise, **New Network Volume.**

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

    RUNPOD_DATACENTERS=EU-RO-1:<volume-id>

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
RUNPOD_DATACENTERS=EU-RO-1:<volume-id>          # 1c
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

### When `resume` cannot get a GPU

A stopped pod keeps its **host machine**, and resuming needs a GPU free on that
specific machine. Rest long enough and someone else takes it:

```
ERROR: Resume failed: There are not enough free GPUs on the host machine
       to start this pod.
```

Nothing is wrong with the pod, the volume or `.env` — the host is simply full,
and no amount of retrying moves a pod off it. **Only a new pod can be scheduled
somewhere with capacity**, so `resume` recognises this one failure and falls
through to `start` by itself. Every other resume failure still stops, because
falling back on any error would answer a typo'd pod ID with a billing pod.

The fallback is cheap: the network volume carries the venv, the models and the
repo, so the new pod is a warm start rather than a first one.

It leaves one thing behind. The old pod stays stopped and keeps billing for its
container disk, and `RUNPOD_POD_ID` now names the new pod — so `terminate` can
no longer reach the old one. The orchestrator prints its ID as it falls through;
delete it from the dashboard, **Pods → that ID → Terminate**.

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
RUNPOD_DATACENTERS=EU-RO-1:<volume-id>,US-KS-2:<new-volume-id>
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
| `status` | state, GPU, cost/hr, uptime, URL, ready-to-paste `ssh` line | — | — |
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
the realism knobs, the guard settings, the inference speed levers, the
debug-frame settings, `LOG_LEVEL` and `PHANTOM_TEMP_DIR`. It is a list rather
than "forward everything" on purpose: the pod is a different machine, and
blanket forwarding would send local paths and secrets that mean nothing there.

### Changing settings on a pod that already exists

**`.env` is read at `create_pod` time and never again.** Those values are baked
into the container's environment, so on a pod that is already running:

- editing `.env` changes nothing;
- `git pull` on the pod changes nothing, because **`.env` is gitignored** and
  travels by a different route than the code;
- restarting the pipeline changes nothing either — it inherits the same
  environment.

That is easy to miss, because code and configuration reach the pod by two
separate paths that look like one:

    code   ->  git push  ->  git pull on the pod
    .env   ->  read locally by orchestrator.py  ->  pod env, at creation only

Three ways to change a setting, in ascending cost:

**1. Live, no restart** — `tools/realism.py` against the running pipeline. It
covers the model selection, the realism knobs, the guard thresholds and the
speed levers, and reports what the server applied and what it refused:

```bash
python tools/realism.py --host <ip> --port <port> swapper_model=hyperswap_1a_256
python tools/realism.py --host <ip> --port <port> enhance=false
python tools/realism.py --host <ip> --port <port> --show
```

Switching the swapper this way applies that model's realism profile and drops
temporal state, exactly as a fresh start would. Changing a speed lever or the
restoration model drops the ONNX sessions, which reload on the next frame — a
visible hitch of a second or two, which is why these are not exposed to a
consumer.

**2. Restart the pipeline with different env**, for the settings `set_realism`
does not carry — `DEBUG_FRAMES_DIR` is the one that matters, since debug
capture is read at process start:

```bash
python runpod/orchestrator.py run "pkill -f pipeline.py; sleep 4; \
  [ -f /etc/rp_environment ] && . /etc/rp_environment; \
  [ -f /etc/profile.d/cudnn.sh ] && . /etc/profile.d/cudnn.sh; \
  cd /workspace/Phantom && DEBUG_FRAMES_DIR=/workspace/dbg \
  nohup /workspace/venv/bin/python /workspace/Phantom/pipeline.py \
  --execution-provider cuda > /workspace/phantom-pipeline.log 2>&1 &"
```

**Budget for warm-up.** The first stream after a pipeline start pays model load
inside its own window — tens of seconds. A 40s capture against a freshly
started pipeline produced zero frames and looked like a broken config. Either
run a discarded pass first, or make the window long enough to contain both.

**3. `terminate` then `start`**, which is the only way a `.env` edit takes
effect, and the only way onto a different GPU — a stopped pod stays pinned to
the host it was created on, so `resume` cannot move it.

### Before any measurement on a resumed pod

```bash
python runpod/orchestrator.py run "cd /workspace/Phantom && git pull"
```

A pod resumed after a break is running whatever code it last pulled. This has
already cost a full sweep: every `set_realism` was refused because the pod
predated the fields being asked for, and each of the twelve configurations
reported the same numbers because none of them applied. The sweep exits
non-zero on that now, but pulling first is what avoids it.

### Getting a measurement clip onto the pod

A speed comparison needs **two files on the pod**, and they are different
things:

| | What it is | Where it goes |
|---|---|---|
| **Source** | A *still image* of the face being swapped **in** | `/workspace/face.jpg` |
| **Clip** | A *video* standing in for the webcam feed — the frames being swapped | `/workspace/clip.mp4` |

The clip is what the operator's camera would have sent, so it should be a
person facing the camera as they would on a call. Record it on the actual
webcam: the design target is what a real video call looks like, sensor noise
and compression included, and a clean studio video measures a workload the
product never sees.

Neither can be uploaded through the app. `upload_target` carries photos inline
over the WebSocket at 6 MB each, and `set_target` resolves paths against the
*pipeline's* filesystem, which on a pod is another machine. So copy them over
SSH:

```bash
python runpod/orchestrator.py push clip.mp4
python runpod/orchestrator.py push face.jpg
```

Both land in `/workspace`, which is the network volume and therefore survives
the pod. Pass a second argument to choose the destination.

**Encode the clip at the preset's capture resolution.** With `--input-url` the
pipeline does *not* set capture width, height or frame rate — those are only
applied to a real webcam — so the file plays at whatever it was encoded at, and
that is what decides how much work each frame is. A 1080p clip measures a
workload no preset ever runs:

```bash
ffmpeg -i raw.mp4 -vf scale=640:360 -r 20 -c:v libx264 -crf 20 clip.mp4
```

**Make it longer than the sweep needs.** Frames are read as fast as they decode
rather than paced to real time, so a clip is consumed faster than its running
length, and the stream simply ends at EOF. Two to three minutes covers a 60
second measurement per configuration comfortably.

### The speed levers

Four settings trade inference time against risk, and all four default **off** so
the out-of-the-box path is bit-identical to what it was before they existed.
They matter here more than anywhere else, because a rented GPU is the only place
they can honestly be measured.

```env
FP16=true                       # load -fp16.onnx weights where they exist
CUDA_GRAPHS=true                # capture and replay the kernel launch sequence
TRT=true                        # route through the TensorRT provider
TRT_GPUS=RTX 4090,H100,L40S     # architectures worth a multi-minute engine build
```

`CUDA_GRAPHS` changes no numerics at all — it only removes per-kernel launch
overhead, which is a real share of the cost for batch-1 models made of many
small kernels. It applies to models whose input shapes are fixed (CodeFormer,
XSeg, the swapper) and is ignored for the detector, whose shapes are not.

`FP16` is the largest single win and the only one of the four that can change
what the output looks like. Convert first, then A/B on footage rather than on a
latency number:

```bash
python tools/convert_fp16.py /workspace/models/codeformer.onnx
python pipeline.py --stream --debug-frames fp32/
python pipeline.py --stream --fp16 --debug-frames fp16/
python tools/compare_frames.py fp16/ --against fp32/
```

`TRT` builds an engine per model, which takes minutes and comes out of a paid
hour. The engines are cached on the network volume under `/workspace/trt-cache`,
keyed by GPU, TensorRT and ONNX Runtime versions, model fingerprint and
precision — every property an engine is invalid across. Each architecture pays
its build **once, ever**, so the cache warms itself as `start` lands on
different cards.

`TRT_GPUS` is why that is affordable. Auto-discovery picks across datacenters
because availability is the binding constraint, so pinning to one GPU would
trade "sometimes a slower card" for "sometimes no pod at all". Instead, the
build is only spent on architectures fast enough to earn it; anything else runs
on CUDA and says so. Budget roughly 0.5–1 GB of volume per architecture.

A TensorRT fallback is reported but does **not** stop the session, unlike a CPU
fallback. A model that fell back to CUDA is still on the GPU and still holds a
live call; it is merely not as fast as intended, and stopping would cost the
operator more than the fallback does.

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
| `RUNPOD_MIN_GPU_PERF` | `85` | Speed floor, ranked by `_GPU_PERF`. The tier is 4090 / H200 / H100 / RTX 6000 Ada / L40S. Ignored in manual mode |
| `RUNPOD_GPU_WAIT` | `300` | Seconds to wait for the tier, retrying the whole list each minute. Free — billing starts when a pod runs |
| `RUNPOD_GPU_FALLBACK` | *unset* | What happens when that wait expires: unset fails, `true` accepts a slower card |
| `RUNPOD_IMAGE` | — | **Required.** Must stay in step with the `Dockerfile` |
| `RUNPOD_CONTAINER_DISK` | `20` | GB, ephemeral, lost on stop |
| `RUNPOD_VOLUME_DISK` | `20` | GB, only used when no network volume is attached |
| `RUNPOD_SSH_KEY_PATH` | `~/.ssh/id_ed25519` | ssh mode; must be unencrypted |
| `RUNPOD_REPO_URL` | — | Cloned on first deploy. Embed a token for a private repo |
| `RUNPOD_MAX_UPTIME` | `120` | Minutes before the pod stops itself. `0` disables |
| `RUNPOD_STOP_WARNING` | `5` | Minutes of warning before that |
| `PHANTOM_SESSION_GRACE` | `120` | Seconds with no client before the session is erased. `0` disables |

Leave `RUNPOD_GPU_TYPES` unset unless you have a reason to pin. Auto-discovery
queries RunPod, filters by VRAM and price, drops GPUs whose compute capability
exceeds what the image's PyTorch and ONNX builds support — Blackwell `sm_120` on
an `sm_90` image, say — and tries the cheapest first. A pinned list gets none of
that and goes stale as RunPod's fleet changes.

---

## Session erase

The pipeline holds the operator's face: the source photos, the embedding built
from them, uploaded target photos and videos, and every swapped output. All of
it lives under one tree — `PHANTOM_TEMP_DIR`, else `/workspace/tmp/phantom` when
a network volume is mounted — so one delete covers it.

That matters because the pod is rented and handed on afterwards, and **the
network volume survives a stop**, so an upload outlives the customer who made
it unless something removes it.

Two things trigger the erase. The desktop sends `cleanup_session` as it closes,
which is the prompt path; and the pipeline sweeps on its own once the last
client has been gone for `PHANTOM_SESSION_GRACE` seconds, which is what covers
the app being killed rather than closed.

The grace period is why it is a delay and not an event. `PipelineClient`
reconnects indefinitely by design — a pod can be slow and a laptop can sleep —
so a dropped socket is usually a live session rather than a finished one, and
erasing on the disconnect itself would delete the operator's face mid-call every
time the link hiccuped. A reconnect inside the window calls the sweep off.

Set it to `0` on a pipeline that several clients legitimately come and go from.
A desktop that reconnects after the erase is told: `_restore_state_from_server`
reads back an empty source and clears the sidebar, rather than leaving it
claiming a face the pod no longer has.

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

**`resume` says there are not enough free GPUs.** The host your pod is pinned
to is full. `resume` falls through to `start` on its own — see [When `resume`
cannot get a GPU](#when-resume-cannot-get-a-gpu), and remember to terminate the
old pod from the dashboard.

**SSH timeout.** The proxy accepts TCP before the container is ready. The
orchestrator already retries 12 times at 10-second intervals; a failure past that
usually means the pod itself is unhealthy. Retry `start`.

**Pipeline never binds 9000 (ssh mode).**

```bash
ssh <podHostId>@ssh.runpod.io
tail -f /workspace/phantom-pipeline.log
```

Don't assemble that first line by hand — **`status` prints it**, ready to
paste, whenever the pod is running:

```
$ python runpod/orchestrator.py status
Pod:    phantom (abc123)
Status: RUNNING
...
SSH:    ssh <podHostId>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

`podHostId` is per-pod, changes on every deploy, and is not the pod ID — nor is
it returned by `runpod.get_pod()`, which is why `_get_ssh_command` asks GraphQL
for it (see [runpod/TROUBLESHOOTING.md](runpod/TROUBLESHOOTING.md)). The pod's
own page in the dashboard shows the same line under **Connect → SSH**; `status`
just saves the trip, and reads `RUNPOD_SSH_KEY_PATH` so the `-i` matches your
actual `.env`.

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

**Proxy URLs are pod-specific — treat them as credentials and do not share
them.** That is the only mitigation currently standing between a running pod
and anyone on the internet.

The full posture, including what is deliberately accepted and what must close
before a paying customer, lives in **[docs/ACCEPTED_RISKS.md](docs/ACCEPTED_RISKS.md)**.
The short version: the WebSocket API has **no authentication**, frames are
broadcast to every connected client, and the forwarded `RUNPOD_API_KEY` is
account-wide. The desktop's access-code gate does not change any of this — it
is client-side and gates the UI, not the pod.
