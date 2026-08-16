# Session Plane — Architecture Assessment

Assessment of the proposed session-based, remotely executed facial-augmentation
platform: GPU pooling, regional redundancy, persistent storage, fault-tolerant
execution.

Reviewed against `main` @ `08f252d`.

> Illustrated version: published as a private artifact. This document is the
> version-controlled copy and is the source of truth for the numbers.

---

## The verdict

The proposal's stack diagram bottoms out in a box labelled *facial pipeline*.
That box is this entire repository. Everything above it is new, and almost
nothing below it needs to change conceptually.

**This is a control plane to be built above the pipeline, not a replacement for
it.**

Roughly a third of what the proposal describes already exists, mostly inside
`runpod/orchestrator.py` — GPU discovery filtered by VRAM, price and compute
capability; multi-datacenter fallback with paired volumes; cheapest-first
ordering; persistent model storage. That is a scheduler. It just runs once, from
a laptop, for one session. Much of the work is to *relocate* that knowledge into
a service, not to invent it.

### Keep verbatim

- **Session ≠ GPU attempt.** The best idea in the document, and a
  billing-correctness issue rather than an aesthetic one.
- **Heartbeats, leases, and a pipeline watchdog.** The observation that a live
  process does not imply a healthy workload is exactly right — a CUDA call can
  hang while the worker happily answers pings.
- **The provider abstraction.** Cheap to build now while there is one provider,
  expensive to retrofit once scheduling logic has grown RunPod-shaped
  assumptions.

### Needs correcting

- **Session packing cannot be built yet.** Two customers on one worker process
  would currently see each other's video. Hard blocker, and it is in our code.
- **VRAM is the wrong capacity metric.** For this workload the binding
  constraint is compute and host CPU.
- **Live sessions cannot be "recovered" the way batch jobs can.** The machinery
  is sound; what it delivers for a live call is a visible outage.

And one priority inverted — see below.

---

## The economics say cold start, not packing

Run the proposal's pricing through its own infrastructure numbers.

```
Session price          $10.00 per 5 minutes
GPU                    $1.00 / hour

GPU cost per session   1 session/GPU   →  $0.083   (120x revenue)
                       2 sessions/GPU  →  $0.042   (240x revenue)

Saving from packing 1 → 2               =  $0.042  per session
90s of cold start (30% of the purchase) =  $3.00   per session
                                           ─────────────────────
                                           ~72x more valuable
```

Compute is between a 120th and a 240th of revenue. The headline infrastructure
idea — packing two sessions onto one card — is worth about **four cents** per
session. Meanwhile the pod must be provisioned, the repo pulled, dependencies
checked, and four ONNX models loaded before the customer sees a single frame.

On a five-minute minimum purchase, that startup is not overhead. It is the
product.

```
                0:00                                              5:00
                 │                                                  │
TODAY            ├──────────┬────┬─────┬──────────────────────────────┤
                 │ provision│setup│models│      usable — 3:30         │
                 │◀─── ~90s paid for, not delivered ───▶│

WARM SLOT        ├┬─────────────────────────────────────────────────┤
                 ││              usable — 4:57                       │
                 │└─ attach to pre-loaded worker (~3s)
                 │
                 └────────▶ +87 seconds of sold time recovered
```

> **Estimate, not a measurement.** The 90-second figure is assembled from the
> steps in `runpod/startup.sh` — provisioning, `apt-get` for ffmpeg,
> `git pull`, a conditional `pip install`, then parallel warm-up of four ONNX
> models, two of which download on first use. It has never been timed end to
> end. On a cold volume it is considerably worse. Measuring it is stage 1 for
> exactly this reason.

---

## What already exists

| Proposal component | Status | Where it lives today |
|---|---|---|
| Persistent storage / disposable GPU | **have** | Network volume; models resolve to `/workspace/models` before local paths |
| Regional redundancy | **have** | `RUNPOD_DATACENTERS=DC1:vol1,DC2:vol2` with per-region volume pairing |
| GPU selection by cost and capability | **have** | `_discover_gpus()`, `_resolve_gpu_candidates()` — VRAM, price, compute-cap filters, cheapest first |
| Session expiry / purchased time | partial | Auto-stop timer + `keep_alive` — protects *us* from billing overrun, not the customer's clock |
| Worker readiness | partial | WebSocket `health` check, polled once at provision time |
| Heartbeat | partial | 30s ping/pong exists, but *client ↔ worker* — the backend learns nothing from it |
| Scheduler as a service | **missing** | Logic exists; runs once, in a CLI, on a developer's laptop |
| Session manager, queue, state machine | **missing** | — |
| Leases, watchdog, retry classification | **missing** | — |
| Multi-tenant worker | **blocked** | Prevented by module-level singletons |

```
                                              ┌─ status ─────────────┐
      Customer app          ◀── new           │ new  = to build      │
           │  POST /sessions                  │ proto= CLI prototype │
      Session API           ◀── new           │ have = works today   │
           │                                  └──────────────────────┘
      Session manager       ◀── new
           │  state machine · billing clock
      Scheduler             ◀── proto   runpod/orchestrator.py
           │  slots · regions · retries
      Worker supervisor     ◀── new
           │  heartbeat · lease · watchdog
      WebSocket API server  ◀── have
           │  frames in / out
      ProcessingPipeline    ◀── have   pipeline/
           │
      Volume · models       ◀── have
```

---

## Three assumptions that need correcting

### 1. Packing is blocked in our code, not the control plane

The worker process is single-tenant by construction, in three separate ways:

- `CONFIG` is a module-level singleton — `pipeline/config.py:205`
- `BUS` is a module-level singleton — `pipeline/events.py:122`
- The API server holds one `Set` of clients and broadcasts every encoded frame
  to all of them — `pipeline/api/server.py`

That last one is not a performance problem. It is a privacy incident.

```
TODAY — ONE PROCESS                    TARGET — SAME PROCESS
┌────────────────────────────┐         ┌────────────────────────────┐
│ worker process             │         │ worker process             │
│  ┌────────┐  ┌────────┐    │         │  ┌──────────────────────┐  │
│  │ CONFIG │  │  BUS   │    │         │  │ shared models (once) │  │
│  └────────┘  └────────┘    │         │  └──────────────────────┘  │
│  ┌──────────────────────┐  │         │  ┌───────────┐┌─────────┐  │
│  │ _clients: Set        │  │         │  │ Session A ││Session B│  │
│  │   → broadcast to all │  │         │  │    ctx    ││   ctx   │  │
│  └──────────┬───────────┘  │         │  └─────┬─────┘└────┬────┘  │
└─────────────┼──────────────┘         └────────┼───────────┼───────┘
        ┌─────┴─────┐                           │           │
        │  ╳ crossed│                           │           │
        ▼           ▼                           ▼           ▼
      ( A )       ( B )                       ( A )       ( B )

  each receives the other's             routed by session_id,
  video and mutates the                 models still shared
  other's config
```

The tempting shortcut is one OS process per session. That works and it is safe,
but it throws away the reason to pack: each process loads its own copy of all
four models — roughly a gigabyte of VRAM duplicated per session, for no benefit.

The version worth building keeps one set of loaded models and gives each session
its own config, event bus and frame route.

The concurrency works out: OpenCV, NumPy and ONNX Runtime all release the GIL
during their heavy calls, so sessions running as threads in one process
genuinely execute in parallel rather than queueing behind the interpreter.

### 2. Compute and CPU bind before VRAM does

The proposal reasons at length about high-VRAM cards and sketches four sessions
inside 128 GB. For this pipeline that framing is backwards. Once models are
shared, the resident set is roughly a gigabyte total — detection, swap,
restoration, occluder — and each additional session adds only working buffers.

What actually binds, in order:

| Constraint | Per session | Why it binds |
|---|---|---|
| GPU compute | ~80 inferences/s | Four models per frame at 20 fps. Detection runs every frame and is the most expensive of the four. |
| Host CPU | 0.26–1.14 cores | Compositing is pure OpenCV — measured at ~13 ms/frame at 256, ~38 ms for a close-up at 320. The GPU does not help with any of it. |
| VRAM | a few MB | Only if models are shared. Without sharing, ~1 GB per session and this moves to the top. |

The CPU line is the one to watch, and the proposal does not mention CPU at all.
A pod is rented with a fixed vCPU allocation attached to the GPU, and it is
entirely possible to hit the CPU wall with the GPU half idle. **Instrument
both.**

> **Already banked.** The preset work in `6aa7350` raises session density
> directly: detector input dropped from a fixed 640² to 320/448/640 by preset,
> and compositing now sizes itself to the face rather than always running at the
> ceiling. On `fast` that is a quarter of the detector pixels and ~2.5× cheaper
> compositing for a distant subject — against the most expensive per-frame
> operation. Whatever `max_sessions` turns out to be, it is higher than it was.

### 3. A live session cannot be recovered, only restarted

The fault-tolerance section is written in the vocabulary of batch inference:
detect the failure, find another GPU, restore execution. For a job that takes a
file and returns a file, that genuinely is recovery.

A live call is different. Restoring execution means provisioning a pod, loading
four models, and reconnecting the client. Even warm that is seconds; cold it is
the 90 seconds above. The customer does not experience a recovered session —
they experience their face falling off during a call.

This does not make the machinery wrong. It changes what you build it to *do*:

- **A warm standby slot is the only real mitigation.** Recovery time is
  dominated by model loading, so spare capacity must be pre-loaded, not merely
  provisioned.
- **Bias the scheduler toward stability over price.** The cheapest card that
  gets reclaimed mid-call costs far more than the cents saved.
- **Make the client hold the session, not the connection.** Frames are already
  pushed from the client over WebSocket, so the worker holds almost no
  unrecoverable state — reconnect-and-resume is cheap once a warm worker exists.
- **Compensate in billing, not in fiction.** Credit interrupted minutes
  automatically. Smaller build than seamless migration, and what the customer
  actually wants.

---

## Session, attempt, and the billing clock

"Don't make the customer pay three times because your infrastructure had to
recover" is correct. The part left unspecified is **when the clock starts** — and
on a five-minute minimum, that detail is most of the customer's experience.

```
SESSION #ABC123 — one purchase, one identity
┌──────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐ │
│ │ Attempt 1    │ → │ Attempt 2    │ → │ Attempt 3 · GPU C          │ │
│ │ GPU A        │   │ GPU B        │   │ RUNNING                    │ │
│ │ alloc timeout│   │ worker died  │   │ heartbeat · lease · watchdog│ │
│ └──────────────┘   └──────────────┘   └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
 session                                 │
 created                                 ▼
    ├────────────────────────────────────┼█████████████████████████████
                                         │        billed time
                                         └─ first delivered frame:
                                            the clock starts HERE

 two failed attempts — not billed, not the customer's problem
```

Starting the clock at first frame is the decision that ties the architecture
together. It puts cold start on **our** side of the ledger, which means the
scheduler, the warm pool and the retry policy all optimise for the same thing
the customer cares about.

If the clock instead starts at session creation, every one of those systems is
free to be slow at the customer's expense — and eventually will be.

---

## A staged path

Ordered by dependency and by what each stage makes sellable.

### 1. Measure the two unknown numbers

*Ships: nothing customer-facing. Unblocks every decision below.*

`max_sessions` and time-to-first-frame are both guesses today, and every
capacity and pricing decision depends on them.

- Load harness driving *N* concurrent synthetic clients pushing frames at a
  fixed rate, ramping 1 → 2 → 3 → 4.
- Record per session: end-to-end latency, achieved fps, dropped frames. Per
  host: GPU utilisation, VRAM, **CPU per core**.
- Separately, time a cold provision end to end, broken down by phase, on both a
  warm and an empty volume.

Define the ceiling by a **latency budget**, not by absence of crashes — the
session is unusable long before it OOMs.

### 2. Attack cold start

*Ships: a session that starts in seconds. Largest revenue impact in the plan.*

- Bake a Docker image with dependencies and model weights inside, so `apt-get`,
  `git pull` and `pip install` leave the critical path. `orchestrator.py`
  already has a docker deploy mode.
- Keep a small **warm pool**: workers provisioned with models loaded and no
  session assigned. Same mechanism as the five-minute grace period, generalised
  — hold capacity ahead of demand, not only behind it.
- Pre-seed both regional volumes, so a fallback region is not silently the slow
  path.

### 3. Control plane, one session per GPU

*Ships: the product. Customers can buy and run sessions.*

- Session manager with the explicit state machine, a session store, and the
  session/attempt split.
- Billing clock starting at first delivered frame.
- Promote `orchestrator.py`'s discovery and multi-datacenter fallback into a
  scheduler service. A port, not a rewrite — and it brings regional redundancy
  along for free.
- Deliberately **no packing yet**. One session per worker is correct, safe, and
  costs eight cents.

### 4. Resilience

*Ships: sessions that survive infrastructure failure, or refund themselves.*

- Worker → backend heartbeat and session leases, so a dead GPU stops being
  advertised as available.
- Watchdog around the frame loop specifically — **progress-based, not
  liveness-based**, because the CUDA-hang case leaves the process responsive.
- Retry classification and bounded attempts, as proposed.
- Automatic credit for interrupted minutes.

### 5. Multi-tenancy and packing

*Ships: margin improvement. Only worth doing at volume.*

- Remove the `CONFIG` and `BUS` singletons in favour of per-session context
  objects — the largest single change to existing code in this plan.
- Route frames by `session_id` instead of broadcasting to a client set.
- Share one set of loaded models across sessions; ONNX Runtime sessions are
  already safe to call from multiple threads.
- Enforce the measured `max_sessions` in the scheduler's slot accounting.

At $10 per five minutes this stage is a rounding error until many sessions run
concurrently. It is listed fifth because it earns fifth place.

### 6. Provider abstraction

*Ships: insurance against a single vendor's availability and pricing.*

Define the provider interface — provision, status, terminate, list capacity —
and move RunPod specifics behind it. Cheap now, because there is exactly one
implementation to conform to it and its shape is already visible in
`orchestrator.py`.

---

## Open questions

**Who runs the client, and where do frames come from?**
Today `desktop.py` captures the webcam and pushes JPEG frames to the pod. If
customers use their own application, the frame source and virtual-camera output
move into their environment, and what we sell becomes a protocol rather than an
app. This shapes the session API more than anything else in the proposal.

**Is a session one face, or one seat?**
Source embedding, quality preset and realism settings are currently
per-process. If a customer expects to switch source faces mid-session, that is
per-session state; if a session is one identity for its duration, embeddings can
be cached per worker and reused, which meaningfully cheapens packing.

**What happens at the five-minute boundary?**
Hard cut, or top-up? A call ending mid-sentence is a bad experience; an
auto-extending session is a billing surprise. This is the auto-stop warning we
already have, pointed at the customer instead of at us — the mechanism
transfers, the policy does not.

**Does batch belong in the same plane?**
Batch video is still unimplemented, and it is genuinely well-suited to
everything in the proposal — queueing, retrying and recovering a file job is
what that machinery is good at, without the live-session caveats. Possibly worth
treating batch as a second session type rather than a separate product.

---

## The short version

Build it. The proposal is sound, most of the hard thinking is already done in
it, and it is additive to a pipeline that works.

But reorder the first moves: **measure the two unknown numbers, then attack cold
start, then build the control plane at one session per GPU.** Packing — the idea
the proposal is organised around — is worth roughly four cents a session and is
blocked behind the largest refactor in the plan. It belongs fifth, not first.

And take the one decision that costs nothing today and is painful to retrofit:
**start the billing clock at the first delivered frame.** Everything else then
optimises in the customer's direction by default.
