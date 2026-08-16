# Session Plane — Architecture Assessment

Assessment of the proposed session-based, remotely executed facial-augmentation
platform: GPU pooling, regional redundancy, persistent storage, fault-tolerant
execution.

Reviewed against `main` @ `08f252d`.

**This document is the assessment** — what the design gets right, what needs
correcting, what it costs, and the order to build it in. The design itself is
specified in [SESSION_ARCHITECTURE.md](SESSION_ARCHITECTURE.md), which stands
alone and is implementable without this one.

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

## What the product is

Settled, and it removes two of the original open questions:

- **The client is our desktop app.** Customers do not integrate against a
  protocol. `desktop.py` is the product surface; the WebSocket API stays an
  internal detail between our app and our workers.
- **A session is time, not a mode.** A customer buys a block — five minutes, an
  hour — connects, and within that block uses either live video call *or* batch
  processing, switching freely.

Both have consequences the original proposal does not cover, because it assumes
a session is one kind of work.

The first is uncomfortable: **batch video is a launch prerequisite, not a
follow-up.** It is currently unimplemented
(`ProcessingPipeline._process_target_batch()` handles images only). You cannot
sell "an hour of the platform, live or batch" while half of that sentence
returns an error.

The second is that **the desktop app now needs to become a client of a service**
rather than of a pod. Today `PHANTOM_API_URL` points at a specific pod, set by
hand in `.env`. It needs identity, session purchase and remaining-time state,
worker assignment, and reconnection.

The third is file transfer, and it is bigger than it looks. Batch on a remote
GPU means getting the target video *to* that GPU and the result back.
`handle_upload_source` exists but is base64 inside a single JSON WebSocket
message — fine for a 200 KB source face, unusable for a 2 GB video: 33% inflation,
held entirely in memory at both ends, no chunking, no resume, no progress. A
real transfer path is required work, and every second of it is billed session
time unless it is deliberately excluded.

---

## The economics, corrected for session length

My first pass assumed the five-minute minimum was typical. At hour-long sessions
the picture changes enough to restate.

```
Assuming $2.00/min ($10 per 5 minutes) holds at all durations, GPU at $1.00/hr

                              5-min session      60-min session
  revenue                          $10.00             $120.00
  GPU cost, 1 session/card          $0.083              $1.000
  GPU cost, 2 sessions/card         $0.042              $0.500
  packing saves                     $0.042              $0.500
  90s cold start, unbilled          $3.00               $3.00
    as % of the purchase             30.0%                2.5%
  ─────────────────────────────────────────────────────────────
  cold start : packing                72x                  6x
```

Cold start still wins, but by **6×, not 72×**. The absolute loss is identical —
90 seconds at $2/min is $3 whatever the session length — but as a share of the
purchase it collapses from 30% to 2.5%. Long sessions amortise startup; short
ones are dominated by it.

### I under-valued packing

Reading packing as a cost saving was the wrong lens. Four cents, or fifty, is
noise either way. The right lens is capacity:

```
  revenue per GPU-hour, 1 session   $120
  revenue per GPU-hour, 2 sessions  $240      ← this is the argument
```

The proposal's own central worry is that GPU availability fluctuates between
regions — that supply, not price, is the constraint. If that is true, packing is
not a 50-cent saving. It **doubles the demand you can serve with the cards you
can actually get.**

That does not move it earlier in the plan, because it stays blocked behind the
largest refactor here. But it changes why it matters, and it should be
re-elevated the moment availability rather than cost becomes the limiter.

### Cold start, drawn against the short session

Where startup hurts most is the five-minute purchase, so that is the case worth
picturing. The same 90 seconds against an hour is a thin sliver at the left
edge — real money, but no longer the shape of the product.

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

## Two workloads, one session

A session that can be either live or batch is really a lease on capacity that
runs two workloads with opposite profiles. The scheduler has to know the
difference.

```
                 LIVE                         BATCH
  latency        hard budget (33ms/frame)     irrelevant
  duration       holds the slot throughout    bursty: saturate, then idle
  interruption   visible failure              retryable, resumable
  scaling        bounded by worst frame       bounded by throughput
  preemptible    no                           yes
```

Three consequences:

- **Do not co-schedule batch onto a card serving live** until measurement says
  it is safe. A batch job saturating the GPU is exactly the spike that blows a
  live session's frame budget. Batch is preemptible; live is not.
- **Batch is where the fault tolerance actually pays.** Everything the proposal
  describes — queue, retry, recover on another GPU — works properly for a file
  job and only partially for a call. The machinery is not wasted; it is simply
  better matched to the half of the product that was going to be built second.
- **An idle live session is not idle capacity.** A customer who has paid for an
  hour and is between calls still holds their slot. Slot accounting must track
  purchased time, not activity.

### Jobs that outlive the session

Customer buys an hour, starts a 90-minute export at minute 55. This needs a
decided policy, and all four options are defensible:

1. Refuse jobs that cannot finish in the remaining time (needs a duration
   estimate, and a wrong estimate is worse than no estimate).
2. Run it, bill the overflow.
3. Run to completion as a courtesy, absorb the cost.
4. Detach the job from the session — finish it, hold the result for download.

Option 4 is the most generous and the most work, and it is the one that makes
batch genuinely different from live rather than an awkward guest inside a
live-shaped session.

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

**Revision:** I originally proposed starting the clock at the first delivered
frame. That was written assuming live-only. A session that may open in batch
mode has no first frame to wait for, and a customer who attaches and then spends
two minutes choosing a source image has not been idle — they have been using the
product.

The rule that generalises is **the clock starts when the session becomes
usable**: worker ready, models loaded, client attached. Before that point the
customer cannot do anything, so it is our cost. After it, their time is their
own regardless of which mode they pick or whether they are actively streaming.

That preserves the property that matters — cold start sits on **our** side of the
ledger, so the scheduler, the warm pool and the retry policy all optimise for the
same thing the customer cares about — while covering both workloads.

If the clock instead starts at session creation, every one of those systems is
free to be slow at the customer's expense, and eventually will be.

Upload time is the one case that needs an explicit decision rather than a
default. A 2 GB video transfer is minutes of wall clock during which the GPU is
doing nothing. Billing it is defensible and will feel like theft; excluding it
invites abuse. The middle position — bill it, but transfer while the worker is
still cold, so it overlaps startup rather than eating session time — is probably
right and needs the transfer path designed for it from the start.

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

### 3. Close the product gap: batch video and file transfer

*Ships: the other half of what a session is sold as. Launch prerequisite.*

Not control-plane work, but nothing above can be sold without it.

- Wire `_process_target_batch()` for video. The FFmpeg building blocks already
  exist in `pipeline/io/ffmpeg.py` — `extract_frames`, `create_video`,
  `restore_audio`, `clean_temp` — and batch reuses the same compositor as live,
  so most of the work is frame iteration plus audio and FPS restoration.
- Build a real file transfer path: chunked, resumable, progress-reporting, in
  both directions. `upload_source`'s base64-in-a-JSON-message approach does not
  extend to video.
- Fixes the CI end-to-end test and desktop VIDEO mode as a side effect; both are
  currently broken against this gap.

### 4. Control plane, one session per GPU

*Ships: the product. Customers can buy and run sessions.*

- Session manager with the explicit state machine, a session store, and the
  session/attempt split.
- Billing clock starting at first delivered frame.
- Promote `orchestrator.py`'s discovery and multi-datacenter fallback into a
  scheduler service. A port, not a rewrite — and it brings regional redundancy
  along for free.
- Session state must carry the active mode, since a customer switches between
  live and batch inside one session.
- Deliberately **no packing yet**. One session per worker is correct and safe.

### 5. Resilience

*Ships: sessions that survive infrastructure failure, or refund themselves.*

- Worker → backend heartbeat and session leases, so a dead GPU stops being
  advertised as available.
- Watchdog around the frame loop specifically — **progress-based, not
  liveness-based**, because the CUDA-hang case leaves the process responsive.
- Retry classification and bounded attempts, as proposed.
- Automatic credit for interrupted minutes.

### 6. Multi-tenancy and packing

*Ships: margin improvement. Only worth doing at volume.*

- Remove the `CONFIG` and `BUS` singletons in favour of per-session context
  objects — the largest single change to existing code in this plan.
- Route frames by `session_id` instead of broadcasting to a client set.
- Share one set of loaded models across sessions; ONNX Runtime sessions are
  already safe to call from multiple threads.
- Enforce the measured `max_sessions` in the scheduler's slot accounting.

At $10 per five minutes this stage is a rounding error until many sessions run
concurrently. It is listed fifth because it earns fifth place.

### 7. Provider abstraction

*Ships: insurance against a single vendor's availability and pricing.*

Define the provider interface — provision, status, terminate, list capacity —
and move RunPod specifics behind it. Cheap now, because there is exactly one
implementation to conform to it and its shape is already visible in
`orchestrator.py`.

---

## Settled

- **Who runs the client.** Our desktop app. No public protocol, no third-party
  integration surface. The WebSocket API stays internal.
- **Whether batch belongs in the same plane.** Yes — in the same *session*. A
  customer buys time and chooses the mode, which promotes batch video from a
  follow-up to a launch prerequisite.

## Still open

**What happens at the session boundary?**
Hard cut, or top-up? A call ending mid-sentence is a bad experience; an
auto-extending session is a billing surprise. This is the auto-stop warning we
already have, pointed at the customer instead of at us — the mechanism
transfers, the policy does not. Sharper now that sessions may be an hour: the
sunk cost of losing a session at minute 59 is much larger.

**What happens to a batch job that outlives its session?**
Four defensible options, listed under *Two workloads, one session*. Needs a
decision before batch ships, because it determines whether a job is owned by the
session or merely started by it.

**Is upload time billed?**
A 2 GB transfer is minutes during which the GPU does nothing. Billing it feels
like theft; excluding it invites abuse. Overlapping transfer with worker startup
is probably the answer, but it has to be designed in from the start.

**Is a session one face, or one seat?**
Source embedding, quality preset and realism settings are currently
per-process. If a customer switches source faces mid-session, that is per-session
state; if a session is one identity throughout, embeddings can be cached per
worker and reused, which meaningfully cheapens packing.

**Does the price hold at length?**
$2/min is $120/hour. Everything in the economics section assumes the five-minute
rate scales linearly; if hour-long sessions are discounted, the GPU-cost ratios
shrink and packing gets more important than shown here.

**What is the concurrency shape?**
Many short sessions and few long ones imply different systems. Short sessions
make cold start and the warm pool dominant; long sessions make packing and
capacity planning dominant. This is the single biggest unknown remaining, and it
decides whether stage 2 or stage 6 is where the leverage is.

**Where does authentication and payment live?**
Not addressed in the proposal or here. The desktop app currently has no concept
of a user. Purchase, entitlement and remaining-time state all have to originate
somewhere before the session API means anything.

---

## The short version

Build it. The proposal is sound, most of the hard thinking is already done in
it, and it is additive to a pipeline that works.

Reorder the first moves: **measure the two unknown numbers, close the batch and
file-transfer gap, attack cold start, then build the control plane at one session
per GPU.** Packing stays late — not because it is low value, but because it is
blocked behind the largest refactor in the plan.

Two things determine how much of the rest is right. **Whether sessions are
typically minutes or hours** decides whether cold start or capacity is the real
constraint; they differ by an order of magnitude and the answer is not known.
And **batch is no longer optional** — it is half of what a session is sold as.

Then take the one decision that costs nothing today and is painful to retrofit:
**start the billing clock when the session becomes usable**, not when it is
created. Everything else then optimises in the customer's direction by default.
