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

- **Packing is not blocked.** Two sessions on one GPU means two pipeline
  processes, isolated by construction, and that works today. What is blocked is
  *sharing loaded models* between them — an optimisation, not a prerequisite.
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
- **A session is time, not a mode.** A customer buys a block — an hour PAYG, a
  4- or 10-hour pack, a 24-hour Day Pass — connects, and within that block uses
  either live video call *or* batch processing, switching freely.

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

## The economics

> **Revised for the tiered pricing.** Earlier drafts assumed $10 per 5 minutes —
> $120/hour. The proposed tiers run $10/hour down to $4.17/hour. That is a **12×
> to 29× drop in revenue per hour**, and it inverts the central conclusion.
> Everything in this section replaces what was here before.

```
Tiers, against a $1.00/hr GPU

  tier             price     $/hr   disc   GPU 1x   GPU 2x   margin 1x  2x
  ──────────────────────────────────────────────────────────────────────────
  PAYG            $10.00   $10.00     0%    $1.00    $0.50        90%  95%
  4-Hour Pack     $35.00    $8.75    12%    $4.00    $2.00        89%  94%
  10-Hour Pack    $70.00    $7.00    30%   $10.00    $5.00        86%  93%
  Day Pass       $100.00    $4.17    58%   $24.00   $12.00        76%  88%
```

Margins are healthy — 76% to 95% — but they are *margins* now, not rounding
errors. GPU cost was a 240th of revenue under the old pricing. On a fully-used
Day Pass it is nearly a quarter.

### This reverses the priority I recommended

The previous conclusion was that cold start beats packing by 72×. At these
prices the comparison flips, and not marginally:

```
  tier            $/min   90s cold start   packing saves      winner
  ─────────────────────────────────────────────────────────────────────
  PAYG            0.167            $0.25           $0.50   packing   2x
  4-Hour Pack     0.146            $0.22           $2.00   packing   9x
  10-Hour Pack    0.117            $0.17           $5.00   packing  29x
  Day Pass        0.069            $0.10          $12.00   packing 115x
```

**Packing is now the dominant financial lever, by up to two orders of
magnitude.** Cold start's financial argument has collapsed — 90 seconds costs
between 10 and 25 cents, against $3.00 before.

Cold start still matters, but as a **user-experience** problem rather than a
revenue one, and it arguably matters more than it did: a Day Pass holder
connects and disconnects repeatedly across a day, so they meet the cold start
many times rather than once.

### Two consequences worth acting on

**Payment is Bitcoin over Lightning**, which removes what would otherwise be
the second-largest cost line. On cards, a $10 PAYG charge costs $0.59 to collect
against $0.50 of packed GPU — the fee would exceed the compute.

```
  rail                PAYG $10        Day Pass $100
  ─────────────────────────────────────────────────
  Stripe card      $0.59  (5.9%)     $3.20  (3.2%)
  BTC on-chain     $3.00 (30.0%)     $3.00  (3.0%)   ← unusable at $10
  BTC Lightning    $0.02  (0.2%)     $0.20  (0.2%)   ← chosen
```

Switching PAYG from cards to Lightning saves $0.57 per session — **more than
packing 1→2 saves ($0.50)**, for an integration rather than a refactor. It is
the cheapest margin win available.

Two knock-on effects worth noting. Fee overhead is now flat at 0.2% across all
tiers, so **the tier ladder can no longer be justified on payment costs** — it
rests on commitment and on utilisation, which is fine, but the earlier argument
that packs reduce payment overhead no longer applies. And on-chain remains
viable only for the Day Pass, so Lightning cannot be the sole rail if you want
an on-chain fallback at the top tier.

### One cost line is still unmodelled: bandwidth

Every margin figure in this document treats bandwidth as free, which is an
assumption rather than a finding.

Live streaming moves roughly **2.2 GB/hour outbound** — processed frames going
from the pod back to the customer — and the same again inbound. Over a 24-hour
Day Pass that is about **53 GB of egress**.

```
  egress billed at    cost on a $100 Day Pass    share of revenue
  ────────────────────────────────────────────────────────────────
  $0.00/GB  (included)          $0.00                     0%
  $0.05/GB                      $2.65                   2.7%
  $0.10/GB                      $5.31                   5.3%
```

At $0.05/GB this would be the **second-largest cost line after the GPU**, about
thirteen times the Lightning fee. RunPod is believed to include bandwidth on
pods, which is why it is modelled at zero — but that has not been verified, and
it is the only cost here set to zero without evidence. Worth confirming before
the Day Pass margin is trusted.

**PAYG's "full hour deducted at session start" is a gift to the architecture.**
Once the hour is paid, holding that customer's worker for the remainder of it
costs at most $1 against $10 collected — and it eliminates reconnect cold starts
entirely inside the paid window. The grace period for a PAYG session should
therefore be *the remainder of the paid hour*, not five minutes.

That does not generalise to a Day Pass: holding a worker for 24 hours costs $24
of $100. **Grace period policy should differ by tier**, which the architecture
currently treats as a single global constant.

### The Day Pass, now that consumption is settled

The session rule resolves what used to be the biggest open number here. A Day
Pass is **24 hours of balance, drawn down one wall-clock hour at a time**, each
deducted when a session becomes usable and consumed whether or not the customer
does anything with it. So GPU cost is bounded by hours actually *connected*, not
by 24 hours of standing availability — and any hour the customer buys but never
connects is pure margin.

That makes 24 the worst case rather than the expected one:

```
  hours actually connected   cost 1x   cost 2x   margin 1x   margin 2x
  ──────────────────────────────────────────────────────────────────────
                        4      $4.00     $2.00         96%         98%
                        8      $8.00     $4.00         92%         96%
                       12     $12.00     $6.00         88%         94%
                       24     $24.00    $12.00         76%         88%   ← worst case
```

Not loss-making at any point. The remaining exposure is a $2/hr GPU held
unpacked across a fully-consumed pass, which is a **52% margin** — and that is
before bandwidth, storage, orchestration and failed sessions.

Two things still follow. `max_sessions` is a **pricing input**, not just a
capacity number: a fully-consumed Day Pass yields 76% at one session per card
and 88% at two. And one question survives — whether Day Pass hours **expire**.
"Day Pass" implies they do; if they do not, it is simply a 24-hour pack under
another name, and the tier ladder has three prepaid blocks rather than two plus
a pass.

### Cold start, drawn

Still worth picturing, now as a UX cost rather than a revenue one. The timeline
below uses the old five-minute session because that is where startup is most
visually dominant; at $10/hour the same 90 seconds is 2.5% of the purchase.

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
| Two sessions per GPU | **have** | Two pipeline processes, isolated by construction. Needs three shared paths fixed — see below |
| Shared models across sessions | **missing** | Prevented by module-level singletons. An optimisation of packing, not a prerequisite for it |

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

### 1. Packing works today; only *shared models* are blocked

> **Correction.** An earlier draft claimed packing was blocked and that two
> customers would see each other's video. That was wrong, and wrong in a way
> worth recording: it took a property of one implementation and stated it as a
> property of the idea.

Two sessions on one GPU means **two pipeline processes**. Each gets its own
`CONFIG`, its own `BUS`, its own WebSocket server on its own port, its own
client set. They are isolated by construction and need no refactor.

```
  PACKING — available now              SHARED MODELS — needs the refactor
┌──────────────────────────────┐     ┌──────────────────────────────────┐
│ GPU                          │     │ GPU                              │
│ ┌────────────┐┌────────────┐ │     │ ┌──────────────────────────────┐ │
│ │ process A  ││ process B  │ │     │ │ one process                  │ │
│ │ :9000      ││ :9001      │ │     │ │ ┌──────────────────────────┐ │ │
│ │ own CONFIG ││ own CONFIG │ │     │ │ │ models — loaded ONCE     │ │ │
│ │ own BUS    ││ own BUS    │ │     │ │ └──────────────────────────┘ │ │
│ │ own models ││ own models │ │     │ │ ┌─────────┐  ┌─────────┐     │ │
│ │  ~1 GB     ││  ~1 GB     │ │     │ │ │ ctx A   │  │ ctx B   │     │ │
│ └─────┬──────┘└──────┬─────┘ │     │ │ └────┬────┘  └────┬────┘     │ │
└───────┼──────────────┼───────┘     └──────┼─────────────┼───────────┘ │
        ▼              ▼                     ▼             ▼
      ( A )          ( B )                 ( A )         ( B )

  isolated, ~1 GB VRAM each,          isolated, ~1 GB total,
  N model loads                       one model load
```

What the singleton state (`pipeline/config.py:205`, `pipeline/events.py:122`,
and the shared client `Set` in `pipeline/api/server.py`) actually prevents is
running two customers **inside one process** — which is the model-sharing
optimisation, not packing.

#### Three shared paths to fix first

Two-process packing is safe except where the processes touch the same
filesystem or the same pod:

| Collision | Consequence | Fix |
|---|---|---|
| `_UPLOAD_DIR = '/tmp/phantom_uploads'` | Two customers upload `face.jpg`; one overwrites the other | Scope by session id |
| Batch temp dirs derived from the target filename | Same-named targets collide | Scope by session id |
| Auto-stop calls `runpod.stop_pod()` (`api/server.py`) | Whichever process's timer fires first kills the pod **and every session on it** | Move pod lifecycle to the control plane; workers must not stop pods |

The third is architectural rather than a bug. In the target design the scheduler
owns the GPU, so that code leaves the worker entirely.

#### What the refactor is actually worth

Not packing. Two narrower things:

- **VRAM per session.** Two processes duplicate all four models — roughly 1 GB
  each, plus a CUDA context. Sharing frees that, which raises how many sessions
  fit per card *once compute allows it*.
- **Second-session start time.** A second process on a warm pod still loads
  models from scratch: another 10–30 seconds. Shared models make the second
  session on an existing GPU near-instant. This is really a cold-start win.

If the refactor is done, the concurrency works out: OpenCV, NumPy and ONNX
Runtime all release the GIL during their heavy calls, so sessions running as
threads in one process genuinely execute in parallel.

### 2. Compute and CPU bind before VRAM does

The proposal reasons at length about high-VRAM cards and sketches four sessions
inside 128 GB. For this pipeline that framing is backwards. Even duplicating all
four models per process, a session costs about a gigabyte; with sharing it is a
few megabytes on top of a single ~1 GB resident set. Neither figure makes VRAM
the limit on a 16 GB card.

What actually binds, in order:

| Constraint | Per session | Why it binds |
|---|---|---|
| GPU compute | ~80 inferences/s | Four models per frame at 20 fps. Detection runs every frame and is the most expensive of the four. |
| Host CPU | 0.26–1.14 cores | Compositing is pure OpenCV — measured at ~13 ms/frame at 256, ~38 ms for a close-up at 320. The GPU does not help with any of it. |
| VRAM | ~1 GB, or a few MB | Two processes duplicate all four models, so ~1 GB per session. With the sharing refactor, a few MB. On the 16 GB+ cards `RUNPOD_MIN_VRAM` already selects, duplication is affordable for several sessions — compute and CPU still bind first. |

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

## What low concurrency changes

Concurrency is expected to sit below one — two customers connected at the same
time is unlikely, and growth gradual. That is the answer the previous section
said stage 3 would produce, and it arrives early enough to save building things
that would not have paid.

**The margin holds up without packing.** Provision on demand, release promptly
when the paid hour ends, and the arithmetic works at every volume:

```
  sessions/day   revenue/mo   GPU/mo    fees        net   margin
  ───────────────────────────────────────────────────────────────
             1         $300      $36    $0.6       $263      88%
             3         $900     $108    $1.8       $790      88%
             5       $1,500     $180    $3.0     $1,317      88%
            10       $3,000     $360    $6.0     $2,634      88%
```

(assuming 1.2 GPU-hours paid per hour sold, covering provisioning and the gap
before release)

Utilisation is high *by construction* here, because nothing is held idle. The
low-utilisation risk in the earlier analysis came entirely from warm pools and
long grace periods — neither of which is affordable at this scale.

**Packing is worth $0.** Not "a little" — zero. There is never a second session
on the card to pack.

**A warm pool is unaffordable.** One always-on standby GPU costs $720/month,
which is 80% of revenue at three sessions a day. It does not become sensible
until roughly ten sessions a day, and even then it buys back seconds on a wait
the customer is not billed for.

### Cold start stops being urgent too

At low concurrency every session is a cold start — there is no previous
customer's warm pod to inherit. That sounds like it makes cold start worse, and
in UX terms it does. But the billing rule already absorbs the cost: the clock
starts when the session becomes *usable*, so the customer never pays for the
wait, and we pay about 1.7 cents of GPU for it.

So cold start is a patience cost, not an economic one. It deserves the cheap fix
— bake the image so `apt-get`, `git pull` and `pip install` leave the critical
path — and not the expensive one.

### What this defers — and what it must not foreclose

The original proposal is organised around GPU pooling and session packing. At
the expected load that centrepiece does not pay yet. But adoption is unknown,
and *unlikely* is not *impossible* — designing as though low concurrency is
certain is the same mistake as designing as though high concurrency is certain,
pointed the other way.

So the distinction is between **building packing** (defer) and **being able to
turn it on** (do not defer). The second costs almost nothing if it is designed
in, and is expensive to retrofit at exactly the moment it becomes urgent.

| Component | Now |
|---|---|
| Slot-shaped scheduling | **Build it** — with `max_sessions = 1`, identical behaviour, zero cost |
| `max_sessions` as per-GPU-type config | **Build it** — never hardcode 1 |
| Session → worker assignment | **Many-to-one capable** in the data model, 1:1 in practice |
| The three shared-path fixes | **Do them** — two are correctness bugs regardless |
| `max_sessions` benchmark | Defer. Nothing depends on the answer yet |
| Actually running two sessions per pod | Defer. Worth $0 today |
| Warm pool / standby capacity | Skip. Costs more than the revenue it serves |
| Regional fallback | **Keep** — availability matters more, not less, with no spare capacity |
| Session lifecycle, billing clock, balance | **Keep** — this is the product |
| Fault attribution, hour reversal | **Keep** |
| Heartbeat, lease, watchdog | **Keep** — with few customers each one matters disproportionately |
| Provider abstraction | Keep. Cheap insurance |

A scheduler that counts **slots** behaves identically to one that counts GPUs
while `max_sessions` is 1. A scheduler that counts GPUs has to be rewritten to
count slots. Same code today, very different code the week demand arrives — and
it would arrive with customers already on the system.

The asymmetry is what decides it:

```
  cost of keeping the option open   ~0   (a config value and a plural)
  cost of retrofitting under load   high (scheduler data model, live traffic)
```

**Trigger to enable:** average concurrency approaching 2, or GPUs becoming hard
to rent in a region. At that point packing is a benchmark plus a config change,
not a project.

---

## A staged path

Ordered by dependency and by what each stage makes sellable.

### 1. Measure time to first frame

*Ships: nothing customer-facing. Sizes the only number that still matters.*

Scope reduced now that concurrency is known to be low. `max_sessions` no longer
gates anything — with fewer than one concurrent session there is nothing to pack
— so the load harness can wait.

What remains worth measuring is startup, because every session pays it:

- Time a cold provision end to end, broken down by phase — provisioning,
  `apt-get`, `git pull`, `pip install`, model load — on both a warm and an empty
  volume.
- Confirm one session comfortably holds its latency budget at each preset. A
  single session missing frame deadlines is a quality problem regardless of how
  many others there are.

The `max_sessions` benchmark — ramping concurrent sessions while recording
latency, GPU utilisation, VRAM and **CPU per core** — moves to stage 7 with the
packing work it exists to inform.

### 2. Close the product gap: batch video and file transfer

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

### 3. Control plane, one session per GPU

*Ships: the product. Customers can buy and run sessions.*

- Session manager with the explicit state machine, a session store, and the
  session/attempt split.
- Billing clock starting at first delivered frame.
- Promote `orchestrator.py`'s discovery and multi-datacenter fallback into a
  scheduler service. A port, not a rewrite — and it brings regional redundancy
  along for free.
- Session state must carry the active mode, since a customer switches between
  live and batch inside one session.
- **Move pod lifecycle out of the worker.** The scheduler refcounts sessions per
  pod and releases when the last one leaves — the worker's own `stop_pod()` call
  cannot see its neighbours and would kill them. Keep only a long,
  no-active-session failsafe in the worker against a control-plane outage.
- **Build every path N-shaped with `max_sessions = 1`**: session-scoped upload
  and temp directories, control-plane-allocated worker ports, a registry keyed
  by (pod, slot), routing to a session's worker rather than a pod. Raising the
  number must then require no code change — see
  [SESSION_ARCHITECTURE.md §4](SESSION_ARCHITECTURE.md).
- Deliberately **no packing yet**. Running one session per worker is correct and
  safe; the paths for two already exist and go untested until stage 7.

### 4. Attack cold start

*Ships: a session that starts in tens of seconds rather than minutes. Image baking only — no warm pool.*

- Bake a Docker image with dependencies and model weights inside, so `apt-get`,
  `git pull` and `pip install` leave the critical path. `orchestrator.py`
  already has a docker deploy mode.
- Keep a small **warm pool**: workers provisioned with models loaded and no
  session assigned. Same mechanism as the five-minute grace period, generalised
  — hold capacity ahead of demand, not only behind it.
- Pre-seed both regional volumes, so a fallback region is not silently the slow
  path.

### 5. Resilience

*Ships: sessions that survive infrastructure failure.*

- Worker → backend heartbeat and session leases, so a dead GPU stops being
  advertised as available.
- Watchdog around the frame loop specifically — **progress-based, not
  liveness-based**, because the CUDA-hang case leaves the process responsive.
- Retry classification and bounded attempts, as proposed.
- Fault attribution, and hour reversal when the fault is ours. The heartbeat
  gives the discriminator: worker gone is ours, client gone with the worker
  still healthy is theirs. Reversal must be idempotent and audited — on an
  irreversible rail the ledger is the only account of what happened.
- Standby capacity with models pre-loaded, so a displaced session has somewhere
  to go inside the interruption threshold rather than waiting on a cold
  provision.

### 6. Provider abstraction

*Ships: insurance against a single vendor's availability and pricing.*

Define the provider interface — provision, status, terminate, list capacity —
and move RunPod specifics behind it. Cheap now, because there is exactly one
implementation to conform to it and its shape is already visible in
`orchestrator.py`.

### 7. Packing, then shared models

*Ships: capacity, when concurrency asks for it. A config change plus a benchmark, because stage 3 kept the shape.*

Split in two, because the first half is nearly free:

**4a — packing (small).** Run *N* pipeline processes per pod, isolated as they
already are. Fix the three shared paths: session-scope `_UPLOAD_DIR`,
session-scope batch temp dirs, and move `runpod.stop_pod()` out of the worker
into the control plane. Enforce the measured `max_sessions` in slot accounting.

**4b — shared models (larger, optional).** Remove the `CONFIG` and `BUS`
singletons in favour of per-session context, and route frames by `session_id`
instead of broadcasting to a client set. Buys VRAM headroom and a near-instant
second session on a warm pod; ONNX Runtime sessions are already safe to call
from multiple threads. Worth doing when VRAM or second-session latency actually
binds — not before.

Worth between 2x and 115x more than fixing cold start **per overlapping
session-hour** — which is the catch, since below one concurrent session there
are none. It is listed last for that reason alone, not because it is hard or
low value.

Because stage 3 builds every path N-shaped with `max_sessions = 1`, this stage
is a benchmark and a config change rather than a rewrite. If it turns out to be
more than that, stage 3 was built wrong — and running N=2 in staging is how that
gets caught early rather than under load. The trigger is
concurrency approaching 2, or GPU scarcity in a region — and `max_sessions`
becomes a pricing input at that moment, since a fully-consumed Day Pass is 76%
margin at one session per card and 88% at two.

---

## Settled

- **Who runs the client.** Our desktop app. No public protocol, no third-party
  integration surface. The WebSocket API stays internal.
- **Whether batch belongs in the same plane.** Yes — in the same *session*. A
  customer buys time and chooses the mode, which promotes batch video from a
  follow-up to a launch prerequisite.
- **How a session is consumed.** One wall-clock hour, deducted when the session
  becomes usable, spent whether used or not, no returns. An extend modal fires
  before expiry and deducts another hour if the balance allows.
- **Payment.** Bitcoin over Lightning. No refund system; manual payout if one is
  genuinely demanded.
- **Outages are our cost.** If a session fails because of us, the hour is
  reverted to the customer's balance. Atomic, never pro-rated. Ambiguous cases
  resolve in the customer's favour.
- **Concurrency is expected to be low.** Two customers connected at once is
  unlikely, at least initially, and growth is expected to be gradual. Packing
  was conceived as cost minimisation rather than a response to load.

## Still open

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

**Do Day Pass hours expire?**
The consumption rule settles how hours are *spent* — one wall-clock hour per
connection, no returns — but not whether unspent ones lapse. "Day Pass" implies
they do. If they do not, it is a 24-hour pack under another name and the ladder
is three prepaid blocks rather than two plus a pass.

**How long an interruption counts as an outage?**
Settled that our-fault failures revert the hour; not settled where the line
sits. A three-second reconnect to a standby worker is not an outage, ninety
seconds of cold provisioning is. The threshold is what standby capacity is
bought to stay under, so it sets how much standby is worth.

**Does the scheduler need to know the tier?**
Under contention a PAYG session earns $10 per GPU-hour and a Day Pass session
$4.17. That is a legitimate admission-control input when capacity is scarce, and
an unpleasant one to discover after the fact. Worth deciding deliberately rather
than by omission.

**Where does authentication and payment live?**
Not addressed in the proposal or here. The desktop app has no concept of a user,
and purchase, entitlement and remaining-time state all have to originate
somewhere before the session API means anything.

**Deliberately deferred to last.** It is the largest unwritten piece, but none
of it matters until the pipeline is worth selling — a billing system attached to
a product that cannot yet do batch video, or cannot hold a session together, is
effort spent on the wrong end. Answer it when stages 1-4 are done, not before.

---

## The short version

Build it — but build considerably less of it than the proposal describes.

**Order: measure startup, close the batch and file-transfer gap, build the
control plane, make cold start reasonable, then make it survive failure.**
Packing goes last — enabled by a trigger rather than scheduled.

That last point is a deferral, not a rejection. Build the control plane
slot-shaped with `max_sessions = 1`: identical behaviour today, and packing
becomes a config change rather than a scheduler rewrite if adoption arrives.
Adoption is the one number nobody can forecast, and it would arrive with
customers already on the system.

The proposal is organised around GPU pooling and session packing. At the
concurrency actually expected — below one, with gradual growth — that
centrepiece is the least valuable part of it, and the supporting machinery is
the product. Session lifecycle, the billing clock, fault attribution and the
hour reversal are what make this sellable; slot arithmetic is what makes it
sellable *at scale*, and scale is not the problem to solve yet.

The margin does not need packing. Provision on demand, release promptly when
the paid hour ends, and it holds at **88% across every volume** — because
nothing is being held idle. The two things that would erode it, warm pools and
long grace periods, are exactly the two this scale cannot afford.

Deferring costs almost nothing. Packing is two pipeline processes plus three
shared-path fixes, not a refactor, so it can be added the week concurrency or
GPU scarcity makes it worth having.

What still matters regardless of scale: **batch video is half of what a session
is sold as** and does not exist yet; **outages must revert the hour**, which
needs fault attribution the spec did not have; and the billing clock starts when
the session becomes **usable**, which is the one decision that is free today and
painful to retrofit.
