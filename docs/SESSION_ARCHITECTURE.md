# Session Architecture — Specification

A session-based, remotely executed facial-augmentation platform: GPU pooling,
regional redundancy, persistent storage, fault-tolerant execution.

**Status: proposed. None of this is built.** The pipeline described in
[ARCHITECTURE.md](ARCHITECTURE.md) is the execution layer this sits above; today
it runs single-tenant on a pod provisioned by hand through
`runpod/orchestrator.py`.

## How to read this document

This is the specification — what the system is and how it behaves. It is written
to be implementable without reference to any other source.

Assessment of the design, its economics, and the order to build it in lives
separately, in [SESSION_PLANE.md](SESSION_PLANE.md). Where a design decision here
has a known objection or an unresolved question, it is marked inline:

> **Annotation —** commentary, correction, or a caveat raised during review.
> These are not part of the original design; they are flagged so the design and
> the objections to it stay distinguishable.

---

## 1. Product model

The customer does not need a powerful local machine. Our desktop application
connects to our backend, which places the work on a remote GPU.

```
Customer
   │  starts a session
   ▼
Desktop app  (our client — not a public protocol)
   │
   ▼
Backend
   │
   ▼
Remote GPU
   │
   ▼
Facial augmentation pipeline
```

### What the customer buys

**Time, not inference operations.** Three tiers, of ascending commitment and
descending effective rate:

| Tier | Price | Effective | Notes |
|---|---|---|---|
| **PAYG** | $10 / hour | $10.00/hr | No commitment. One hour at a time. **The full hour is deducted at session start.** For occasional and first-time users. |
| **4-Hour Pack** | $35 | $8.75/hr | Prepaid block. |
| **10-Hour Pack** | $70 | $7.00/hr | Prepaid block. |
| **Day Pass** | $100 / 24 hours | $4.17/hr | Heavy users. Strongest discount. |

```
  PAYG  →  Session Packs  →  Day Pass
   ────────────────────────────────▶
   rising commitment, falling rate
```

A session stays active until either:

- the user explicitly ends it, or
- the purchased time expires.

This is a **long-lived compute session**, deliberately not the conventional
`upload → inference → result` API.

### Payment

**Bitcoin over the Lightning Network.** At 0.2% it removes what would otherwise
be the second-largest cost line: a card charge on a $10 PAYG purchase costs
$0.59, more than the compute it pays for. On-chain Bitcoin is unusable at the
small tiers — a ~$3 network fee is 30% of a PAYG purchase — so Lightning is the
primary rail, with on-chain viable only as a Day Pass fallback.

Three consequences the architecture has to absorb:

- **Payments are irreversible, and there is no refund system.** No chargeback
  to fear, which suits a face-swapping product. Refunds are not a product
  feature: if one is genuinely demanded, Bitcoin is sent manually to the
  customer's address. That is an exception handled by a person, not a code path.
  Nor is unused time returned — see §11.
- **Invoices are USD-denominated and short-lived.** Quote in sats against a USD
  price with a expiry of roughly fifteen minutes, and decide explicitly whether
  revenue is auto-converted or held. Holding is an unhedged position on the
  entire top line.
- **Liquidity is operational work.** Receiving over Lightning needs inbound
  capacity and a node that stays online. A managed provider costs about 1% and
  removes that; self-hosting via BTCPay costs about 0.2% and adds channel
  management. Start managed, move in-house if volume justifies it.

> **Annotation — purchases top up a balance; sessions draw it down.** Money
> never flows back out. Hours do, but only in one case: a session we broke
> ourselves. See §11 for the consumption rule, the reversal rule, and why the
> balance is denominated in hours rather than currency.

> **Annotation — three things the tiers imply for the architecture.**
>
> **`max_sessions` becomes a pricing input, not just a capacity number.** A
> fully-consumed Day Pass costs $24 of GPU against $100 at one session per card,
> and $12 at two. That is the difference between a 76% and an 88% margin, so the
> benchmark in §4 now feeds the price list.
>
> **The Day Pass is bounded by hours connected, not by standing availability.**
> Since every tier draws down the same hour blocks (§11), a Day Pass costs us GPU
> only for the hours a customer actually connects. Twenty-four is the worst case
> — $24 of GPU against $100 at one session per card, $12 at two — rather than the
> expected one, and hours bought but never connected are pure margin. What
> remains undecided is whether unspent Day Pass hours **expire**; if they do not,
> the tier is a 24-hour pack under another name.

### What the customer does inside a session

A session is a block of purchased time, not a mode of work. Within it the
customer uses either:

- **live** — realtime video call augmentation, or
- **batch** — processing a video or image file,

switching freely between the two.

> **Annotation —** batch video is currently unimplemented
> (`ProcessingPipeline._process_target_batch()` handles images only), which makes
> it a launch prerequisite rather than a follow-up. Batch on a remote GPU also
> requires a file transfer path that does not yet exist; `handle_upload_source`
> is base64 inside a single JSON message, workable for a source face and not for
> a video.

### Client scope

The desktop app is the product surface. Customers do not integrate against a
protocol, and the WebSocket API between app and worker stays an internal
contract.

> **Annotation —** the app must therefore grow a concept of a user: identity,
> purchase, entitlement, remaining time, worker assignment, reconnection. Today
> `PHANTOM_API_URL` names one pod, set by hand in `.env`. Where authentication
> and payment live is unresolved.

---

## 2. Vocabulary

These terms are used precisely throughout.

| Term | Meaning |
|---|---|
| **Session** | What the customer bought. One identity, one block of time, survives infrastructure failure. |
| **Attempt** | One try at placing that session on a GPU. A session may consume several. Invisible to the customer. |
| **Slot** | One unit of session capacity on a GPU. A GPU sustains *N* slots. |
| **Worker** | The process on a GPU that runs the pipeline and reports health. |
| **Lease** | A renewable claim a worker holds on a session. Expiry means the worker is presumed lost. |
| **Grace period** | The window a GPU with zero sessions is kept alive before release. |

The session/attempt distinction is load-bearing and appears again in §11.

---

## 3. Session lifecycle

A session has explicit states. Nothing is implicit.

```
                          ┌─────────┐
                          │ QUEUED  │  no slot available yet
                          └────┬────┘
                               ▼
                     ┌──────────────────┐
                     │ ALLOCATING_GPU   │──── timeout ─────┐
                     └────────┬─────────┘──── unavailable ─┤
                              ▼         ──── provider fail ┤
                     ┌──────────────────┐                  │
                     │    GPU_READY     │                  │
                     └────────┬─────────┘                  │
                              ▼                            │
                     ┌──────────────────┐                  │
                     │  LOADING_MODEL   │──── timeout ─────┤
                     └────────┬─────────┘──── crash ───────┤
                              ▼                            │
                     ┌──────────────────┐                  ▼
                     │     RUNNING      │          ┌───────────────┐
                     │  live │ batch    │──fault──▶│ next attempt  │
                     └────┬────────┬────┘          └───────┬───────┘
                          │        │                       │ attempts
          user ends /     │        │ attempts              │ remaining
          hour expires    │        │ exhausted             │
                          │        │                       └──▶ ALLOCATING_GPU
                          ▼        ▼
                 ┌───────────┐   ┌──────────────────┐
                 │ COMPLETED │   │    ATTRIBUTE     │
                 │hour spent │   │   whose fault?   │
                 └───────────┘   └────┬────────┬────┘
                                 ours │        │ theirs
                                      ▼        ▼
                            ┌────────────┐  ┌───────────┐
                            │  ABORTED   │  │ COMPLETED │
                            │ hour       │  │hour spent │
                            │ REVERTED   │  └───────────┘
                            └────────────┘
```

**A GPU provisioning attempt is not a customer session.** Failure paths out of
`ALLOCATING_GPU` and `LOADING_MODEL` consume an *attempt*; the session stays
alive and is retried against another candidate until the retry budget (§10) is
exhausted.

> **Annotation —** `RUNNING` must carry the active mode, since the customer
> switches between live and batch inside one session. The two are not
> interchangeable to the scheduler — see §6 and SESSION_PLANE.md.

---

## 4. Capacity: slots and packing

The central infrastructure idea is **GPU session packing**. Rather than
dedicating a GPU per customer, a GPU is benchmarked to establish how many
simultaneous sessions it sustains reliably, and sessions are packed into those
slots.

The initial target is **one moderate GPU → two simultaneous sessions**.

First customer arrives:

```
GPU #1
┌─────────────────┐
│ Session A       │
│ EMPTY           │
└─────────────────┘
```

Second customer arrives — no new GPU is provisioned:

```
GPU #1
┌─────────────────┐
│ Session A       │
│ Session B       │
└─────────────────┘
```

Third customer arrives — now a second GPU is needed:

```
GPU #1                 GPU #2
┌─────────────────┐    ┌─────────────────┐
│ Session A       │    │ Session C       │
│ Session B       │    │ EMPTY           │
└─────────────────┘    └─────────────────┘
```

The scheduler therefore reasons about **available session capacity**, not GPU
count.

### Establishing max_sessions

`max_sessions` is measured, never assumed. Ramp concurrent sessions and record
at each step:

```
1 session → 2 → 3 → 4 → ...

  measuring   VRAM consumption
              GPU utilisation
              processing performance
              latency
              stability
              OOM behaviour
              failure rate
```

The result is a per-GPU-type constant: `max_sessions = 2`, or `4`, depending on
the card and the pipeline.

> **Annotation — three corrections to this section.**
>
> **VRAM is not the binding constraint.** With models shared across sessions the
> resident set is roughly 1 GB total; each further session adds only working
> buffers. What binds first is GPU compute (~80 inferences/sec/session: four
> models per frame at 20 fps) and then **host CPU** — compositing is pure OpenCV,
> measured at ~13 ms/frame at aligned 256 and ~38 ms for a close-up at 320, none
> of which the GPU touches. A pod rents a fixed vCPU allocation; it is possible
> to hit the CPU wall with the GPU half idle. The benchmark must instrument CPU
> per core, which the original method does not list.
>
> **Define the ceiling by a latency budget, not by absence of crashes.** A live
> session is unusable long before it OOMs.
>
> **Packing itself is available now.** Two sessions on one GPU means two
> pipeline processes, each with its own config, event bus, port and client set —
> isolated by construction. Three shared paths need session-scoping first:
> `_UPLOAD_DIR`, the batch temp directories, and — most importantly —
> `runpod.stop_pod()` in the auto-stop timer, since whichever process fires
> first would kill the pod and every session on it. Pod lifecycle belongs to the
> scheduler (§6), not the worker.
>
> What *is* blocked is sharing one set of loaded models between sessions, which
> `CONFIG` (`pipeline/config.py:205`), `BUS` (`pipeline/events.py:122`) and the
> shared client set prevent. That is an optimisation — it buys VRAM headroom and
> a near-instant second session on a warm pod — not a prerequisite for packing.

---

## 5. GPU selection

Do not choose a GPU because it is cheap, or because its VRAM is large. The
metric is:

```
            GPU cost per hour
    ───────────────────────────────────    →  minimise
    sustainable concurrent sessions
```

Worked example:

```
  GPU A                      GPU B
  $1/hour                    $2/hour
  2 sessions                 5 sessions
  ─────────────────          ─────────────────
  $0.50/session-hour         $0.40/session-hour   ← cheaper capacity
```

GPU B costs twice as much per hour and delivers cheaper customer capacity. The
selection filter therefore ranks on cost per session-hour, not on sticker price
or VRAM.

> **Annotation —** this metric is right, and under the tiered pricing it matters
> considerably more than an earlier draft of this note suggested. At $10/hour
> PAYG a packed session costs $0.50 of GPU; on a fully-consumed Day Pass at
> $4.17/hour it is $12 against $100. Cost per session-hour is therefore a direct
> margin lever, not a rounding error, and it compounds with the capacity argument
> — packing doubles the demand servable from whatever cards are actually
> available (§8). See SESSION_PLANE.md for the full economics.

---

## 6. The scheduler

The scheduler decides where sessions run. It does not ask "do we have a GPU?"
It asks:

> Do we have a **healthy** GPU with an **available session slot** capable of
> running this pipeline?

```
                 Session request
                        │
                        ▼
                    Scheduler
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       GPU A          GPU B          GPU C
        2/2            1/2            0/2
        FULL        AVAILABLE      AVAILABLE
                        │
                        ▼
                  Assign session
```

Selection inputs, in eventual scope:

- available slots
- GPU type, VRAM, compute capability
- current utilisation
- GPU health
- provisioning state
- region
- provider
- cost
- availability

The customer must not care, and must not be able to tell, which GPU executes
their session.

> **Annotation — live and batch are not interchangeable workloads.** They have
> opposing profiles and the scheduler has to distinguish them:
>
> ```
>                  LIVE                         BATCH
>   latency        hard budget (33ms/frame)     irrelevant
>   duration       holds the slot throughout    bursty: saturate, then idle
>   interruption   visible failure              retryable, resumable
>   preemptible    no                           yes
> ```
>
> Consequences: do not co-schedule batch onto a card serving live until
> measurement says it is safe — a batch job saturating the GPU is exactly the
> spike that blows a live frame budget. And an idle live session is *not* idle
> capacity: a customer between calls still holds their slot, so slot accounting
> tracks purchased time, not activity.

---

## 7. Persistent storage and GPU disposability

Persistent storage is what makes the GPU disposable.

```
Persistent volume
├── Models
├── Pipeline assets
├── Configuration
└── Other large assets
          │
          ▼
    Disposable GPU
          │
          ▼
      Load models
          │
          ▼
      Run sessions
```

When a GPU disappears:

```
GPU #1  ✗
    │
    ▼
New GPU
    │
    ▼
Mount persistent storage
    │
    ▼
Load pipeline
    │
    ▼
Recover session
```

This already holds in part: models resolve to `/workspace/models` before local
paths, and the venv lives on the network volume so it survives pod restarts.

> **Annotation — standby capacity is a requirement, not an optimisation.**
> Hour reversal (§11) makes every slow recovery cost real revenue, so the system
> needs somewhere to put a displaced session *now*. That means keeping at least
> one worker provisioned with models already loaded and no session on it, and
> the scheduler treating it as reserved rather than available. Without it,
> "recover the session" means a cold provision, which is far past any sane
> interruption threshold and therefore reverts the hour by definition.
>
> **Recovery still means something weaker for live sessions than this diagram
> implies.** Restoring execution requires provisioning, loading four
> ONNX models, and reconnecting the client. Warm that is seconds; cold it is
> a minute and a half or worse. A batch job genuinely recovers — the customer
> sees a slower result. A live customer sees their face fall off mid-call and
> come back. The machinery is still what *detects* the failure; what it should
> drive is (a) a warm pre-loaded standby slot, since recovery time is dominated
> by model loading, (b) scheduler bias toward stability over price, and
> (c) a decision about the customer whose hour was consumed by an outage —
> unused time is never returned by design, but a failure is not the same as a
> choice not to use (see §11).

---

## 8. Regional redundancy

GPU availability fluctuates significantly between regions. Depending on one
region is a capacity risk.

```
                     Backend
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Region A   Region B   Region C
             │          │          │
          Volume A   Volume B   Volume C
             │          │          │
          GPU pool   GPU pool   GPU pool
```

Network volumes are datacenter-local, so each region needs its own, pre-seeded
with models and pipeline assets. Fallback is by region:

```
Region A → no suitable GPU → Region B → GPU available → run session
```

Already implemented in prototype form: `RUNPOD_DATACENTERS=DC1:vol1,DC2:vol2`
pairs each datacenter with its volume, and `_resolve_gpu_candidates()` walks
regions as the outer loop. That logic is the seed of the scheduler.

> **Annotation —** ensure every regional volume is genuinely pre-seeded.
> Otherwise the fallback region is silently the slow path, and failover costs a
> cold model download on top of everything else.

---

## 9. Fault tolerance

The governing principle:

> **Do not try to make the GPU reliable. Make the system resilient to unreliable
> GPUs.**

A GPU is a disposable execution resource.

### Failures the system must anticipate

- GPU provisioning failure
- GPU becoming unavailable
- GPU becoming unresponsive
- worker crash
- pipeline crash
- pipeline hanging
- GPU OOM
- CUDA / runtime failure
- network failure
- provider API failure
- excessively slow provisioning
- lost client connection
- lost worker heartbeat

### Timeouts

Nothing external may hang forever. Every external boundary carries a timeout.

```
GPU allocation          Worker startup         Pipeline execution
      │                       │                       │
   timeout                 timeout            no progress / frozen
      │                       │                       │
      ▼                       ▼                       ▼
 mark attempt            worker unhealthy         watchdog fires
   failed                       │                       │
      │                         ▼                       ▼
      ▼                  try another worker      terminate process
try another candidate                                   │
                                                        ▼
                                                 recover session
```

### Heartbeats

A worker periodically tells the backend it is alive.

```
Worker ──heartbeat──▶ ──heartbeat──▶ ──heartbeat──▶  ✗

                            no heartbeat
                                 │
                                 ▼
                           grace period
                                 │
                          still nothing
                                 │
                                 ▼
                       worker presumed dead
                                 │
                                 ▼
                          recover session
```

Without this a dead GPU stays marked available indefinitely.

> **Annotation —** the heartbeat that exists today is WebSocket ping/pong
> between *client and worker*. The backend learns nothing from it. The required
> direction is worker → backend.

### Watchdogs

The worker runs a watchdog around the pipeline itself.

```
Pipeline ──progress──▶ ──progress──▶ ──progress──▶  frozen
                                                       │
                                                       ▼
                                                   watchdog
                                                       │
                                                       ▼
                                                 kill / restart
                                                       │
                                                       ▼
                                                recover session
```

This is separate from the heartbeat and cannot be replaced by it: **a process
being alive does not mean the workload is healthy.** A CUDA operation can hang
while the worker process remains perfectly responsive.

The watchdog must therefore be **progress-based, not liveness-based** — it
watches frames advancing, not the process existing.

### Leases

A session holds a worker lease.

```
Session
   └── worker lease
          ├── renewed  → worker healthy
          └── expired  → worker considered lost
```

If a worker disappears without cleanly reporting failure, the lease expires, the
session becomes recoverable, and the scheduler places it elsewhere. This is what
prevents orphaned sessions.

---

## 10. Retry policy

Failures are classified. Infrastructure failures retry; user and input errors do
not.

| Retryable | Not retryable |
|---|---|
| GPU unavailable | Invalid input |
| GPU provisioning failure | Invalid configuration |
| Worker crash | Unsupported operation |
| Network failure | |
| Provider timeout | |

Retries are **bounded**. Not this:

```
GPU A fails → GPU B fails → GPU C fails → GPU D fails → ... ∞
```

But this:

```
Attempt 1 → Attempt 2 → Attempt 3 → failure / queue / notify
```

with the bound and the terminal action varying by failure class.

---

## 11. Identity, idempotency, and billing

### Session identity

Every session carries a unique identifier, e.g. `session_id = abc123`.

If a GPU crashes and the client reconnects, the backend must check:

```
Does abc123 already have a completed result?
        │
        ├── YES → do not execute duplicate work
        │
        └── NO  → recover / retry
```

This exists because **network failure ≠ execution failure**. A client losing its
connection does not mean the GPU stopped working.

> **Annotation —** this matters most for batch. Re-running a finished
> ninety-minute export because a client reconnected is both expensive and
> visible. For live there is no completed artifact to deduplicate, so the check
> is cheap and always worth making.

### Session versus attempt

A customer has one session. That session may internally consume several
attempts:

```
SESSION #ABC123 — one purchase, one identity
┌──────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐ │
│ │ Attempt 1    │ → │ Attempt 2    │ → │ Attempt 3 · GPU C          │ │
│ │ GPU A        │   │ GPU B        │   │ RUNNING                    │ │
│ │ alloc timeout│   │ worker died  │   │ heartbeat · lease · watchdog│ │
│ └──────────────┘   └──────────────┘   └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**The customer must not pay three times because our infrastructure had to
recover.** Attempts are an infrastructure concern; the session is the customer's.

### The session clock

**Settled.** An hour is deducted the moment the session becomes usable — worker
running, models loaded, client connected — and then runs as **wall clock**.

```
  balance >= 1 hour?  ──no──▶  cannot connect
        │ yes
        ▼
  connect · worker ready · models loaded
        │
        ▼  DEDUCT 1 HOUR, countdown starts
  12:00 ────────────────────────────────────▶ 13:00
        │                                    │
        │  used for live, batch, or not      │  hour exhausted
        │  at all — the customer's choice    │
        │                                    ▼
        │                          extend modal fires before this
        │                          point; extend if balance allows
```

Two rules follow, and they are deliberate:

- **Before the session is usable, nothing is charged.** The customer cannot do
  anything yet, so provisioning and model loading are our cost. This is what
  keeps the scheduler, the warm pool and the retry policy all optimising for the
  thing the customer actually experiences.
- **After that, the hour is consumed whether it is used or not.** Idle time
  inside a paid hour belongs to the customer. There is no metering, no partial
  consumption, and **no return of unused time**.

Before the hour expires, a modal offers to extend. Extending deducts another
hour if the balance can sustain it.

> **Annotation — this already exists in prototype.** The `auto_stop_warning`
> event, the `keep_alive` command, and the desktop's countdown dialog with
> Extend and Dismiss were built for pod uptime. The mechanism transfers directly
> to session time; what changes is what it counts down and that Extend must
> check and deduct balance.

> **Annotation — balance should be denominated in hours, not currency.** Packs
> sell hours at a discount, so a dollar balance cannot represent them: $35 of
> balance against a $10 hourly price yields 3.5 hours, not the 4 that were
> bought. Purchases convert money to hours at the tier's rate; sessions deduct
> one hour. A Day Pass is then 24 hours of balance drawn down in hour blocks,
> which keeps one consumption rule for every tier.

> **Annotation — infrastructure failure is a different case and is not yet
> covered.** "No return of unused time" answers the customer who chooses not to
> use their hour. It does not obviously answer the customer whose GPU died at
> minute five. Deciding this deliberately is worth more than it costs, because
> it is the difference between an outage and a grievance.

Two related questions remain open: whether upload time is billed (a 2 GB
transfer is minutes with the GPU idle — overlapping it with worker startup is
the likely answer), and what happens to a batch job that outlives its session.
See SESSION_PLANE.md.

### Outages are our cost, not the customer's

Unused time is never returned when the customer simply chose not to use it. An
outage is the opposite case, and it is handled explicitly: **if the session
failed because of us, the hour is reverted to the customer's balance.**

The hour is atomic. It is not pro-rated against how much of it elapsed before
the failure — that would reintroduce the metering the consumption rule
deliberately avoids. Either the customer got a usable hour or they did not.

#### Attributing the fault

The heartbeat and lease machinery in §9 already produces the discriminator, and
this is a second reason to build it:

```
  worker heartbeat   client connection   verdict
  ─────────────────────────────────────────────────────────────
  stopped            —                   OURS    → revert hour
  alive              dropped             THEIRS  → hour stands
  alive              alive               not a failure
  stopped            dropped             OURS    → revert hour
```

| Cause | Attribution |
|---|---|
| GPU reclaimed, unresponsive, or OOM | ours |
| Worker crash, pipeline crash or hang | ours |
| Provider API failure, provisioning failure | ours |
| Watchdog fired on a frozen pipeline | ours |
| Client closed the app, or ended the session | theirs |
| Client network dropped, worker still healthy | theirs |
| Customer idle for the whole hour | theirs — nothing failed |

Ambiguity resolves in the customer's favour. On an irreversible payment rail the
cost of being wrong in their favour is one hour of GPU; the cost of being wrong
against them is a customer who cannot get their money back and knows it.

#### What a revert must guarantee

- **Idempotent.** A session reverts at most once, keyed by `session_id`. A
  retried failure notification must not credit twice.
- **Audited.** Every deduction and every reversal is recorded with its
  attribution and the evidence for it. There is no chargeback process to appeal
  to, so the ledger is the only account of what happened.
- **Balance-only.** A reversal restores hours to the balance. It never moves
  money — see §1.

#### The interruption threshold

A three-second reconnect to a standby worker is not an outage; ninety seconds of
cold provisioning is. Reverting on every blip is expensive and reverting on none
is dishonest, so the revert triggers past a threshold of unavailability.

That threshold is what makes standby capacity pay: its job is to keep recoveries
underneath it.

> **Annotation — this gives resilience a direct financial value again.** With no
> payout and no time returned, failures cost only retention. With hour reversal
> they cost revenue: at $10 per reverted hour, plus the GPU already spent on the
> broken session. If our-fault failures ran at 5% of sessions, that is roughly
> 5.5% of net income — which is what stage 6 is buying back.

---

## 12. GPU lifecycle and the idle grace period

A GPU serving one session with one empty slot is **kept alive**. The empty slot
is valuable capacity that can be handed to the next customer immediately.

Only when *all* slots are empty does the release timer start:

```
Both slots empty
       │
       ▼
Start 5-minute timer
       │
       ├── new user arrives ──▶ reuse GPU  (timer cancelled)
       │
       └── nobody arrives ────▶ release GPU
```

**The timer starts only when zero sessions remain.** Any occupied slot cancels
it.

This exists to prevent thrashing:

```
provision → terminate → provision → terminate
```

when customers arrive close together.

> **Annotation —** this is the same mechanism a warm pool needs, generalised.
> The grace period holds capacity *behind* demand; a warm pool holds it *ahead*
> of demand. Both are "keep a GPU with models loaded and no session on it," and
> building the second is mostly a matter of deciding when to start one
> speculatively.
>
> **For an in-progress session there is no bet to make.** The consumption rule
> in §11 settles it: the customer has already paid for the whole wall-clock hour
> and may reconnect at any point inside it, so the worker is **held until their
> hour expires** — not for five minutes. Releasing it early only buys a cold
> start we would then have to pay for, on time the customer already owns.
>
> The five-minute timer therefore applies to a genuinely different case: a GPU
> whose sessions have all *expired*, being kept warm on the chance a new customer
> arrives. That is speculative capacity, and it is the same mechanism as a warm
> pool — the grace period holds it behind demand, a warm pool holds it ahead.

---

## 13. Complete architecture

```
                         CUSTOMER
                            │
                            ▼
                       DESKTOP APP
                            │
                            ▼
                          API
                            │
                            ▼
                    SESSION MANAGER
                            │
                    ┌───────┴───────┐
                    ▼               ▼
               Session DB        Billing
                    │
                    ▼
                   QUEUE
                    │
                    ▼
                SCHEDULER
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    REGION A     REGION B    REGION C
        │           │           │
    GPU POOL     GPU POOL    GPU POOL
        │           │           │
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
   GPU     GPU GPU     GPU GPU     GPU
    │       │   │       │   │       │
    └───────┴───┴───────┴───┴───────┘
                    │
                    ▼
             FACIAL PIPELINE
                    │
                    ▼
                 SESSION
```

with persistent storage alongside each regional pool:

```
Region A                    Region B
┌───────────────┐           ┌───────────────┐
│ Volume A      │           │ Volume B      │
│ Models        │           │ Models        │
│ Pipeline      │           │ Pipeline      │
└───────┬───────┘           └───────┬───────┘
        │                           │
     GPU pool                    GPU pool
```

> **Annotation —** everything from `FACIAL PIPELINE` downward exists today. The
> scheduler exists as a CLI prototype in `runpod/orchestrator.py`. Everything
> else is new. See SESSION_PLANE.md for the mapping and the staged path.

---

## 14. Complete session flow

```
USER STARTS SESSION
        │
        ▼
Create session (session_id)
        │
        ▼
Scheduler searches for a healthy
GPU with an available slot
        │
        ├───────────────┬───────────────┐
        ▼               ▼               │
     Slot found      No slot            │
        │               │               │
        ▼               ▼               │
   Assign GPU      Queue request ───────┘
        │
        ▼
   GPU ready?
        │
        ├── NO ──▶ timeout / failure ──▶ next candidate (new attempt)
        │
        ▼
   Load pipeline
        │
        ▼
   Session usable ──▶ BILLING CLOCK STARTS
        │
        ▼
     RUNNING  (live │ batch)
        │
        ├── heartbeat
        ├── lease renewal
        ├── watchdog
        └── monitoring
        │
        ▼
User ends session / time expires
        │
        ▼
Free the session slot
        │
        ├── customer waiting? ──▶ YES ──▶ assign immediately
        │
        ▼
Any other active session on this GPU?
        │
        ├── YES ──▶ keep GPU alive
        │
        └── NO
             │
             ▼
       Start 5-minute grace period
             │
        ┌────┴────┐
        ▼         ▼
    New user    Nothing
        │         │
        ▼         ▼
      Reuse     Release GPU
```

---

## 15. Design principles

The philosophy the whole design reduces to:

> **Treat GPU compute as a dynamically allocated, disposable resource rather
> than as a permanent server.**

The customer sees a simple persistent session:

```
Start → use pipeline → time remaining → end
```

The infrastructure sees:

```
Session → GPU slot → GPU worker → pipeline
```

And on failure:

```
GPU failure → detect → recover → find another GPU → restore execution
```

with regional redundancy absorbing capacity problems, pooling absorbing normal
demand, and the grace period absorbing short idle gaps.

The consequence worth stating explicitly: this is **a small fault-tolerant
compute orchestration layer over GPU providers**, in which RunPod is initially
one execution backend. That abstraction is the point. If RunPod availability or
pricing becomes a problem later, another provider goes in behind the same
scheduler rather than forcing a redesign.

---

## Related documents

- [SESSION_PLANE.md](SESSION_PLANE.md) — assessment of this design, economics,
  gap analysis against the current code, and the staged migration path.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the pipeline this orchestrates.
- [../runpod/TROUBLESHOOTING.md](../runpod/TROUBLESHOOTING.md) — accumulated
  RunPod API behaviour, much of which constrains what the scheduler can do.
