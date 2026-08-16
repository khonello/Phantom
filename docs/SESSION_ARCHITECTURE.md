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
- **Invoices are USD-denominated and short-lived.** Quote in sats against a USD
  price with a expiry of roughly fifteen minutes, and decide explicitly whether
  revenue is auto-converted or held. Holding is an unhedged position on the
  entire top line.
- **Liquidity is operational work.** Receiving over Lightning needs inbound
  capacity and a node that stays online. A managed provider costs about 1% and
  removes that; self-hosting via BTCPay costs about 0.2% and adds channel
  management. Start managed, move in-house if volume justifies it.

> **Annotation — a balance of hours is the right model, and it is not a refund
> mechanism.** Two distinct things are easy to conflate here:
>
> - **Paying money out** — not supported. Manual, exceptional, handled by a
>   person.
> - **Returning unused time** — putting hours back on a balance the customer
>   already holds. No payment rail, no payout, no reversal.
>
> Purchases should top up a per-customer balance of hours, and sessions draw it
> down. PAYG's "full hour deducted at session start" becomes a balance
> deduction; packs are larger top-ups. This is simpler than per-purchase
> accounting regardless of the refund position.
>
> It also makes the compensation question cheap to answer. A Day Pass that dies
> at hour 2 of 24 leaves 22 hours owed. Sending that back as Bitcoin is a
> **$91.67 payout**; returning it to the balance costs the GPU time to serve it
> — about **$11 at two sessions per card**, eight times less, and the customer
> stays rather than leaves. Whether to do that automatically is a policy
> decision, but the mechanism should exist because it is far cheaper than the
> alternative it replaces.

> **Annotation — three things the tiers imply for the architecture.**
>
> **`max_sessions` becomes a pricing input, not just a capacity number.** A
> fully-consumed Day Pass costs $24 of GPU against $100 at one session per card,
> and $12 at two. That is the difference between a 76% and an 88% margin, so the
> benchmark in §4 now feeds the price list.
>
> **Grace period policy should vary by tier** (§12 currently specifies one global
> constant). PAYG deducts the full hour at session start, so holding that
> customer's worker for the remainder of their paid hour costs at most $1 against
> $10 collected — and removes reconnect cold starts entirely inside the window.
> Holding a Day Pass worker for 24 hours costs $24 of $100, so the same
> generosity does not transfer.
>
> **"24 Hours" is undefined and the answer moves the margin by 20 points.** A day
> of *access* means we may hold or repeatedly re-provision a worker across 24
> hours; 24 hours of *accumulated session time* is metered and bounded. This
> needs settling before the Day Pass ships.

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
                     │  live │ batch    │───────▶  │ next attempt  │
                     └────┬────────┬────┘  fault   └───────┬───────┘
                          │        │                       │
              user ends / │        │ unrecoverable         │ attempts
              time expiry │        │ or attempts exhausted │ remaining
                          ▼        ▼                       │
                   ┌───────────┐ ┌────────┐                │
                   │ COMPLETED │ │ FAILED │                └──▶ ALLOCATING_GPU
                   └───────────┘ └────────┘
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

> **Annotation — recovery means something weaker for live sessions than this
> diagram implies.** Restoring execution requires provisioning, loading four
> ONNX models, and reconnecting the client. Warm that is seconds; cold it is
> a minute and a half or worse. A batch job genuinely recovers — the customer
> sees a slower result. A live customer sees their face fall off mid-call and
> come back. The machinery is still what *detects* the failure; what it should
> drive is (a) a warm pre-loaded standby slot, since recovery time is dominated
> by model loading, (b) scheduler bias toward stability over price, and
> (c) returning interrupted time to the customer's hour balance — not a
> payout, and roughly eight times cheaper than one (see §1).

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

### When the billing clock starts

The clock starts when the **session becomes usable**: worker ready, models
loaded, client attached.

Before that point the customer can do nothing, so the time is our cost. After
it, their time is their own — whether they are streaming, choosing a source
face, or between calls.

This places cold start on our side of the ledger, which is deliberate: it makes
the scheduler, the warm pool and the retry policy all optimise for the thing the
customer experiences. If the clock started at session creation instead, every one
of those systems would be free to be slow at the customer's expense.

> **Annotation —** an earlier draft proposed starting at the first delivered
> frame. That assumed live-only; a batch session has no first frame, and setup
> time is use rather than idleness. Two related questions remain open: whether
> upload time is billed (a 2 GB transfer is minutes with the GPU idle —
> overlapping it with worker startup is the likely answer), and what happens to a
> batch job that outlives its session. See SESSION_PLANE.md.

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
> **The five minutes should not be one global constant.** It is a bet that a
> customer returns before the hold costs more than the cold start it saves, and
> that bet differs sharply by tier. A PAYG customer has already paid for the full
> hour at session start, so holding their worker until that hour expires costs at
> most $1 against $10 collected and guarantees an instant reconnect. A Day Pass
> customer at $4.17/hour cannot be held on the same terms. Make the grace period
> a function of the tier and of paid-but-unused time.

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
