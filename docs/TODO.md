# Implementation TODO

Everything the design documents call for that does not exist in the code yet,
in dependency order.

Sources: [SESSION_ARCHITECTURE.md](SESSION_ARCHITECTURE.md) (the spec),
[SESSION_PLANE.md](SESSION_PLANE.md) (assessment and staging),
[INPUT_GUARDS.md](INPUT_GUARDS.md) (input validation).

**Nothing below is built.** Items are grouped by stage; stages are ordered by
what unblocks what, and by what makes the product sellable. Payment and auth are
last on purpose — a billing system attached to a product that cannot yet do
batch video, or hold a session together, is effort spent on the wrong end.

---

## Stage 0 — Repository health

Small, independent, and they stop the other stages building on sand.

- [ ] **Fix the CI test job.** It runs the batch-video command, which hits the
      unimplemented path. Currently fails on every push. Resolved by stage 2.
- [ ] **Clear the 12 remaining mypy errors** — `Queue` and `dict` type arguments,
      platform-specific module attributes, a `cv2` constant. All trivial; none
      are in code written recently.
- [ ] **Add a test suite.** There is none. Start with the pieces that are pure
      functions: `estimate_similarity`, the compositor's colour and detail
      matching, `FaceDatabase` averaging, the guard predicates.
- [ ] **Verify whether RunPod bills egress.** ~2.2 GB/hour outbound per session,
      ~11 GB over a 5-Hour Pack. The only cost line in the economics set to zero
      without evidence. A pricing-page lookup.

---

## Stage 1 — Measure time to first frame

*Ships: nothing customer-facing. Sizes the only number that still matters.*

- [ ] **Time a cold provision end to end**, broken down by phase: provisioning,
      `apt-get`, `git pull`, `pip install`, model load. Run it against both a
      warm volume and an empty one.
- [ ] **Confirm a single session holds its latency budget** at each preset. One
      session missing frame deadlines is a quality problem regardless of how many
      others exist. The per-stage timings behind `--log-level debug` already
      exist for this.

The `max_sessions` load harness is deliberately *not* here — nothing depends on
that number until stage 7.

---

## Stage 2 — Close the product gap

*Ships: the other half of what a session is sold as. Launch prerequisite.*

A session is sold as "live **or** batch". Half of that currently returns an
error.

- [ ] **Wire `_process_target_batch()` for video.** The FFmpeg pieces already
      exist in `pipeline/io/ffmpeg.py` — `extract_frames`, `create_video`,
      `restore_audio`, `clean_temp` — and batch reuses the same compositor as
      live, so the work is frame iteration plus audio and FPS restoration.
- [ ] **Build a real file transfer path**: chunked, resumable,
      progress-reporting, both directions. `upload_source` is base64 inside one
      JSON message — fine for a 200 KB face, unusable for a 2 GB video.
- [ ] **Decide and implement the overflow policy** for a batch job that outlives
      its session: refuse / bill overflow / absorb / detach and hold the result.
- [ ] **Decide whether upload time is billed.** Overlapping transfer with worker
      startup is the likely answer, but it has to be designed in.
- [ ] Desktop VIDEO mode and the CI test both start working as a side effect.

---

## Stage 3 — Input guards

*Ships: the pipeline stops producing confidently wrong output.*

Independent of the control plane, and worth doing before customers arrive.

### Fixes that stand on their own

- [ ] **Replace `detect_one`'s leftmost rule with largest-face.**
      `min(detections, key=lambda d: d.bbox.x)` is an arbitrary tie-break, not a
      heuristic.
- [ ] **Reset `LandmarkStabilizer` when the selected face changes identity.** The
      existing centroid-jump reset does not catch a switch between two people who
      are standing still.

### Source guards

Applied at upload, before any embedding is built.

- [ ] Reject images containing **more than one face**
- [ ] Reject **no face** (exists; the reason must reach the UI)
- [ ] Reject faces under ~110 px on the shorter side
- [ ] Reject blurred sources below a Laplacian-variance floor
- [ ] Reject extreme pose beyond roughly ±35° yaw
- [ ] **Identity outlier check** — leave-one-out cosine similarity against the
      group mean, three images minimum
- [ ] **Report per-image reasons**, not a bare failure

### Runtime guards

- [ ] Guard frames with more than one detection while `many_faces` is off
- [ ] Guard on low detection confidence
- [ ] Guard faces under ~80 px
- [ ] Guard extreme pose
- [ ] Guard heavy occlusion (XSeg coverage below ~40% of the hull)

### Guard behaviour

- [ ] **Emit the last good swapped frame**, unchanged — nothing drawn on it
- [ ] **Fail closed**: an un-evaluable guard is a guarded frame
- [ ] **Never update temporal state** on a guarded frame — call
      `FaceCompositor.reset()` and `LandmarkStabilizer.reset()`
- [ ] **`_run_vcam` holds and re-sends the last frame** when the queue is empty,
      rather than stopping `cam.send()`. A stalled device can surface as
      "camera disconnected"; a frozen picture reads as a network hiccup.
- [ ] Add guard thresholds to `FaceSwapConfig` and the `set_realism` validator
- [ ] Calibrate the two unknown thresholds — sharpness floor, outlier cosine
      floor — against real uploads

---

## Stage 4 — Control plane, one session per GPU

*Ships: the product. Customers can run sessions.*

### Session management

- [ ] **Session state machine**: `QUEUED → ALLOCATING_GPU → GPU_READY →
      LOADING_MODEL → RUNNING → COMPLETED | ABORTED`, with the `ATTRIBUTE` step
      before the terminal states
- [ ] **Session store** with a durable record per session
- [ ] **Session ≠ attempt.** Several GPU attempts, one customer session, billed
      once
- [ ] **Idempotency by `session_id`** — a reconnect must not re-run completed
      work, and a retried failure notification must not credit twice
- [ ] **Session state carries the active mode**, since a customer switches
      between live and batch inside one session
- [ ] **Audit ledger** of every deduction and reversal with its attribution.
      There is no chargeback process to appeal to

### Scheduler

- [ ] **Promote `orchestrator.py` into a service.** GPU discovery, VRAM/price/
      compute-cap filtering, cheapest-first ordering and multi-datacenter
      fallback already exist — this is a port, and it brings regional redundancy
      with it
- [ ] **Slot-based capacity**, not GPU counting
- [ ] **Timeouts at every external boundary**: allocation, worker startup,
      model load
- [ ] **Retry classification** — infrastructure retries, user error does not —
      with a bounded attempt budget

### Everything N-shaped, with `max_sessions = 1`

Raising the number later must require no code change.

- [ ] **Pod release refcounted** — the *last* session out starts the grace
      period; the pod is released only if it expires with the count at zero
- [ ] **Move `runpod.stop_pod()` out of the worker.** A worker cannot see its
      neighbours and would kill them
- [ ] **Keep a worker failsafe**: stop the pod after a long period with no
      control-plane contact *and* no active session, so an outage does not leave
      pods billing forever
- [ ] **Session-scoped `_UPLOAD_DIR`** — currently the fixed `/tmp/phantom_uploads`
- [ ] **Session-scoped batch temp directories** — currently derived from the
      target filename
- [ ] **Control-plane-allocated worker ports**, not a hardcoded 9000
- [ ] **Worker registry keyed by (pod, slot)**, not by pod
- [ ] **Route clients to a session's worker**, not to "the pod"
- [ ] **`max_sessions` as per-GPU-type config**, defaulting to 1

### Billing

- [ ] **Balance denominated in hours**, not currency — packs sell hours at a
      discount, so a dollar balance cannot represent them
- [ ] **Deduct one hour when the session becomes usable** — worker running,
      models loaded, client connected — then run a wall clock
- [ ] **No return of unused time.** The hour is spent whether used or not
- [ ] **Extend modal before expiry**, deducting another hour if the balance
      allows. `auto_stop_warning`, `keep_alive` and the desktop countdown dialog
      already exist; they need repointing from pod uptime to session time
- [ ] **On expiry the pipeline disconnects and the virtual camera holds the last
      augmented frame** — never the raw camera, never nothing

### Desktop

- [ ] Client talks to the **control plane**, not a pod URL in `.env`
- [ ] Session state, remaining time, reconnection

---

## Stage 5 — Cold start

*Ships: sessions that start in tens of seconds rather than minutes.*

- [ ] **Bake a Docker image** with dependencies and model weights inside, so
      `apt-get`, `git pull` and `pip install` leave the critical path.
      `orchestrator.py` already has a docker deploy mode
- [ ] **Pre-seed every regional volume**, so a fallback region is not silently
      the slow path
- [ ] **No warm pool.** $720/month against roughly $900 of revenue at three
      sessions a day, to save a wait the customer is not billed for

---

## Stage 6 — Resilience

*Ships: sessions that survive infrastructure failure, or return the hour.*

- [ ] **Worker → backend heartbeat**, so a dead GPU stops being advertised as
      available. The existing 30s ping/pong is client↔worker; the backend learns
      nothing from it
- [ ] **Session leases**, renewed by the worker, expiring into recoverable
- [ ] **Progress-based watchdog** around the frame loop. A CUDA call can hang
      while the process answers pings perfectly — liveness is not health
- [ ] **Fault attribution.** Worker heartbeat gone is ours; client gone with the
      worker still healthy is theirs. Ambiguity resolves in the customer's favour
- [ ] **Hour reversal when the fault is ours** — atomic, never pro-rated,
      idempotent, audited
- [ ] **Decide the interruption threshold**: how long an outage must last before
      it reverts the hour
- [ ] **Build reserved standby capacity but leave it disabled.** Standby costs $1
      per hour held; a reverted hour costs $10 and only on sessions that actually
      fail. Trigger-gated like packing

---

## Stage 7 — Provider abstraction

*Ships: insurance against one vendor's availability and pricing.*

- [ ] Define the interface: provision, status, terminate, list capacity
- [ ] Move RunPod specifics behind it. Cheap now, with one implementation to
      conform to and its shape already visible in `orchestrator.py`

---

## Stage 8 — Packing

*Trigger-gated: average concurrency approaching 2, or GPUs hard to rent in a
region. Neither is true today.*

- [ ] **Run N=2 in staging even while production runs N=1.** The N-shaped paths
      cannot be proven correct at N=1 — only two sessions on one pod exercise the
      collisions
- [ ] **`max_sessions` benchmark**: ramp concurrent sessions recording latency,
      achieved fps, dropped frames, GPU utilisation, VRAM and **CPU per core**.
      Define the ceiling by a latency budget, not by absence of crashes
- [ ] **Raise `max_sessions`** to what the benchmark supports. If this needs more
      than a config change, stage 4 was built wrong
- [ ] *Optional, later:* shared models — remove the `CONFIG` and `BUS`
      singletons for per-session context, route frames by `session_id`. Buys VRAM
      headroom and a near-instant second session on a warm pod

---

## Stage 9 — Payment

*Deliberately late. Nothing here matters until the pipeline is worth selling.*

- [ ] **Bitcoin over Lightning.** 0.2% against 5.9% on cards, where the fee would
      exceed the compute it pays for
- [ ] **Start with a managed provider** (~1%) rather than self-hosted BTCPay
      (~0.2%). Lightning needs inbound liquidity and a node that stays online;
      move in-house when volume justifies the channel management
- [ ] **USD-denominated invoices** with a short expiry, roughly fifteen minutes
- [ ] **Decide hold or auto-convert.** Holding is an unhedged position on the
      entire top line
- [ ] **Lightning only.** With no tier above $40, a ~$3 on-chain network fee is
      at least 7.5% of any purchase and 30% of a PAYG hour
- [ ] **Purchases top up the hour balance.** Money never flows back out
- [ ] **No refund system.** If one is genuinely demanded, Bitcoin is sent
      manually — an exception handled by a person, not a code path
- [ ] Tier pricing: PAYG $10/hr, 2-Hour Pack $18, 5-Hour Pack $40


---

## Stage 10 — Authentication

*Last. The largest unwritten piece, and worth nothing before the product works.*

- [ ] User identity — the desktop app has no concept of one today
- [ ] Entitlement and balance ownership
- [ ] Session authorisation: which user may start a session, against which
      balance
- [ ] Reconnection with proof of identity, so a dropped client resumes its own
      session and nobody else's

---

## Decisions still open

Recorded here so they are not rediscovered mid-implementation.

| Question | Blocks | Notes |
|---|---|---|
| Batch job outliving its session | Stage 2 | Four defensible options |
| Is upload time billed? | Stage 2 | Overlap with startup is the likely answer |
| How aggressive should source rejection be? | Stage 3 | Friction at signup versus wrong output |
| Is pose available from `buffalo_l`? | Stage 3 | Else approximate yaw from the five keypoints |
| Is largest-face enough, or is continuity needed? | Stage 3 | Guarding may make it moot |
| Is a session one face, or one seat? | Stage 4 | Fixed identity makes packing cheaper |
| Does the scheduler need to know the tier? | Stage 4 | PAYG earns $10/GPU-hour against a 5-Hour Pack's $8.00 |
| Interruption threshold for hour reversal | Stage 6 | Sets what standby is worth |
| Does RunPod bill egress? | Stage 0 | 1.4% of a 5-Hour Pack at $0.05/GB |
