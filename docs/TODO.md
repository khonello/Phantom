# Implementation TODO

Everything the design documents call for that does not exist in the code yet,
in dependency order.

Sources: [SESSION_ARCHITECTURE.md](SESSION_ARCHITECTURE.md) (the spec),
[SESSION_PLANE.md](SESSION_PLANE.md) (assessment and staging),
[INPUT_GUARDS.md](INPUT_GUARDS.md) (input validation).

For the *order* to work through this in, with the commands, see
[PENDING_WORK.md](PENDING_WORK.md). This file stays the record of what is
outstanding; that one is the runbook for getting through it.

Items are grouped by stage; stages are ordered by what unblocks what, and by
what makes the product sellable. Payment and auth are last on purpose — a
billing system attached to a product that cannot hold a session together is
effort spent on the wrong end.

Checked items are built. Everything else is not.

---

## Stage 0 — Repository health

Small, independent, and they stop the other stages building on sand.

- [x] **Fix the CI test job.** Two separate defects, both fixed; a green run has
      not been observed yet, so confirm on the next push.
      - `requirements-ci.txt` did not exist in any commit, so the job failed at
        `pip install` and never reached the batch command at all. Now present,
        referencing `requirements-pipeline-cpu.txt` so the two cannot drift
      - The batch-video path it invokes is implemented (stage 2)
      - The job wrote its output over `.github/examples/output.mp4`, a *tracked*
        file, so a run that produced nothing could still pass by comparing the
        committed copy against the snapshot. Output now goes to `$RUNNER_TEMP`
        and is checked for existence before the comparison
- [x] **Clear the remaining mypy errors** — `mypy pipeline` is clean. This was
      not cosmetic: the CI `lint` job runs it and exits non-zero on any error, so
      that job had been failing on every push. Each was fixed at its site rather
      than blanket-ignored — a real `Dict[str, Face]` annotation, parameterised
      `Queue`s, and narrow per-line ignores where the cause is a missing stub
      (`cv2.VideoWriter_fourcc`, the runpod SDK) or a platform-only module
      (`resource`, POSIX). `mypy pipeline desktop` still reports 25, all in
      `desktop/`, which CI does not check.

      Superseded detail, kept because the count was quoted in planning: — `Queue` and `dict` type arguments,
      platform-specific module attributes, a `cv2` constant. All trivial; none
      are in code written recently. **11**, not 12 — and that is `mypy pipeline`,
      which is what CI runs. The command in CLAUDE.md, `mypy pipeline desktop`,
      reports **36**: the desktop package accounts for 25 of them and CI never
      looks at it.
- [x] **Add a test suite.** `tests/`, 165 checks across five modules, running
      under pytest in ~40s and wired into CI as a `unit` job. `conftest.py` stubs
      the ML layer into `sys.modules`, which is what lets the suite run with no
      GPU, no weights and no multi-gigabyte install — `pytest numpy
      opencv-python-headless tqdm` is the whole dependency list.

      Covers the parts that fail *silently*: guard predicates and thresholds,
      face selection, the stabilizer's identity reset, source review, held-frame
      behaviour, the FFmpeg batch plumbing (ordering, audio sync, cancellation,
      cleanup), execution-provider verification, and the realism metrics.

      Each file is still runnable directly for debugging; a `test_everything_passed`
      function is what surfaces it to pytest.

      **Not covered: the models.** Everything ML is stubbed, so this proves the
      logic around the models, never the models themselves. That gap closes on
      the pod, not here.
- [x] **Verify whether RunPod bills egress.** **It does not.** RunPod's Pod
      pricing documentation states "no fees for data ingress or egress", and the
      pricing page repeats it as a differentiator against AWS. The zero in the
      economics was right, and is now evidence rather than assumption.

      What *is* billed, so the model stays honest: compute per second, plus
      storage — container disk $0.10/GB/month (running only), volume disk
      $0.10 running / $0.20 stopped, network volume $0.07/GB/month under 1 TB
      and $0.05 above, charged whether the pod runs or not.

      That last one is the line worth watching instead: a network volume bills
      continuously, including while every pod is stopped, and the multi-datacenter
      fallback needs **one volume per region**. Egress was never the leak — idle
      regional volumes are.

---

## Stage 1 — Measure time to first frame

*Ships: nothing customer-facing. Sizes the only number that still matters.*

**The instrumentation is built; only the pod run is missing.** Both items below
now produce their own numbers, so this stage is a session rather than a
stopwatch exercise.

- [ ] **Time a cold provision end to end**, broken down by phase. `orchestrator.py
      start` and `resume` now print the breakdown themselves — `BootTimer` measures
      the coarse phases (provision, wait-for-ssh, remote-setup, pipeline-ready) and
      `startup.sh` reports its own inner phases as `PHASE <name> <seconds>` lines
      which the orchestrator folds in, so `pip-install` and `model-load` appear as
      separate rows rather than hiding inside "setup". Every run states whether the
      volume was **warm or empty**, so the two cases stop being remembered and
      start being labelled.

      Still to do: run it. Twice — once on a warm volume, once on an empty one.

      This is also what settles the Docker question. If `pip-install` dominates,
      baking an image is the answer; if `model-load` does, it is not, because the
      Dockerfile deliberately leaves weights on the network volume and the real
      fix is pre-seeding regional volumes instead.
- [ ] **Confirm a single session holds its latency budget** at each preset.
      `LatencyBudget` now records every frame's stage timings — not the 1-in-30
      sample at debug level, which cannot answer a question about a distribution —
      and reports p50/p95/p99 per stage against the deadline the preset's own
      capture rate sets (66ms at 15fps, 50 at 20, 33 at 30). It prints a `HOLDS`
      or `MISSES` verdict with the percentage of frames over deadline and the
      headroom at p95.

      A frame over deadline does not borrow time back from a fast neighbour, so
      the number that matters is the fraction over and the p95, never the mean.

      Still to do: run one session per preset and read the verdict.

The `max_sessions` load harness is deliberately *not* here — nothing depends on
that number until stage 7.

---

## Stage 2 — Close the product gap

*Ships: the other half of what a session is sold as. Launch prerequisite.*

A session is sold as "live **or** batch". Batch now runs end to end for both
images and video; what remains is getting large files to and from the worker, and
deciding what a job that outlives its session costs.

Photo mode has since closed the *image* half of the transfer problem, and is the
only target shape that reaches a remote worker at all today. Video is unchanged
and still blocked on a real transfer path.

- [x] **Wire `_process_target_batch()` for video.** Extract → swap in place →
      encode → restore audio, reusing the FFmpeg helpers and the same compositor
      as live. Splits into `_process_image_batch` and `_process_video_batch` over
      a shared `_swap_frame_faces`, so the two cannot drift. Landmark smoothing
      is on for video (consecutive frames) and off for a lone image. Honours
      `_stop_event` between frames, reports throttled progress with an ETA, and
      cleans its scratch space on every exit path.
- [x] **Photo mode** — one to four target photos, each swapped independently,
      failures skipped. No new pipeline stage: it loops the existing image path,
      so every photo goes through the same guards and the same compositor as a
      live frame. `upload_target` carries them base64, capped at 4 x 6 MB and
      enforced on both sides, which sidesteps the transfer problem rather than
      solving it — photos are small, video is not.

      Two things fell out of building it, both worth more than the feature:

      - **`set_target` never worked against a pod.** It validates with
        `os.path.exists` on the *pipeline's* filesystem, so a desktop-chosen
        file only ever resolved when the pipeline ran locally. Photo targets are
        the first that reach a remote worker.
      - **A refused still used to produce an output file.** `_process_image_batch`
        wrote unconditionally, so a guarded or faceless target left a copy of the
        input named like a result — the exact failure the guards exist to
        prevent, and invisible to whoever opens the folder. It now writes nothing
        and reports the reason. Video still passes frames through, which is
        correct there; the two now differ deliberately.

      Covered by `tests/test_photo_batch.py` (59 checks) and a CI step that runs
      the image path with real models for the first time.
- [ ] **Build a real file transfer path** for **video**: chunked, resumable,
      progress-reporting, both directions. `upload_source` is base64 inside one
      JSON message — fine for a 200 KB face, unusable for a 2 GB video. Photo
      mode does not change this; it only removes stills from its scope.
- [x] **Template targets.** Bundled scenes the source face is swapped into —
      the target is ours, the face is theirs. It adds no job shape: a template
      runs as a photo job of one, through the same guards, compositor and
      result path photo mode already proved.

      Cheaper than photo mode was, for one structural reason: templates live on
      the pipeline's filesystem, so there is nothing to transfer. The upload
      machinery photo mode needed simply does not apply.

      Three decisions worth keeping:

      - **`face_point` is a normalised point, not an index.** Detection order is
        not a stable contract, and an index that comes to mean a different
        person is a silent wrong-person swap — no error, no crash.
      - **A named face stands the multi-face guard down.** The guard refuses a
        crowd because the question has no safe default; a template answers it
        offline, so a scene we shipped on purpose is not refused.
      - **Outputs never land in the library.** Writing beside a shared asset
        would leave one user's face there for the next job.

      `tools/validate_templates.py` runs the real guards over the library and
      exits non-zero, so a scene that would be refused never ships. Covered by
      `tests/test_templates.py` (45 checks).

      **Not done: the templates themselves.** The machinery runs against an
      empty library. The assets are a content decision — including where they
      come from and how they are licensed for this use — and that, not the
      code, is the real cost of this feature.
- [ ] **Face selection as a shared predicate.** *Deferred deliberately — recorded
      here so the reasoning is not re-derived later.*

      Every mode already answers "which face?" and answers it the same silent
      way: `select_primary` takes the largest detection, in live, render and
      photo alike. Making that an operator choice belongs **below** the mode
      layer, not inside photo mode — it is a predicate to any augmentation, not
      a feature of one job shape.

      It also resolves the multi-face guard properly. The guard fires because
      "which face did you mean?" has no safe default; today the only escape is
      `many_faces`, which swaps *every* face. A first-class selection makes the
      answer "the one they picked", in every mode.

      One concept, two implementations, and only the first is cheap:

      - **Still** — one act, stable answer, a click on a box. Local-testable.
      - **Video / live** — detection order is not stable frame to frame (which
        is why temporal EMA is bypassed under `many_faces`), so this is picking
        an *identity* and re-identifying it every frame. The primitive already
        exists: `LandmarkStabilizer._identity_changed` runs a cosine over
        `normed_embedding` with a 3-of-6 window. Selecting and following is the
        same dot product asked the other way round.

      Build the selection as something the pipeline consults instead of calling
      `select_primary` directly, do the still case first, and let the video case
      land behind the same interface once there is real footage to tune the hold
      through occlusions against.

      **Partly borrowed already.** Template targets needed the same predicate
      and took the cheap half of it: `config.target_face_point`, consulted by
      `DetectionProcessor` ahead of `select_primary`, plus the guard standing
      down when a face is named. What is still missing is the *operator* naming
      one — a picker on a still, and identity-following on a stream. The seam
      is in place; only the ways of filling it are not.
- [ ] **Decide and implement the overflow policy** for a batch job that outlives
      its session: refuse / bill overflow / absorb / detach and hold the result.
- [ ] **Decide whether upload time is billed.** Overlapping transfer with worker
      startup is the likely answer, but it has to be designed in.
- [x] Desktop VIDEO mode and the CI test both start working as a side effect.
      Not quite free, in the end — see below.

### Found while wiring it

Three defects sitting in the path, none of which were reachable while video
batch returned an error string.

- [x] **`ERROR` events never reached any client.** The server forwards
      `STATUS_CHANGED`, `DETECTION`, `WARNING` and the lifecycle events, but not
      `ERROR` — and `emit_error` publishes only to `ERROR`. A batch reports
      completion *by stopping*, so every failure arrived at the desktop as a bare
      `PIPELINE_STOPPED` and was rendered as "processing complete". The server
      now forwards it as an error-level status, and the bridge holds the message
      and reports `failed: <reason>` instead of marking the job complete.
- [x] **Frame ordering broke past 9999 frames.** `extract_frames` wrote `%04d.png`
      while `get_temp_frame_paths` orders by sorting filenames, so at 10000
      frames `'10000.png'` sorts before `'9999.png'` and every later frame was
      silently reordered. That is 5m34s at 30fps. Now `%06d`.
- [x] **`keep_fps = False` desynchronised the audio.** `create_video` passes `-r`
      *before* `-i`, which sets the image2 demuxer's **input** rate — so the
      frames were consumed at 30fps regardless of the source, rescaling a 4.00s
      24fps clip to 3.20s of video against 4.02s of restored audio. Frames are
      now always read at the source rate and retiming is a separate `fps=` filter,
      which drops or duplicates frames and preserves duration.

---

## Stage 3 — Input guards

*Ships: the pipeline stops producing confidently wrong output.*

Independent of the control plane, and worth doing before customers arrive.

**Built**, except for calibrating two thresholds, which needs real uploads. See
[INPUT_GUARDS.md](INPUT_GUARDS.md) for the design and
`pipeline/services/guards.py` for the implementation.

### Fixes that stand on their own

- [x] **Replace `detect_one`'s leftmost rule with largest-face.** Now
      `FaceDetector.select_primary`, so a caller that needs the whole list to
      count faces does not run detection twice. `DetectionProcessor` calls
      `detect()` and trims, rather than `detect_one()` which threw the count
      away — and the count is what the multi-face guard *is*.
- [x] **Reset `LandmarkStabilizer` when the selected face changes identity.**
      Compares the recognition embedding InsightFace already computed, so it
      costs a dot product. The remembered identity deliberately survives
      `reset()`: clearing it would leave the frame after a reset with nothing to
      compare against, so two people alternating would be caught on only every
      second frame.

### Source guards

Applied at upload, before any embedding is built.

All in `pipeline/services/guards.py`, as pure predicates over data the pipeline
already has, called from `FaceDatabase.review_sources`.

- [x] Reject images containing **more than one face**
- [x] Reject **no face** — the reason now reaches the UI
- [x] Reject faces under ~110 px on the shorter side. The *shorter* side, not the
      longer one or the area: a tall narrow box has too few pixels across the
      features however tall it is
- [x] Reject blurred sources below a Laplacian-variance floor
- [x] Reject extreme pose beyond roughly ±35° yaw. Pose is not reliably exposed
      by `buffalo_l`, so yaw is approximated from the five keypoints — the nose's
      offset along the inter-eye axis, normalised by eye span, so it is scale and
      roll invariant. Coarse by design; it only has to separate "roughly frontal"
      from "far enough that ArcFace degrades", and the alternative is a second
      model on the critical path
- [x] **Identity outlier check** — leave-one-out cosine against the mean of *the
      others*, three images minimum. Including a candidate in the mean it is
      tested against is what lets one wrong photo drag the reference toward
      itself and pass
- [x] **Report per-image reasons**, not a bare failure. `SourceReview` carries
      one outcome per image; `upload_source` returns them and the desktop names
      the refused file. Partial rejection is reported too — a source built from 1
      of 3 photos still succeeds, and the label would otherwise go on claiming
      all three were averaged

### Runtime guards

- [x] Guard frames with more than one detection while `many_faces` is off
- [x] Guard on low detection confidence
- [x] Guard faces under ~80 px
- [x] Guard extreme pose
- [x] Guard heavy occlusion (XSeg coverage below ~40% of the hull). Measured
      against the hull *alone*, not hull × valid-region: a face at the frame edge
      is cropped, not occluded, and including that term would guard it for the
      wrong reason. No second XSeg pass — `FaceMasker` records the coverage from
      the inference it already runs, and `FaceCompositor` builds the mask before
      restoration and smoothing so the guard can refuse the frame before
      anything mutates temporal state

Zero faces is deliberately **not** a guarded frame. That is someone stepping out
of shot, already handled, and guarding it would hold a stale face over an empty
chair.

### Guard behaviour

- [x] **Emit the last good swapped frame**, unchanged — nothing drawn on it
- [x] **Fail closed**: an un-evaluable guard is a guarded frame. This also
      corrected `FaceCompositor.composite`, which returned the *untouched frame*
      when compositing failed — on the live path that is the operator's real
      face, the exact exposure the guards exist to prevent. It now returns None
      and the caller decides: live holds the last good frame, batch passes the
      original through, since a batch target is a file the operator supplied
      rather than their camera
- [x] **Never update temporal state** on a guarded frame — calls
      `FaceCompositor.reset()` and `LandmarkStabilizer.reset()`
- [x] **`_run_vcam` holds and re-sends the last frame** when the queue is empty,
      rather than stopping `cam.send()`. This covers every way frames can stop,
      not just guarded ones — hour expiry, session end, worker death, a crash
- [x] Add guard thresholds to `FaceSwapConfig` and the `set_realism` validator.
      Guard values are *clamped* rather than rejected, so sweeping a threshold
      live gives the nearest legal value instead of an error
- [ ] Calibrate the thresholds against real footage. **The measurement is built;
      only the session is missing.** Run with `--guard-observe --guard-report
      report.json` and every guard is evaluated and recorded while none of them
      act — a session that *enforces* cannot measure itself, since a guarded
      frame emits a held frame and stops being a sample of what the camera was
      doing. The report gives a distribution per metric with the percentage that
      would fail and the margin to the threshold, so the output is a number per
      knob rather than "it seemed to guard a lot".

      It turned out to be **nine** thresholds, not two, and three of them can
      make things actively worse:
      - `guard_min_coverage` (0.4) is compared against XSeg coverage of an
        *expanded* hull. What that reads on a completely clear face was never
        measured, so the floor could sit inside normal range and guard constantly
      - `guard_identity_sim` (0.5) — promoted from a constant on
        `LandmarkStabilizer` to config for this. If it sits above where the same
        person lands under motion blur, the stabilizer resets every frame and the
        shimmer it exists to remove comes back: **a realism regression caused by
        a guard**
      - `guard_min_confidence` (0.5) against a detector threshold of 0.35, so
        everything scoring in between is guarded

      `guard_min_sharpness` (40.0) and `guard_outlier_sim` (0.35) remain the two
      that need *upload* data rather than footage. All are deliberately
      permissive: a guard that turns away a usable photo is friction at the exact
      moment a new customer is deciding whether this works.
- [x] **Guard calibration telemetry** — `guards.GuardTelemetry`, reported to the
      log on stop and written as JSON with `--guard-report`. Records the measured
      value behind every guard, not just the verdict, because watching output
      only ever reveals *that* something is wrong, never which number
- [x] **Capability probe** — logs on first detection which `Face` attributes the
      model pack actually provides. Several guards silently become no-ops when
      their input is missing (no `normed_embedding`, no identity reset; no
      `det_score`, no confidence guard), and a silent no-op looks exactly like a
      guard that never had cause to fire
- [x] **Yaw now prefers `face.pose`.** `buffalo_l` bundles `1k3d68.onnx`, whose
      3D-landmark model sets `face.pose` as a side effect of detection — so the
      real estimate was already being computed and thrown away. The keypoint
      approximation is now the fallback for trimmed packs. The two are not on the
      same scale, so the telemetry records which source was used

---

## Stage 3.5 — Call realism

*Ships: output that behaves like a video call, not just a face that looks like one.*

Most video-call character is inherited — real webcam, real network — and the
existing per-frame colour, detail and grain matching is what stops the face
undoing it. Two physical cues are still unmatched, and both appear during
movement.

- [ ] **Motion blur matching.** During head movement the real frame smears and
      the generated face does not, so the swap is *sharper than what it
      replaces*. Estimate displacement from the stabilised landmarks the pipeline
      already computes, and apply matching directional blur to the aligned crop.
      Magnitude and angle both fall out of the landmark delta
- [ ] **Drop frames evenly when falling behind**, rather than accumulating
      latency. Even dropping reads as bandwidth; growing lag reads as a machine
      struggling, and desynchronises from the audio
- [ ] **Fall back a quality preset under load** rather than missing deadlines at
      the current one — which is what a call client does
- [ ] *Noted, not scheduled:* rolling-shutter skew. Much subtler than motion
      blur and much harder to model
- [x] **Debug frame capture.** `--debug-frames DIR` writes lossless
      `NNNNNN_in.png` / `NNNNNN_out.png` pairs from a live session, on a
      background thread so it does not change the latency it exists to measure.
      `--debug-frames-stride` and `--debug-frames-limit` bound the volume
- [ ] **Record one clip of real output and watch it.** Nothing above has been
      checked against a real call. This gates every other item in this stage and
      is the highest-value outstanding task on the project. The capture
      (`--debug-frames`) and the measurement (below) both exist now, so this is
      one session away
- [x] **Write the comparison script.** `tools/compare_frames.py`, measuring a
      `--debug-frames` capture for all five: noise sigma inside vs outside the
      mask, high-frequency energy ratio, LAB distribution, gradient
      discontinuity across the mask boundary, and blur anisotropy face-vs-frame
      **during motion specifically** — averaging that over a still clip hides
      the very effect it exists to find.

      Every ratio is inside-over-outside taken from the *output* frame, because
      the question is not whether the swap resembles the original face but
      whether it belongs in the picture it is now part of.

      The face region is derived from the input/output difference rather than
      from a detector: the swap marks its own extent, so this needs no models
      and no GPU. `--against DIR` diffs two captures, which is how a realism
      change gets judged rather than argued about.

      It prints a plain-language reading, not just numbers — "TOO CLEAN: the
      face carries 14% of the sensor noise the rest of the frame has" — and is
      tested against synthetic frames with each defect deliberately present, so
      the metrics are known to detect what they claim rather than merely
      producing a number that would get quoted in decisions

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
- [x] **Session-scoped batch temp directories** — done early, since stage 2 was
      writing these call sites for the first time and scoping them later would
      have meant editing them twice. Scratch space is now
      `<root>/<session>/temp/<target>`: keyed by session so two sessions handed
      the same filename cannot collide, with the target name kept as the leaf so
      one session can process several targets. Root is `PHANTOM_TEMP_DIR` or the
      system temp — no longer a `temp/` directory created next to the target,
      which on a pod is inside the upload directory. `set_temp_scope()` is what
      the control plane will call; until then the scope defaults to
      `PHANTOM_SESSION_ID`, else a per-process token, which is exactly one
      session while `max_sessions` is 1. `reset_temp()` clears scratch before a
      run unconditionally, so frames from an aborted job cannot be picked up and
      re-encoded into the next one — `clean_temp` could not do that job, since
      `keep_frames` would have turned it into silent corruption.
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
| How aggressive should source rejection be? | Stage 3 | **Open.** Started permissive; needs real upload data to settle |
| ~~Is pose available from `buffalo_l`?~~ | — | **Settled:** not reliably, so yaw is approximated from the five keypoints |
| ~~Is largest-face enough, or is continuity needed?~~ | — | **Settled:** enough. Two comparable faces guard the frame anyway, and the identity check on the stabilizer catches a switch the size rule misses |
| Is a session one face, or one seat? | Stage 4 | Fixed identity makes packing cheaper |
| Does the scheduler need to know the tier? | Stage 4 | PAYG earns $10/GPU-hour against a 5-Hour Pack's $8.00 |
| Interruption threshold for hour reversal | Stage 6 | Sets what standby is worth |
| ~~Does RunPod bill egress?~~ | — | **Settled: no.** Documented as "no fees for data ingress or egress". Watch idle per-region network volumes instead |
