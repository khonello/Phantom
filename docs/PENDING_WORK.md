# Pending Work

Everything left to do, in the order it should happen, starting from a cold
machine.

This is a runbook, not a backlog. [TODO.md](TODO.md) is the backlog and stays the
source of truth for *what* is outstanding; this document is the sequence for
getting through it, with the commands.

**Where things stand.** A large amount was built without ever running against a
GPU, a model, or a real face. The build-out is finished for now, and the binding
constraint has moved: nothing further should be built on top of the recent work
until a single pod session has confirmed it does what it claims. Phase 2 is
therefore the whole of the near-term plan, and it is roughly one hour of pod
time.

---

## Phase 0 — Before spending anything

All local, all free. Twenty minutes.

### 0.1 Confirm the tree is green

```bash
python -m pytest tests/ -q                              # expect: 7 passed, ~50s
flake8 pipeline.py pipeline desktop tests tools runpod  # expect: silent
mypy pipeline                                           # expect: 11 errors (baseline)
bash -n runpod/startup.sh && bash -n runpod/entrypoint.sh
```

Seven test modules, ~245 checks. `tests/test_wiring.py` is the one to watch: it
asserts the seams *between* files — that every forwarded env var is read, that
`Dockerfile` and `.env.example` pin the same image, that both deploy paths
pre-warm. Every historical break in this repo lived in one of those gaps.

The 11 mypy errors are known and unrelated to recent work — `Queue` and `dict`
type arguments, platform-specific attributes. `mypy pipeline desktop` reports 36;
the extra 25 are all in `desktop/` which CI does not check. Do not treat either
number as a regression unless it moves.

### 0.2 Push and watch CI

The CI file gained two jobs (`unit`, `docker`), had two defects fixed, and its
lint step now covers `tests`, `tools` and **`runpod`** — which it did not before,
which is exactly how an `F821` sat undetected in `orchestrator.py`. It also runs
`bash -n` on both shell scripts, since a syntax error there is a failed
provision discovered after paying for a GPU.

**No run has been observed yet.** This is the cheapest possible verification, so
do it before the pod.

Expect four jobs: `lint`, `unit`, `test`, `docker`.

| Job | If it fails |
|---|---|
| `lint` | flake8/mypy regression, or a shell syntax error — fix before continuing |
| `unit` | the test suite; reproduce locally with `pytest tests/ -q` |
| `test` | end-to-end CPU swap. **First real check of batch video with models in the loop** |
| `docker` | image build. Catches a dead base tag, an unresolvable package, or the cuDNN check failing |

The `test` job matters more than its name suggests: it is the first time the
batch-video path runs with real InsightFace and ONNX rather than a stub. If it
passes, batch video is verified on CPU and only the GPU path remains open.

### 0.3 Have a source face ready

One clear, frontal, well-lit photo of one person, face at least 110px on the
shorter side. The source guards will reject anything else, which is the point —
but discovering that on the pod wastes billed time.

If testing the guards deliberately, prepare a second set: a photo with two
people, a blurry one, a profile shot, and three photos of one person plus one of
someone else.

---

## Phase 1 — Decide the cold-start experiment

Read this before starting anything, because one of the two measurements is
destructive and the choice affects what Phase 2 can answer.

Cold start needs timing under two conditions:

| Condition | Cost | How |
|---|---|---|
| **Warm volume** | Free | Your existing volume already has venv, models and repo. Just `start` |
| **Empty volume** | A full re-provision | Requires a volume with nothing on it |

`RUNPOD_DATACENTERS` currently lists **one** datacenter, so there is no second
region whose volume is already empty to measure for free.

Three options, in order of preference:

1. **Add a second datacenter and volume.** Gets the empty-volume number as a side
   effect of the regional redundancy Stage 4 wants anyway, and leaves the working
   volume untouched. Costs a small standing storage charge (see the note on
   volume billing below).
2. **Measure warm only for now.** Perfectly reasonable. The warm number is what a
   returning customer experiences and is the more common case. Record that the
   empty case is unmeasured rather than guessing at it.
3. **Wipe the existing volume.** Cheapest in storage, worst in risk: everything
   re-downloads, and a failed re-provision leaves nothing to fall back to. Only
   do this deliberately.

> **Volume billing.** Network volumes bill at $0.07/GB/month **whether or not any
> pod is running**. Egress is free (confirmed in RunPod's pricing docs), so
> bandwidth is not the cost line to watch — idle per-region volumes are. If a
> second datacenter is added for option 1, that volume bills continuously.

---

## Phase 2 — The measurement session

**One pod session answers every open question.** Read the whole phase before
starting it; the ordering exists to avoid paying for a second session.

`RUNPOD_MAX_UPTIME` will stop the pod automatically, so an abandoned session is
capped rather than open-ended. It is still worth running `stop` when done.

### 2.1 Start the pod, and capture the cold-start breakdown

Set what the run should do in `.env` **before** starting — the orchestrator
forwards those settings into the pod, so the pipeline it auto-starts is already
configured. Without that, configuring a run means SSHing in and restarting the
pipeline by hand, which is most of a session.

```env
SWAPPER_MODEL=hyperswap_1a_256
GUARD_OBSERVE=true
GUARD_REPORT=/workspace/guards.json
DEBUG_FRAMES_DIR=/workspace/clip
DEBUG_FRAMES_STRIDE=3
```

```bash
python runpod/orchestrator.py start
```

This now prints its own phase table at the end. **Save the output** — it is the
Stage 1 deliverable and is not written to a file.

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

Rows marked `*` are exclusive; indented rows break down the row above. The
`volume:` label states warm or empty, so the two runs are distinguishable later
without relying on memory.

**Watch for two things in the log before anything else:**

- `Execution provider: CUDAExecutionProvider confirmed on ...` — if instead the
  pipeline **exits** with an `ExecutionProviderError`, that is deliberate. It
  means ONNX would have run on CPU, which is seconds per frame and a wasted GPU
  hour. See `runpod/TROUBLESHOOTING.md` §5b.
- The **pre-warm report**, which now covers all four models rather than two:
  `ok detection` / `ok swap` / `ok restoration` / `ok occluder`. A `REQUIRED`
  line means that model will download on the first frame of a paid session
  instead. Note this block previously called `Enhancer()` with no arguments and
  silently skipped restoration on every deploy, so a first green run here is new
  information.
- `Model provides: det_score, kps, landmark_2d_106, landmark_3d_68,
  normed_embedding, pose` — the capability probe. **If `pose` is missing**, yaw
  falls back to a keypoint approximation on a different scale and the
  `guard_max_yaw` threshold means something different. **If `normed_embedding` is
  missing**, the stabilizer's identity reset silently never fires.

### 2.2 Verify batch video with real models

The first thing to check, because it is the only completely unverified feature
rather than an unverified threshold.

```bash
# On the pod
cd /workspace/Phantom
/workspace/venv/bin/python pipeline.py \
    --execution-provider cuda \
    -s <source>.jpg -t <target>.mp4 -o /workspace/out.mp4
```

Check, in order:

- [ ] It completes without error
- [ ] Progress lines appear with an ETA, roughly once a second — not once a frame
- [ ] `ffprobe /workspace/out.mp4` — duration matches the source
- [ ] Audio stream present, and **the same length as the video** (this is the
      `keep_fps` desync that was fixed; it is worth confirming on real footage)
- [ ] The scratch directory is gone: `ls /tmp/phantom/` should be empty or absent
- [ ] Watch it. Do the faces track, or does the swap drift?

Then a long clip — over 9999 frames, i.e. more than ~5m34s at 30fps — to confirm
frame ordering past the four-digit rollover that was fixed. If a long clip is
inconvenient, this is the one check that can be deferred.

### 2.3 Compare swap models on one clip

The highest-value measurement available, and the reason to do it *before*
calibrating anything: the realism knobs differ per model, so calibrating on
inswapper and then switching would throw the calibration away.

```bash
# Same clip, one model per run. --debug-frames into separate directories.
/workspace/venv/bin/python pipeline.py --stream --execution-provider cuda     --swapper-model inswapper_128 --debug-frames /workspace/clip-inswapper

/workspace/venv/bin/python pipeline.py --stream --execution-provider cuda     --swapper-model hyperswap_1a_256 --debug-frames /workspace/clip-hyperswap
```

Then locally:

```bash
python tools/compare_frames.py clip-inswapper/ --against clip-hyperswap/
```

Judge on `hf_ratio` and `noise_ratio` landing **nearer 1.0**, not on which crop
looks crisper zoomed in — a 256 model will always look sharper in a still, and
sharper than the frame is one of the three failure modes.

Note the download cost: **384 MB per model**, so pulling all three hyperswap
variants is ~1.2 GB before a frame is served. Pull the one under test.

Expect hyperswap to need its knobs re-swept afterwards. The profile
(`enhance_strength` 0.5, `enhancer_weight` 0.8) is a mechanically-reasoned
starting point, not a measured one.

### 2.4 Guard calibration and latency, in one stream run

`--guard-observe` evaluates every guard and records what it *would* have done
without any of them acting. This matters: a session that **enforces** cannot
measure itself, because a guarded frame emits the held frame and stops being a
sample of what the camera was doing.

```bash
/workspace/venv/bin/python pipeline.py --stream \
    --execution-provider cuda \
    --quality optimal \
    --guard-observe \
    --guard-report /workspace/guards-optimal.json \
    --debug-frames /workspace/clip-optimal \
    --debug-frames-stride 3
```

Then connect the desktop (`python desktop.py` locally) and **use it like a real
call for two to three minutes**:

- Sit still and talk — the baseline
- Turn your head slowly left and right, past profile — exercises yaw, and is the
  only way to get motion-blur data
- Move closer and further — exercises face size and the compositing step ladder
- Put a hand across your face — exercises occlusion coverage
- Have a second person enter frame briefly — exercises multi-face and the
  stabilizer's identity reset
- Step out of frame entirely and come back

Stop the pipeline cleanly (not `kill -9`) so the reports print.

**Repeat per preset.** Change `--quality` to `fast` and then `production` and run
again. Restart the pipeline between presets rather than switching quality live:
the latency budget accumulates across the whole run and would otherwise mix three
different frame deadlines into one verdict.

That gives three guard reports, three latency verdicts and three clips.

### 2.5 Collect everything before stopping

```bash
scp -r <pod>:/workspace/guards-*.json .
scp -r <pod>:/workspace/clip-* .
scp <pod>:/workspace/out.mp4 .
scp <pod>:/workspace/phantom-pipeline.log .
```

The log matters as much as the reports — it holds the cold-start table, the
capability probe, and the provider confirmation.

### 2.6 Stop the pod

```bash
python runpod/orchestrator.py stop
```

`stop` preserves the volume and the container disk. Use `terminate` only to
delete the pod; the network volume survives either way.

---

## Phase 3 — Read the results

All local and free. This is where the session turns into decisions.

### 3.1 Guard thresholds

Each report gives, per metric, a distribution and the **margin** to the
configured threshold. A negative margin means the threshold sits inside normal
operating range and will fire on ordinary frames.

Three of the nine can make the product actively worse, and none is visible by
watching output:

| Threshold | Default | What to look for |
|---|---|---|
| `guard_min_coverage` | 0.4 | What XSeg reads on a *clear* face is unknown. If p1 coverage is near 0.4, ordinary frames will guard |
| `guard_identity_sim` | 0.35 | Check the `identity_sim` p1 against 0.35. The mechanism now tolerates isolated dips, but a floor above where the same person actually lands under motion would still cost smoothing |
| `guard_min_confidence` | 0.5 | Detector threshold is 0.35, so everything between is guarded. If a real webcam commonly lands there, this is too aggressive |

Set each from the data, then re-run one short session with guards **enforcing**
(drop `--guard-observe`) and confirm the guard rate is low and the frames that do
guard deserve it.

`guard_min_sharpness` and `guard_outlier_sim` need *upload* data rather than
footage — test them by uploading the deliberately-bad photos from step 0.3 and
checking each is refused with the right reason.

### 3.2 Latency

Each session prints `HOLDS` or `MISSES` with the fraction of frames over deadline
and the headroom at p95. Deadlines are 66ms at 15fps (fast), 50ms at 20fps
(optimal), 33ms at 30fps (production).

If `production` misses, that is expected and useful — it is the preset most
likely to. The decision it feeds is whether to make the preset cheaper or to stop
offering it.

### 3.3 Realism

```bash
python tools/compare_frames.py clip-optimal/
python tools/compare_frames.py clip-optimal/ --against clip-production/
```

Then **watch the clip**. The TODO calls this the highest-value outstanding task
on the project, and the numbers do not replace it — they tell you *which* of the
three failure modes to look for.

The script's reading will say things like *"TOO CLEAN: the face carries 14% of
the sensor noise the rest of the frame has"* or *"MOTION MISMATCH: during
movement the frame smears (2.28) while the face does not (1.14)"*. The second
would confirm motion-blur matching is worth building; the first would point at
grain and restoration strength instead.

### 3.4 The Docker decision

From the cold-start table:

- **`pip-install` dominates** → bake an image. The Dockerfile already exists and
  builds in CI; switching is `RUNPOD_DEPLOY_MODE=docker`
- **`model-load` or downloads dominate** → do **not** bake. The Dockerfile
  deliberately leaves weights on the network volume, so it would not help. The
  fix is pre-seeding regional volumes instead
- **`provision` dominates** → neither helps; that is RunPod scheduling, and the
  lever is `support_public_ip`, which is why Docker mode schedules more freely

Keep both modes regardless. The rule that makes that safe is already in place:
CI builds the image on every push, so the unused path cannot rot silently the way
it did before.

---

## Phase 4 — Implementation, in order

Only after Phase 3. Each item's priority depends on what the data says.

### 4.1 Immediate follow-ups

- [ ] **Act on the guard calibration** — set the nine thresholds from the report
- [ ] **Clear the 11 mypy errors** (Stage 0). Trivial, and the only remaining
      Stage 0 item
- [ ] **Stop the pod on a fatal startup error.** If the pipeline now refuses to
      start — a provider fallback, say — the pod keeps billing until
      `RUNPOD_MAX_UPTIME`, up to ~$2 wasted. Belongs with Stage 4's "move
      `runpod.stop_pod()` out of the worker"

### 4.2 Stage 3.5 — call realism

Gated on the clip. Build in the order the measurements justify:

- [ ] **Motion blur matching** — if `compare_frames` reports a motion mismatch.
      Displacement comes from the stabilised landmarks already computed
- [ ] **Drop frames evenly when falling behind** — if the latency budget reports
      `MISSES`. Even dropping reads as bandwidth; growing lag desynchronises from
      audio
- [ ] **Fall back a preset under load** — same trigger, and what a real call
      client does
- [ ] *Noted, not scheduled:* rolling-shutter skew

### 4.3 Stage 2 — the rest of the product gap

- [ ] **Chunked, resumable file transfer.** `upload_source` is base64 in one JSON
      message: fine for a face, unusable for a 2 GB video. **This is what makes
      batch video usable over the API rather than only over the CLI**
- [ ] Decide the overflow policy for a batch job outliving its session
- [ ] Decide whether upload time is billed

### 4.4 Stage 4 and beyond

Unchanged from [TODO.md](TODO.md), and all substantially larger than anything
above: the control plane and session state machine, then cold start, resilience,
provider abstraction, packing, payment, authentication.

Nothing in Stage 4 should start until Phase 3 is done. It is the largest body of
work in the project, and building a session plane around a pipeline whose
realism is unconfirmed is effort spent on the wrong end — the same argument the
TODO already makes for putting payment last.

---

## Deliberate behaviour — do not soften these

Decisions that look like bugs if you meet them without context. Each one exists
because the alternative is worse, and each is easy to "fix" into uselessness by
someone who does not know why it is there.

### Requesting a GPU and not getting one is fatal

If CUDA is asked for and the models are not actually running on it, the pipeline
**raises and refuses to start**. It does not warn and continue.

This is a deliberate product decision, not a safety default that got left on:
a pod running on CPU is the worst possible outcome. It bills a full GPU hour and
produces output at seconds per frame — unusable — while looking like it is
working. Running on CPU is not a degraded version of the product; it is a bill
with nothing attached, and it defeats the entire reason for renting the pod.

Enforced at three layers, so it cannot slip through whichever way it is deployed:

| Layer | Where | Effect |
|---|---|---|
| Runtime | `execution.py::verify`, `strict=True` | Pipeline refuses to start |
| SSH setup | `startup.sh` step 6b, `exit 1` | Provisioning stops |
| Docker build | `Dockerfile`, `CDLL('libcudnn.so.9')` | Image cannot be built |

`--execution-provider cpu` remains fully supported and does **not** raise — an
explicit choice to run without an accelerator is a different thing from silently
falling back to one.

**Do not downgrade any of these to a warning.** ONNX Runtime already warns, and
that warning is precisely what let this ship broken once: it was found by reading
the Dockerfile, not by anything failing. The whole value here is that it stops.

### A guarded frame holds the last good frame

Never the raw camera, and nothing is drawn on it. If nothing has been swapped yet
there is no frame to hold, so **nothing is sent at all** — which looks like a bug
and is not. The operator is on the call precisely because they do not want their
own face transmitted; showing it would turn the guard into the exposure it exists
to prevent.

### An identity change takes several frames to be believed

`LandmarkStabilizer` does not drop its smoothing the first time an embedding
falls below `guard_identity_sim`. It waits for 3 low readings within the last 6
frames.

This looks like sloppiness and is the opposite. An embedding comes from a crop
that can be motion-blurred, half-turned or badly lit for a single frame and
recover on the next — and resetting on that drops the landmark EMA *during
movement*, which is exactly when shimmer is most visible. A guard reinstating
the shimmer it exists to remove would be a realism regression caused by a safety
feature, which is the worst kind.

The window rather than a consecutive run matters: a detector flickering between
two people gives good, bad, good, bad, and a consecutive counter is zeroed by
every good frame and never fires at all.

Cost of waiting is bounded. Any frame with two faces is already refused by the
multi-face guard before it reaches the stabilizer, so this is a backstop against
detection flicker rather than the primary defence — and `reset()` discards the
smoothing history wholesale, so the few blended frames are thrown away rather
than lingering.

### The pipeline pre-warms before reporting ready

Both deploy paths load all four models during setup rather than on the first
frame. Model weights live on the network volume by design, not in the image, so
they are the one thing still fetched at run time — and hyperswap is a 384 MB
download that would otherwise land on a customer.

`runpod/prewarm.py` is shared by `startup.sh` (SSH) and `entrypoint.sh` (Docker).
A pre-warm failure is a warning, not a stop: it costs a slow first frame, and
the pipeline downloads on demand anyway. The checks that genuinely halt are the
cuDNN test and the execution-provider check.

### The virtual camera re-sends rather than stopping

`_run_vcam` holds and re-sends the last frame when its queue empties. A stalled
device can be reported by a call application as a *disconnected camera*, which is
louder and stranger to the other participants than a frozen picture.

---

## Landmines

Things that will genuinely bite, all found the hard way.

| Landmine | Detail |
|---|---|
| **`runtime` image tag does not exist** | Only `devel` is published for `runpod/pytorch`. Four files referenced a dead tag; all now pinned to `2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` |
| **`Dockerfile` and `RUNPOD_IMAGE` must stay in step** | Drifting them means production runs a different Python, torch and CUDA than anything ever tested over SSH |
| **Batch scratch needs real disk** | Extracted PNGs are ~4 MB per 1080p frame — ~36 GB for five minutes at 30fps. Scratch defaults to `/workspace/tmp` (the network volume) rather than the 20 GB container disk; raise `RUNPOD_VOLUME_DISK` before long jobs |
| **Volumes bill while stopped** | $0.07/GB/month per region, running or not. Egress is free; idle per-region volumes are the cost line to watch |
| **Guard thresholds still uncalibrated** | Eight of nine are guesses. `guard_min_coverage` is the one most likely to fire on ordinary frames — what XSeg reads on a clear face has never been measured. Phase 3.1 sets them from data |
| **Tests stub the ML layer** | `tests/` proves the logic around the models, never the models. Only the pod and the CI `test` job cover those |
| **Both deploy paths must stay in step** | SSH runs `startup.sh`; Docker runs `entrypoint.sh`. Both must pre-warm, both must source `/etc/rp_environment`, both must fail on cuDNN. `tests/test_wiring.py` asserts this — do not let one path gain a step the other lacks |
| **The unused deploy mode rots** | Docker mode shipped unbuildable and would have run every model on CPU. CI now builds the image on every push; keep that job |

---

## Command reference

```bash
# Pod lifecycle
python runpod/orchestrator.py start        # new pod; prints the cold-start breakdown
python runpod/orchestrator.py resume       # existing pod (RUNPOD_POD_ID)
python runpod/orchestrator.py stop         # pause; volume and container disk survive
python runpod/orchestrator.py terminate    # delete pod; network volume survives
python runpod/orchestrator.py status
python runpod/orchestrator.py gpus         # VRAM, price, eligibility

# Pipeline — batch
python pipeline.py -s <src>.jpg -t <target>.mp4 -o <out>.mp4 --execution-provider cuda

# Pipeline — stream, fully instrumented
python pipeline.py --stream --execution-provider cuda \
    --quality optimal \
    --guard-observe --guard-report guards.json \
    --debug-frames clip/ --debug-frames-stride 3 \
    --log-level debug

# Guards off entirely
python pipeline.py --stream --no-guards

# Analysis, local and free
python tools/compare_frames.py clip/ [--against clip2/] [--json report.json]

# Local checks
python -m pytest tests/ -q
flake8 pipeline.py pipeline desktop tests tools
mypy pipeline
```

**Remote paths:** repo `/workspace/Phantom`, python
`/workspace/venv/bin/python`, log `/workspace/phantom-pipeline.log`, models
`/workspace/models`, batch scratch `/tmp/phantom/<session>/`.
