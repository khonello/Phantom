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

**Photo mode has since been added**, deliberately *beside* that work rather than
on top of it: it loops the existing image path instead of adding a stage, so it
does not depend on anything the pod session is meant to settle. It earns its
place in the session for a different reason — it is the only target shape that
can reach a remote worker at all, because `set_target` resolves paths on the
pipeline's own filesystem. See 2.2b.

---

## Phase 0 — Before spending anything

All local, all free. Twenty minutes.

### 0.1 Confirm the tree is green

```bash
python -m pytest tests/ -q                                       # expect: 10 passed, ~35s
flake8 pipeline.py pipeline desktop tests tools runpod firebase  # expect: silent
mypy pipeline desktop                                            # expect: clean
bash -n runpod/startup.sh && bash -n runpod/entrypoint.sh
```

Ten test modules, ~310 checks. `tests/test_wiring.py` is the one to watch: it
asserts the seams *between* files — that every forwarded env var is read, that
`Dockerfile` and `.env.example` pin the same image, that both deploy paths
pre-warm. Every historical break in this repo lived in one of those gaps.

`mypy pipeline desktop` is now clean. The 11 errors in `pipeline/` were fixed
first, which turned the CI `lint` job green; `desktop/` carried 53 more that CI
never checked, and those are fixed too. All were annotation strictness rather
than defects — bare `np.ndarray`, `deque` and `Queue` needing parameters, and
sounddevice stream handles typed `object` (a type with *no* attributes) where
`Any` was meant, which had accumulated stale `type: ignore` comments that no
longer matched the errors mypy emitted. Worth clearing rather than tolerating:
53 cosmetic errors is where a real one hides.

### 0.2 Push and watch CI

The CI file gained two jobs (`unit`, `docker`), had two defects fixed, and its
lint step now covers `tests`, `tools` and **`runpod`** — which it did not before,
which is exactly how an `F821` sat undetected in `orchestrator.py`. It also runs
`bash -n` on both shell scripts, since a syntax error there is a failed
provision discovered after paying for a GPU.

**First run: `unit` and `test` passed; `lint` and `docker` failed and are
fixed.** Details below — both failures were real and both were worth catching
here rather than on a pod.

This is the cheapest possible verification, so keep doing it before the pod.

Expect four jobs: `lint`, `unit`, `test`, `docker`.

| Job | If it fails |
|---|---|
| `lint` | flake8/mypy regression, or a shell syntax error — fix before continuing |
| `unit` | the test suite; reproduce locally with `pytest tests/ -q` |
| `test` | end-to-end CPU swap. **First real check of batch video with models in the loop** |
| `docker` | image build. Catches a dead base tag, an unresolvable package, or the cuDNN check failing |

The `test` job matters more than its name suggests: it is the first time the
batch-video path runs with real InsightFace and ONNX rather than a stub.

**It passed.** Batch video is therefore verified on CPU with real models —
extraction, per-frame swap, encode, audio restore, and a PSNR comparison against
the reference render. Only the GPU path remains open for it.

What the first run caught, both fixed:

- **`lint`** — `mypy pipeline` exits non-zero on any error, and the 11
  pre-existing errors were enough. That job had been failing on `main` all
  along. All 11 are now fixed at their sites rather than blanket-ignored, and
  `mypy pipeline` reports clean.
- **`docker`** — the cuDNN step used `nvidia.cudnn.__file__`, which is **None**
  for a namespace package, so `os.path.dirname` raised TypeError and the build
  failed. `startup.sh` had the identical bug, hidden behind `2>/dev/null`, where
  it would have produced an unset library path and a silent CPU fallback — and
  with the new fatal cuDNN check, a failed provision. Both now use
  `runpod/cudnn_path.py`, which handles namespace and regular packages and
  reports rather than swallowing.

### 0.3 Model weights

`pipeline/models/` is gitignored, so weights travel by disk image rather than by
clone. Present locally:

| Weight | Size | Why it is kept locally |
|---|---|---|
| `inswapper_128.onnx` | 529 MB | The incumbent swapper |
| `hyperswap_1a_256.onnx` | 384 MB | The candidate Phase 2 compares against it |

Keeping hyperswap on disk is **archival, not a speed optimisation**. The pod
downloads it straight from GitHub, and uploading a local copy from here would be
slower than letting it do that. The reason to hold it is supply chain: the
release tag is load-bearing and demonstrably mutable — `models-3.0.0` and
`models-3.4.0` both 404 for these exact files, and only `models-3.3.0` serves
them. If facefusion retags or removes that release, a pipeline that depends on
those weights has no way to obtain them.

`1b` and `1c` are not held. They are one `curl` away and nothing depends on them
until there is a reason to compare all three:

```bash
curl -L -o pipeline/models/hyperswap_1b_256.onnx   https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx
```

### 0.4 Have a source face ready

One clear, frontal, well-lit photo of one person, face at least 110px on the
shorter side. The source guards will reject anything else, which is the point —
but discovering that on the pod wastes billed time.

If testing the guards deliberately, prepare a second set: a photo with two
people, a blurry one, a profile shot, and three photos of one person plus one of
someone else.

---

## Phase 1 — Set up the cold-start experiment

Cold start needs timing under two conditions, and they cost very differently:

| Condition | What it measures | Cost |
|---|---|---|
| **Warm volume** | What a returning customer waits for. The common case | Free — the existing volume already has venv, models and repo |
| **Empty volume** | First-ever deploy into a region. Also what a *fallback* region costs | A second volume that bills continuously — **deferred, see 1.4** |

Current config is a single datacenter: `RUNPOD_DATACENTERS=EU-RO-1:z8now7p5ts`.
Its volume is warm, so there is no second region whose volume is already empty to
measure for free.

**Do the warm measurement now; defer the empty one.** A second volume bills from
the moment it exists, and paying for regional redundancy before the pipeline is
proven on a GPU is buying resilience you cannot use yet. The warm number is the
one that describes what a returning customer waits for.

### 1.1 Check what already exists

```bash
python runpod/orchestrator.py status        # is there a pod, and is it running?
python runpod/orchestrator.py datacenters   # every datacenter RunPod offers
python runpod/orchestrator.py gpus          # GPUs matching MIN_VRAM / MAX_PRICE
```

`status` reads `RUNPOD_POD_ID` from `.env`. Three possible answers:

- **RUNNING** — a pod is live and billing. `stop` it before measuring, or the
  cold-start number is meaningless
- **EXITED / STOPPED** — the pod exists with its container disk intact. `resume`
  is the fast path; it is *not* a cold start
- **not found** — it was terminated. `start` is the only option, and that is a
  genuine cold start

> `start` **always creates a new pod**. `resume` restarts the existing one. They
> measure different things: `resume` skips provisioning and keeps the container
> disk, so `apt-get` does not re-run.

### 1.2 Resize the volume — do this before provisioning anything

`RUNPOD_VOLUME_DISK=20` is too small for this session. Budget:

| Item | Size |
|---|---|
| venv (torch, tensorflow, onnxruntime) | ~6–8 GB |
| Models (inswapper, CodeFormer, XSeg, GFPGAN, buffalo_l) | ~1.6 GB |
| hyperswap_1a_256, if measured | 0.4 GB |
| Debug frames at stride 3, limit 1500 | ~2 GB |
| Batch scratch, if a video is processed | 4 MB per 1080p frame |

That is **~12 GB before any batch job**, and batch scratch now lands on
`/workspace/tmp`, which is this volume. A five-minute 1080p clip alone wants
~36 GB of scratch.

**Do this:**

1. RunPod dashboard → **Storage** → volume `z8now7p5ts` → **Edit** → raise to
   **30 GB**. Volumes can be grown in place and cannot be shrunk, so 30 is a
   deliberate middle: comfortable for this session, not paying for batch
   headroom you may never use.
2. Update `.env` so a future `start` requests the same:

   ```env
   RUNPOD_VOLUME_DISK=30
   ```

Cost: $0.07/GB/month, billed **whether or not a pod is running**. 30 GB is
about $2.10/month. Going to 100 GB "just in case" would be $7/month standing for
capacity that sits idle — worth avoiding until a real batch job needs it.

> If you later run long 1080p batch jobs, raise it again or point
> `PHANTOM_TEMP_DIR` at the container disk for that job. Scratch is transient, so
> it does not need to live on the persistent volume — it lives there only because
> the container disk is the smaller of the two.

### 1.3 Warm-volume measurement — the one to actually run

Nothing to set up. It is the common case and it costs one pod session.

```bash
python runpod/orchestrator.py status
python runpod/orchestrator.py stop      # only if it reports RUNNING
python runpod/orchestrator.py start     # always a fresh pod
```

The phase table prints at the end and says `volume: warm`. Save the output.

Note this still runs `apt-get` and re-clones nothing — the container disk is new
even though the volume is warm, which is exactly the case a returning customer
hits.

### 1.4 Empty-volume measurement — deferred on purpose

The empty-volume number tells you what a *first-ever* deploy into a region
costs, which matters for two things: the Docker decision, and knowing what a
fallback region costs the first time it is used.

Getting it needs a second network volume in a second datacenter — **and that
volume bills continuously from the moment it exists, whether or not anything
uses it.**

**Do not do this yet.** Spending on regional redundancy before the pipeline is
proven is paying for resilience you cannot yet use. The warm number is the one
that describes what a returning customer actually waits for, and it is free.

Revisit when **both** are true:

- The pipeline is confirmed working end to end on a GPU — hyperswap produces a
  face, guards behave, latency holds
- You are either taking real sessions, or `provision` has actually failed to
  find a GPU in `EU-RO-1` and the fallback is no longer theoretical

When that day comes, the procedure is:

1. `python runpod/orchestrator.py datacenters`, pick one that is **not**
   `EU-RO-1`
2. RunPod dashboard → **Storage** → **New Network Volume** in that datacenter,
   30 GB. Copy the volume ID
3. Put the new pair **first**, because the orchestrator tries datacenters in the
   order listed and only falls through when no GPU is free. Listing it second
   would land you back on the warm volume and silently measure the wrong thing:

   ```env
   RUNPOD_DATACENTERS=<NEW_DC>:<NEW_VOL_ID>,EU-RO-1:z8now7p5ts
   ```

4. `python runpod/orchestrator.py start` → the phase table reports
   `volume: empty`
5. Swap the order back so normal sessions use the warm volume. Keep both
   entries — that is the fallback working as intended, and the second volume is
   warm now too

**In the meantime**, record the empty case as unmeasured rather than estimating
it. An invented number in the cold-start budget is worse than an admitted gap.

**The one alternative that costs nothing** is wiping the existing volume, and it
is worse than waiting: everything re-downloads, and a failed re-provision leaves
nothing to fall back to.

```bash
# Only if you have decided the number is worth the risk. On the pod, over SSH.
rm -rf /workspace/venv /workspace/models /workspace/Phantom
```

### 1.5 What you should have at the end

- A phase table labelled `volume: warm`, saved somewhere — it is printed, not
  written to a file
- The empty-volume row explicitly marked *not measured*, with the reason

That warm number plus the phase breakdown is what settles the Docker question in
Phase 3.4. It is a weaker input than having both, and that is the right trade
for now: if `pip-install` dominates even a warm start, the answer is already
clear without paying for a second volume to confirm it.

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

### 2.2b Verify photo mode, and that a refusal writes nothing

Cheap — seconds of GPU, not minutes — and it checks the one thing no other step
does: that a target chosen on the desktop actually arrives at the worker.

Run it **from the desktop, not over SSH**. Over SSH the files are already on the
pod and the upload path is never exercised, which is the entire point of the
check.

- [ ] IMAGE tab, pick two to four photos, PROCESS
- [ ] Each tile resolves on its own as its photo finishes, rather than all at
      the end
- [ ] The swapped photos land beside the originals with `_swapped` in the name
- [ ] Include one photo that **should** be refused — two faces in frame is the
      easiest — and confirm it is reported with a reason **and leaves no output
      file**. A file appearing there is the regression this mode exists to
      prevent
- [ ] Include one large camera original (over 6 MB) and confirm it still
      uploads, having been re-encoded rather than rejected

Also worth one look: a photo swap has no frame deadline, so it composites at the
full `aligned_size` ceiling. If stills look noticeably better than the live path
at the same settings, that is the compositing ceiling talking, and it sizes what
a higher preset would buy.

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

## Phase 2b — The inference speed levers

Runs inside the same paid session as Phase 2, and reads the same latency report.
The code landed in `c3dd55f`; **none of it has run on a GPU**, so every figure in
[docs/COMPILATION.md](COMPILATION.md) is a prediction until this phase replaces
it with a number.

The hardware question is already closed. `orchestrator.py gpus` puts **RTX 4090
first at $0.34/hr**, comfortably inside the $1.00 cap, so auto-discovery is
already selecting the fastest eligible card and there is no cheaper win to take
before optimising. See 2b.5 for the one hardware move that remains.

All levers default off, so a baseline run needs no flags.

### 2b.0 Where this got to (2026-08-30, RTX 4090)

**The speed question is answered. The remaining questions are about looks.**
See CLAUDE.md for the tables; the short version:

- Baseline **58.9ms/frame** against a 50ms deadline. Restoration is 39.5ms of
  it, ~68%.
- **Restoration off is 17.7ms and HOLDS** — the first `[HOLDS]` this project
  has produced. `restore_min_face=200` reaches the same floor (18.0ms), which
  is the shippable version of the same saving.
- `aligned_size` does nothing. **hyperswap is slightly worse**, not better.
- CodeFormer is **fixed at 512x512** in the graph, so `restore_size` cannot
  shrink it — that needs a different model, not a config change.
- **fp16 has never actually run.** The conversion fails its own load check on a
  Cast node. Do not read any past `fp16` row as evidence; it was a silent
  fallback to fp32 both times.
- On footage, 29.4ms of CodeFormer moves the detail metric **+0.03** and leaves
  noise and seam unchanged.

**The shortest path to a real answer, in full.** Everything below assumes a
resumed pod; take `<ip>` and `<port>` from what `push` prints, as they change on
every stop/resume.

```bash
# 1. The pod runs whatever it last pulled. This is not optional — a sweep has
#    already been lost to a pod that predated the fields it was being asked to
#    set, and every configuration reported identical numbers.
python runpod/orchestrator.py run "cd /workspace/Phantom && git pull"

# 2. Restart the pipeline so it loads the pulled code.
python runpod/orchestrator.py run "pkill -f pipeline.py; sleep 4; \
  [ -f /etc/rp_environment ] && . /etc/rp_environment; \
  [ -f /etc/profile.d/cudnn.sh ] && . /etc/profile.d/cudnn.sh; \
  cd /workspace/Phantom && nohup /workspace/venv/bin/python \
  /workspace/Phantom/pipeline.py --execution-provider cuda \
  > /workspace/phantom-pipeline.log 2>&1 &"

# 3. Wait for it to bind 9000 — model load is ~60-90s, and a stream started
#    before then reports nothing at all.
python runpod/orchestrator.py run "for i in \$(seq 1 20); do \
  ss -ltn | grep -q ':9000' && echo READY && break; sleep 10; done"

# 4. Get the clip and the source face on, if this is a fresh volume.
python runpod/orchestrator.py push clip_h264.mp4

# 5. Then A/B live, with no restart between configurations.
python tools/realism.py --host <ip> --port <port> enhance=false
python tools/realism.py --host <ip> --port <port> enhance=true
python tools/realism.py --host <ip> --port <port> swapper_model=inswapper_128
```

**`.env` does not reach a running pod.** It is read at `create_pod` time and
baked into the container environment, and it is gitignored so `git pull` never
carries it. On a pod that already exists, `tools/realism.py` is how a model
changes; only `terminate` + `start` picks up an edited `.env`. See
RUNPOD_DEPLOYMENT.md, "Changing settings on a pod that already exists".

**Watch the latency badge, not just the frame time.** The desktop now shows RTT
p50/p95, buffer depth and uplink Mbps top-right in the viewport. Read it against
the pipeline's own per-stage report: the pipeline measures only its own work, so
the gap between the two is network and encode. If RTT dominates, no further
compute work will be felt, and the next lever is the datacenter rather than the
GPU.

**Do these next, in order:**

1. **Look at the frames.** 24 pairs each are on the volume at
   `/workspace/dbg/on` and `/workspace/dbg/off`, plus `/workspace/montage.jpg`.
   The statistics say restoration is near-invisible here; a person has not
   checked, and temporal shimmer is not in those numbers at all. **This needs
   an `orchestrator.py pull` first — nothing can currently be copied off the
   pod.** That is the smallest unblocking task in this list.

2. **Wire `gpen_bfr_256` as a third enhancer backend.** 5.4ms against
   CodeFormer's 29.4ms in isolation, already on the volume, already ONNX. The
   `Enhancer.crop_size` machinery exists and would finally do something:
   `_spatial_size` reads 256 from the graph and the compositor builds a 256
   FFHQ crop with no change of its own. `_CodeFormerBackend` already handles a
   model with no `weight` input, so the work is a model registry rather than a
   new backend — mirror `swapper_models.py`, which already pairs a spec with a
   look profile.

3. **Then judge all three on footage**: off, gpen_bfr_256, codeformer_512.
   Restoration is what decides whether output reads as a call or as AI, and
   only one of those three has been looked at.

4. **Re-measure noise on better-matched footage.** The 1.50x face/frame noise
   reading appears with restoration on *and* off, and the clip pairs a fair,
   well-lit source against a dark, under-lit, noisy target — the hardest case
   for colour matching and a sufficient explanation on its own. Decide whether
   grain matching is overshooting only after a fairer pairing.

5. **Only then** the fp16 block list, the XSeg overlap and GPU compositing.
   Compositing is ~10.3ms of the frame now, not 20ms, and it scaled with the
   card — the earlier claim that it was a fixed CPU floor was wrong.

**Not worth re-running:** `restore_256`, `restore_128` (model is static 512),
`cuda_graphs`, `cuda_streams`, `async_encode` (all measured flat), and `trt`
(registering it dropped the models to CPU and correctly halted the stream).
`tools/sweep_levers.py` records each exclusion with its reason.

**Known-good invocation** (host and port change on every stop/resume — take
them from what `push` prints; on Git Bash set `MSYS_NO_PATHCONV=1` or every
`/workspace/...` argument is rewritten into a Windows path):

```
python runpod/orchestrator.py push clip_h264.mp4
python tools/sweep_levers.py --host <ip> --port <port> --input-url <path> --source /workspace/Phantom/.github/examples/source.jpg --seconds 60 --out sweep.json
```

### 2b.1 Baseline — no flags

```bash
python pipeline.py --stream
# ... run a normal session, then stop it
```

The report prints per stage on stop. Record `detect`, `restore`, `smooth`,
`colour`, `detail`, `mask`, `paste`, `swap+composite` and `total`, with the
HOLDS/MISSES verdict, at each preset you care about.

**Read `restore` against `swap+composite` first.** Everything downstream assumes
restoration is the dominant term:

| Reading | What to do |
|---|---|
| `restore` dominates | As predicted. Continue to 2b.2 |
| `detect` dominates | Stop. The answer is `det_size` and the preset, not any lever here |
| `total` >> the sum of stages | The cost is outside the compositor — capture, JPEG encode, or the proxy hop. None of these levers touch it |

The third case is the one worth taking seriously: the desktop pushes webcam
frames to the pod and reads them back through RunPod's proxy, and that round
trip is tens of milliseconds no GPU affects.

### 2b.2 CUDA graphs — free, do it second

```bash
CUDA_GRAPHS=true python pipeline.py --stream
```

Changes no numerics at all, so this is a pure latency read: same output, fewer
kernel launches. Applies to CodeFormer, XSeg and the swapper; the detector is
excluded because `det_size` moves with the preset and a captured graph records
fixed buffer addresses.

If it moves nothing, drop the flag rather than keep a lever nobody reads.

### 2b.3 fp16 — the largest win, and the only risky one

Convert first — it writes a copy, leaving the fp32 weights untouched:

```bash
python tools/convert_fp16.py /workspace/models/codeformer.onnx
python tools/convert_fp16.py /workspace/models/inswapper_128.onnx
```

Then A/B on footage, not on the latency number:

```bash
python pipeline.py --stream --debug-frames /workspace/fp32/
FP16=true python pipeline.py --stream --debug-frames /workspace/fp16/
python tools/compare_frames.py /workspace/fp16/ --against /workspace/fp32/
```

This is the design target itself. `enhancer_weight` and `enhance_strength` were
tuned to `0.7` because full restoration reads as AI, and "2x faster and slightly
different" is not obviously a win in a product whose stated failure mode is *too
clean*. Reverting is dropping the flag; the fp32 weights never moved.

### 2b.4 TensorRT — only if 2b.2 and 2b.3 leave the deadline missed

```bash
TRT=true FP16=true python pipeline.py --stream
```

The first run on a given GPU **builds engines, which takes minutes of a paid
hour**. They cache to `/workspace/trt-cache`, keyed by GPU, TensorRT and ORT
versions, weights and precision, so each architecture pays that once ever.
Budget roughly 0.5–1 GB of volume per architecture.

Watch the startup log. A TensorRT fallback warns rather than halting — a model
that fell back to CUDA is still on the GPU and still holds a call — but it means
the minutes bought nothing, and TensorRT's failure mode is silence.

`TRT_GPUS` bounds which cards are worth the build. The default covers the top
eligible cards; `L40` (ranked 80, $0.69/hr) is the one arguable omission.

### 2b.5 Record the numbers

Replace the predictions in `docs/COMPILATION.md` with what you measured, and
close the questions that the data settles. A lever that moved nothing should be
recorded as such rather than left looking untried.

**The one hardware move left:** `_MAX_SUPPORTED_COMPUTE_CAP = (9, 0)` excludes
Blackwell, which costs a real field — RTX 5090 at $0.69/hr and RTX PRO 4500 at
$0.34/hr are blocked by the image's PyTorch/ONNX support, not by price or
availability. A 5090 is meaningfully faster than a 4090 at batch 1. Unlocking it
means a base image on CUDA 12.8+, a PyTorch with sm_120, and a matching
`onnxruntime-gpu`. Separate work, tracked in 4.1.

---

## Phase 4 — Implementation, in order

Only after Phase 3. Each item's priority depends on what the data says.

### 4.1 Immediate follow-ups

- [ ] **Act on the guard calibration** — set the nine thresholds from the report
- [ ] **Clear the 11 mypy errors** (Stage 0). Trivial, and the only remaining
      Stage 0 item
- [ ] **Unlock Blackwell (sm_120).** Raise `_MAX_SUPPORTED_COMPUTE_CAP` once
      the base image carries CUDA 12.8+, a PyTorch built for sm_120 and a
      matching `onnxruntime-gpu`. It is the only upward hardware move left: RTX
      5090 ($0.69/hr) and RTX PRO 4500 ($0.34/hr, 32GB) are excluded by the
      image, not by price or availability
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

### 4.4 Access codes — built, not yet deployed

**One code buys one hour.** Sell it however you like, the customer types it into
the desktop once, and the machine stays authenticated until the hour is out.
Full description in [ACCESS_CODES.md](ACCESS_CODES.md).

This is deliberately *not* the Stage 4 control plane. It is the smallest thing
that turns the pipeline into something sellable, and it was built beside the
existing work rather than on top of it: nothing in `pipeline/` depends on it,
and leaving `PHANTOM_AUTH_URL` unset removes the feature entirely. That is what
makes it safe to have landed before Phase 3 is finished.

| Piece | Where | State |
|---|---|---|
| Endpoints (`/session`, `/redeem`) | `firebase/functions/main.py` | Written, **not deployed** |
| Deny-all rules | `firebase/firestore.rules` | Written, **not deployed** |
| Code format + checksum | `desktop/codes.py` | Done, tested |
| Machine id, held-code logic | `desktop/auth.py` | Done |
| Gate overlay, countdown | `desktop/main.qml` | Done |
| Bridge wiring, deferred commit | `desktop/bridge.py` | Done |
| Minting tool | `tools/mint_codes.py` | Done (`--dry-run` needs no Firebase) |

**Two rules carry the design, and both are easy to undo by accident:**

- **A code is spent when the pipeline is running, not when it is typed.**
  `Redemption.begin` holds it; `_set_pipeline_running(True)` commits it. A pod
  that fails to provision costs the customer nothing. Do not "simplify" this
  into a single call at the auth screen.
- **The hour belongs to the machine, not to the app session.** `/session` is
  called on every launch, so closing and reopening the desktop inside the hour
  never prompts. The answer lives in Firestore, not in a local file — a local
  file would mean a reinstall costs an hour someone paid for.

**Outstanding, in order:**

- [ ] **Create the Firebase project and deploy.** Needs the Blaze plan (card on
      file; $0 at this volume, but the upgrade is a real step that will stop
      you). `firebase deploy --only firestore:rules,functions`, then put the
      base URL in `PHANTOM_AUTH_URL`
- [ ] **Mint a first batch and redeem one end to end** against a real pod. This
      is the only part never exercised — everything below the network call is
      tested, the network call itself is not
- [ ] **Decide what a mid-session pipeline death costs the customer.** Right now
      the hour keeps running. Deferring the burn until the pipeline is up
      removes the most likely failure, not all of them
- [ ] **Fold into the pod session.** The desktop knows the hour is over; the pod
      is stopped separately by `RUNPOD_MAX_UPTIME`. Two clocks that currently
      agree because both are 60 minutes — see 4.5

### 4.5 Session shutdown — done

The gaps the access codes exposed, all now closed. Verified by driving the real
`Bridge` through every ending: user Stop, `auto_stop`, socket drop, clock
expiry, and app close.

- [x] **The desktop handles `auto_stop`.** `server.py::_stop_pod` broadcasts it
      and sleeps a second so clients receive it; nothing listened, so the end of
      a paid hour presented itself as *"disconnected — reconnecting…"* forever.
      It now ends the session with a reason
- [x] **The reconnect loop distinguishes expected from unexpected.**
      `expect_disconnect()` stops it when the pod was stopped on purpose. It is
      otherwise indefinite, which is correct — a pod can be slow and a laptop
      can sleep. `max_retries = 3` only ever capped the backoff exponent, and
      the docstring that claimed a retry limit was simply wrong
- [x] **The virtual camera is on for the life of the app.** Not a toggle, not
      tied to LIVE, not tied to a session. `cleanup` is the only path that
      releases it; `stopPipeline`, expiry and disconnect all leave it running,
      holding the last swapped frame

**The flow, as built:**

| When | Desktop | What the call sees |
|---|---|---|
| last 10 min | countdown chip in the header | swapped frames, unchanged |
| pod's 5-min warning | existing auto-stop dialog, Extend available | unchanged |
| time ends | **anchored card**: "Session ended", with ENTER A NEW CODE | **last swapped frame, held, indefinitely** |
| socket drops | nothing further — reconnecting has stopped | still the held frame |
| app closed | — | device released |

**Do not soften these:**

- **Never release the virtual camera to signal anything.** A conferencing app
  responds to a device disappearing by showing a placeholder, reporting a
  disconnected camera, or selecting the next available one — the operator's
  real webcam. `tests/test_wiring.py` asserts which functions may call
  `_stop_vcam` and which may not
- **Only the pipeline's swapped stream feeds the device.** Both feed calls sit
  in `_poll_frames`, downstream of `_jitter_buffer.pop_eligible()`. The raw
  webcam preview is handled in the same function and has no path there
- **Nothing is drawn on a frame, ever.** Every notice is desktop chrome

### 4.6 Stage 4 and beyond

Unchanged from [TODO.md](TODO.md), and all substantially larger than anything
above: the control plane and session state machine, then cold start, resilience,
provider abstraction, packing, payment.

Authentication is no longer on that list in its original form — 4.4 covers what
it was for. What remains under Stage 4 is what access codes deliberately do not
do: entitlements that outlive a machine, self-serve purchase, partial-hour
credit, and several customers on one pod.

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
