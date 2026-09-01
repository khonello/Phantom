# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Phantom is a modern, composable face-swapping application for videos and images. It uses deep learning models (ONNX-based face detection and swapping via InsightFace) to replace faces in media with high quality.

**Architecture**: Clean, event-driven, service-oriented design with unified ProcessingPipeline for both batch and realtime modes. No global state.

**Two entry points**:
- `pipeline.py` (headless engine): Supports batch mode (`-s <source> -t <target> -o <output>`) and realtime stream mode
- `desktop.py` (GUI controller): Qt/PySide6 interface, communicates with pipeline via WebSocket API on port 9000

### Design target
**What a real video call looks like** — sensor noise, compression, ordinary
imperfection. Not a high-resolution portrait, and not the poreless "beautified"
look. Three failure modes drive most decisions in this codebase:

1. **Too clean** — perfectly sharp, poreless skin on a 720p webcam feed
2. **A visible seam** — any edge, halo or colour step where the swap meets the head
3. **Wrong motion** — shimmer, jitter, or a face lagging the head it sits on

Several stages deliberately degrade the output (grain, partial restoration,
detail matching downward) because matching the frame beats looking good in
isolation. When changing anything here, judge it on real footage, not stills.

### Current state
Development is focused on the **live call** path; batch follows and reuses the
same compositor.

**Next actions live in [docs/PENDING_WORK.md](docs/PENDING_WORK.md)** — a runbook
from starting the pod through to the outstanding implementation work.
[docs/TODO.md](docs/TODO.md) remains the backlog, and
**[docs/ACCEPTED_RISKS.md](docs/ACCEPTED_RISKS.md)** records what is knowingly
wrong and why — including the unauthenticated WebSocket, which must close
before a paying customer. Read it before assuming a gap is unnoticed.

- Working: realtime stream, aligned-space compositing, RunPod deployment,
  desktop LIVE mode, batch **image** and **video**, **photo mode** (up to four
  uploaded targets), **template targets** (bundled scenes), desktop VIDEO and
  IMAGE tabs
- No templates are bundled yet — the machinery runs against an empty library.
  The assets are a content decision, including licensing for this use
- Not exposed: realism knobs have no desktop UI (API/CLI only)
- Batch video is wired but has only been exercised against a stubbed swapper —
  the FFmpeg plumbing, frame ordering, audio sync, cancellation and cleanup are
  verified; a real run with the models in the loop has not been done locally
- Large-file transfer is still the gap for **video** targets. Photos sidestep it
  (`upload_target`, ≤4, ≤6 MB each, base64 in one message per job), and are so
  far the only target that reaches a remote pod at all — `set_target` validates
  against the *pipeline's* filesystem, so a desktop-chosen file only resolves
  when the pipeline runs locally. A 2 GB video still needs a real transfer path

### Where performance work stands (measured 2026-08-29, RTX 4090)

**The `optimal` preset misses its deadline by 8.3ms.** Not by 96ms, which is
what the L4 said. The card was worth more than every software lever combined.

| Stage | p95 | Share |
|---|---|---|
| detect | 8.5ms | 15% |
| **restore (CodeFormer)** | **39.5ms** | **68%** |
| compositor + paste + encode | ~10.3ms | 18% |
| **total** | **58.3ms** | vs a **50ms** deadline |

RTX 4090, 640x360 @20fps, 1026 frames. Unusually well-supported for a single
run: eleven sweep configurations were recorded before anyone noticed the pod
was refusing every lever, so this is eleven independent samples of the same
stock configuration, spanning 58.1–58.8ms.

For comparison, the same clip on an **L4** (2026-08-28): detect 16.1ms, restore
110ms, CPU ~20ms, total 146ms. So the 4090 is ~2.5x, and — worth correcting —
the non-GPU portion **did** scale with the card, 20ms to 10.3ms. The earlier
claim that ~20ms was a fixed CPU floor was wrong, which weakens the case for
the GPU-compositing work: it is now 18% of the frame and cannot close an 8.3ms
gap by itself.

Restoration is still the dominant term, so the premise holds. And the
arithmetic is now friendly rather than hopeless: **restoration is 39.5ms of a
58.3ms frame against a 50ms deadline**, so removing even part of it holds the
preset.

### What the levers are actually worth (same session, same clip)

Read the **frames processed per run**, not the p95 column. `LatencyBudget` was
never reset between streams, so each report covered every frame since the
process started (1018, 4399, 7740, 8781, 9746, 12775) and every p95 was
diluted by its predecessors — a run with restoration *off* still reported a
`restore` percentile, because those were the baseline's frames. Fixed now
(`LatencyBudget.reset()`, called from `_run_stream_impl`), but every sweep
taken before that fix has to be read this way.

| config | frames in its 60s | ms/frame |
|---|---|---|
| baseline | 1018 | 58.9 |
| **no restoration** | 3381 | **17.7** |
| **`restore_min_face=200`** | 3341 | **18.0** |
| `aligned_size=128` | 1041 | 57.6 |
| hyperswap_1a_256 | 965 | 62.2 |
| hyperswap + no restoration | 3029 | 19.8 |

**Restoration off is 3.3x, and it HOLDS the deadline** — the first `[HOLDS]`
verdict this project has produced. `restore_min_face` lands on the same number,
which is the cross-check it was built for: the shippable, config-level lever
reaches the same floor as switching the stage off wholesale.

Two questions closed, both negative. **`aligned_size` is not the cost** — 128
against 256 changes nothing. **hyperswap is slightly worse, not better**: 62.2
against 58.9, and 19.8 against 17.7 with restoration off. The 256px swap costs
2-3ms and buys back *nothing* in restoration time, so on speed grounds it is
just a bigger swap. Its appearance remains unjudged.

### Does restoration actually help a 101px face?

Measured on the recorded webcam clip, 24 frame pairs at identical indices,
restoration off against CodeFormer at 512:

| `tools/compare_frames.py` (ideal = 1.00) | off | CodeFormer 512 | change |
|---|---|---|---|
| high-frequency detail, face / frame | 0.584 | 0.614 | **+0.030** |
| sensor noise, face / frame | 1.500 | 1.500 | **+0.000** |
| gradient at mask edge | 1.028 | 1.038 | +0.010 |

**29.4ms buys +0.03 on the one metric it moves at all.** The face sits 0.42
short of matching the frame's detail; restoration closes 7% of that gap and
leaves noise and seam unchanged to three decimals. Both configurations produce
the same verdict lines — "softer than the frame", "no seam detected", "motion
blur consistent".

That is what the geometry predicts. 86% of what CodeFormer produces is
discarded at `compositor.py:515`, one warp after it is created, so the stage
cannot move the metric much and does not.

**What this does not cover, and should not be read as covering.** These are
per-frame image statistics over 24 frames. They say nothing about **temporal**
behaviour, and shimmer between frames is a large part of what reads as AI. They
are also not a person looking at a face. Strong evidence, not proof.

The noise row reads 1.50x in **both** configurations, so it is not caused by
restoration. It is also not necessarily a defect: this clip was recorded in
poor light, and the source face is fair-complexioned and well lit against a
dark-complexioned, under-lit, visibly noisy target. That is the hardest case
for colour matching and a plausible cause on its own. Re-measure on
better-matched footage before treating grain matching as overshooting.

### Restoration models, benchmarked in isolation (RTX 4090)

Raw inference only — 100 runs, random input, no compositing:

| model | crop | inference | file |
|---|---|---|---|
| codeformer | 512 | 29.4ms | 377 MB |
| gpen_bfr_512 | 512 | 37.5ms | 284 MB |
| **gpen_bfr_256** | **256** | **5.4ms** | **76 MB** |

Note which way `gpen_bfr_512` falls: **slower** than CodeFormer at the same
resolution. The saving is entirely **resolution**, not architecture. GPEN is
not a lighter model; 256 is simply a quarter of the pixels.

**`gpen_bfr_256` has never run in the pipeline.** It has not been composited,
not judged on footage, and the ~27ms frame estimate for it is arithmetic on the
numbers above rather than a measurement. Both GPEN files are on the volume at
`/workspace/models/`.

Why 256 is the interesting number rather than "off": restoring at 512 and
warping down to a 128-192 aligned space is *supersampling*, and some of that
cost buys antialiasing and stability rather than nothing. At 256 into a 192
aligned space the supersampling margin survives, along with all the
low-frequency work — tone, structure, artifact cleanup — that the downsample
does not destroy. What is given up is the 512-to-256 octave, which is the one
the final resize deletes anyway. It also has **no fidelity weight**: GPEN takes
one input, so `enhancer_weight` would stop meaning anything and only
`enhance_strength` would remain.

### The bottleneck has moved to the transport

With restoration at 256 the frame is ~27ms against a 50ms deadline, and the
reported symptom changed shape with it: **the stutter went away and the lag did
not**. That is the diagnosis. Stutter is throughput — frames arriving faster
than they can be processed. Lag is latency — and halving the compute did not
move it, so compute was not what was holding it.

The chain, end to end, with what each part costs:

    webcam capture
      -> JPEG encode (desktop)
      -> UPLINK          ~30 KB/frame, ~4.8 Mbps at 640x360 q70 @20fps
      -> inbound queue   was 10 deep = 500ms of pure latency
      -> process         ~27ms                    <- no longer the problem
      -> JPEG encode (pod)
      -> DOWNLINK
      -> jitter buffer   started at 400ms, adapted slowly
      -> decode -> display

**None of this was visible.** `RTTTracker` computed true glass-to-glass latency
from a capture timestamp that rides with every frame, and had done all along —
nothing displayed it, logged it to the UI, or reported it. "It feels sluggish"
could not become "RTT is 210ms, the buffer adds 60, the pipeline uses 27".

Fixed, in order of how much they were costing:

- **The readout exists.** `Bridge.latencyText` publishes RTT p50/p95, buffer
  depth and uplink Mbps every two seconds, shown top-right in the viewport
  beside the other badges — never drawn on the frame, for the usual reason.
  Read it against the pipeline's own per-stage report: **the difference between
  the two is network and encode**, and on a remote pod that is most of it.
- **The inbound queue dropped the wrong frame.** On a full queue the handler
  refused the *arriving* frame and kept the backlog, so under pressure the
  pipeline chewed through stale frames while discarding the only current one —
  the face lagged by the whole queue depth and stayed there. It now evicts the
  oldest. Depth went 10 -> 2: anything waiting there is a frame the operator
  has already moved past.
- **The playout buffer started at 400ms** and converged slowly with one
  symmetric alpha, so even a nearby pod felt heavily delayed for the first
  seconds — exactly when an impression forms. Now 120ms initial, a 50ms floor
  (one frame interval at 20fps rather than 80ms), and **asymmetric** smoothing:
  rise fast because a late buffer glitches visibly, fall slow because an early
  one underruns. One alpha has to be slow in a direction; 0.2 was slow in both.

**What is still only a hypothesis: the uplink.** The desktop sends a JPEG per
captured frame, ~4.8 Mbps at the `optimal` preset, and receives about the same
back. Home connections are usually asymmetric with far less upstream. A
saturated uplink queues frames in the OS send buffer, which reads as **latency
while throughput still looks healthy** — the exact reported symptom. The
readout now carries the number; the cheap test is to switch to `fast`
(480x270 q60, ~1.4 Mbps) and see whether latency falls by far more than the
~10ms of compute that saves. If it does, the answer is encoding, not the GPU.

**And the term nothing in this repo can fix: distance.** `RUNPOD_DATACENTERS`
is `EU-RO-1` — Romania — against an operator in West Africa. That is a physical
floor of roughly 80-120ms round trip at best and typically worse; someone had
already met it, since the RTT ceiling's comment reads "accommodates RunPod
RTT". Moving to a western-European datacenter is the only lever, and network
volumes are datacenter-local, so it costs a second volume and a re-seed of the
models. Worth measuring the readout first — if RTT is 200ms and the buffer adds
60, a datacenter move is the largest remaining item by a wide margin.

**The standing conclusion: stop optimising the pipeline for latency.** There is
~23ms of headroom under the deadline and the felt delay is dominated by terms
the GPU does not touch. Further compute work should be justified by the readout
showing compute as the largest term, which it currently is not.

### Playout is fixed, not adaptive

Both streams are presented **550ms after capture**, whatever the network did in
between (`DEFAULT_PLAYOUT_DELAY_NS`; 0 restores the adaptive behaviour).

Adaptive playout is right for video alone — it chases the network and the
viewer sees nothing. It becomes wrong the moment audio is played against the
same number, because every adjustment is a discontinuity: move the read point
forward and samples are skipped, back and silence is inserted. A measured
session had the target swinging 380 -> 500 -> 420 -> 490ms every two seconds,
which is what an operator hears as speech breaking up. **Jitter is far more
damaging than delay** — people adapt to a constant 550ms and never to one that
moves.

550 comes from measurement: RTT p50 ~350ms, p95 ~450ms, 700ms outliers. It
covers p95 with margin and is barely above what the adaptive buffer already
averaged, so it costs almost nothing and removes the variance entirely.

**Only video crosses the network.** Audio is captured into a local ring buffer
and is available in ~23ms; video does a round trip to the pod and takes ~350ms.
So audio spends most of the budget waiting, jitter is entirely video's, and D
is set by the video distribution alone. That changes if voice processing ever
moves server-side — there is a `set_voice_transformer` hook suggesting someone
considered it — at which point a fixed buffer matters more, not less.

`JitterBuffer.next_for_slot` holds the schedule, and the second rule is the one
that is easy to get wrong:

1. **The slot fires on time, always.** Stalling for a late frame slips the
   schedule, and audio is locked to the same clock, so a stall is either a gap
   in speech or a drift out of sync.
2. **A frame that missed its slot is discarded, not shown late.** Playing the
   straggler shifts everything one slot later and the pattern never recovers.
3. **An empty slot repeats the last shown frame.** 33-50ms of an already
   mostly-still face is invisible, which is exactly why video is the cheap
   place to absorb jitter. Always the last *swapped* frame — never the raw
   camera, never black, which is the same invariant `_run_vcam` holds.

Repeats are counted and shown in the badge as `N held`. One is invisible; a
sustained rate is a frozen face while audio continues, which reads as a broken
swap rather than a slow link — so **>20% of slots over ~10s steps the delay up
by 100ms, once, and says so**. That is the only place playout adapts, and it
adapts on evidence rather than per frame.

### Session gotchas worth not rediscovering

- **The first stream after a pipeline start pays model warm-up**, tens of
  seconds, inside its own window. A 40s capture produced zero frames for this
  reason and looked like a broken config. The sweep hides this with a discarded
  warm-up pass; anything else driving the stream needs its own.
- **Nothing can be copied off the pod.** `orchestrator.py push` is local->pod
  only, port 9000 is the only opening, and the SSH proxy carries no SFTP — so a
  45 KB montage of the comparison frames could not be brought home. An
  `orchestrator.py pull` over the same WebSocket path `push` already uses is
  what makes visual review routine instead of impossible.
- **`orchestrator.py run` used `PATH=... <cmd>`**, which binds only to the
  first word of a line, so the second half of any `&&` chain ran under
  `/usr/bin/python`. Now `export PATH=... && <cmd>`.

**Settled by this:**

- Every model is confirmed on `CUDAExecutionProvider`. No silent CPU fallback.
- **`cuda_graphs` and `cuda_streams` measured flat** (144.4 / 146.1ms — noise).
  A 110ms model is not waiting on kernel launch overhead. Both can be dropped.
- **Numba is closed.** The whole compositor is ~20ms; making it free still
  leaves 126ms. Argued against on reasoning before, now on a number.
- `fp16` and `trt` **never ran** — no converted weights existed, and `trt_gpus`
  correctly declined an engine build on an L4.

**hyperswap has never been run.** Every measurement used the default
`inswapper_128`; the weights are on the pod but unused. It matters for speed,
not just looks: hyperswap is 256px native against inswapper's 128, and its
profile asks for *less* restoration (`enhance_strength` 0.5 vs 0.7) because the
swap needs less. A bigger swap that buys a cheaper restore may be a net win, or
may just be a bigger swap — the sweep now covers both.

**The first run of the next session should be `no_restore`.** It bounds
everything: whatever remains with restoration off is what no amount of work on
restoration can remove. `tools/sweep_levers.py` now leads with it, plus
`aligned_128`, `hyperswap` and `hyperswap+no_restore`.

**Continue from [docs/PENDING_WORK.md](docs/PENDING_WORK.md) §2b.0**, in order:

1. **Convert fp16 on the pod** — the only untested lever aimed at the 110ms:
   `orchestrator.py run "python tools/convert_fp16.py /workspace/models/codeformer.onnx"`
   (needs `pip install onnx onnxconverter-common` there first). Then re-sweep.
2. **Judge it on footage**, not latency alone — restoration is what decides
   whether output reads as a call or as AI.
3. **Measure again on a 4090.** Estimated ~72ms total, so a real 2x but still
   short of 50ms alone; **4090 + fp16** is the combination that plausibly
   holds. Needs `terminate` then `start` — `resume` cannot move a pinned pod.
4. **Reconsider restoration decimation.** Declined earlier as too risky to what
   the operator sees; that was decided before knowing restoration is
   three-quarters of the frame.
5. Only then the XSeg overlap and pipelining — both are bounded by the ~20ms
   that is *not* restoration.

Estimates above are labelled as such. This session's lesson was that the
reasoning about *where* time goes held, and the predictions of *how much* each
lever would buy did not survive contact with a measurement.

**Frame rate, estimated from the measured L4 numbers.** GPU stages scale with
the card; the ~20ms of CPU compositing and encode does not, which is what sets
the floor:

| | restore | detect | CPU | total | fps |
|---|---|---|---|---|---|
| L4 (measured) | 110ms | 16ms | ~20ms | 146ms | **~7** |
| RTX 4090 (est.) | ~44ms | ~6ms | ~20ms | ~70ms | **~14** |
| RTX 4090 + fp16 (est.) | ~24ms | ~6ms | ~20ms | ~50ms | **~20** |

So a 4090 roughly doubles the frame rate and still misses 20fps on its own.
**4090 + fp16 is the first combination that plausibly holds the `optimal`
preset**, and it lands right on the deadline rather than comfortably inside it.

### Why 110ms: restoration ignores how big the face is

**The dominant cost is spent on interpolated data.** In the measured session the
face was **101x129 px** in a 640x360 frame. The chain it went through:

    face in frame          101 x 129   <- the only real information
    swap native            128 x 128   <- inswapper_128 output, the ceiling
    aligned space          256 x 256   <- follows face size, has a floor
    FFHQ restore crop      512 x 512   <- ALWAYS 512, regardless

`CROP_SIZE = 512` is hard-coded through `_ffhq_geometry` and `_build_ffhq_crop`
(`compositor.py:496`, `:524`), because CodeFormer is trained on FFHQ 512 crops.
So a 101px face is upsampled about 20x in pixel count, the heaviest model in
the pipeline runs on the result, and the output is squeezed back into a 101px
hole. Conv cost scales with pixel count, so this is roughly **4x the compute of
restoring at 256, and 16x of 128** — spent reconstructing detail that was never
in the source.

**The codebase already holds the correct principle and does not apply it here.**
`_aligned_size` (`compositor.py:373`) says the working resolution "follows how
many frame pixels the face actually covers ... and, more importantly, is not
upsampled to a detail level their webcam never captured." That reasoning is
right, and it governs a stage costing a few milliseconds while the 110ms stage
ignores it entirely.

**This is the largest single lever available, and larger than fp16, TensorRT and
a 4090 combined.** Options, cheapest first:

1. **Skip restoration below a face-size threshold.** Config-level, no new
   model. The question it rests on is a footage question, not a latency one:
   does 512-space restoration visibly improve a 101px face whose swap was
   generated at 128? Test with `--debug-frames` before assuming either answer.
2. **Restore every Nth frame**, letting `temporal_alpha`'s aligned-pixel EMA
   carry the gap. Previously declined as too risky to what the operator sees;
   that was decided before knowing restoration is 75% of the frame.
3. **A restoration model that accepts a smaller input**, or a re-export of
   CodeFormer at 256. Changes what the output looks like, so it is an A/B, not
   a swap.

**Option 3 is closed, and option 1 is the live one.** `codeformer.onnx`
declares:

    INPUT  input   [1, 3, 512, 512]   tensor(float)
    INPUT  weight  []                 tensor(double)
    OUTPUT output  [1, 3, 512, 512]

Static and square. So `restore_size` cannot make this model restore at 256 —
`Enhancer.crop_size` warns once and holds at 512, which is the declining path
working as designed rather than a bug. **Restoring smaller needs a re-export,
not a config change.** `restore_min_face` is therefore the only config-level
lever against the 39.5ms, and it needs no new model because skipping is free.

The general lesson is cheaper than the sweep that would have found it: **read a
model's declared input shape before sweeping a shape lever.** Five seconds of
`InferenceSession(...).get_inputs()` replaced a paid measurement run.

`restore_size` and `restore_min_face` are config fields, on `set_realism`, the
CLI, the env and the sweep. Two properties matter:

- **`restore_size` is a request, and the model answers it.**
  `_spatial_size` reads the ONNX input's declared shape at load;
  `Enhancer.crop_size` honours a fixed export over the config and warns **once**
  rather than throwing per frame on the live path. So option 3's real question —
  *is facefusion's `codeformer.onnx` exported with dynamic spatial dims?* — is
  answered by one line of the pod's startup log, and a `restore_256` run whose
  `restore` equals the baseline exactly is what "no, it is fixed" looks like in
  the sweep. It is not a lever that quietly does nothing.
- **The seam is a fraction of the crop, not a pixel count.** `_FFHQ_ERODE` and
  `_FFHQ_FEATHER` reproduce the old 5px erode and 6.0 sigma exactly at 512, so
  nothing changes at the default, and a 256 crop gets the same *seam* rather
  than twice as hard an edge. Otherwise a resolution A/B would also be a
  feathering A/B and neither would be readable.

`_ffhq_geometry` at 256 is exactly half the matrix it is at 512 — the framing is
identical, only the sampling rate changes — so this is a resolution comparison
and nothing else. Both default to current behaviour: 512, never skip.

Note what is *not* the problem, so it does not get optimised by mistake:
transfers are trivial (a 512x512x3 fp32 tensor is 3MB, ~0.1ms over PCIe 4.0,
even six round trips are under 2ms), and the Python layer does not appear in
the measurement at all.

### GPU compositing, revisited with numbers

Worth doing, but second-order. The whole compositor is ~20ms — 14% of the frame
on an L4, but ~28% on a 4090, since it is the one stage that does not scale with
the card. The route is **torch, not `cv2.cuda`**: torch is already a GPU
dependency, `affine_grid`/`grid_sample` cover the warps and elementwise ops
cover colour, detail and grain, whereas PyPI OpenCV has no CUDA build. Expect
~20ms to become single digits.

Do it *after* the restoration resolution question, not before: 20ms is worth
having, 110ms is worth having first.

### Refusing mediocre GPUs, and waiting instead

Auto-discovery sorted by `_GPU_PERF` and took the fastest *available* card,
which silently accepts a 34-ranked L4 when a 100-ranked 4090 is busy. That is
not a hypothetical: a whole measurement session was spent on an L4 by accident,
and the card turned out to matter more than every software lever combined.

`start` now restricts itself to a **top tier**, and when none of it is free it
**waits and retries the whole list every minute** rather than dropping down.

| Setting | Default | Effect |
|---|---|---|
| `RUNPOD_MIN_GPU_PERF` | `85` | Floor in `_GPU_PERF`. The tier is 4090, H200, H100, RTX 6000 Ada, L40S |
| `RUNPOD_GPU_WAIT` | `300` | Seconds to keep retrying before giving up |
| `RUNPOD_GPU_FALLBACK` | *unset* | What the timeout does: unset fails, `true` accepts a slower card |

Five names, and in practice three — the H-series usually breaches
`RUNPOD_MAX_PRICE` and never reaches the floor at all.

This deliberately reverses the reasoning in commit 45ba27c and in
docs/COMPILATION.md, which argued that pinning trades "sometimes a slower card"
for "sometimes no pod at all", and that no pod is the worse failure on a paid
session. That argument assumed pinning meant *failing*. **A bounded wait is not
a pin: billing starts when a pod runs, not while you are waiting for one**, so
the wait costs nothing and the thing it avoids costs a session.

Four properties worth keeping:

- **The whole list is retried each pass, not just what was untried.** A card
  taken a minute ago is the most likely one to have come free, so a pass that
  skipped it would skip the answer. `_try_deploy_pass` is split out of
  `_deploy_new_pod` for exactly this: the pass repeats, the setup around it
  does not.
- **Only capacity is waited out.** A bad volume id, a dead image or a rejected
  key fails identically every sixty seconds, and spending five minutes proving
  that is worse than saying so immediately. `_is_capacity_error` — already
  needed by the resume path, since it is the same API saying the same thing —
  decides, and a single non-capacity failure ends the wait.
- **What happens at the timeout differs by purpose**, which is why it is a
  setting and not a constant. A measurement session should fail rather than
  accept a slower card, since a comparison across two architectures is not a
  comparison. A customer session should fall back, since some service beats
  none. The default is *fail*, because that is the failure this was built for.
- **Manual mode is exempt.** `RUNPOD_GPU_TYPES` naming cards is already a
  statement about which are acceptable; a floor that silently removed one would
  make the setting mean something other than what it says.

A narrow tier is also what makes the TensorRT engine cache pay. Engines are
keyed per architecture and each pays its build once; across a dozen eligible
cards the cache rarely hits, across three or four it is warm almost always. The
same holds for any future per-GPU tuning — a small, known set is what makes
"optimise for this card" a sentence that means something.

`gpus` lists the whole eligible field and marks the tier with `T` rather than
hiding what `start` refuses, since "why is there no pod" has to be answerable
from that command. Lives in `_discover_gpus`, `_resolve_gpu_candidates`,
`_try_deploy_pass` and `_deploy_new_pod` in `runpod/orchestrator.py`;
`tests/test_gpu_tier.py` pins the tier membership against the prose above.

## Quick Commands

### Operator machine setup
**[docs/SETUP_CHECKLIST.md](docs/SETUP_CHECKLIST.md)** — what has to be
installed locally for a call to work, and where every output file lands. Two
third-party drivers, and neither can be created from Python: **OBS Studio** for
the virtual camera, and **VB-Audio Virtual Cable** (or **BlackHole** on macOS)
for the virtual microphone.

The audio one is the easy one to skip and the worst one to skip. The desktop
delays the operator's microphone to match video that arrives ~350-400ms late;
without a virtual output that delayed audio goes to their *speakers* while the
call still receives the real microphone undelayed — so the delay makes the
desync worse rather than better. The app now says so at startup rather than
appearing to work.

### Running
- **Pipeline engine**: `python pipeline.py`
- **Desktop GUI**: `python desktop.py`
- **CLI batch mode**: `python pipeline.py -s <source_image> -t <target_video> -o <output_path>`
- **With CUDA**: `python pipeline.py --execution-provider cuda`

### Development
- **Lint**: `flake8 pipeline.py pipeline desktop`
- **Type check**: `mypy pipeline desktop` — clean, keep it that way (CI runs `mypy pipeline` only)
- **Unit tests**: `python -m pytest tests/ -q` — ~32s, no GPU or model weights
  needed (the ML layer is stubbed in `tests/conftest.py`). Ten modules;
  `test_photo_batch.py` covers the photo path, including that a refused photo
  leaves no output file behind, and `test_templates.py` covers the bundled
  library and the face its manifest names
- **Validate templates**: `python tools/validate_templates.py` — runs the real
  guards over the library, non-zero if any scene would be refused
- **End-to-end**: `python pipeline.py -s=.github/examples/source.jpg -t=.github/examples/target.mp4 -o=/tmp/output.mp4`

### Measurement
- **Lever sweep**: `python tools/sweep_levers.py --host <ip> --port <port>
  --input-url <clip> --source <face> --seconds 60 --out sweep.json` — measures
  every speed lever against one clip in a single pod session. Take the host and
  port from what `orchestrator.py push` prints; they change on every
  stop/resume
- **What is it running?**: `python tools/stats.py --host <ip> --port <port>`
  — GPU, swap model, restoration model and crop, whether restoration is on,
  requested vs available execution providers, capture settings, active speed
  levers, uptime and minutes left before auto-stop. Reports **resolved**
  values, since both registries fall back on an unknown name and the gap
  between requested and loaded is usually the bug. Exits non-zero when a
  requested accelerator is not available — the silent CPU fallback. `--json`
  for the raw reply
- **Change settings live**: `python tools/realism.py --host <ip> --port <port>
  key=value ...` — the only way to reach `set_realism` without writing a
  WebSocket client. Covers model selection, realism knobs, guard thresholds and
  speed levers; prints what was applied and what was refused. `--show` reads
  the pipeline's status instead. **`.env` reaches a pod only at creation**, so
  on a running pod this is the way to change a model rather than editing `.env`
- **On-pod work**: `python runpod/orchestrator.py run "<command>"`, and
  `logs [n]` for the pipeline log. Only port 9000 is exposed and the SSH proxy
  drops `exec_command`, so both drive the interactive shell the deploy opens
- **Cold start**: `python runpod/orchestrator.py start` prints a phase breakdown
  (provision / setup / pip / model load), labelled warm or empty volume
- **Latency budget**: reported per preset when a stream stops — p50/p95/p99 per
  stage against the frame deadline, with a HOLDS/MISSES verdict
- **Guard calibration**: `python pipeline.py --stream --guard-observe --guard-report r.json`
- **Realism**: `python pipeline.py --stream --debug-frames clip/` then
  `python tools/compare_frames.py clip/ [--against clip2/]`

### Building for distribution
- **Desktop standalone**: `python tools/build_desktop.py` (add `--print-only`
  to see the command, `--release` to hide the console once it works)

Only `desktop/` is compiled, and the reason is **distribution, not speed** — it
ships to customers and the access-code gate, session clock and Firestore session
plane are all enforced inside it, which as `.py` files comes out with a text
editor. `pipeline/` is deliberately excluded: it runs inside a Docker image on a
rented pod where nobody reads the source, its startup is dominated by model
load, and compiling it would add twenty minutes to every image build while
making tracebacks worse on the layer under active development.

`--standalone`, never `--onefile`: unsigned single-file builds are routinely
flagged as malware on Windows, which reads worse to a customer than a `.py`
file. Code signing is the step after a build that works.

The failure mode worth knowing: a build that loses `main.qml` or a QML module
**compiles fine**, starts, produces no root object and exits `-1` saying
nothing. `desktop/resources.py` resolves bundled files across source and frozen
layouts and names every location it tried; `tests/test_desktop_build.py` checks
the recorded QML module list against what `main.qml` actually imports, so the
two cannot drift.

### Install Dependencies
- **CPU (local dev)**: `pip install -r requirements-pipeline-cpu.txt`
- **GPU (CUDA)**: `pip install -r requirements-pipeline-gpu.txt`
- **CI/Testing**: `pip install -r requirements-ci.txt`

## Architecture

### New Core Modules (Phase 7 Migration Complete)

**Configuration & Infrastructure:**
- **pipeline/config.py**: `FaceSwapConfig` dataclass, observable (replaces globals.py)
- **pipeline/types.py**: Typed dataclasses (`Bbox`, `Detection`, `VideoProperties`, `SwapResult`)
- **pipeline/events.py**: `EventBus` pub/sub system, event constants
- **pipeline/logging.py**: Structured logging with event emission

**Services Layer (ML/CV components):**
- **pipeline/services/face_detection.py**: `FaceDetector` (InsightFace wrapper)
- **pipeline/services/face_swapping.py**: `FaceSwapper` (ONNX face swap)
- **pipeline/services/enhancement.py**: `Enhancer` (CodeFormer ONNX, or GFPGAN)
- **pipeline/services/face_tracking.py**: `LandmarkStabilizer` (EMA on face landmarks)
- **pipeline/services/masking.py**: `FaceMasker` (landmark hull + optional XSeg occlusion)
- **pipeline/services/database.py**: `FaceDatabase` (embedding cache, averaging, source review)
- **pipeline/services/guards.py**: Input guards — pure predicates plus `GuardResult`

**Processing Pipeline:**
- **pipeline/processing/frame_processor.py**: `FrameProcessor` ABC + implementations
  - `PreprocessingProcessor`, `DetectionProcessor`, `SwappingProcessor`, `OutputProcessor`
- **pipeline/processing/compositor.py**: `FaceCompositor` (aligned-space compositing)
- **pipeline/processing/pipeline.py**: `ProcessingPipeline` (orchestrator, replaces monolithic stream.py)

**I/O Layer:**
- **pipeline/io/capture.py**: `InputSource` ABC + implementations (Webcam, Network, File, ImageSequence)
- **pipeline/io/output.py**: `OutputSink` ABC + implementations (File, HTTP, WebSocket, RTMP)
- **pipeline/io/ffmpeg.py**: FFmpeg utilities (extract_frames, create_video, restore_audio, etc.)

**API & Control:**
- **pipeline/api/server.py**: `WebSocketAPIServer` — real WebSocket server on single port 9000
  - Text frames: JSON commands and events
  - Binary frames: JPEG-encoded video frames pushed to all clients
  - Health check: `{"action": "health"}` → `{"status": "healthy", "uptime": <seconds>}`
  - Heartbeat ping/pong every 30s
  - Auto-stop timer: background thread stops pod after `RUNPOD_MAX_UPTIME` minutes
- **pipeline/api/handlers.py**: Type-safe command handlers; `HandlerContext` dataclass (no globals)
- **pipeline/api/schema.py**: Message types, command/event constants, quality presets

**Simplified Entry Points:**
- **pipeline/core.py**: Argument parsing, headless orchestration; supports `--stream`, `--log-level`
- **pipeline/stream.py**: Stream mode wrapper
- **desktop/bridge.py**: Push-based frame display (no HTTP polling, no 2s status timer)
- **desktop/controller.py**: WebSocket client (`websockets` library, single connection, auto-reconnect)

### Session shutdown
The paid hour ends in one of two ways, and both land in
`Bridge._end_session`: the pipeline broadcasts `auto_stop` before stopping the
pod, or the session's own clock runs out. Three things deliberately do **not**
happen there:

- **The virtual camera is not touched.** See above — releasing it is the one
  action that can expose the operator's real face.
- **Nothing is drawn on the frame.** The notice is a card in the desktop
  window. What reaches the call is the last swapped frame, unchanged.
- **The full auth gate does not take over.** The operator may still be in a
  call and needs to see the app. The gate returns when they ask for it
  (`enterNewCode`) or on the next launch.

`PipelineClient.expect_disconnect()` stops the reconnect loop, because a pod
stopped on purpose is not a network fault and must not present as one. The loop
is otherwise indefinite — the cap is on the delay, not the attempt count.

### Removed Files (Dead Code Deleted)
The following files were deleted in the Phase 2 cleanup:
- `pipeline/processors/frame/face_swapper.py` → replaced by `pipeline/processing/frame_processor.py::SwappingProcessor`
- `pipeline/processors/frame/face_enhancer.py` → replaced by `EnhancementProcessor`
- `pipeline/processors/frame/core.py` → orphaned
- `pipeline/processors/` directory → fully removed
- `pipeline/face_analyser.py` → replaced by `pipeline/services/face_detection.py::FaceDetector`
- `pipeline/typing.py` → replaced by `pipeline/types.py`
- `pipeline/ws_server.py` → replaced by `pipeline/api/server.py`
- `pipeline/capturer.py` → replaced by `pipeline/io/capture.py`
- `pipeline/utilities.py` → functions migrated to `pipeline/io/ffmpeg.py`

### Data Flow (Event-Driven)
1. `pipeline.py` → `core.run_headless()` parses args → loads `.env` → updates `CONFIG`
2. `WebSocketAPIServer` starts on port 9000 (`ws://host:9000/ws`), single port
3. **Batch mode**: `ProcessingPipeline.run_batch()` → detects faces → swaps → composites → outputs
4. **Stream mode**: `ProcessingPipeline.run_stream()` → captures frames → detects (every frame) → stabilizes landmarks → swaps → composites → emits `FRAME_READY` event
5. `FRAME_READY` → server encodes JPEG → pushes binary to all WebSocket clients (no polling)
6. `STATUS_CHANGED`, `DETECTION` events → server pushes JSON text to all clients
7. `desktop/bridge.py` receives push callbacks, updates frame buffers and UI state

**Event Flow:**
```
ProcessingPipeline (coordinator)
  ↓ emits events to BUS
EventBus (pub/sub)
  ↓ broadcasts to
WebSocketAPIServer
  ↓ sends to
desktop/bridge.py (UI updater)
  ↓ updates
QML display
```

### Quality Presets
Desktop quality dropdown controls capture resolution, frame rate, and processing parameters. Defined in `pipeline/api/schema.py::PRESETS` and `desktop/bridge.py::_QUALITY_CAPTURE`.

Presets trade latency against realism. Defined once in
`pipeline/api/schema.py::PRESETS`, applied via `FaceSwapConfig.apply_preset()`.

A preset picks **how much compute to spend. It does not change how the face
looks.** `enhancer_weight` and `enhance_strength` decide whether the output reads
as a real call or as AI, and neither costs anything to compute — so varying them
per preset only meant "production" restored hardest and therefore looked most
synthetic, while presenting itself as the best option. They are identical in
every preset now.

|                         | Fast      | Optimal (default) | Production |
|-------------------------|-----------|-------------------|------------|
| **Capture resolution**  | 480x270   | 640x360           | 960x540    |
| **Frame rate**          | 15 fps    | 20 fps            | 30 fps     |
| **JPEG quality**        | 60        | 70                | 85         |
| **Detector input**      | 320       | 448               | 640        |
| **Compositing ceiling** | 192       | 256               | 320        |
| **Occlusion masking**   | Off       | On                | On         |
| **Landmark EMA**        | 0.7       | 0.6               | 0.5        |
| **Temporal EMA**        | 0.7       | 0.6               | 0.5        |
| **Restore strength**    | 0.7       | 0.7               | 0.7        |
| **Fidelity weight**     | 0.7       | 0.7               | 0.7        |
| **Grain matching**      | On        | On                | On         |

The EMA factors vary with frame rate rather than with quality: smoothing across
frames is smoothing across time, so the same factor reaches twice as far back at
15fps as it does at 30.

Capture settings live in `PRESETS` and are read by both the pipeline's own
`VideoCapture` loop and the desktop's webcam thread, so local and push mode
cannot diverge. Changing quality restarts the capture device to apply them.

Presets deliberately **do not** set `enhance`: it has an explicit toggle in the
desktop header, and a preset must not silently undo something the operator just
clicked. `color_correction` is left alone for a different reason — it is on and
stays on (see below).

### Header toggles — what a consumer is allowed to change
**None.** The header is status only: the app name on the left, the status
message, connection state and the media tabs on the right. `VCAM`, `COLOR`,
`PREPROC` and finally `ENHANCE` were all removed, and the distinction between
the reasons is worth keeping straight — a toggle implies a choice worth making.

- **VCAM** was removed because it was never a quality knob: it is where the
  output *goes*. The earlier note here said it "is the control that makes the
  whole thing work", which was the argument against it being a control at all.
  Its only distinct effect was releasing the virtual camera device, and the only
  moment anyone would do that is mid-call — where a conferencing app responds by
  showing a placeholder, reporting a disconnected camera, or **selecting the
  next available camera, which is the operator's real webcam**. That is the
  exact failure the product exists to prevent, reached through a button that
  reads like a convenience.

  The camera is now simply **on**: opened when the app opens, released only when
  it closes — not tied to a mode or a session. An open device nobody has
  selected costs nothing, while a device that comes and goes is what makes a
  conferencing app go looking for another one. The **VCAM badge in the
  viewport's bottom-left corner** shows its state — the header carried a
  second copy of the same bit and it was cut as redundant.
- **ENHANCE** was removed last, and it is the subtlest of the four because the
  opinion behind it is legitimate: restoration is what decides whether the
  output reads as a real call or as AI, so "too plastic" is a real complaint.
  What made it wrong was the *shape*. The believability axis is `enhancer_weight`
  and `enhance_strength`, both already at a tuned `0.7` — restoration is
  deliberately partial, keeping some of the input's imperfection. So the
  toggle's off position was never "less plastic", it was **no restoration at
  all**: a 128-native swap upscaled into a sharp frame, which is a soft face
  that does not match the picture around it. A switch across an axis that is
  not binary. It belongs behind a strength slider; until that exists,
  `set_realism` is the honest way to A/B it.
- **COLOR** was removed because it is correctness, not preference. It matches
  the swapped face's skin tone to the target; off produces a colour step at the
  boundary, which is failure mode 2. Turning it off never makes output better.
- **PREPROC** was removed because it defaults *off* and, by its own docstring,
  "the output stops looking like the operator's real camera" — the opposite of
  the design target. It is a rescue knob for terrible lighting.

All of them remain reachable via `set_realism`, the CLI and env: they were
removed from the consumer's surface, not from the product. That escape hatch is
only real if the desktop stops asserting its own defaults — `startPipeline`
used to fire `set_enhance`, `set_color_correction` and `set_preprocessing` on
every start, which silently reverted a pipeline launched with `--no-enhance`.
It now pushes **only** `set_quality`, the one setting the desktop actually owns
a control for, and reads the rest back via `_sync_state_from_server`.

Note also that the toggles were realtime-only, while the settings they set are
global — a photo or render job inherits whatever the config holds, which is
another reason not to expose knobs there that nobody should be turning.

`tracker`, `blend`, `luminance_blend` and `redetect_interval` remain on
`FaceSwapConfig` so the `set_blend` / `set_alpha` API commands keep working, but
nothing reads them and they are no longer in `PRESETS` — face tracking was
replaced by per-frame detection plus landmark EMA, and blending is handled by the
compositor's mask.

### Compositing (FaceCompositor)
Everything after the swap happens in **aligned face space** at 256x256, not on
whole frames. `FaceSwapper.swap_aligned` returns the swapper's raw crop plus its
affine (via `paste_back=False`), and `FaceCompositor` owns the rest: enhancement,
temporal smoothing, colour matching, detail matching, masking and grain. This is
what lets the mask follow the real jawline, keeps colour from pulsing, and stops
the enhanced face reading as sharper than the frame around it.

Detection runs on **every** frame, so the swap is always warped with current
landmarks. Temporal continuity comes from EMA — on landmarks
(`LandmarkStabilizer`) and on aligned pixels (`FaceCompositor`) — both of which
release under motion and reset on face loss or source change. Both are bypassed
when `many_faces` is set, since per-frame detection order is not stable.

### Face restoration
`pipeline/services/enhancer_models.py` is a registry of restoration models, the
sibling of `swapper_models.py` and for the same reason: the model owns facts
about itself — crop size, whether it has a fidelity weight, where to fetch it —
and hard-coding one model's answers is what made a second model impossible to
add. Select with `--enhancer-model`, `ENHANCER_MODEL`, or `set_realism`.

| | crop | inference (4090) | file | fidelity weight |
|---|---|---|---|---|
| **`gpen_bfr_256`** (default) | **256** | **5.4ms** | 76 MB | no |
| `codeformer` | 512 | 29.4ms | 377 MB | **yes** |
| `gpen_bfr_512` | 512 | 37.5ms | 284 MB | no |
| `gfpgan` | 512 | — | 340 MB | no |

**Why the default is the small one.** Restoration was 68% of a 58.9ms frame and
is the one stage whose cost ignores how big the face is — it runs on a fixed
crop either way. On a 101px webcam face, CodeFormer's 29.4ms buys **+0.03** on
the face/frame detail ratio and leaves noise and seam unchanged to three
decimals, because the 512 result is warped straight back down into a 128-192
aligned space at `compositor.py:515`, discarding ~86% of it one operation after
it was made.

Note which way `gpen_bfr_512` falls: **slower than CodeFormer at the same
resolution**. The saving is entirely the crop, not the architecture — GPEN is
not a lighter model. And 256 rather than 128 or off because restoring above the
aligned size is *supersampling*: at 256 into a 192 aligned space that margin
survives, along with all the low-frequency work — tone, structure, artefact
cleanup — that a downsample does not destroy. What is given up is the
512-to-256 octave, which the final resize into a ~101px face deletes anyway.

**This was adopted on speed evidence and has not been judged on footage.**
`gpen_bfr_256` has never been composited or looked at. The measured comparison
was restoration-off against CodeFormer-512 only.

**What changes with a model without a fidelity weight.** `enhancer_weight` is
CodeFormer's input and nothing else's, so under GPEN it means nothing and
`enhance_strength` — the compositor-side blend — is the only remaining control.
That is a real loss of an axis: CLAUDE.md called the fidelity weight the knob
believability lives on, and it is why CodeFormer stays registered rather than
being deleted.

Two backends run these. The **`codeformer` backend** is the ONNX path and
despite its name runs any single-input ONNX restorer, because it introspects
the graph rather than assuming it — the weight input is wired only if declared,
and the crop size is read from the declared input shape. The **`gfpgan`
backend** needs torch plus the `gfpgan` package. If the selected model cannot
load, the registry is walked — requested, then default, then the rest — so a
missing weight file degrades to "restoration still works" rather than "off".

All of them are trained on **FFHQ-framed crops** and rely on features sitting
where FFHQ puts them, so `FaceCompositor` warps into FFHQ space around the
restore call rather than handing them the swapper's tighter arcface crop. FFHQ
framing is ~28% wider than arcface, so the crop given to the restorer is the
real frame in FFHQ framing with the swapped face composited over it — otherwise
the edges would be empty. Only what the swap covers survives the mask, so the
real face at the edges never reaches the output.

Geometry uses a closed-form Umeyama similarity fit (`estimate_similarity`), not
`cv2.estimateAffinePartial2D` — the OpenCV estimators are randomized and anything
that varies frame to frame feeds straight back into shimmer.

**Restoration is not tied to the swapper, deliberately.** It would be easy to
put `enhance: False` in hyperswap's look profile and call the pairing settled —
a 256-native swap needs less repair than a 128 one, which is the belief the
profile's `enhance_strength` 0.5 already encodes. Two reasons not to. It is an
axis, not a switch, which is the same argument that removed the ENHANCE toggle
from the header; and it is unmeasured — hyperswap's output has never been
looked at with restoration or without. Encoding an untested belief as a hard
rule also removes the ability to test it, since three of the four cells in
{inswapper, hyperswap} x {restore, don't} would become unreachable. The graded
version already exists in `enhance_strength`; leave the binary to the footage.

### Realism knobs (`FaceSwapConfig`)
| Field | Default | Effect |
|-------|---------|--------|
| `enhance` | `True` | Face restoration on/off |
| `enhancer_model` | `gpen_bfr_256` | Restoration model — see `enhancer_models.py`. The registry owns crop size and fidelity support |
| `enhancer_weight` | `0.7` | CodeFormer fidelity: `0`=most restoration, `1`=closest to input. **Inert on models without a weight input**, which is every model but `codeformer` |
| `enhance_strength` | `0.7` | How much of the restored face to keep. Full strength reads as AI; partial keeps believable imperfection |
| `restore_size` | `512` | Edge of the FFHQ crop fed to the restorer. A model with fixed spatial dims overrides it and says so once — see below |
| `restore_min_face` | `0` | Skip restoration below this face size (px, shorter side). `0` never skips |
| `aligned_size` | `256` | **Ceiling** on compositing resolution (clamped 128–512). The size actually used follows the face's own size in frame, in steps, with hysteresis — a distant face is not upsampled to detail its webcam never captured, and costs proportionally less |
| `temporal_alpha` | `0.6` | EMA on aligned pixels, kills shimmer (`1.0` disables) |
| `color_correction` | `True` | LAB transfer, sampled inside the mask, ramped by colour distance |
| `color_strength` | `1.0` | Scales that transfer |
| `grain` | `True` | Matches sensor noise on the composited face |
| `occluder` | `True` | XSeg mask so hands/mics are not overpainted |

### Swap models
`pipeline/services/swapper_models.py` is a registry of swap models, each
carrying both a **spec** (kind, alignment template, native size, normalisation,
URL) and a **look profile** (`enhancer_weight`, `enhance_strength`,
`aligned_min`).

| | inswapper_128 | hyperswap_1a/1b/1c_256 |
|---|---|---|
| Source input | ArcFace embedding via `emap` | ArcFace embedding, direct |
| Template | `arcface_128` | `arcface_128` (identical) |
| Native size | 128 | 256 |
| `enhance_strength` | 0.7 | 0.5 |
| `enhancer_weight` | 0.7 | 0.8 |
| `aligned_min` | 128 | 256 |

The appearance knobs used to live in `PRESETS._LOOK`, identical in every preset.
That reasoning was right (a preset picks compute, not looks) but the location
was wrong: **how much restoration a face needs depends on what generated it**,
not on the frame rate. Ownership is now:

    quality preset  ->  compute     (capture, det_size, aligned ceiling, EMA)
    model profile   ->  appearance  (restoration burden, aligned floor)
    CLI / env       ->  explicit override of either
    set_realism     ->  live A/B on top

Select with `--swapper-model`, `SWAPPER_MODEL`, or `set_realism` at runtime —
switching applies the new profile, drops the session and resets temporal state.

Both registered families take an **embedding**, which is what preserves
multi-photo averaging, `.npy` embeddings and the identity-outlier guard.
`uniface_256` and `blendswap_256` take a source *image* and are deliberately
absent for that reason. Both use the same alignment template, which is why
switching needs no change to the compositor, masker or guards.

`FaceSwapper` runs inswapper through InsightFace's `INSwapper` (which owns the
`emap` projection) and everything else on its own onnxruntime session. Note the
naming trap recorded there: facefusion's `embedding_norm` is the normalised
512-d *vector*, while InsightFace's attribute of that name is a *scalar* — the
code reads `normed_embedding` deliberately.

Weights: 384 MB each, pinned to release tag `models-3.3.0` (verified; `3.0.0`
and `3.4.0` both 404 for these files).

### Input guards
`pipeline/services/guards.py` refuses inputs that would produce a wrong swap
instead of swapping them badly — confidently wrong output is worse than none, and
a stranger's face swapped in *looks like it worked*. See
[docs/INPUT_GUARDS.md](docs/INPUT_GUARDS.md).

Two call sites, differing in whether a human is present to be told:

- **Source images, at upload** — `FaceDatabase.review_sources` rejects multi-face,
  no-face, too-small (<110px shorter side), blurred, extreme-pose (>±35° yaw) and
  identity-outlier images, reporting **which** image and **why** per file.
  Outliers use leave-one-out cosine against the mean of *the others*, three
  images minimum; two that disagree are both refused, since there is no majority
  to identify the intruder.
- **Runtime, per frame** — `guards.check_frame` guards multi-face, low
  confidence, small faces (<80px), extreme pose; occlusion is checked from the
  coverage `FaceMasker` records during the XSeg pass it already runs.

A guarded frame emits **the last good swapped frame, unchanged** — nothing drawn
on it, since it reaches every participant on the call. Guards fail closed, and
never update temporal state. `FaceCompositor.composite` returns `None` rather than
the untouched frame when it cannot produce a swap: on the live path the untouched
frame is the operator's real face.

Batch splits by what an unswapped output would mean, and **multiple faces is
the one guard that splits again**:

- **Video** passes the original frame through — for pose, confidence and
  occlusion. The target is a file the operator supplied, not their camera, and
  one unswapped frame mid-clip is a smaller defect than a hole. Those guards
  describe a single frame: a turn of the head, a hand, a blurred moment.
- **Video, multiple faces** stops the job and says where. That guard describes
  the *target*, not a frame of it — a second person in shot will almost
  certainly persist, so every frame they appear in is written unswapped and the
  render silently stops being a swap partway through while reporting success.
  The reason names the frame, the timecode and the count, and goes out through
  `emit_error`: the desktop reads a batch's success from whether an error
  arrived, so a warning would render as "processing complete". No partial file
  survives — the abort precedes `create_video` and the existing `finally`
  cleans the extracted frames.
- **A still** writes nothing at all. There is no surrounding footage to carry
  it, so an unswapped photo is a copy of the input wearing the output's name —
  indistinguishable from success to whoever opens the folder, which is the
  "confidently wrong output" the guards exist to prevent. `PhotoResult` carries
  the refusal and its reason instead.

`_run_vcam` holds and re-sends the last frame when the queue empties, so the
virtual camera never shows the raw camera and never shows nothing — covering hour
expiry, session end, worker death and crashes alike. It keeps doing so until the
app closes: `cleanup` is the only path that releases the device. `stopPipeline`,
an expired session and a dropped socket all deliberately leave it running.

Yaw prefers `face.pose` — `buffalo_l` bundles `1k3d68.onnx`, which computes it as
a side effect of detection — and falls back to a keypoint approximation on packs
that lack it. The two are not on the same scale, so which was used is recorded.

Thresholds are `guard_*` fields on `FaceSwapConfig`, settable via `set_realism`
(clamped, not rejected). `guards` disables the runtime guards wholesale;
`many_faces` bypasses them.

### Naming a face
The multi-face guard fires because "which face did you mean?" has no safe
default, so **anyone who answers the question dismisses the guard**. Two can:

| Who | How | Field |
|---|---|---|
| A template's author | `face_point` in the manifest, offline | `target_face_point` |
| The operator | Clicks a face over their own photo | `target_face_points[i]` |

Both are **normalised points, not indices** — detection order is not a stable
contract, and an index that comes to mean a different person is the silent
wrong-person swap the guards exist to prevent. `templates.select_by_point`
resolves by containment, then nearest centre; `DetectionProcessor` consults it
ahead of `select_primary` and `guards.check_frame` stands down when one is set.

The operator's is a **list** because photo mode carries up to four targets and
each asks separately; the config's single point would name a face in photos
nobody looked at. Aligned with `target_paths`, `None` where nothing was asked,
cleared by every new `upload_target` and by `_clear_template`. It is threaded
explicitly through `_process_photos_batch` → `_process_image_batch` →
`_swap_frame_detail` rather than by mutating config mid-loop.

Faces are counted at **upload** (`handle_upload_target` → `face_boxes`), not at
swap time, because that is where the person is: a photo refused mid-job tells
them only that they already picked the wrong one.

**Sources are deliberately excluded.** `check_source` has no such escape and
should not get one — a source builds the identity every frame is swapped *to*,
and averaging that out of a crowd is unrecoverable downstream.

### The operator is told
A guarded frame is silent by design on the *call* — nothing is drawn on it,
because the frame reaches every participant. It was silent in the app too,
which was an oversight: the pipeline broadcasts the reason as a `STATUS_CHANGED`
with `scope='GUARD'`, and the bridge dropped it while the pipeline was running.
It now sets `guardReason`, shown as a badge in the viewport beside the
detection badge. Both edges arrive (`_guard_frame` on transition,
`_clear_guard` when it lifts), so it is not a message that needs timing out.

### Photo mode
A third job shape beside live and batch video: **one to four target photos, each
swapped independently, failures skipped**. It adds no stage to the pipeline — it
loops the image path that already existed, so every photo goes through the same
`_swap_frame_detail`, the same guards and the same `FaceCompositor` as a video
frame or a live frame.

Three things are specific to it:

- **Targets are uploaded, not referenced.** `upload_target` carries the images
  base64 in one message, capped at `MAX_PHOTO_TARGETS` (4) and
  `MAX_PHOTO_BYTES` (6 MB) each, both defined in `pipeline/api/schema.py` and
  enforced on the server as well as in the desktop. This exists because
  `set_target` validates with `os.path.exists` against the *pipeline's*
  filesystem: on a pod that is another machine, so a chosen file is simply not
  there. Photos are small enough to carry inline; a video is not, which is why
  this is image-only and not a general transfer path.
- **A photo that cannot be swapped writes no file.** See the guards section
  above — this is the one place batch behaviour deliberately differs between
  video and stills.
- **Failures are per photo.** Unreadable file, no face, a guard, or an exception
  out of the swap all record against that photo and the loop continues; one bad
  photo must not cost the operator the other three. Each result goes out as a
  `PHOTO_RESULT` event as it lands, and `get_photo_results` returns the whole
  set with the swapped images inline — the outputs live on the pipeline's
  filesystem, so a path alone would be useless to a remote operator.

The desktop writes each returned image beside the original the operator picked,
with the `_swapped` suffix batch video already uses. A photo that already fits
under the cap is uploaded byte-for-byte; only a camera original over it is
re-encoded, quality first and dimensions only after, in 10% steps with a floor
of 1600px on the long side. Losing detail defeats the point of a photo swap, so
the transfer budget gives way before the image does.

### Output format, and the commands that answer for it
`keep_fps` and `keep_audio` decide what *file* the operator gets back, not how
the face looks, which is why they are settable at runtime while `many_faces`
and `keep_frames` are not. Both were declared in `COMMANDS` with no handler
behind them, so a client method written against either returned
`Unknown command`.

Their defaults were worse than the missing handlers. `--keep-fps` was
`store_true`, so **every render retimed to 30fps** unless asked otherwise —
duplicating frames on a 24fps source, discarding motion on a 60fps one, and
routing everyone through the branch the audio-desync bug lived on. Both
`docs/USAGE.md` and `docs/TROUBLESHOOTING.md` already described it as "enabled
by default", so the docs had the intended behaviour and the code disagreed.
`--keep-audio` was `store_true` with `default=True`, which can only ever
produce True — there was no way to drop audio at all.

Both are now `argparse.BooleanOptionalAction` with `default=True`, so
`--no-keep-fps` and `--no-keep-audio` exist and work, and the default hands
back what was handed in. Note the CLI is what actually decides this:
`core.py` runs `CONFIG.set('keep_fps', args.keep_fps)` unconditionally, so the
dataclass default never reaches a CLI run — it is set to match rather than to
lead.

**The desktop deliberately does not push either on a render.** It has no
control for them, and asserting a default it never chose is exactly the
`set_enhance` mistake recorded above — `startPipeline` used to revert a
pipeline launched with `--no-enhance`. Fixing the default fixes it for every
path at once; sending it from the desktop would re-break the CLI's escape
hatch.

`COMMANDS` is now checked against `dispatch_command` in both directions by
`tests/test_wiring.py`, along with the rule that every command the client can
send is one the server answers. It had drifted both ways because nothing read
it — it is documentation only, and documentation nothing checks is a comment.

### What counts as a photo
`IMAGE_EXTENSIONS` in `pipeline/io/ffmpeg.py` is the single list: `.jpg`,
`.jpeg`, `.png`, `.webp`, `.bmp`. Every file dialog builds its filter from it
(`_IMAGE_FILTER` in `desktop/bridge.py`) and `is_image` checks against it, so
the picker cannot offer something the check refuses.

It is a fixed tuple rather than `mimetypes.guess_type`, which is what it used
to be. That silently dropped **webp**: the mapping only arrived in Python 3.11,
and on Windows the module also reads `HKEY_CLASSES_ROOT`, so the same file
resolved on one machine and not the next — while the dialog offered `*.webp`
either way. Selecting one did nothing at all, with no message. A supported
format failing by environment is worse than one failing outright, because
nothing about it looks broken.

`.gif` and `.heic` are excluded deliberately, not pending: OpenCV decodes
neither, so accepting one means uploading a file certain to be refused after a
round trip. Better to say so while the picker is still open. An *animated*
webp lands on the other side of that line for the same reason rather than an
opposite one — OpenCV reads its first frame, and the first frame is a real
still of the person, which the guards then judge normally. It is accepted
silently; nobody is told the animation was flattened.

`tests/test_wiring.py` round-trips every extension on the list through
`imwrite`/`imread` and asserts a well-formed gif still fails, so the list stays
a checked claim about what OpenCV reads rather than a remembered one. `is_video` stays a
mimetype lookup on purpose — video is whatever FFmpeg can demux, never
enumerated in a dialog, so a fixed list there would reject working files.

Two related rules the same bug exposed: a dialog that yields nothing usable
**says why** (`_unusable_reason` names the file and whether it was a video or
an unsupported format) rather than returning quietly, and `_set_status` takes
an explicit `error` flag that colours the header line — every refusal used to
render in the same grey as "idle".

### The one-face notice
Shown once, on a first run, over a blurred window; dismissed by the button or a
click outside, and reopened from the `?` beside the media tabs. The flag lives
in `prefs.json` under `Bridge._cache_dir()` — the first purely local state the
desktop owns, since session state deliberately lives in Firestore so a reinstall
does not cost the customer their hour. Unreadable prefs mean the card shows
again, which is a repeat rather than a fault.

It says **three rules, not one**. "Exactly one face" stopped being true when the
picker landed: a target photo may hold several so long as the operator says
which. A rule stated more strictly than the app enforces it teaches people to
distrust the next one.

The blur is why `main.qml` has an `appBody` wrapper: `MultiEffect` needs the
window's content in one item to render through. The gate, the session card, the
auto-stop dialog and the notice itself sit **outside** it — an overlay that
blurred itself would be unreadable. `layer.enabled` is toggled rather than
`blurEnabled`, so the effect costs nothing while the notice is closed, which is
almost always: there is a live 30fps viewport underneath.

### Desktop navigation
Two levels. The header's far right picks the **media tab** — VIDEO or IMAGE —
and the sidebar picks the job within it:

| Tab | Modes | The difference |
|---|---|---|
| **VIDEO** | LIVE, RENDER | Streamed now, or a file processed offline |
| **IMAGE** | UPLOAD, TEMPLATES | Whose picture the face goes into |

LIVE and offline video are one family because they share the video pipeline and
the compositor; a still is a different kind of job, not a third peer. Switching
tabs moves the mode with it, and setting a mode directly pulls the tab back into
step, so the two can never disagree.

The image pair is named for what actually differs. The source face is the
operator's in both — **UPLOAD** is a picture they bring, **TEMPLATES** is one we
ship. PHOTOS/SCENES reads better but the two words are near-synonyms, so the
distinction would not survive a first reading.

The video label is **RENDER**, not "batch". Batch reads as *many*, and this has
always been one video processed offline rather than streamed; the word sent
readers looking for a multi-file feature that never existed. The code still says
`run_batch`, correctly — there it means "not streaming", and it now covers
photos and templates too.

### Filters and effects — the last layers
Two decorative layers over the finished swap, in this order:

    swap  ->  filter (regrades the picture)  ->  effect (draws on top of it)

A **filter** (`desktop/filters.py`) is a grade — Warm, Mono, Noir. An **effect**
(`desktop/effects.py`) is an overlay — Confetti, Snow, Hearts, Bubbles, Sparkle.
Filter first, because an effect is meant to sit *on* the picture rather than be
part of it; grading confetti would tint it to match a look it is supposed to be
separate from.

Both are shown by one control. Pressing FILTERS shrinks the viewport and reveals
a **horizontal strip** of grades along the bottom and a **vertical rail** of
overlays down the right; HIDE gives the space back. APPLY sits beside HIDE
rather than at the end of the chips, since it acts on the whole panel and not on
any one chip.

**Effects are a function of the clock, not of call count.** Every particle's
position is computed from a timestamp and wraps with a modulo, so nothing holds
state between frames. That is what makes the overlay safe to render from two
places at two different rates — the webcam thread's local preview and the
display timer's pipeline frames — which a `step()`-style animator could not be:
advanced by both, it would run at the sum of their rates. Nothing is loaded from
disk either; these are drawn, not decoded, so there are no assets to bundle and
no GIF to keep in step with a frame rate.

Measured at 960x540: filters worst 7.5ms (Soft), effects worst 2.8ms (Bubbles),
against a 33ms display tick — and **zero** when nothing is on, since the
undecorated path still lets Qt load the JPEG itself.

That layout is not a style choice. The first version was a separate window with
its own preview, and it was wrong the moment the image tab was open — it showed
the live camera in a mode that has no live camera. A strip has nothing to
preview: whatever the mode was already showing is what a filter is judged
against, so the body is identical in every mode and only the panels move.
`filterStrip.reserved` and `effectRail.reserved` are the single numbers both
viewports read, so the body and the panels cannot disagree about the split.

Showing the strip persists across a media-tab switch — it is a preference, not
a detour.

Two properties do the work for both:

- **Last, always.** A filter is applied after the swap has fully composited.
  Grading first would have `FaceCompositor` match the face to an already-graded
  frame and then grade it again; applied last, a filter cannot break the swap
  underneath it. The webcam frame sent *upstream* is deliberately ungraded for
  the same reason — only the local preview gets the filter.
- **Desktop-side, never the pipeline.** Filters need nothing the face models
  provide, so they must not compete for a latency budget the swap has not been
  measured against, and changing one should be a local variable rather than a
  round trip to a rented GPU. `desktop/filters.py` is lookup tables and one
  cached multiply — worst case ~7ms at 960x540 against a 33ms timer, and
  **exactly zero** when nothing is enabled, since the unfiltered path still
  lets Qt decode the JPEG itself.

Choosing is not applying. The picker sets the look and the preview shows it, but
nothing leaves the machine until **APPLY** is pressed — so a look can be
auditioned without it reaching a call. The same key is read by the display, the
virtual camera and a saved photo through one accessor, so those three can never
disagree about whether a filter is on.

Filters default **off**, and should stay off during the pod session: that
session exists to judge whether the swap reads as real, and a grade on top
changes what is being looked at.

Not covered: **RENDER**. A video is written pipeline-side, so the desktop never
holds those frames. Filtering a render needs either a pipeline stage or a local
FFmpeg pass, and neither is built.

### Template targets
Bundled scenes the source face is swapped into — **the target is ours, the face
is theirs**. Not a new job shape: `set_template` points `target_paths` at a
library image and the job runs as a photo job of one, through the same guards,
the same `FaceCompositor` and the same `PHOTO_RESULT` / `get_photo_results`
return path that uploaded photos use.

Being bundled is what makes it small:

- **No transfer.** The library lives on the pipeline's filesystem, so
  `set_target`-style path resolution works as designed. The upload machinery
  photo mode needed does not apply.
- **Ambiguity is resolved offline.** A scene with several people would be
  refused by the multi-face guard, which exists because "which face did you
  mean?" has no safe default. A template answers it once, in its manifest, as
  `face_point` — and `check_frame` stands down when one is set, since there is
  nothing left to protect against.
- **Failure is a build problem.** `tools/validate_templates.py` runs the real
  guards over the library and exits non-zero, so a scene that would be refused
  never ships. A user must never meet a refusal caused by an asset we chose.

`face_point` is a **normalised point, not an index**. Detection order is not a
stable contract — it can shift with a model pack — and an index that quietly
comes to mean a different person is exactly the confidently-wrong output the
guards exist to prevent. `select_by_point` matches by containment, then by
nearest centre; the validator flags a point that only resolves by proximity,
because that means the manifest is asserting something it does not point at.

An optional `foreground` RGBA layer is composited *over* the finished swap, for
hair, glasses or a hand that belongs in front of the face. XSeg already does
this from the frame, but a template's occlusion is fixed and known, so it can be
drawn once by hand and be right every time instead of approximately right per
run. PSD is the authoring format for that layer; the runtime consumes a flat
PNG, so no PSD is parsed and no blend-mode fidelity is at stake.

Outputs never land in the library — a template's target is a shared asset, and
writing `_swapped` beside it would leave one user's face there for the next job.
`config.output_dir` sends them to a per-job directory instead, and the desktop
saves the returned image to `Pictures/Phantom/`.

Library location follows the model weights: `/workspace/templates` when the
network volume is mounted, else `pipeline/templates/`. Gitignored for the same
reason weights are — a scene library would bloat every clone and image build.

### Execution providers — fails closed
ONNX Runtime does not error when a provider cannot initialise; it silently uses
CPU. Every model that decides how the output looks is ONNX (swapper, CodeFormer,
XSeg), so that fallback is seconds per frame, not a degraded live call — and on a
rented GPU it is a bill with nothing usable attached.

`pipeline/services/execution.py::verify` runs after `_warm_up_models` and
**raises `ExecutionProviderError`** if an accelerator was requested and the
sessions are not using it. It catches both shapes: the provider missing from the
build entirely, and a single model falling back while others did not.
`--execution-provider cpu` is the supported way to run without one and does not
raise.

Both deploy paths also check at build/setup time: the Dockerfile fails the build
if `libcudnn.so.9` will not load, and `runpod/startup.sh` exits non-zero after
installing cuDNN if it still cannot. See runpod/TROUBLESHOOTING.md section 5b —
this shipped broken once and was found by reading, not by failing.

**Do not downgrade any of these three to a warning.** Stopping is the requested
behaviour, not a conservative default: a pod on CPU bills a full GPU hour and
produces unusable output while appearing to work, which defeats the reason for
renting it. ONNX Runtime already emits a warning, and that warning is exactly
what let this ship broken — the value here is that it halts.

### ONNX sessions — one owner, four levers
`pipeline/services/onnx_session.py` builds **every** ONNX session in the
pipeline. Services say which model they want and whether its shapes are static;
everything about how the session is constructed is decided in one place.

It exists because nothing owned that moment. `face_swapping.py` constructed a
`SessionOptions`, set two fields on it, and never passed it to the model — dead
for as long as it had been there, and invisible to flake8 because the attribute
assignments count as uses. Four speed levers all hook session construction, and
bolting each onto three call sites independently is how a codebase acquires
three subtly different answers to the same question.

**All four default off.** The out-of-the-box path is bit-identical to what it
was before they existed; each is opted into and measured rather than assumed.

| Lever | Flag / env | Changes numerics | Notes |
|---|---|---|---|
| Pre-allocated IOBinding | always on | No | `BoundRunner`; falls back silently |
| CUDA graphs | `--cuda-graphs` / `CUDA_GRAPHS` | No | Static shapes only |
| fp16 weights | `--fp16` / `FP16` | **Yes** | **No valid weights — see below.** A/B on footage before shipping |
| TensorRT | `--trt` / `TRT` | Via fp16 | Per-architecture engine cache |

`static_shapes` is the caller's declaration, not a guess. CUDA graph capture
records **fixed device buffer addresses**, so a model whose input size changes
between calls would replay a graph describing the previous shape. CodeFormer
(always 512), XSeg (always its own input size) and the swapper (always
`model.size`) qualify; the detector does not, because `det_size` moves with the
preset.

`BoundRunner` reuses output buffers rather than letting ORT allocate one per
call. The copies themselves are not removable — the compositor is OpenCV on the
CPU, so pixels come home between models regardless — but the allocation and the
pageable-memory penalty are, at four to six inferences a frame. It degrades
silently to a plain `run` on a symbolic output shape or a build without the
binding API: this is a performance path, and a warning per frame would cost more
than it reports.

**fp16 does not currently convert.** Attempted on the pod against
`codeformer.onnx`: it runs, halves the file (359 MB -> 180 MB), and then fails
its own `--check` load with

    Type (tensor(float16)) of output arg (/fuse_convs_dict.32/Cast_output_0)
    of node (/fuse_convs_dict.32/Cast) does not match expected type (tensor(float))

A Cast the block list leaves in fp32 is being fed an output the conversion
moved to fp16. The tool refused to ship it, which is the behaviour that matters
— but it means **`FP16=true` has never actually run**, on either card. Both
sessions where the `fp16` row read flat were reading a silent fallback to the
fp32 weights, not a measurement of fp16. Fixing the block list is the work; do
not read the existing `fp16` numbers as evidence either way.

**fp16 is a copy, never a replacement.** `tools/convert_fp16.py` writes
`<name>-fp16.onnx` beside the original with `keep_io_types=True`, so callers
still hand it float32 and no second edit is needed in a second file per model.
Reverting is a config flag rather than a 384 MB download. The op block list is
not optional — reductions and normalisations accumulate across a whole feature
map, which is where fp16's exponent runs out, and a model that produces NaN is
not a faster model.

**TensorRT engines are cached per architecture, not pinned to one.** The cache
key is GPU, TensorRT and ORT versions, model fingerprint and precision — every
property an engine is invalid across. Pinning to a single GPU would defeat
`RUNPOD_DATACENTERS`, which exists because availability is the binding
constraint; it would trade "sometimes a slower card" for "sometimes no pod at
all", and on a paid session no pod is the worse failure. Each architecture pays
its build once, ever, so the cache warms itself.

`trt_gpus` bounds which cards are worth that build. Minutes of a paid hour with
an operator waiting is a good trade amortised on a fast card and a bad one on a
card that was never going to hold the deadline. It is a substring list rather
than a copy of the orchestrator's `_GPU_PERF` — the two answer different
questions and would drift.

**A TensorRT fallback warns; it does not halt.** This is the one deliberate
departure from the rule above, and it rests on the same reasoning. A model on
CPU is a paid GPU hour producing nothing usable, so that halts. A model that
fell back from TensorRT to CUDA is still on the GPU and still holds a live call
— stopping the session over it would cost the operator more than the fallback
does. It still has to be *said*, because TensorRT's failure mode is silence: the
provider registers, declines the graph, and CUDA runs it.

**What makes any of this falsifiable.** `swap+composite` used to be one number
covering inference, restoration, smoothing, colour, detail, masking and the
paste — enough to answer "does this preset hold", not "what is worth speeding
up". `FaceCompositor.last_stage_ms` now carries the breakdown and
`LatencyBudget.record` takes it as `extra`. Same pattern as
`masker.last_coverage`: the stage that measures a thing owns the number, and
whoever needs it reads it afterwards rather than having a timer threaded
through. Read `restore` against `swap+composite` first — if restoration is not
the dominant term, the premise behind fp16 and TensorRT is wrong here and
should be rewritten rather than defended.

Full reasoning, including why Nuitka is a distribution decision rather than a
performance one and why Numba has almost nothing to do here, is in
[docs/COMPILATION.md](docs/COMPILATION.md).

### Guard calibration
Nine thresholds were chosen without data. `--guard-observe` evaluates and records
every guard **without any of them acting**, because a session that enforces
cannot measure itself: a guarded frame emits a held frame and stops being a
sample of what the camera was doing. `--guard-report PATH` writes the JSON.

`GuardTelemetry` records the measured value behind each guard, not just the
verdict, and reports a distribution with the percentage that would fail and the
margin to the threshold — so a session returns a number per knob instead of "it
guarded a lot". A negative margin means the threshold sits inside normal
operating range and will fire on ordinary frames.

`FaceDetector` probes the model pack on first detection and logs which `Face`
attributes exist. Guards silently become no-ops when their input is missing — no
`normed_embedding` means no identity reset, no `det_score` means no confidence
guard — and that is indistinguishable from a guard that never had cause to fire.

Three thresholds can make things actively worse if mis-set, all invisible without
this: `guard_min_coverage` (unknown what XSeg reads on a clear face),
`guard_identity_sim`, and `guard_min_confidence` (0.5 against a detector
threshold of 0.35).

**Realism protection in the stabilizer.** An identity change needs 3 low
readings within the last 6 frames before smoothing is dropped — a single
motion-blurred embedding must not reset the landmark EMA mid-movement, which is
when shimmer is most visible. A window rather than a consecutive run, since
alternating detections would zero a consecutive counter every other frame. See
`LandmarkStabilizer._IDENTITY_CONFIRM`.

Three ways to set them:
- **Quality preset** — the desktop dropdown; see the table above.
- **CLI / env** — `--enhancer-model`, `--enhancer-weight`, `--enhance-strength`,
  `--aligned-size`, `--restore-size`, `--restore-min-face`, `--temporal-alpha`, `--color-strength`, `--no-enhance`,
  `--no-grain`, `--no-occluder`. Each also reads an env var
  (`ENHANCER_MODEL`, `ENHANCER_WEIGHT`, …) since the pod is configured via `.env`.
  Precedence: preset first, then CLI/env overrides.
- **`set_realism` API command** — `{"action": "set_realism", "values": {...}}`,
  or `controller.set_realism(enhancer_weight=0.5)`. Validates and clamps; unknown
  fields are reported back rather than silently ignored. Use this to A/B live.

Models (`codeformer.onnx`, `dfl_xseg.onnx`) download on first use to
`/workspace/models/` or `pipeline/models/`. If one is unavailable the pipeline
degrades — masking falls back to landmark hull + valid-region, restoration falls
back to the other backend or off — rather than failing.

### Entry Points
- **pipeline.py**: Headless engine; starts WebSocket API server + ProcessingPipeline (batch or stream)
- **desktop.py**: Qt/PySide6 GUI; connects to pipeline via WebSocket, never processes frames

## Code Style & Standards

### Architecture First
- **Service-oriented design**: Each service encapsulates one responsibility (FaceDetector, FaceSwapper, Enhancer, etc.)
- **Composable processors**: `FrameProcessor` subclasses chain operations without side effects
- **Observable config**: Use `CONFIG.set()` and `CONFIG.on_change()` instead of global mutable state
- **Event-driven coordination**: Use `BUS.emit()` and `BUS.on()` for inter-module communication, not direct function calls

### Naming & Comments
- Use clear, self-documenting names
- Comments only for non-obvious logic
- Docstrings for all classes and public methods (brief, concise)
- Private methods/attributes: prefix with `_`

### Type Checking
- Strict mypy enabled (`disallow_untyped_defs = True`, `disallow_any_generics = True`)
- All functions and methods must have complete type annotations
- All dataclass fields must be typed
- `ignore_missing_imports = True` allows third-party stubs to be optional

### Linting & Testing
- flake8 checks: E3, E4, **E9**, F. E9 is not a style class — it is "could not parse this file". Without it a syntax error lints clean, which is how a broken string literal reached a paid pod session
- Exception: `pipeline/core.py` ignores E402 (imports after code) for performance-critical initialization
- Run before commit: `mypy pipeline desktop` and
  `flake8 pipeline.py pipeline desktop tests tools runpod firebase`

## Dependencies & Environment

### Runtime
- **Python**: 3.9+ (required for type annotations)
- **Deep Learning**: `torch`, `onnxruntime`, `tensorflow`, `insightface`
- **Computer Vision**: `opencv-python`, `pillow`
- **Restoration**: CodeFormer runs on `onnxruntime` (no extra dependency);
  `gfpgan` is only needed for the alternate backend, graceful fallback if missing
- **GUI**: `PySide6` / Qt Quick (for desktop.py)
- **External**: FFmpeg (required for video encoding/decoding)

### Platform-Specific
- **GPU**: CUDA-enabled variants for torch/onnxruntime on Linux/Windows
- **macOS**: M1/M2 arm64 support via `torch::mps` acceleration (if available)
- **Execution providers**: CUDA, ROCm (AMD), DML (DirectML on Windows), CPU fallback

### Development
- **Type checking**: `mypy` (strict mode)
- **Linting**: `flake8`
- **Testing**: pytest (run examples through full pipeline)
- **Virtual environment**: Recommended (Python venv or conda)

## PR Guidelines

### Before You Start
- Check existing issues/PRs to avoid duplicate work
- For major features, open an issue first to discuss approach
- Prioritize bug fixes and correctness over features

### During Development
- Keep PRs focused: one feature or bug fix per PR
- Write complete type annotations; run `mypy pipeline desktop` locally
- Run linting: `flake8 pipeline.py pipeline desktop`
- Test with example files: `python pipeline.py -s=.github/examples/source.jpg -t=.github/examples/target.mp4 -o=/tmp/test.mp4`
- Use `.on_change()` for config updates, `BUS.emit()` for events, not global state mutations

### What We Value
- Clear, minimal changes (prefer small fixes over refactoring)
- New services: follow existing pattern (init + 1-3 public methods)
- New processors: inherit from `FrameProcessor` ABC, implement `process()`
- New handlers: add to `dispatch_command()`, validate all inputs
- Event-driven architecture: emit events instead of direct calls between modules

### What We Avoid
- Long classes with many responsibilities (split into services)
- Direct access to other modules' globals (use CONFIG or events)
- Monolithic functions (refactor into reusable processors/services)
- Proof-of-concepts without tests
- Undocumented behavioral changes

## Key Files

### Configuration & Infrastructure
- `pipeline/config.py`: `FaceSwapConfig` dataclass, observable pattern (source of truth for all settings)
- `pipeline/events.py`: `EventBus`, event type constants (inter-module communication backbone)
- `pipeline/logging.py`: Structured logging with event emission (debugging & monitoring)

### Services (ML/CV Models)
- `pipeline/services/face_detection.py`: `FaceDetector` wraps InsightFace
- `pipeline/services/face_swapping.py`: `FaceSwapper` ONNX model orchestration
- `pipeline/services/enhancement.py`: `Enhancer` face restoration, CodeFormer (ONNX) or GFPGAN backend
- `pipeline/services/masking.py`: `FaceMasker` landmark hull + optional XSeg occlusion
- `pipeline/services/face_tracking.py`: `LandmarkStabilizer` EMA on kps/106 landmarks, resets on identity change
- `pipeline/services/database.py`: `FaceDatabase` embedding cache, averaging, `review_sources`
- `pipeline/services/guards.py`: Source and runtime input guards, threshold validation

### Processing Pipeline
- `pipeline/processing/pipeline.py`: `ProcessingPipeline` orchestrator (batch & stream modes)
- `pipeline/processing/frame_processor.py`: `FrameProcessor` ABC + 4 implementations
- `pipeline/processing/compositor.py`: `FaceCompositor` aligned-space compositing

### I/O & API
- `pipeline/io/capture.py`: Input sources (webcam, file, network)
- `pipeline/io/output.py`: Output sinks (file, HTTP, WebSocket)
- `pipeline/api/server.py`: WebSocket API server, auto-stop timer
- `pipeline/api/handlers.py`: Command dispatching & business logic (`keep_alive`, `set_enhance`, etc.)

### Entry Points & Config
- `pipeline/core.py`: CLI argument parsing, headless orchestration
- `pipeline/stream.py`: Stream mode convenience wrapper
- `.flake8`: Linting configuration (E3, E4, F only)
- `mypy.ini`: Type checking (strict mode)
- `.github/workflows/ci.yml`: CI pipeline (mypy → flake8 → test)

### RunPod Deployment
- `runpod/orchestrator.py`: CLI tool for managing GPU pods (start, resume, stop, terminate, status, gpus, datacenters)
- `runpod/startup.sh`: Pod setup script (ffmpeg, venv, pip install)
- `runpod/TROUBLESHOOTING.md`: Detailed log of every RunPod API gotcha and fix
- `RUNPOD_DEPLOYMENT.md`: Setup and operation guide — one-time account/volume/`.env`
  setup, the command set, what each `start` phase does, and the full `.env`
  reference. `tests/test_wiring.py` asserts it stays in step with the code

## RunPod Orchestrator

### Commands
```bash
python runpod/orchestrator.py start        # deploy fresh pod → setup → pipeline → update .env
python runpod/orchestrator.py resume       # resume stopped pod (RUNPOD_POD_ID)
python runpod/orchestrator.py stop         # pause pod (volume preserved)
python runpod/orchestrator.py terminate    # delete pod (network volume survives)
python runpod/orchestrator.py status       # show pod state + URL
python runpod/orchestrator.py gpus         # list GPUs with VRAM, pricing, eligibility
python runpod/orchestrator.py datacenters  # list all datacenters
```

### How It Works
- `start` always creates a new pod; `resume` resumes an existing one
- **Multi-datacenter fallback**: `RUNPOD_DATACENTERS=DC1:vol1,DC2:vol2` — tries all GPUs in DC1 first, falls back to DC2 with its paired volume. Network volumes are datacenter-local, so each datacenter needs its own volume.
- Legacy single-datacenter config (`RUNPOD_DATACENTER_ID` + `RUNPOD_NETWORK_VOLUME_ID`) still works as fallback
- **GPU auto-discovery**: By default, queries RunPod API for GPUs matching `RUNPOD_MIN_VRAM` (default 16GB), `RUNPOD_MAX_PRICE` (default $1.00/hr), and architecture compatibility, tries cheapest first. Set `RUNPOD_GPU_TYPES` to override with specific GPUs.
- **Architecture filtering**: GPUs whose compute capability exceeds the image's PyTorch/ONNX support are automatically excluded (e.g. Blackwell sm_120 GPUs are skipped when the image only supports up to sm_90). Controlled by `_MAX_SUPPORTED_COMPUTE_CAP` in `orchestrator.py` — update when the base image upgrades.
- GPU display names (e.g. `RTX 4090`) are resolved to API IDs via GraphQL
- SSH uses RunPod's proxy: `{podHostId}@ssh.runpod.io` (podHostId from GraphQL `machine.podHostId`, NOT from SDK `get_pod()`)
- WebSocket uses RunPod's proxy: `wss://{pod_id}-9000.proxy.runpod.net/ws`
- Only port `9000/tcp` is exposed (no 8888 — that triggers slow JupyterLab init)
- Image must be `devel` tag — `runtime` tag doesn't exist for `runpod/pytorch`

### Critical API Notes
- `runpod.create_pod(gpu_type_id=...)` needs the GPU **ID** (e.g. `NVIDIA GeForce RTX 4090`), not display name
- `runpod.get_pod()` does NOT return `machine.podHostId` — must query GraphQL directly for SSH username
- RunPod SSH proxy silently drops commands sent via `exec_command` — must use `invoke_shell()` for interactive sessions
- RunPod GraphQL does NOT support schema introspection or per-datacenter GPU filtering
- `support_public_ip=True` severely constrains pod scheduling — only enable for SSH mode
- Never pass both `volume_in_gb` and `network_volume_id` to `create_pod()`
- **Auto-stop**: Pipeline stops the pod after `RUNPOD_MAX_UPTIME` minutes (default 120) to prevent billing overruns. Sends `auto_stop_warning` event 5 minutes before. Desktop shows a dialog; user can click "Extend" (sends `keep_alive` command) or let it stop. Works even with no desktop connected — the pipeline calls `runpod.stop_pod()` directly.
