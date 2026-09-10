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

### Start here in a new session

**Identity is now measurable, and nothing has been measured yet.** That is the
newest and largest queued item — see
**[docs/IDENTITY_WORK.md](docs/IDENTITY_WORK.md)**. The reported complaint was
that the output resembles the source "good, but not very good", raised about the
swap model; nothing here could have said whether the swap model was the cause,
because identity was the one realism quantity nothing measured. Six things were
built for it and **every one is unjudged on footage**: `identity_probe=N`
(ArcFace cosine at five points in the chain, naming the stage that cost most —
restoration at `enhance_strength` 0.7 is the prime suspect);
`tools/identity_probe.py` (sweep a lever against one still, no pod);
`hififace_unofficial_256` (the only registered model that moves the face
*contour*); `mask_shape_growth` (stops the target's landmark hull clipping that
contour off); `identity_push`; `complexion_keep` (the one item the cosine cannot
judge); and `source_blend` — sweep that one **only** with `--holdout`, or every
strategy wins. Run them in the order docs/IDENTITY_WORK.md ends with: **measure
the current default first**, then the free levers, then the models.

**A seventh was added 2026-09-09, and it changes what the first step reads.**
The complaint above was raised about the swap model; a separate observation off
footage says results are much better when the source and target **head shapes**
are similar and degrade when they are not. Nothing measured that either, and
the existing instrument could not — ArcFace is trained to be invariant to most
of that geometry. `pipeline/services/shape.py` now reports `shape_mismatch`
(how far apart the two heads are, a property of the *pairing* that no setting
moves), `shape_shift` and `outline_shift` (**0 kept the target's head shape, 1
took the source's**), on the same `identity_probe=N` interval and in the same
REALISM block and `tools/identity_probe.py` table. It is verified on synthetic
shapes and **unmeasured on real footage**. Read `shape_mismatch` before
anything else: under ~0.02 the heads are close and shape is not the cost. See
"Head shape" below.

Two other things are queued, in this order because the first is free.

**1. The pod run — half done, and the half that remains is the half that
matters.** Run 2026-09-05 against the Denmark 4090. *Settled:* the uplink test
(OPERATING_ENVIRONMENT.md §2), the `REALISM` block (the detail clamp binds on
**0%** of 2267 frames, so the cheap lever does not exist and the texture work was
necessary), both new layers' cost on a real pod CPU (texture 1.3ms, scatter 0.3ms
marginal, inside a 25.9ms frame against 66.7ms), and `gpen_bfr_256` composited at
7.7ms. *Outstanding, and it is the whole point:* **nobody has looked at a face.**
That needs §2.4 — a live stream from the operator's own camera, two to three
minutes per preset, with `--debug-frames`. Mind the `guard_min_frame_px` trap in
§2.3b when running `fast`. Note what statistics cannot settle by construction:
**texture at 0.3 and 0.5 read identically on every metric**, because texture is
added in frame space inside `_paste`, after `_match_detail` has run in aligned
space. Only footage separates them.

**2. Nothing else.** [docs/PERFORMANCE_AUDIT.md](docs/PERFORMANCE_AUDIT.md) §3 is
**done** — 0.74ms saved at `optimal` and 6.77ms at `production`, *net* of a
correctness fix in §3.7 costing ~3.3ms. **§4 is measured and closed**: 0.47ms at
`fast`, ~1.3ms at `optimal`, ~4.6ms at `production` — 0.7% and 2.6% of budget on
the live presets. Revive it only if `production` becomes a live target. One thing
from that measurement is worth carrying: **`fast` is not cheaper than `optimal`
in the compositor.** At a seated distance both clamp to the same aligned size
(`_ALIGNED_MIN` is 128; a 76px and a 101px face are both under it), so smoothing,
colour, detail and scatter do byte-identical work at either preset. What `fast`
buys is uplink, a smaller detector input, and XSeg off — none of it compositor
work, which is why the `fast` uplink test in §2.3b has only ~5-6ms of compute
saving to subtract. §5 is the torch port and the video codec, both decided and
both gated on the pod's own report.

**Next actions live in [docs/PENDING_WORK.md](docs/PENDING_WORK.md)** — a runbook
from starting the pod through to the outstanding implementation work.
[docs/TODO.md](docs/TODO.md) is the backlog, and
**[docs/ACCEPTED_RISKS.md](docs/ACCEPTED_RISKS.md)** records what is knowingly
wrong and why — including the unauthenticated WebSocket, which must close before
a paying customer. Read it before assuming a gap is unnoticed.

- Working: realtime stream, aligned-space compositing, Vast.ai deployment,
  desktop LIVE mode, batch **image** and **video**, **photo mode** (up to four
  uploaded targets), **template targets**, desktop VIDEO and IMAGE tabs
- No templates are bundled yet — the machinery runs against an empty library.
  The assets are a content decision, including licensing for this use
- Not exposed: most realism knobs are API/CLI only. The exception is the
  **TUNING** strip in the viewport — `texture_strength`, `diffuse_strength`,
  `texture_band`, plus BYPASS. It is an instrument for deciding defaults, not a
  consumer control, and what earns a place is a value that **cannot be settled by
  reasoning**. For `texture_band` the stated reason — that it follows the
  operator's face size and mark strength — **was measured and is wrong**: swept
  on two people with very different face sizes and mark strength, the curve has
  the same shape and the same optimum. Its real justification is a trade nobody
  has priced: from band 2.0 to 4.0 the mark contrast delivered rises 14% while
  the share of the map above `_SCATTER_SIGMA` — shading, which belongs to the
  source photograph's lighting rather than the target's — rises from 27% to 38%.
  More of their marks, and more of the wrong light with it. A footage question,
  which is why it keeps a slider. **1.0 is simply wrong** and is not the low end
  of a range: marks arrive with their middles cut out (freckle contrast 3.5x
  plain skin against 9.2x at 2.0), and it silently makes `texture_relief` and
  `texture_contrast` inert. Those two stay on `set_realism`; so does
  `texture_strength` above 1.0, a diagnostic overshoot rather than a slider
- Batch video is wired but has only been exercised against a stubbed swapper —
  FFmpeg plumbing, frame ordering, audio sync, cancellation and cleanup are
  verified; a real run with the models in the loop has not been done locally
- Large-file transfer is still the gap for **video** targets. Photos sidestep it
  (`upload_target`, <=4, <=6 MB each, base64 in one message per job), and are so
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

Read the **frames processed per run**, not the p95 column — `LatencyBudget` was
never reset between streams, so every p95 in that session was diluted by its
predecessors. Fixed since (`LatencyBudget.reset()`, from `_run_stream_impl`);
sweeps taken before the fix have to be read this way.

| config | frames in its 60s | ms/frame |
|---|---|---|
| baseline | 1018 | 58.9 |
| **no restoration** | 3381 | **17.7** |
| **`restore_min_face=200`** | 3341 | **18.0** |
| `aligned_size=128` | 1041 | 57.6 |
| hyperswap_1a_256 | 965 | 62.2 |
| hyperswap + no restoration | 3029 | 19.8 |

**Restoration off is 3.3x, and it HOLDS the deadline** — and `restore_min_face`
lands on the same number, which is the cross-check it was built for: the
shippable config-level lever reaches the same floor as switching the stage off.
Two questions closed, both negative. **`aligned_size` is not the cost.**
**hyperswap was slightly worse, not better** — the 256px swap costs 2-3ms and buys
back nothing in restoration time. Its appearance remains unjudged. Full working
in [docs/PERFORMANCE_AUDIT.md](docs/PERFORMANCE_AUDIT.md) §11.

### Does restoration actually help a 101px face?

Measured on the recorded webcam clip, 24 frame pairs at identical indices,
restoration off against CodeFormer at 512:

| `tools/compare_frames.py` (ideal = 1.00) | off | CodeFormer 512 | change |
|---|---|---|---|
| high-frequency detail, face / frame | 0.584 | 0.614 | **+0.030** |
| sensor noise, face / frame | 1.500 | 1.500 | **+0.000** |
| gradient at mask edge | 1.028 | 1.038 | +0.010 |

**29.4ms buys +0.03 on the one metric it moves at all**, which is what the
geometry predicts: 86% of what CodeFormer produces is discarded at
`compositor.py:515`, one warp after it is created. These are per-frame image
statistics over 24 frames — they say nothing about **temporal** behaviour, and
shimmer is a large part of what reads as AI, nor are they a person looking at a
face. The 1.50x noise row appears in **both** configurations, so it is not caused
by restoration; that clip was also poorly lit, with a fair, well-lit source
against a dark, under-lit, noisy target, which is the hardest case for colour
matching. Re-measure on better-matched footage before treating grain matching as
overshooting. Full working in docs/PERFORMANCE_AUDIT.md §11.

### Restoration models, benchmarked in isolation (RTX 4090)

Raw inference only, 100 runs, no compositing: codeformer 512 **29.4ms**,
gpen_bfr_512 512 **37.5ms**, gpen_bfr_256 256 **5.4ms**. Note which way
`gpen_bfr_512` falls — **slower** than CodeFormer at the same resolution. The
saving is entirely **resolution**, not architecture.

**`gpen_bfr_256` has now run in the pipeline** (2026-09-05, RTX 4090, 2267
frames): `restore` p50 **7.7ms** inside a 25.9ms frame. It still **has not been
judged on footage**, which is what decides whether it belongs. Both GPEN files
are on the volume at `/workspace/models/`. Why 256 rather than off: restoring at
512 into a 128-192 aligned space is *supersampling*, and at 256 that margin
survives along with all the low-frequency work the downsample does not destroy.
See "Face restoration" below for the registry and what changes without a
fidelity weight.

### The bottleneck has moved to the transport

With restoration at 256 the frame is ~27ms against a 50ms deadline, and the
reported symptom changed shape with it: **the stutter went away and the lag did
not**. Stutter is throughput; lag is latency, and halving the compute did not
move it. **None of it was visible** — `RTTTracker` had computed true
glass-to-glass latency all along and nothing displayed it.

Fixed, in order of what they cost:

- **The readout exists.** `Bridge.latencyText` publishes RTT p50/p95, buffer
  depth and uplink Mbps every two seconds, top-right in the viewport — never
  drawn on the frame. Read it against the pipeline's own per-stage report: **the
  difference between the two is network and encode.**
- **The inbound queue dropped the wrong frame.** On a full queue the handler
  refused the *arriving* frame and kept the backlog, so the face lagged by the
  whole queue depth and stayed there. It now evicts the oldest, and depth went
  10 -> 2.
- **The playout buffer started at 400ms** and converged slowly. Now 120ms
  initial, a 50ms floor, and asymmetric smoothing — see
  [docs/PLAYOUT_AND_SYNC.md](docs/PLAYOUT_AND_SYNC.md).

**Still only a hypothesis: the uplink.** ~4.8 Mbps up at `optimal`, on home
connections that are usually asymmetric. A saturated uplink queues frames in the
OS send buffer, which reads as **latency while throughput still looks healthy**
— the exact reported symptom. The cheap test is to drop to `fast` and see whether
latency falls by far more than the ~10ms of compute that saves.

**The term that dominated everything: distance — and it is what the move to Vast
was for.** EU-RO-1 against an operator in West Africa is a physical floor of
80-120ms round trip at best. A nearer RunPod datacenter does not exist: EU-FR-1
carries no eligible GPU, EU-NL-1 a single L40S, and RunPod has no UK datacenter
at all. Vast has verified 4090s in the UK at $0.31/hr on 885 MB/s uplinks —
[docs/VAST_MIGRATION.md](docs/VAST_MIGRATION.md).

**The standing conclusion: stop optimising the pipeline for latency.** There is
~23ms of headroom under the deadline and the felt delay is dominated by terms
the GPU does not touch. Further compute work should be justified by the readout
showing compute as the largest term, which it currently is not.

### Playout is measured once, then fixed

Both streams are presented **D after capture**, whatever the network did in
between. D is measured from the link over the first full RTT window (~3s),
committed once, and then held (`_maybe_calibrate`; `PHANTOM_PLAYOUT_DELAY_MS`
pins a value, and `0` there restores adaptive behaviour). Adaptive playout is
right for video alone and becomes wrong the moment audio is played against the
same number, because every adjustment is a discontinuity. **Jitter is far more
damaging than delay.**

The rules that are easy to get wrong, in `JitterBuffer.next_for_slot`:

1. **The slot fires on time, always** — audio is locked to the same clock.
2. **A frame that missed its slot is discarded, not shown late.**
3. **An empty slot repeats the last shown frame** — always the last *swapped*
   frame, never the raw camera, never black.

**Both streams read one accessor**, and that is not decoration: video once read
the adaptive estimate while audio read the fixed value, so the sound trailed the
lips by the difference, and escalation made it worse rather than better.
`[SYNC] av video=+Xms audio=+Yms skew=±Zms` now reports what each stream
**actually presented**; a negative skew is the sound behind the lips.

**A repeat is not the evidence to step D on** — the display ticks at 30/s while
`optimal` streams at 20fps, so a third of slots have no new frame due on a
perfect link, forever. The signal is **starvation**: a slot with nothing eligible
*and nothing waiting*. >20% of slots starving over ~10s, while frames are
actually arriving, steps D up by 100ms, once, and says so. That is the only place
playout adapts, and it adapts on evidence rather than per frame.

Full record — the four one-directional audio timebase errors, the calibration
warm-up rule, `MAX_FRAMES` capacity, `delay_epoch`, and why only video crosses
the network — in [docs/PLAYOUT_AND_SYNC.md](docs/PLAYOUT_AND_SYNC.md).

### Session gotchas worth not rediscovering

- **The first stream after a pipeline start pays model warm-up**, tens of
  seconds, inside its own window. A 40s capture produced zero frames for this
  reason and looked like a broken config. The sweep hides this with a discarded
  warm-up pass; anything else driving the stream needs its own.
- **Nothing can be copied off the pod** under the old proxy. `orchestrator.py
  push` was local->pod only, port 9000 the only opening, and the SSH proxy
  carried no SFTP — so a 45 KB montage of comparison frames could not be brought
  home. Vast's `ssh_direct` is what makes `pull` possible, and visual review
  routine rather than impossible.
- **`orchestrator.py run` used `PATH=... <cmd>`**, which binds only to the first
  word of a line, so the second half of any `&&` chain ran under
  `/usr/bin/python`. Now `export PATH=... && <cmd>`.

**Settled, and still true:** every model is confirmed on
`CUDAExecutionProvider`, with no silent CPU fallback. **`cuda_graphs` and
`cuda_streams` measured flat** (144.4 / 146.1ms — noise); a 110ms model is not
waiting on kernel launch overhead, so both can be dropped. **Numba is closed** —
the whole compositor is ~20ms, and making it free still left 126ms. And **`fp16`
and `trt` never actually ran**: no converted weights existed, and the conversion
still fails its own check (see "ONNX sessions" below), so do not read either
row of any sweep as evidence.

The plan that session wrote for itself, and the L4-based frame-rate estimates it
rested on, are archived in docs/PERFORMANCE_AUDIT.md §12 — superseded by the
4090 measurements above, and kept only for the lesson that the reasoning about
*where* time goes held while every prediction of *how much* did not.

### Why 110ms: restoration ignores how big the face is

**The dominant cost is spent on interpolated data.** In the measured session the
face was **101x129 px** in a 640x360 frame:

    face in frame          101 x 129   <- the only real information
    swap native            128 x 128   <- inswapper_128 output, the ceiling
    aligned space          256 x 256   <- follows face size, has a floor
    FFHQ restore crop      512 x 512   <- ALWAYS 512, regardless

`CROP_SIZE = 512` is hard-coded through `_ffhq_geometry` and `_build_ffhq_crop`,
so a 101px face is upsampled ~20x in pixel count, the heaviest model runs on the
result, and the output is squeezed back into a 101px hole. `_aligned_size`
already holds the correct principle — the working resolution "is not upsampled to
a detail level their webcam never captured" — and governs a stage costing a few
milliseconds while the 110ms stage ignores it entirely.

**Option 3 — a re-export at 256 — is closed for CodeFormer**, which declares
static `[1, 3, 512, 512]`. So `restore_size` cannot make *that* model restore
smaller; `Enhancer.crop_size` warns once and holds at 512, which is the declining
path working as designed. **`restore_min_face` is the only config-level lever**
against the 39.5ms, and needs no new model because skipping is free.
`gpen_bfr_256` answers the same question by being a different model.

The general lesson is cheaper than the sweep that would have found it: **read a
model's declared input shape before sweeping a shape lever.** Two properties keep
that lever honest — `restore_size` is a request the model answers
(`_spatial_size` reads the declared shape at load, and warns once rather than
throwing per frame), and `_FFHQ_ERODE` / `_FFHQ_FEATHER` reproduce the old erode
and sigma exactly at 512, so a 256 crop gets the same *seam* rather than twice as
hard an edge and a resolution A/B is not also a feathering A/B.

Not the problem, so it does not get optimised by mistake: transfers are trivial
(a 512x512x3 fp32 tensor is 3MB, ~0.1ms over PCIe 4.0) and the Python layer does
not appear in the measurement. Full working in docs/PERFORMANCE_AUDIT.md §11.

### GPU compositing, revisited with numbers

Worth doing, but second-order. The whole compositor is ~20ms — 14% of the frame
on an L4, but ~28% on a 4090, since it is the one stage that does not scale with
the card. The route is **torch, not `cv2.cuda`**: torch is already a GPU
dependency, `affine_grid`/`grid_sample` cover the warps and elementwise ops
cover colour, detail and grain, whereas PyPI OpenCV has no CUDA build. Expect
~20ms to become single digits.

Do it *after* the restoration resolution question, not before: 20ms is worth
having, 110ms is worth having first.

### Refusing a slow card, and waiting instead

Auto-discovery took the fastest *available* card, which silently accepts a weak
one when the fast one is busy. That is not hypothetical: a whole measurement
session was spent on an L4 by accident, and the card turned out to matter more
than every software lever combined.

`start` therefore applies a **speed floor** and, when nothing clears it, waits
and retries the whole search every minute rather than dropping down.

The floor is no longer a hand-typed ranking. Vast publishes **`dlperf`**, a
measured score on every offer, so `tests/test_gpu_tier.py` and the `_GPU_PERF`
table it pinned are both gone.

| Setting | Default | Effect |
|---|---|---|
| `VAST_MIN_DLPERF` | `90` | Measured in western Europe: RTX 6000 Ada 113, RTX 4090 97, RTX 5080 84, RTX 3090 45. So 90 is "4090 or better" |
| `VAST_GPU_WAIT` | `300` | Seconds to keep retrying before giving up |
| `VAST_RELAX` | `true` | Relax in bounded steps when nothing matches — CPU, then GPU, then both. **Every rung still holds 20fps**; the ladder stops rather than renting hardware that cannot deliver |

**A bounded wait is not a pin: billing starts when an instance runs, not while
you are waiting for one**, so the wait costs nothing and the thing it avoids
costs a session.

Four properties worth keeping:

- **The preferred host is retried on every pass**, not just the first. It is
  the one whose disk is warm and whose address the desktop already has, so a
  pass that skipped it after one miss would skip the answer.
- **Only capacity is waited out.** `_is_capacity_error` decides, and a single
  non-capacity failure ends the wait — spending five minutes proving a key is
  rejected is worse than saying so immediately.
- **What happens at the timeout differs by purpose.** A measurement session
  should fail rather than accept a slower card, since a comparison across two
  architectures is not a comparison. A customer session should fall back. The
  default is *fail*, because that is the failure this was built for.
- **A pinned host is exempt from the quality floors, but not from the two
  filters about whether the software can run at all.** `compute_cap <= 900` and
  `gpu_arch == nvidia` always apply: a Blackwell card would schedule happily
  against a CUDA 12.1 image and fail after billing started, and every ONNX model
  here needs CUDAExecutionProvider.

**Ordering is by distance, then price — not by speed.** That is only safe
*because* of the floor: once everything below a 4090 is gone, the remaining
offers differ in the two things that decide felt quality, and speed is not one
of them. Sorting on price alone had a French host at $0.336 beating a British
one at $0.350 — three cents to give back part of the round trip the whole
migration exists to remove. `VAST_GEOLOCATIONS` is a **strict** priority list:
reorder it if a nearer host is not worth its price.

Lives in `_search_offers`, `_rank`, `_find_offer` and `_create_instance` in
`vast/orchestrator.py`; `tests/test_offer_selection.py` pins all of it.

## Quick Commands

### Machine setup
**[OPERATING_ENVIRONMENT.md](OPERATING_ENVIRONMENT.md)** — the constraints that
are **ours, not the product's**: the operator's uplink, their machine, and the
measurements that separate a bad link from a bad pipeline. Read it before
concluding anything is broken. It carries the uplink each preset needs, what the
badge readings mean, and a log of things that looked like product faults and
were not.


**[docs/VAST_MANUAL_SETUP.md](docs/VAST_MANUAL_SETUP.md)** — everything on
Vast.ai that no script here can do: account, credit, the two API keys, the SSH
key, and the four ongoing duties the orchestrator cannot take over. Leads with
autobilling, because a zero balance with no card on file destroys the instance
disk — which is the only copy of the venv and weights.

**[docs/SETUP_CHECKLIST.md](docs/SETUP_CHECKLIST.md)** — what has to be
installed **on which machine**, and where every output file lands.

The split is the part that gets got wrong: **OBS Studio and VB-Audio Cable
belong on the operator machine, never the pipeline machine.** The pipeline is
headless — it receives JPEG frames, swaps, and sends them back. It has no
virtual camera, no conferencing app, and no audio path at all, since **audio is
never uploaded to it**; `pyvirtualcam` and `sounddevice` appear nowhere in its
imports or requirements. Installing either driver on a pod does nothing.

The audio one is the easy one to skip and the worst one to skip. The desktop
delays the operator's microphone to match video that arrives ~350-400ms late;
without a virtual output that delayed audio goes to their *speakers* while the
call still receives the real microphone undelayed — so the delay makes the
desync worse rather than better. The app now says so at startup rather than
appearing to work.

**And the same fault reached from the other end was not checked at all until
2026-09-07.** The cable is selected as the microphone *inside* the conferencing
app; the natural mis-reading is to also make `CABLE Output` the **Windows
default recording device**, at which point `AudioCapture` — which took the
system default — read from the same cable `AudioPlayback` writes into. A loop
with no microphone anywhere in it, and a call receiving silence while the
stream, the connection and the virtual camera all look healthy. Found on the
development machine, where the default input measured an RMS of **0.000015**
against the real microphone's **0.0093**, and reported by no part of the app.

`resolve_input_device` now detects it, captures from the lowest-latency real
microphone instead, and says which and why. Resolved *before* the sample rate,
because the rate check compares the two ends and comparing against a device
that will not be recorded from answers the wrong question — on that machine the
cable was 44.1kHz on MME while the output was 48kHz on WASAPI, so the loop came
with a rate mismatch riding along behind it. The fallback is deliberate rather
than a refusal: a VoiceMeeter user routing a real microphone through a virtual
device would rather lose their routing than lose the call, and both cases are
told exactly what happened.

**The machine had both ends of the cable as system defaults at once** — default
recording `CABLE Output`, default playback `CABLE Input` — so the second is
checked too, and reported rather than routed around: this app resolves the cable
explicitly and its audio reaches it either way, so what is broken there is the
operator's machine. On that setting everything the machine plays, the call's own
incoming audio included, is pushed into the pipe the conferencing app records as
its microphone; the operator hears nothing and the other party hears themselves.

Four settings carry the whole path, and the rule under them is that **the
system defaults are the operator's real hardware, and the cable appears only
inside the conferencing app, as its microphone**:

| Setting | Where | Value |
|---|---|---|
| Default **recording** device | Windows Sound settings | the real microphone |
| Default **playback** device | Windows Sound settings | headphones / speakers |
| Microphone | inside the conferencing app | `CABLE Output` — **by name** |
| Speaker | inside the conferencing app | headphones — by name |

**"By name" is load-bearing, and fixing the defaults is what makes it so.**
While the default recording device was the cable, a conferencing app left on
"Default" resolved to the right device by accident — correct for the wrong
reason. Restore the default to a real microphone and that app immediately sends
the real, undelayed voice: audible, badly out of sync with the swapped video,
and sounding like it works, which is worse than the silence it replaced. Full
symptom-by-symptom version in
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### Running the pipeline on your own GPU
**[docs/LOCAL_GPU_SETUP.md](docs/LOCAL_GPU_SETUP.md)** and
`python tools/setup_local_gpu.py`. `requirements-pipeline-gpu.txt` is written
for the rented instance and is not sufficient locally: it installs torch on **macOS only**,
insightface overwrites `onnxruntime-gpu` with the CPU wheel, and cuDNN 9 lands
in site-packages where the loader does not look. Each gap ends in a
working-looking install running silently on CPU, which for an all-ONNX pipeline
is seconds per frame. The script mirrors `vast/startup.sh` and verifies at the
end; `--check` is safe anywhere.

Worth knowing why it is attractive: a local 4090 is not faster than a rented
4090, it is **closer**. It removes the ~350ms round trip that dominates felt
latency, which is the largest single improvement available and one no amount of
GPU work can reach.

### Running
- **Pipeline engine**: `python pipeline.py`
- **Desktop GUI**: `python desktop.py`
- **CLI batch mode**: `python pipeline.py -s <source_image> -t <target_video> -o <output_path>`
- **With CUDA**: `python pipeline.py --execution-provider cuda`

### Development
- **Lint**: `flake8 pipeline.py pipeline desktop`
- **Type check**: `mypy pipeline desktop` — clean, keep it that way (CI runs `mypy pipeline` only)
- **Unit tests**: `python -m pytest tests/ -q` — ~80s, no GPU or model weights
  needed (the ML layer is stubbed in `tests/conftest.py`). 35 modules;
  `test_uplink_governor.py` pins the two rules the adaptive uplink exists to
  keep — the dropdown is a ceiling automation never exceeds, and adapting down
  is fast while adapting up is slow —
  `test_photo_batch.py` covers the photo path, including that a refused photo
  leaves no output file behind, `test_identity.py` covers the identity work —
  including the two claims that would otherwise fail silently: that the
  recognition probe re-frames a crop by the template it was *given* (reading one
  with the wrong template must **not** land on target, or the check is vacuous),
  and that the shape mask is a byte-for-byte no-op when the generated face has
  the target's outline, `test_templates.py` covers the bundled
  library and the face its manifest names, and `test_texture.py` covers the
  source-texture layer — including that its band follows the *target* face's
  size rather than the crop it was cached at, which is the mistake that would
  make the whole layer a silent no-op, and `test_live_exposure.py` pins the one
  rule the product exists for — while a stream runs, a frame leaves the pipeline
  only if it was swapped or is a held swapped frame
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
  the pipeline's status instead. **`.env` is re-read on every boot — `start`
  *and* `resume`** — because the forwarding happens as a shell prefix on the
  pipeline launch command (`_remote_env_exports`), not as docker env at
  creation. What it cannot reach is a pipeline **already running**, which is
  what this tool is for. The older note here said "only at creation", which
  wrongly implied a new rental was needed to change a model
- **On-instance work**: `python vast/orchestrator.py run "<command>"`, and
  `logs [n]` for the pipeline log. Only port 9000 is exposed and the SSH proxy
  drops `exec_command`, so both drive the interactive shell the deploy opens
- **Cold start**: `python vast/orchestrator.py start` prints a phase breakdown
  (provision / setup / pip / model load), labelled warm or empty volume
- **Latency budget**: reported per preset when a stream stops — p50/p95/p99 per
  stage against the frame deadline, with a HOLDS/MISSES verdict
- **Realism readings**: reported beside it under a `REALISM` scope, **when a
  stream stops and when a batch job finishes**. The two used to share one
  early return on the guard telemetry, which only the stream path records, so
  a render or a photo job printed no `REALISM` block at all — every reading
  built to answer "did this layer have anything to spend" was dark on the one
  job shape a still can be studied on.
  `detail_ratio` is the correction `_match_detail` *wanted* before its clamp,
  with the share of frames that hit it — the only way to see a clamped quantity,
  since percentiles of the clamped value cannot exceed the clamp. This was the
  reading that decided how much of the texture work was necessary — if 1.6 bound
  on most frames, part of the detail gap would be the clamp rather than the swap,
  and raising a constant would be the cheapest lever in the project.
  **Measured 2026-09-05: it binds on 0% of 2267 frames.** That lever does not
  exist, so the texture layer was the necessary answer rather than an expensive
  substitute for a one-line change. `texture_headroom` and `texture_confidence`
  say whether the texture layer had anything to spend and whether pose let it
  spend it — measured p50 **0.78** headroom.
  **That 0.78 was read the wrong way round at the time**, as "so
  `texture_strength=1.0` reaches parity". True inside the model and misleading
  in fact: parity had *already* been reached by `_match_detail` one stage
  earlier, so 0.78 was not a budget, it was the rounding error left after
  another stage spent it. See "Two stages, one budget" below. `detail_reserve`
  is the new reading that says whether the fix is engaged
- **Guard calibration**: `python pipeline.py --stream --guard-observe --guard-report r.json`
- **Realism**: `python pipeline.py --stream --debug-frames clip/` then
  `python tools/compare_frames.py clip/ [--against clip2/]`
- **Identity**: `python tools/identity_probe.py -s <face> -t <photo>
  [--sweep field=a,b,c ...] [--save-frames dir/]` — runs the **real**
  compositing path on one still and reports ArcFace similarity between the
  source and the output at each stage, so "which stage lost the likeness" has an
  answer instead of an opinion. Sweeps are a full product across every
  `--sweep`, and `--save-frames` is not optional in practice: a cosine measures
  one axis of three, and a higher number that reads as plastic is not a better
  swap. On a running stream the same readings come from `identity_probe=5` via
  `tools/realism.py`. **The same run now also reports head shape** — see below

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
  - Auto-stop timer: background thread stops the instance after `VAST_MAX_UPTIME` minutes
- **pipeline/api/handlers.py**: Type-safe command handlers; `HandlerContext` dataclass (no globals)
- **pipeline/api/schema.py**: Message types, command/event constants, quality presets

**Simplified Entry Points:**
- **pipeline/core.py**: Argument parsing, headless orchestration; supports `--stream`, `--log-level`
- **pipeline/stream.py**: Stream mode wrapper
- **desktop/bridge.py**: Push-based frame display (no HTTP polling, no 2s status timer)
- **desktop/controller.py**: WebSocket client (`websockets` library, single connection, auto-reconnect); carries the uplink counters, including time blocked inside `send()`
- **desktop/pacing.py**: `FramePacer` — which captured frames are sent upstream
- **desktop/uplink.py**: `UplinkGovernor` — which gear to send them at, from what the link is doing

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

A preset picks **how much uplink and compute to spend. It does not change how
the face looks.** `enhancer_weight` and `enhance_strength` decide whether the
output reads as a real call or as AI, and neither costs anything to compute —
so varying them per preset only meant "production" restored hardest and
therefore looked most synthetic, while presenting itself as the best option.
They are identical in every preset now.

|                         | Fast      | Optimal (default) | Production |
|-------------------------|-----------|-------------------|------------|
| **Capture resolution**  | 480x270   | 640x360           | 640x360    |
| **Frame rate**          | 15 fps    | 15 fps            | 20 fps     |
| **JPEG quality**        | 60        | 60                | 70         |
| **Uplink**              | 1.58 Mbps | 2.45 Mbps         | 3.96 Mbps  |
| **Detector input**      | 320       | 448               | 448        |
| **Compositing ceiling** | 192       | 256               | 256        |
| **Occlusion masking**   | Off       | On                | On         |
| **Landmark EMA**        | 0.7       | 0.7               | 0.6        |
| **Temporal EMA**        | 0.7       | 0.7               | 0.6        |
| **Restore strength**    | 0.7       | 0.7               | 0.7        |
| **Fidelity weight**     | 0.7       | 0.7               | 0.7        |
| **Grain matching**      | On        | On                | On         |

**The ladder is uplink, not compute, and that is a change.** Measured
2026-09-05, West Africa to a Denmark RTX 4090: the pipeline held its 50ms
deadline at p50 **38.8ms** with 7ms of headroom and guarded nothing across 1105
frames. The GPU was never the constraint. What the operator saw as sluggish was
the uplink — the old `optimal` at 3.96 Mbps delivered **84%** of frames, while
`fast` at 1.58 Mbps delivered **91%**, and switching by hand read as "much
smoother" immediately.

So `optimal` now keeps everything that decides how the output *looks* —
640x360, det_size 448, occlusion masking, aligned 256 — and pays in frame rate
and JPEG quality, the two axes that cost uplink without costing detail. That is
2.45 Mbps against 3.96.

**15fps is a floor, not a target.** 12fps was considered and rejected: speech
runs at 4-8 syllables a second, so 12fps gives one to three frames per syllable
and lip movement stutters — on a product whose whole subject is a talking face.
The virtual camera ticks at 30/s and repeats to fill, so at 12fps three frames
in five reaching the call are repeats.

**The old `production` — 960x540 at 30fps, det_size 640, aligned 320 — was
deleted rather than renamed.** docs/PERFORMANCE_AUDIT.md had it at 39ms of a
33ms deadline *before* detection, swap, restoration or encode, so it missed its
budget on the compositor alone; and at q85 and 30fps it asked roughly 11 Mbps
of the one leg that is asymmetric. It was never a usable live preset.
`production` is now what `optimal` used to be, for a connection that can carry
it.

**`fast` keeps occlusion masking OFF, and that was tried and reverted.** The
argument for turning it on is sound on paper - at 15fps the deadline is 66.7ms
against a measured 38.8ms *with* occlusion, and the mask costs nothing on the
uplink. It was switched on and put back within the hour, for a reason worth
recording: `fast` is the gear an operator drops to when the link is failing, and
it is the one configuration measured to hold on theirs. A fallback that has
drifted from what was tested is not a fallback. Revisit it on a good link, as a
deliberate A/B, rather than folding it in alongside other changes.

So `fast` gives up resolution, detector input and occlusion; `production` spends
frame rate and JPEG quality. Nothing that decides whether the output reads as
real varies between them.

The EMA factors vary with frame rate rather than with quality: smoothing across
frames is smoothing across time, so the same factor reaches twice as far back at
15fps as it does at 30. `fast` and `optimal` share 0.7 because they share 15fps.

**Do not delete `fast` until `optimal` has proven itself.** `fast` is the only
configuration measured to hold on the operator's own connection - 91% of frames
delivered against `optimal`'s old 84%. The new `optimal` at 2.45 Mbps is an
estimate that it lands on the right side of that link, not a measurement, and
without a lower gear a bad day has no remedy.

Capture settings live in `PRESETS` and are read by both the pipeline's own
`VideoCapture` loop and the desktop's webcam thread, so local and push mode
cannot diverge.

**Changing quality no longer restarts the capture device**, and that change is
what makes the ladder drivable rather than merely present. See "The uplink
drives itself" below.

**The frame rate is enforced on the send, not on the camera.** A camera is
free to ignore `CAP_PROP_FPS` and Windows Media Foundation does — asked for 20
it reports 20 and delivers 30 — so the uplink was carrying half again as many
JPEGs as the preset's own budget assumes, on the one leg of the path that is
asymmetric.

**Measured again 2026-09-05 on the development machine, and that is not what
happens there** — see OPERATING_ENVIRONMENT.md §3. The camera delivers 15.1fps
whatever it is asked for, MSMF cannot open it at all (the backend resolves to
DSHOW), and so the pacer drops nothing: 15.1 against a 15 target is inside its
own margin. Keep both readings. The point of the pacer is that the camera's rate
is *not a contract*, and a camera that undershoots proves that as well as one
that overshoots — it just makes `production`'s 20fps unreachable, and its real
uplink cost ~2.6 Mbps rather than 3.5.

`desktop/pacing.py` chooses which captured frames to send;
capture itself runs at whatever the device gives, which keeps the local preview
smooth and costs one comparison per frame.

The target is a target, not a ratio: the camera's real rate is measured, and
frames are dropped only while it is actually overshooting. A camera already
delivering 20 has no spare frames, so nothing is discarded — which is also what
keeps ordinary jitter from bleeding frames off a nominally-correct camera.

Setting the rate on the device was measured and rejected. DirectShow *does*
honour the request, by snapping to its nearest mode — 15fps on the test camera
— so asking for 20 delivered less than asking for nothing. The pacer's schedule
advances by whole intervals for the same reason: `now - last_sent >= interval`
aliases a 30fps camera down to 15 rather than 20.

Presets deliberately **do not** set `enhance`: it has an explicit toggle in the
desktop header, and a preset must not silently undo something the operator just
clicked. `color_correction` is left alone for a different reason — it is on and
stays on (see below).

### The uplink drives itself
The ladder was already the most valuable lever in the project and it was driven
by hand: dropping a gear took delivery from 61% to 94% and p50 from 1222ms to
366ms, worth more than every compute lever combined. `desktop/uplink.py`
does it on evidence, within seconds instead of whenever someone notices.

**The dropdown became a ceiling.** `UplinkGovernor` may move below what the
operator chose and back up to it, and never past it. Automation can protect
them from a link that cannot carry their choice and can never hand them
something they did not ask for — and picking a gear still applies immediately in
both directions, because someone who selects one expects to see it rather than
be climbed towards it.

**The precondition was decoupling capture from the preset.** Applying a preset
to the device costs ~3.9s to first frame on Windows MSMF (`_configure_capture`
measured it), so a governor that reconfigured the camera per gear change would
black the call out for four seconds every time it acted — worse than not
adapting. Every gear is 640x360 or smaller, so the device is opened **once** at
`capture_ceiling()` and each gear is a `cv2.resize` of what it already delivers.
Software downscaling is also the better picture: 640x360 resampled to 480x270
supersamples, where asking the camera for 270p does not.

**A gear is uplink only** — resize, JPEG quality, send rate. It deliberately
cannot reach `det_size`, `aligned_size` or `occluder`, because those decide how
the face *looks* and an automatic change must not alter the swap mid-call.
`tests/test_uplink_governor.py` asserts a `Gear` has no such attribute.

**The signal is send-buffer backpressure, not bandwidth.** `websockets.sync`
blocks inside `send()` until the kernel accepts the bytes, so the time the
uplink thread spends in there measures directly whether the link is absorbing
frames as fast as they are produced — microseconds with headroom, tens of
milliseconds when saturated. Two reasons to prefer it to RTT: it is local, so it
reports at the moment of the event rather than one round trip later; and it
cannot confuse congestion with distance. A 350ms RTT to Europe is geography, and
a controller steering on RTT alone would read that floor as a fault and shift
down forever. It was **not measured before** — `PipelineClient` now carries
`send_frames`, `send_blocked_ns` and `send_contended`.

Delivery ratio corroborates it, as a ratio of **rates** rather than of counts:
frames sent in the last round trip have not come back yet, so cumulative totals
under-report by the whole RTT.

Asymmetric, for the same reason `RTTTracker` is — one bad window drops a gear,
twenty seconds of clean ones raise it, with an 8s cooldown because a gear change
alters the very thing being measured. An idle window is explicitly **not**
evidence of health, or the governor would climb precisely when it has learned
nothing.

`PHANTOM_UPLINK_ADAPT=0` leaves it observing without steering — for a
measurement run, for the reason `PHANTOM_PLAYOUT_DELAY_MS` exists: two sessions
cannot be compared if the thing under test chose itself differently in each.

**Both drop-oldest queues now count what they discard.** A frame evicted from
the pod's depth-2 inbound queue never comes back, so from the desktop it was
indistinguishable from one the network lost — and the two want opposite
remedies. Loss means the uplink is over budget and wants a lower bitrate;
eviction means frames arrived in a *burst*, which is what a saturated link does
after each stall, and wants a smoother send schedule. `measure_link.py` reported
39% of frames missing at the old `optimal` with no way to tell which it was.
`get_stats` now carries `frames.inbound_evicted`, and `tools/stats.py` prints it.

**The block-time denominator is measured, not nominal, and that was a
correction.** The fraction asks what share of the time available per frame was
spent blocked, and the gear's own interval is only that figure when the camera
can reach its rate. It cannot here — 15.1fps whatever it is asked — so dividing
`production`'s blocked time by its nominal 50ms rather than the real 66ms
overstates pressure by a third and would shift down on a link that was coping.
`observe(interval_ms=...)` takes the measured one; the gear's is the fallback.

**Measured 2026-09-05, and the premise came out stronger than the thresholds.**
A clean run against the Denmark 4090, order reversed to control for it, gave
`production` 97% / p50 286ms, `optimal` 99% / 276ms, `fast` 100% / 255ms against
a 201ms network floor — every gear holding, the whole ladder worth 31ms.
**The same `optimal` had delivered 61% at p50 1222ms earlier the same day.**

So on that evening the governor would have sat at the ceiling and never moved,
which is correct behaviour and also means its down-shift path is still unproven
against a genuinely saturated link. What the pair of readings does prove is the
premise: the link moved from "cannot carry 2.45 Mbps" to "carries 3.96
comfortably" in hours, with nothing in the repository changing, and **no fixed
preset is right on both**. That is the argument for adapting at all, and it is
now measured rather than reasoned.

One threshold is suspect as a result. `DOWN_DELIVERED` at 0.85 would not have
fired at any gear that evening — correct — but delivery barely separated the
gears at all (97/99/100), while p95 did more work. If the next saturated link
also shows pressure in **jitter before loss**, delivery is the blunter of the
two signals and the block-time fraction is carrying the decision. Read them
apart before retuning either.

**None of this is judged yet.** The thresholds are starting points reasoned from
measured endpoints, not from a sweep of the space between them, and no session
has yet watched the governor shift when a person would have.

### Restoration strength — the one appearance control
A dropdown in the sidebar under QUALITY: **auto / off / subtle / balanced /
full** (`RESTORATION_PRESETS` in `pipeline/api/schema.py`).

This is the ENHANCE toggle's replacement, and the difference is the shape
rather than the placement. The toggle was removed because "off" was never
"less plastic" — it was *no restoration at all*, a 128-native swap dropped into
a sharp frame — a switch across an axis that is not binary. **In a list, `off`
is the bottom of a scale rather than one of two states**, which is what it
actually is. The objection dissolves instead of being worked around.

Named steps rather than a raw 0-1 slider because `0.7` means nothing to an
operator and "balanced" does, and because a support question has an answer.

**`auto` is the default and is what makes the control safe.** It means "follow
this swap model's profile", which is the behaviour that existed before the
dropdown. It exists because `apply_model_profile` sets `enhance_strength` per
model — 0.7 for inswapper_128, 0.5 for the 256-native ones — and the desktop applies a
profile on start, so without `auto` an operator's choice would be silently
reverted by a model change. That is exactly the `set_enhance` mistake recorded
below, and `config.apply_model_profile` now skips the restoration fields when
the preset is pinned.

**It is global, deliberately.** `enhance_strength` governs LIVE, RENDER and
photo output alike, and scoping the control to one of them would make the other
two disagree with the UI. This is the one place the desktop asserts an
appearance default at all — `startPipeline` still pushes only `set_quality` —
and it does so because the operator asked for the control rather than because
the desktop has an opinion.

The values are starting points chosen on the design target, not measured:
full-strength restoration is what makes a swap read as AI, so `full` is named
honestly rather than as the best option. Retune them from footage; that is what
a named step is for.

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

**This was adopted on speed evidence and has still not been judged on footage.**
It has now been composited — 7.7ms per frame over 2267 frames on a 4090 — but
nobody has *looked* at the result. The measured image comparison remains
restoration-off against CodeFormer-512 only.

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
put `enhance: False` in a 256-native model's look profile and call it settled —
a 256-native swap needs less repair than a 128 one, which is the belief the
profile's `enhance_strength` 0.5 already encodes. Two reasons not to. It is an
axis, not a switch, which is the same argument that removed the ENHANCE toggle
from the header; and it is unmeasured — no 256-native model's output has been
looked at with restoration or without. Encoding an untested belief as a hard
rule also removes the ability to test it, since three of the four cells in
{inswapper, alphaface} x {restore, don't} would become unreachable. The graded
version already exists in `enhance_strength`; leave the binary to the footage.

### Source skin texture

The swapped face carries **58% of the frame's high-frequency energy** against an
ideal of 1.00. That deficit is what reads as plastic, and **restoration is not
what causes it**: turning restoration off entirely moves the number by 0.03, 7%
of a 0.42 gap. The detail was never there — the swapper generates at 128 or 256
native and everything downstream resamples that.

So it is taken from where it does exist: the operator's own source photograph.
`pipeline/processing/texture.py` high-passes one source image, stores it in
canonical FFHQ framing, and `FaceCompositor._add_texture` warps it onto the face
every frame. Full reasoning, and the assessment of the design it came from, in
[docs/TEXTURE_PIPELINE.md](docs/TEXTURE_PIPELINE.md).

Four properties carry it:

- **It runs in frame space, inside `_paste`, not in aligned space.** This is the
  one that decides whether the layer exists at all. The compositor works at
  128-320 and `_paste` warps the finished crop down onto a face that is often
  ~100px; a high-frequency field added before that warp is decimated by it —
  pores land under the destination's Nyquist limit and average away. The
  codebase already made this argument about a field with the same spectral
  character: `_add_grain` says grain "would filter into blobs" if it were added
  earlier. **Pores are grain with structure.** Same place, texture first.
- **The band is chosen at the size it will be displayed at.** The *crop* is
  cached at 512; the *map* is derived per working size and memoised, with the
  high-pass sigma scaled against the same 256px reference `_match_detail` uses.
  Both stages therefore mean the same physical detail — one adds to the band the
  other scales, and stages describing adjacent-but-different bands would fight.
- **The band spans octaves, because skin is more than pores.** `texture_band`
  is the span as a multiple of `DETAIL_SIGMA`, and at the default 2.0 it lands
  on `_SCATTER_SIGMA`: this layer owns everything finer than the distance light
  diffuses under skin, the scatter pass owns everything coarser. Surface against
  shading, which is a physical line rather than an arbitrary one. At 1.0 — what
  shipped — the high-pass keeps roughly the finest two pixels, which is pore
  noise and *the rim of everything else*. A freckle, a spot, a mole, a scar or a
  fine crease is 2-8px at working resolution, so most of each one was being
  subtracted as "shape" at extraction. That is the mechanism behind a face that
  measures as textured and reads as smooth.

  The octaves are cut, normalised and weighted **separately** — `texture_relief`
  is the mark octave's share against the pore octave — so this is a pass per
  kind of skin feature, keyed on the thing that actually separates them, which
  is scale. Not on a classifier: telling a mole from a spot on one uploaded
  photograph under unknown light is the unreliable part, and getting it wrong
  would put the wrong gain on real detail. They are summed into one map at
  extraction, so the live path still does **one** warp per frame regardless of
  how many octaves there are.
- **Energy is redistributed toward structure, because RMS is the wrong
  statistic for a mark.** The map is normalised to unit deviation and spent
  against a budget denominated in deviation — and a *sparse* field spends that
  budget badly. A dozen spots on an otherwise flat cheek barely move a second
  moment, so matching second moments scales them down to the level of the dense
  pore noise they sit among. `texture_contrast` expands the amplitude
  distribution before renormalising: same total energy, more of it in the marks.
  Applied to the mark octave only — expanding the pore octave is how a pore
  field turns into speckle, which is the monochrome rule's failure reached from
  the other direction.
- **One source image, not the average.** Every accepted photo feeds the identity
  embedding, because identity is a distributed representation. Texture is not —
  it is spatially localised, so blending maps taken at different angles and focal
  lengths makes misaligned pores cancel instead of reinforce.
  `FaceDatabase.select_texture_source` picks the best on sharpness, size,
  frontality and clipping, all scored during the review that already read and
  detected every image. The weights are starting points chosen on the design
  target, not measured.
- **Bounded by measurement, not by a constant.** `_match_detail` runs in
  aligned space *before* `_paste`, so nothing downstream ever saw what texture
  added and the knob was an open-ended gain — past parity the face becomes
  noisier than the camera that supposedly shot it, which is failure mode 1
  approached from the other side. `_texture_headroom` measures the operator's
  real face in the same pixels (a real face, right size, right lens, right
  light — a better statement of "what skin looks like here" than the background
  could give), subtracts what the swap already carries and what grain is about
  to add, and `texture_strength` is the fraction of what remains. Independent
  fields add in quadrature, so the arithmetic is `f² + g² + t² = r²`.
- **Monochrome, masked to skin, normalised.** Monochrome for the reason grain is:
  independent per-channel high frequency reads as coloured speckle. Masked with
  the landmark hull minus eye, nose and mouth exclusions taken from the FFHQ
  template — reprojected eyelashes over the swap's own eyes are worse than no
  texture. Normalised to unit deviation inside that mask, so `texture_strength`
  means the same thing for a contrasty photograph and a flat one.

**Two stages, one budget — and the one that ran first took all of it.** This is
why the layer read as present-but-soft on the first real footage it was tried
on, and it was not the strength knob and not the source photograph.

`_match_detail` (aligned space) and `_add_texture` (frame space) both aim at the
same quantity: the target face's high-frequency energy. `_match_detail` runs
first and **reaches** it — the clamp binds on 0% of 2267 frames, so parity is
not merely attempted but achieved, every frame. What `_texture_headroom` then
measured was whatever the warp down to frame space happened to lose: p50 **0.78**
of an 8-bit unit against a real face carrying several. At the 0.3-0.5 working
range that is well under 1% of the face's high-frequency energy, which is why
0.3 and 0.5 produced identical readings on every metric — both were invisible.

Worse than the amount is what the amount bought. `_match_detail` can only
amplify the band the swap already has, and that band is upsampled 128-native
output with no structure in it. The face arrived at the correct energy and the
wrong content.

So a share is now held back: `_texture_reserve` asks detail matching for
`sqrt(1 - r²)` of the target's energy and leaves `r` for real skin. Total energy
still lands at parity and overshoot is still impossible — what changes is the
composition. Three properties make it safe rather than merely bolder:

- **It reserves only when the layer will actually run.** A reservation is a
  deliberate undershoot; if texture then declines (no source, pose too far, no
  map at this size) nothing fills the gap and the face comes out *softer* than
  before any of this existed. Every gate `_add_texture` applies is applied in
  `_texture_reserve` first, against the same clamped values.
- **The band travels with the reserve.** Reserving one octave of a field
  measured over three has almost no leverage — measured on a synthetic face in
  exactly the starved regime, it moved the headroom 0.30 to 0.31. So when
  reserving, `_match_detail` scales the same span the map fills. With no reserve
  it splits where it always did, bit-identically.
- **`_DETAIL_RATIO` moves with the target it clamps.** The clamp bounds
  deviation from what the stage is aiming at, and reserving lowers that; a fixed
  floor refused the very attenuation the reservation asked for, quietly keeping
  a quarter of the room at `reserve` 0.8.

**Measured on a synthetic face in the starved regime** (`_match_detail`
unclamped and reaching parity, which is the real condition):

| | headroom | reaching the picture |
|---|---|---|
| as shipped | 0.30 | 0.364 |
| + wider band alone | 0.78 | 0.378 |
| + reserve alone (narrow band) | 0.31 | 0.375 |
| both, strength 0.4 | 0.80 | 0.455 |
| both, strength 1.0 | 0.86 | 0.647 |

Read the attribution honestly: **the band widening is the dominant term and the
reservation is secondary.** Reserving without widening does essentially nothing,
which is the trap the second property above exists to name. And this is a
synthetic fixture — it establishes the mechanism and the direction, not the
magnitude on a real face.

### The reservation was made and never filled (2026-09-07)

The layer was run on a real still — `source/Two`, IMG_3745, a 275px freckled
face — and the obvious freckles did not appear. Not the donor, not the band, not
the picker, not chroma: **headroom came out at exactly zero across the whole
documented working range.** Two defects, both fixed:

- **The grain budget was spent twice.** The target's band is skin *and* sensor
  noise, and `_add_grain` supplies the noise separately in frame space. Aiming
  detail matching at the *total* amplified the swap's own structureless band
  until it covered the noise, then grain added the noise again on top.
  `_match_detail` now discounts the noise once when reserving, on the grayscale
  band so it matches `_estimate_noise` and `_texture_headroom`. Without a reserve
  the stage is bit-identical, so the pre-existing over-parity is untouched and
  stays an open question.
- **`texture_strength` was applied twice**, so delivered amplitude went as its
  *square*: 0.4 asked detail matching to stand back 40%, then filled 40% of what
  that freed, delivering 16%. It is spent once now, in the reserve.
  `f² + g² + t² = r²` still puts the total exactly at parity.

**Then the two stages were made to measure the same thing.** A reserve of 0.2-0.3
still found no room, because the stages read the same face through three
different instruments — pooled per-channel deviation against grayscale, the full
compositing mask against the skin-weighted one, and the whole crop against a
centred 160px window. Together they left detail matching overshooting its aim by
~2.5%: nothing at a large reserve, the entire reservation at a small one. **The
region was the dominant term**, not the mask or the colour convention. When
reserving, `_match_detail` now measures exactly as `_texture_headroom` will.
**The dead zone is gone and the knob is linear from 0.2 up**, at ~1.0ms (aligned
128) to ~1.2ms (256), paid only while the layer is on. **A reservation that finds
no room now says so**, once, naming both numbers — that state leaves the face
*softer* than with the layer switched off, and it was completely silent.

**Chroma was measured and dropped.** Freckles are ~9% of their own ΔE in chroma,
so a colour-carrying texture channel would buy almost nothing. The whole-skin
figure looks far better (chroma 0.58 of luminance) but that is uncorrelated
chroma *noise*, not freckle signal.

**`texture_contrast` and `texture_relief` do work**, and p99 was the wrong way to
ask — scored at the freckles (mean |map| there against plain skin): band 1.0
gives 5.2 at *any* setting of either knob; band 2.0 gives 10.0 at relief 0.65 /
contrast 1.0, **13.5 at the 0.65/1.6 default**, 23.7 at 0.90/1.6 and 28.6 at
0.90/2.4. **At `texture_band` 1.0 both knobs are completely inert**, by
construction — there is no mark octave to act on. The defaults are conservative.

**It is not a freckle layer.** Nothing in it knows what a freckle is; the
delivered map correlates +0.97 to +0.99 with the source photograph's own band. At
band 2.0 it keeps pores, freckles, moles and fine creases at ~2.5x their source
share, wrinkles and scars at ~1.2x, and suppresses shading to 0.36x. **The
exclusions are of kind, not scale, and there are three**: colour (grayscale map,
so a pimple contributes only its dark rim); anything expression-dependent (a
smiling source's crow's foot is painted on regardless — no band setting fixes
this); and three-dimensional relief, which arrives carrying the source's light.
The colour half was measured and **closed** — redness is excluded because it is
*low-frequency*, so an RGB layer recovers 8% of a red spot and 2% of a rash, and
carrying it would mean a low-frequency colour stage fighting `_match_color` over
the one thing the eye reads as skin tone. docs/TEXTURE_PIPELINE.md §6.7 and §6.8.

### Judged on footage at last, and it is net-negative (2026-09-10)

**`texture_strength` currently makes the face smoother, not more textured.**
Swept 0.0 / 0.5 / 1.0 / 2.0 on two stills, alphaface, RTX 5880 Ada. High-pass
deviation on a flat cheek patch, `IMG_3623`:

| `texture_strength` | 0.0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| high-frequency sigma | **4.46** | 3.67 | 2.98 | 3.16 |
| `detail_ratio` | 1.181 | 0.863 | 0.598 | 0.598 |
| `id_out` | 0.761 | 0.755 | 0.738 | 0.723 |

Monotonic **down**, and by eye the face gets waxier at 1.0 and grows painted-on
crease lines at 2.0. The mechanism is visible in the pipeline's own reading:
`_match_detail` stands back exactly as designed (`detail_ratio` 1.181 → 0.598)
and `_add_texture` **does not fill what it gave up**. Even at 2.0 — deliberate
overshoot past parity — it does not recover the strength-0 level.

So this is the failure docs/TEXTURE_PIPELINE.md §15 records as fixed, recurring
by a different route. The 2026-09-07 fix stopped the layer *declining* after a
reservation; it does not stop the layer **delivering less than it promised**.

The cause is named in the tool's own output: the donor is a **369px face,
upsampled** into the 512 canonical crop, so its fine octave is interpolated
rather than photographed — "the band is thinner than it looks". `_SIZE_FULL` is
400px, so this donor is below the size at which the texture term even
saturates. Headroom was **not** the problem: it measured 5.3–6.5 here against
the 0.78 recorded on the live clip, so the budget existed and the map could not
spend it.

**The fix is not a knob.** `_texture_reserve` gates on whether the layer will
*run*; it needs to gate on whether the map can *deliver at the working size* —
an upsampled donor, or one whose native face is smaller than the target's,
should reserve proportionally less or not at all. Until that exists,
`texture_strength` above 0 is a net loss and the default stays 0.

Read `texture headroom` alongside `detail` in `tools/identity_probe.py`: a
falling `detail_ratio` with no corresponding gain is this defect.

**Off by default, and the knob still does not mean what it says.** 0.5 is the place to start; A/B `texture_band`, `texture_relief` and
`texture_contrast` one at a time, since they are separately switchable precisely
so one cannot be blamed for another's artefact. **`texture_strength` reaches 2.0
over the API and the desktop slider stops at 1.0** — above 1.0 deliberately
exceeds measured parity and exists to separate "the map is weak" from "the budget
is small". If marks stay soft at 2.0, read the `Texture source:` line, which
reports the chosen photograph's own pore and mark deviations alongside its face
size.

**Subsurface scatter is built and off** (`diffuse_strength`). Real skin is
translucent, softening *shading* while the texture on top stays sharp; a
generated face reads **hard**, which is a different complaint from plastic in a
different band. It runs in aligned space immediately before `_match_color` so the
colour stages can reconcile it, on the L channel only, with feature exclusions
built from `face.kps` (not the 106 landmarks, whose layout varies by pack) and
feathered, since an unfeathered exclusion is a disc of "sharp" in softened skin.
Two things that look wrong and are not: the blur attenuates part of the texture
band on its way past, but `_match_detail` runs *after* it and scales that band
back against the real crop; and the LAB round trip was kept over the cheaper
monochrome-delta trick because that variant saves 5% and drifts chroma three
times as far.

**Re-examining the CPU work found ~15ms a frame of waste, which was a better
answer than moving anything to the GPU.** One LAB conversion serves both shading
stages; the scatter feature weight is built at a quarter resolution; grain reuses
a cached noise tile at a random offset; `_estimate_noise` bounds its sample
*before* the colour conversion and Laplacian. Net: scatter 3.18ms -> 0.07ms
marginal at 256, grain at a 500px region ~15ms -> 2.83ms. **These are laptop-CPU
numbers and the pod prints its own** — a GPU touches neither layer. Texture is
~1.0ms on a 101px face and ~7.2ms on a 460px one; extraction is 28.2ms once per
source, off the live path. Headroom statistics are taken over a bounded 160px
window because measuring every pixel of a 500px face cost **16.1ms**, more than
the rest of the frame, and was paid even when the answer was "add nothing". Judge
realism at `optimal`.

**The seam came first.** A live run reported the swap as "very noticeable, like
the face pasted on target" — failure mode 2, seen rather than measured, and the
thing the eye finds before it finds texture. Three causes were arithmetic rather
than hypothesis; two are fixed:

- **The transition was ~1.4% of the face's width.** Both feathers were fractions
  of *their own space*, and both spaces are bigger than the face: 5% of a 256
  aligned crop is 0.91px on a 101px face. Neither constant was wrong alone;
  nothing was looking at the product. `mask_feather` now measures against the
  face's own extent, and the ROI pad grows with it, since a blur wider than its
  padding reflects off the border and never reaches zero. On a 100px face:
  **5px -> 10px**.
- **The 50%-alpha line sat outside the face.** `_HULL_EXPAND` grows the hull 10%
  radially *before* the blur, putting the midpoint of the transition on neck at
  the chin and hair at the temples. `mask_erode` pulls it back onto skin first.
  Deliberately not "extend the mask": growing *coverage* puts swapped skin where
  hair should be, which is the worse tell.
- **A convex hull has no concave points** and cannot follow a jawline at any
  expansion. Not yet addressed — phase A3, held back so it is not confounded
  with the two changes above.

The colour deadband went with them: `_COLOR_FLOOR` was 4.0, so a sub-4-unit LAB
mean difference got **zero** global correction and a 10-unit one only half. The
anti-snapping property it was protecting is delivered by the *ramp*, so it only
has to clear estimator noise — now 1.5, with the range 12.0 -> 8.0.

**`compare_frames.py` said there was no seam**, because it measures gradient
*magnitude* — which a 3-unit step over two pixels barely moves — and divides by
the ring *outside* the mask, which contains hair. It now reports
**`seam_excess`**: the LAB step across the boundary in the output, less the step
the untouched input already had at the same rings, so it measures what the
composite *added*. Medians, and blurred first, so grain and hair do not register
as a seam. `seam_ratio` is retained and demoted.

Two things the texture layer deliberately does **not** do:

- **It does not correct pose — it withdraws instead.** Canonical space is a
  similarity transform, so source->canonical->target has identical error to
  source->target in one step; what it buys is that extraction runs *once*. An
  angled source yields a foreshortened map, so `_pose_confidence` scales the
  layer from full at 12° of disagreement to nothing at 45°. Magnitude only — the
  directional term needs `face.pose`'s sign convention pinned against footage
  first, and applied backwards it would attenuate the good half of the face. A
  pack without `pose` gets **full** confidence, never zero: a capability gap must
  not become a silent behaviour change.
- **It does not remove texture swimming**, but pose confidence is the one lever
  against it that is not "turn the strength down". The map's content is fixed, so
  there is no content flicker; fixed content warped by a per-frame affine slides
  across the face as the head turns, caused by *correct* landmark motion rather
  than by noise, so `LandmarkStabilizer` does not address it. Swimming is worst
  where pose has moved furthest from the source — exactly where the confidence
  term takes the detail away.

Full session record, with every table: docs/TEXTURE_PIPELINE.md §15.

### Realism knobs (`FaceSwapConfig`)
| Field | Default | Effect |
|-------|---------|--------|
| `enhance` | `True` | Face restoration on/off |
| `enhancer_model` | `gpen_bfr_256` | Restoration model — see `enhancer_models.py`. The registry owns crop size and fidelity support |
| `enhancer_weight` | `0.7` | CodeFormer fidelity: `0`=most restoration, `1`=closest to input. **Inert on models without a weight input**, which is every model but `codeformer` |
| `enhance_strength` | `0.7` | How much of the restored face to keep. Full strength reads as AI; partial keeps believable imperfection |
| `restore_size` | `512` | Edge of the FFHQ crop fed to the restorer. A model with fixed spatial dims overrides it and says so once — see below |
| `restore_min_face` | `0` | Skip restoration below this face size (px, shorter side). `0` never skips |
| `texture_strength` | `0.0` | Skin detail lifted from the operator's own source photo and warped onto the face each frame. **The fraction of the measured gap to close** — the compositor measures the real face's high-frequency energy in the same pixels, less what the swap and grain already carry, so `1.0` is parity and overshoot is impossible. Reaches `2.0` over the API for diagnosis only; the desktop slider stops at 1.0. **Off by default, never judged on footage.** Linear from 0.2 up; start at 0.5 |
| `texture_band` | `2.0` | How far up in scale that layer reaches, as a multiple of `DETAIL_SIGMA`. `1.0` is pores and the rims of everything else — what shipped, and why marks vanished. `2.0` lands on `_SCATTER_SIGMA`, so texture owns surface and scatter owns shading |
| `texture_relief` | `0.65` | The mark octave's share of the budget against the pore octave, in quadrature. The two are normalised separately, so this is a share of the budget rather than of whatever the photo held most of. `0.0` is the pores-only behaviour exactly, and so is **any value at all when `texture_band` is 1.0** — there is no mark octave to weight. Measured at the freckles, 0.65 gives 13.5x plain skin and 0.9 gives 23.7x |
| `texture_contrast` | `1.6` | Amplitude shaping on the mark octave, at constant total energy. A sparse mark is invisible to a second moment, so an RMS budget flattens it into the dense noise around it; this moves the same energy back into it. Mark octave only — shaping the pore octave makes speckle. Inert at `texture_band` 1.0, for the same reason `texture_relief` is |
| `mask_feather` | `0.04` | Frame-space seam transition, as a fraction of the face's extent in frame (floor 2px). Was effectively 1%, giving a ~1.4px transition on a 101px face — a hard edge, and the reported "pasted on" look |
| `mask_erode` | `0.015` | Pulls the mask in, in aligned space, **before** it is feathered, so the transition sits on skin rather than straddling the expanded hull onto neck and hair. **Halved from 0.03 on measurement** — see "What the mask costs" below |
| `mask_shape_growth` | `0.0` | How far the mask may follow the **generated** face's outline rather than the target's, as a fraction of the face's extent. The mask is a hull of the *target's* landmarks, so the output silhouette is always the target's — free for `inswapper`, which does not move the contour, and destructive for a shape-aware model, which does. Bounded, lower-face only, and self-neutralising. See "Identity" below |
| `identity_push` | `0.0` | Extrapolates the source identity away from the target's in ArcFace space before the swapper is conditioned on it (`src*(1+k) - tgt*k`, renormalised). Every model lands *between* the two faces; this moves the point it aims at. 0.2-0.3 to start, hard-clamped at 0.6 |
| `identity_probe` | `0` | Measure source-to-output ArcFace similarity every Nth frame. Reports `id_swap`/`id_restore`/`id_final`/`id_out`/`id_target` in the REALISM block — **and the `shape_*` / `outline_*` readings**, which cost one further landmark inference. One interval rather than two because a cosine and a shape residual are only interpretable together. Off on a call, `5` for a measurement session |
| `complexion_keep` | `0.0` | Keep this fraction of the source's own skin tone through colour matching. **Chroma only** — luminance is still corrected in full, because a brightness step at the jaw is the most visible seam there is — and bounded at `_COMPLEXION_RESIDUAL` LAB units, so it gives way entirely as the two complexions diverge. The cosine is blind to this one; judge it by eye and by `seam_excess`. Start at 0.4 |
| `source_blend` | `weighted` | How several source photographs become one identity vector: `mean` / `weighted` / `norm` / `best` / `median`. Averaging is a low-pass filter on identity — the average is the most *typical* version of the person — and it is validated for *recognition*, which is not this pipeline's objective. Sweep it **only** with `identity_probe.py --holdout`: scored against the identity it builds, every strategy wins by construction |
| `diffuse_strength` | `0.0` | Subsurface scatter — softens *shading* the way light under skin does, on LAB's L channel only, with eyes/nose/mouth cut out. Answers "the skin reads hard", which is a different complaint from "plastic" and a different band. **Off by default, never judged on footage.** 0.2-0.4 expected |
| `aligned_size` | `256` | **Ceiling** on compositing resolution (clamped 128–512). The size actually used follows the face's own size in frame, in steps, with hysteresis — a distant face is not upsampled to detail its webcam never captured, and costs proportionally less |
| `temporal_alpha` | `0.6` | EMA on aligned pixels, kills shimmer (`1.0` disables) |
| `color_correction` | `True` | LAB transfer, sampled inside the mask, ramped by colour distance |
| `color_strength` | `1.0` | Scales that transfer |
| `grain` | `True` | Matches sensor noise on the composited face |
| `occluder` | `True` | XSeg mask so hands/mics are not overpainted |

### What the mask costs — measured 2026-09-09

First real run of the identity and shape probes. One still, `source/one` (21
photographs accepted of 30), `source/two/IMG_3623.jpg`, alphaface_256 on an RTX
5880 Ada.

**The mask is the largest identity loss in the whole chain, and restoration is
free.** For the baseline configuration: restoration `+0.001`, colour/detail
/texture `-0.006`, **mask and paste `-0.101`**. `gpen_bfr_256` costs nothing in
identity, which retires a standing suspicion about it.

`mask_erode` is the dominant lever, and it costs on **both** axes rather than
trading between them:

| erode | feather | `id_out` | `id_target` |
|---|---|---|---|
| **0.00** | **0.02** | **0.800** | **0.192** |
| 0.00 | 0.04 | 0.790 | 0.211 |
| 0.03 | 0.04 (old default) | 0.713 | 0.293 |
| 0.06 | 0.08 | 0.604 | 0.429 |

Feather is secondary — at erode 0 it runs 0.800 / 0.790 / 0.761 across
0.02/0.04/0.08.

**What the knob actually controls is how much of `_HULL_EXPAND` survives.** The
hull is grown 10% radially and the erode takes a constant number of pixels back
off; at a hull radius of about a third of the crop those coincide at every
working size:

| aligned size | 128 | 192 | 256 | 320 |
|---|---|---|---|---|
| expansion (px) | 4.2 | 6.3 | 8.4 | 10.5 |
| erode 0.030 | 4 | 6 | 8 | 10 | 
| erode 0.015 | 2 | 3 | 4 | 5 |
| erode 0.0075 | 1 | 1 | 2 | 2 |

So **0.03 cancelled the expansion outright**, leaving the mask at the bare
landmark hull — and that expansion exists precisely because "the 106 points stop
at the eyebrows and hug the jaw, so a bare hull clips the swap". The default was
undoing its own correction, and the 0.087 of identity was the bill.

**0.015 is a floor, not a midpoint.** Below it the constant-pixel erode is
dominated by rounding — 0.0075 is 1px at both 128 and 192, a quarter of the
expansion at one size and a sixth at another — so the mask would behave
differently by preset and by how close the operator sits.
`tests/test_identity.py` pins this.

**Not zero**, for three reasons the still cannot test. XSeg gates the mask
*before* the erode, so hair and background are normally excluded — but
`occluder` is **off on `fast`**, the gear an operator drops to when the link is
failing, so there the erode is the only protection left. A turned head pushes a
radial expansion past the visible silhouette into background, and this still is
frontal. And zero leaves no margin for landmark error. Read the 0.800 with that
in mind: `id_out` embeds a crop framed on the face, so more coverage puts more
source-derived pixels in it **whether or not the extra coverage landed on
skin** — the cosine rewards reaching onto hair.

The
frames show the only place the settings visibly differ is the **hairline**,
where a smaller erode lets the smoothed swap reach into fine hair at the
temples. There is **no colour seam at the jaw at any setting** — the colour
stages are doing their job. That still is the *easy* case for the hairline
question, since the subject's hair is tied back; a target with loose hair across
the temple is what would decide between 0.015 and 0.0, and has not been run.

**XSeg costs identity, and compounds with the erode.** Measured the next day on
the same still, `mask_erode` against `occluder`:

| erode | `occluder` | `id_out` | `id_target` |
|---|---|---|---|
| 0.0 | on | 0.791 | 0.215 |
| 0.0 | **off** | **0.803** | **0.184** |
| 0.015 | on | 0.764 | 0.249 |
| 0.015 | off | 0.792 | 0.205 |
| 0.03 | on | 0.719 | 0.295 |
| 0.03 | off | 0.765 | 0.228 |

XSeg costs 0.012 / 0.028 / 0.046 as the erode grows — the two both shrink the
mask and the losses compound. With the occluder off, erode spans only 0.038
against 0.072 with it on.

**But this still cannot price XSeg's benefit**, only its cost: the subject's
hair is tied back, so there is nothing over the face for it to exclude. Read
the table as "what occlusion masking costs on a clean frontal frame", never as
an argument for turning it off. The frame that would answer the other half —
loose hair across the temple, or a hand — has not been run.

**What the frames say that no number did.** The visible defect is not a seam —
it is that the swap is conspicuously *smoother* than the target, which carries
freckles, cheek redness and real skin texture. Failure mode 1, and
`texture_strength` is still `0.0`. At the jaw the discontinuity that does exist
is a **texture** step, not a colour step.

**The shape metric agreed with the eye, which is the validation it needed.**
`outline_swap` `+0.001` says the generator did not move the contour, and in the
frames the jaw and chin outline is unchanged from the target while eyebrows,
eyes, nose and lips are all clearly the source's. `mask_shape_growth` is
confirmed inert on alphaface. Note also that the *frontal* shape reference
raised `shape_mismatch` from 0.099 to **0.142** — the 28° reference had been
understating the true head-shape difference, not inflating it.

### Head shape — the axis the cosine cannot see

`pipeline/services/shape.py`. Built 2026-09-09 from an observation off footage
rather than off a number: **swaps read better when the source and target head
shapes are similar, and badly when they differ.** Nothing measured that, and
nothing could — `identity_probe` reports ArcFace cosines, and recognition models
are *trained* to be invariant to much of the geometry involved. A swap can move
the jawline visibly and shift `id_out` by almost nothing.

The head's outline is also the strongest identity cue a viewer reads, and the
one this pipeline structurally loses in three separate places:

    alignment   a similarity fit has 4 DOF, so the crop the swapper is handed
                is framed by the TARGET's five points and no amount of source
                identity reshapes it
    generator   most models repaint the interior and leave the contour; only
                3D-supervised ones move it
    mask        the silhouette is a hull of the TARGET's landmarks, so any
                contour movement that survived is clipped back off

Two readings, from 106-point landmarks on the source photograph, the target and
the finished frame, each reduced to pure shape:

| Reading | Means |
|---|---|
| `shape_mismatch` | How far apart the source's and target's head shapes are, as a fraction of face size. **A property of the pairing, not of any setting** — this is the quantity behind the observation. Under ~0.02 the heads are close and shape is not the problem |
| `shape_shift` | How far the output moved off the target's shape toward the source's. **0 kept the target's, 1 took the source's**, negative moved away |
| `outline_swap` | The same at the silhouette, measured on the **generated crop** — what the swap model actually produced |
| `outline_final` | And after restoration and the aligned-space stages, before the mask |
| `outline_shift` | And on the finished frame. **The number to act on** |

**The three outline readings are an attribution, and without it the measurement
is inert.** A final `outline_shift` near zero has two causes with opposite
remedies, and they are indistinguishable from the finished frame alone:

    generator produced +0.400, +0.050 survived   -> the mask ate it
                                                    raise mask_shape_growth
    generator produced +0.010, +0.010 survived   -> nothing to clip
                                                    mask_shape_growth is inert;
                                                    only a different model moves this

The middle reading exists because restoration is the third candidate nobody
would suspect — it regresses a face toward its training manifold at
`enhance_strength`, which is a plausible way to lose a widened jaw. The report
names whichever stage took the most rather than assuming the mask; pointing at
the wrong knob is worse than pointing at none. Only the outline is recorded at
the intermediate stages: the whole-face residual there is dominated by interior
features every stage repaints, and no lever is attached to it.

The generated contour exists **only** in the swapper's own crop — the frame
holds the target's face — so `ShapeProbe.landmarks_aligned` reads it there.
Nothing has to be re-derived to compare across spaces: each set is normalised
independently before the fit, so an aligned crop and a frame are directly
comparable, which is the property the invariance tests pin.

Four properties carry it:

- **Similarity-invariant by construction.** The three landmark sets come off
  three different images at three different scales, so each is normalised to
  unit radius *before* the fit rather than the residual being divided
  afterwards. Dividing at the end was tried and ties the answer to whichever
  frame supplied the denominator — an output whose face measures slightly larger
  than the target's then reports a shape difference for that reason alone.
  Pinned: scaling the output 3.7x and rotating it 23° must not move the reading.
- **The outline subset is the convex hull, not an index range.** The 106-point
  layout is a property of the model pack, so "0-32 is the jaw" would be right
  for `buffalo_l` and silently wrong for the next one. Points are ranked by
  distance to the hull of the landmarks — which is *literally* what `FaceMasker`
  fills — and the nearest third taken. A radial ranking from the centroid was
  tried first and is wrong: a face is taller than it is wide, so the outer brow
  ends outrank the chin and the subset drifts off the jaw, which is the one part
  a shape-aware model moves.
- **The outline reading earns its place, measured.** On a fixture where only the
  contour moved to the source against one where only the interior did, the
  outline reading separates them by **0.821** against the whole-face reading's
  0.535. Fitting that measurement on the interior alone reads better on paper
  and measured worse on both halves — 0.841 vs 0.891 when the contour moved,
  and 0.117 vs 0.069 of leakage when it did not — because an interior-anchored
  fit is recomputed between the two residuals, so an interior change moves the
  frame the outline is measured in.
- **Source shape is one photograph, never the average** — but **not the same
  one the texture layer takes.** One pick was tried for both and measured
  wrong: on a real 21-image source set `select_texture_source` returned a photo
  at **-28° of yaw**, because sharpness carries twice the weight of frontality
  there. Right for pores, wrong for a silhouette — yaw foreshortens the face's
  horizontal extent by `1 - cos(yaw)`, 12% at 28°, and `shape_mismatch` reports
  that as head-shape difference because it cannot tell the two apart.
  `select_shape_source` weights frontality 0.65 against sharpness 0.10 and
  scores **yaw and pitch together** (`off_axis`), since a lowered chin
  foreshortens vertically exactly as a turned head does horizontally. Roll is
  excluded — the metric fits rotation away before measuring. Still one
  photograph either way: identity is a distributed representation and averages
  soundly, geometry does not.

**Read `shift`, not the absolutes.** Out-of-plane pose changes a face's apparent
2D shape, so an angled source inflates `mismatch` for a reason that is not head
shape — but it inflates the gap by nearly the same factor, and `shift` is their
ratio. Same argument the identity probe makes for reading its drops.

**Nothing has been measured on real footage yet.** What this unblocks first is
the experiment queued since `mask_shape_growth` shipped and never run, because
until now there was no instrument that could report its result: growth at 0
against 0.08, under `hififace_unofficial_256`, read on `outline_shift`.

### Swap models
`pipeline/services/swapper_models.py` is a registry of swap models, each
carrying both a **spec** (kind, alignment template, native size, normalisation,
URL) and a **look profile** (`enhancer_weight`, `enhance_strength`,
`aligned_min`).

| | inswapper_128 | alphaface_256 | hififace_unofficial_256 |
|---|---|---|---|
| Source input | ArcFace embedding via `emap` | ArcFace embedding, direct | ArcFace embedding through a **converter** |
| Template | `arcface_128` | `arcface_128` (identical) | **`mtcnn_512`** |
| Native size | 128 | 256 | 256 |
| `enhance_strength` | 0.7 | 0.5 | 0.5 |
| `enhancer_weight` | 0.7 | 0.8 | 0.8 |
| `aligned_min` | 128 | 256 | 256 |

**hififace is the one that moves face *shape*.** It is trained with a 3DMM in
the loop — the source's identity coefficients recombined with the target's
expression and pose — so the generator learns to move the face **contour**
toward the source rather than only repainting the interior. Face outline is one
of the strongest identity cues a viewer has, and it is the one thing every other
model here leaves at the target's.

Three things about it that are easy to get wrong:

- **The 3D is training-time.** The export takes an embedding and a crop, nothing
  else. No 3DMM fit at inference, no per-frame reconstruction cost.
- **`mask_shape_growth` has to be on or half of it is discarded.** The mask is a
  hull of the target's landmarks; a contour this model widens is clipped
  straight back off. Measuring it at growth 0 and 0.08 on the same clip *is* the
  experiment.
- **`models-3.1.0`, not `models-3.3.0`.** The tag is load-bearing and the asset
  does not exist under 3.3.0. It also pulls a second 21 MB file — the converter
  that maps ArcFace's embedding space into its own, which runs **once per
  source**, not per frame, and is fed the *raw* vector rather than the
  normalised one.

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

### The rest of the registry, and what each was added to answer

Six more models were registered on 2026-09-08, all faithful to facefusion's
reference integration — templates, normalisation, embedding contract and output
denormalisation were each diffed against it rather than assumed.

| | native | template | source form | notes |
|---|---|---|---|---|
| **`alphaface_256`** | 256 | arcface_128 | **raw** | 98.77 ID retrieval, 24.1ms |
| `ghost_1_256` | 256 | arcface_112_v1 | raw | **Apache-2.0**, the only one |
| `simswap_256` | 256 | arcface_112_v1 | raw | ImageNet input, `[0,1]` output |
| `simswap_unofficial_512` | **512** | arcface_112_v1 | raw | forces 512 compositing |

**alphaface is the one to measure against inswapper.** Its conditioning injects
the source code at *every encoder stage* rather than once at the bottleneck,
which is a mechanism against target leakage rather than a claim about it — and
leakage is the quantity the identity work exists to move. It reports 0.471 CSIM
on pose-hard MPIE against FaceDancer 0.401 and SimSwap 0.180.

**Every convention differs per family and none of them raises when wrong**,
which is why `source_form`, `normalise_source` and `denormalize_output` are
registry fields pinned by `tests/test_swapper_models.py`. ghost reads its
converter's output **un-normalised** while hififace and simswap read it
normalised — same converter architecture, opposite convention. simswap is fed
ImageNet mean and deviation but emits `[0,1]`, so undoing it would tint a
correct crop. alphaface wants the raw vector, whose magnitude is part of the
signal. Each of these produces a blander identity or a tinted face, never an
error, so none would be caught while judging a model on footage.

**The source contract is now per model, not per architecture.** `blendswap_256`
and `uniface_256` were previously excluded on the grounds that an image source
"would break multi-photo averaging, `.npy` embeddings and the identity-outlier
guard" — that was the pipeline deciding which models were allowed to exist. None
of the three survives contact: the guards run at **upload** over every
photograph and are untouched by what the swapper is later handed; averaging is
*inapplicable*, not broken; and an all-`.npy` source is told once rather than
swapping badly. `source_kind` carries it, and an image model gets
`select_texture_source`'s single best photograph — the same pick the texture
layer uses, so pores and identity come from one face. Note `identity_push` is
inapplicable to them: there is no identity space to extrapolate in. Note also
each wants a **different** framing for its source than for its target
(blendswap: arcface_112_v2 at 112 against an FFHQ target), which is why
`source_template`/`source_size` are separate fields rather than reused ones.

**The averaged raw embedding lost its magnitude, and that is fixed.** Averaging
vectors that disagree yields a resultant shorter than any input, so the mean of
raw ArcFace vectors came out **5-20% under** a real embedding's norm — further
under the more photographs were added, and the more the person varied between
them. Uploading a fourth photograph made the conditioning vector *weaker*.
Invisible under inswapper, which divides by the norm and is scale-invariant; a
**different input** to `alphaface` and to the `crossface` converters, which were
fitted on ArcFace's own output scale. The raw vector is now pointed along the
normalised one and rescaled to the weighted mean of the input magnitudes, so the
two cannot disagree about direction. Pinned in `tests/test_identity.py`.

**simswap is registered to be falsified.** The 2026 survey puts it at 0.61 ID
similarity against inswapper's 0.73 and its documented failure is exactly target
leakage — it does not carry the source's face shape. It is here because
`simswap_unofficial_512` is the only 512-native swapper, which is a different
axis from the one it is expected to lose on.

hififace's converter moved to `crossface_hififace` (`models-3.4.0`), which is
what facefusion moved to. Different map, so a hififace result from before is not
comparable with one from after — worth knowing when re-testing it with
`mask_shape_growth` on, which is the experiment that has never actually been run.

Pulling every model is **~3.4 GB**. Pre-seed the one being measured.

**Removed 2026-09-09: `hyperswap_1a/1b/1c_256`**, and it is worth recording
why rather than quietly dropping three entries. It was facefusion's own
default and it lost on all three axes that matter here: tuned to blend well
and *respect the target*, which is the opposite of low leakage; **62.2ms**
against inswapper's 58.9 on a 4090, so slower as well; and judged worst by
eye on real footage. Three 384 MB entries nobody should reach for is a trap,
not an option. The measurements above are kept — they are the evidence.

### Studio swap backends — the non-live path
`pipeline/services/studio_swappers.py` and `pipeline/processing/offline.py`.
A **second registry**, deliberately not a `kind` on the first one. Everything in
`swapper_models.py` returns an aligned crop for `FaceCompositor` to finish;
these return a finished picture and must not be composited at all.

| | media | native | swaps | licence |
|---|---|---|---|---|
| `reface` | image | 512 | **head** | research |
| `ghost_2` | image | 512 | **head** | Apache-2.0 |
| `dreamid_v` | **video** | 480 | face | Apache-2.0 |

Four properties carry it:

- **They bypass the compositor entirely.** Colour matching, detail matching and
  a hull of the *target's* landmarks would put back the target information these
  exist to remove — and a hull mask would clip a head swap back to a face swap.
  **That head swap is the point:** it is the only route past the ceiling every
  ONNX model here shares, since our silhouette is always the target's.
- **They are refused on a stream, before the models warm.** The fastest is
  ~0.6s per image against a 50ms deadline, so a live session would emit nothing
  while the connection, the virtual camera and every badge read healthy.
  `is_live_safe()` is False for all three and `_run_stream_impl` reads it.
- **Subprocess, not import.** Each is a checkout rather than a package and each
  vendors its own diffusion stack — REFace an old latent-diffusion tree,
  DreamID-V a Wan fork needing torch >= 2.4. No single environment satisfies all
  three, so each names its own interpreter and checkout, and only the *command
  line* is depended on, which is the part their own docs pin.
- **A failure writes no file.** Crash, timeout, no output, wrong media kind, or
  an unconfigured backend all return False and leave the destination absent —
  the same rule photo mode already keeps, and deliberately **not** a silent
  fallback to the ONNX path, which would be a result from a model nobody chose.

The source handed over is `select_texture_source`'s pick rather than the
averaged embedding: these take an image, and that picker already scores
sharpness, size, frontality and clipping.

**Nothing here is bundled or verified.** Every path comes from the environment,
all of it is forwarded to the pod (the GPU is there, so a backend configured
only on the operator's laptop is configured on the machine that will never run
it), and no integration has been run against real weights.

### Source contracts — what each model is conditioned on

Declared, never inferred. Seven fields on `SwapperModel` carry it, the values
were diffed against facefusion's reference integration, and
`tests/test_swapper_models.py` pins them against a hard-coded table so they
cannot drift. **Every one of them fails silently when wrong** — a blander
identity or a tinted crop, never an exception — which is the whole reason they
are declared rather than assumed.

The one thing introspected at runtime rather than declared is ONNX **input
names**: exports disagree about what they call things, and a wrong key is a
`KeyError` on every frame. Same for a TRAINED model's crop size, read from its
declared input shape.

**LIVE tier — source side**

| model | source | framing / form | normalise | converter | `identity_push` |
|---|---|---|---|---|---|
| `inswapper_128` | embedding | normed | yes | — | yes |
| `hififace_unofficial_256` | embedding | **raw** | yes | `crossface_hififace` | yes |
| `alphaface_256` | embedding | **raw** | **no** | — | yes |
| `ghost_1_256` | embedding | **raw** | **no** | `crossface_ghost` | yes |
| `simswap_256` | embedding | **raw** | yes | `crossface_simswap` | yes |
| `simswap_unofficial_512` | embedding | **raw** | yes | `crossface_simswap` | yes |
| `blendswap_256` | **image** | `arcface_112_v2` @112 | — | — | **no** |
| `uniface_256` | **image** | `ffhq_512` @256 | — | — | **no** |

**LIVE tier — target side**

| model | target framing | size | mean | deviation | undo on output |
|---|---|---|---|---|---|
| `inswapper_128` | `arcface_128` | 128 | 0 | 1 | yes |
| `hififace_unofficial_256` | `mtcnn_512` | 256 | 0.5 | 0.5 | yes |
| `alphaface_256` | `arcface_128` | 256 | 0 | 1 | **no** |
| `ghost_1_256` | `arcface_112_v1` | 256 | 0.5 | 0.5 | yes |
| `simswap_256` | `arcface_112_v1` | 256 | **ImageNet** | **ImageNet** | **no** |
| `simswap_unofficial_512` | `arcface_112_v1` | **512** | 0 | 1 | **no** |
| `blendswap_256` | `ffhq_512` | 256 | 0 | 1 | **no** |
| `uniface_256` | `ffhq_512` | 256 | 0.5 | 0.5 | yes |

**STUDIO tier** — one contract for all three: a source image **file path** plus a
target file, returning a finished picture. No crop, no framing, no
normalisation; they do their own. `reface` and `ghost_2` take an image target,
`dreamid_v` a video.

**TRAINED tier** — **no source at all.** Target warped to `dfl_whole_face` at
the size read from the export; NHWC, BGR, unsharp-masked first.

Four things the tables make visible:

- **Source framing is not target framing.** `blendswap_256` reads its source in
  `arcface_112_v2` at 112 and its target in `ffhq_512` at 256. `uniface_256`
  happens to use one space for both — which is exactly why they cannot be
  generalised from each other.
- **`identity_push` is inapplicable to image models**, not ignored. There is no
  identity space to extrapolate in; `source_image_blob` never touches `_push`.
- **Image models get one photograph** — `select_texture_source`'s pick, the same
  one the texture layer uses, so pores and identity come from one face.
- **The source guards and `identity_probe` are unaffected by any of this.**
  Guards run at *upload*, over every photograph. The averaged embedding is still
  built for every model, so `id_swap`/`id_restore`/`id_final`/`id_out`/`id_target`
  measure against the same reference regardless of what conditions the model.

### Which models to trust, and on what evidence

Ranked on the stated objective — **low target leakage, high source
preservation** — with the *kind* of evidence named, because "measured
elsewhere" and "measured here" are not the same claim and only one model has
both.

**Read this second, not first.** The audit found the two dominant leak paths are
the **mask** (the silhouette is a hull of the *target's* landmarks, so hair, jaw
outline, ears and neck are never swapped) and **restoration** (`enhance_strength`
0.7, plus a global encode that sees the target's border). Both cost more than
the spread between any two models below. A model ranking is a ranking inside a
constraint that beats it.

**LIVE tier**

| confidence | model | evidence |
|---|---|---|
| **High** | `inswapper_128` | 0.73 ID sim / 96.9% retrieval — top of the open field in the 2026 survey — **and** confirmed by eye on our own footage. The only model with two independent lines agreeing. Its weakness is resolution, not identity |
| **High on mechanism** | `alphaface_256` | Identity injected at *every* encoder stage rather than only the bottleneck — a mechanism against leakage, not a claim about it. 98.77 ID retrieval FF++; best CSIM on pose-hard MPIE (0.471 against FaceDancer 0.401, SimSwap 0.180, HifiFace 0.092). Unjudged here. **The one to try against inswapper** |
| **Moves a channel nothing else does** | `hififace_unofficial_256` | The only model that moves the face *contour*, which is a leakage channel no other model touches. But 0.62 ID sim, below inswapper, and **half of it is clipped unless `mask_shape_growth` > 0**. Test it; do not trust it yet |
| **Low** | `ghost_1_256` | Kept **only** for its Apache-2.0 licence — the one permissively licensed model here, and every other is non-commercial, ResearchRAIL or unlicensed. Not an identity argument. Variants 2 and 3 were dropped: 1.5 GB of untested siblings of a model nothing here rates |
| **Low** | `simswap_256`, `simswap_unofficial_512` | 0.61 ID sim, and its documented failure is precisely face-shape leakage. The 512 is here for resolution, not identity |
| **Unknown** | `blendswap_256`, `uniface_256` | No benchmark worth quoting. Conditioned on one photograph, which could cut either way. `blendswap_256` is 1.6 GB and unmeasured for speed, so its LIVE declaration is an assumption rather than a measurement |

**STUDIO tier**

| confidence | model | evidence |
|---|---|---|
| **High** | `reface` | 98.8% ID retrieval top-1 on CelebA, the strongest identity figure in the open literature, **and** it swaps the head — the only thing that raises the mask ceiling |
| **High on quality, thin on identity numbers** | `dreamid_v` | Best-in-class claims and video-native, so temporal coherence is by construction rather than by smoothing. No independent identity number extracted |
| **Moderate** | `ghost_2` | Head transfer with explicit background inpainting. Least direct evidence on identity of the three |

**TRAINED tier** — structurally the strongest thing here and the least evidenced
in our chain. No generic-face manifold to regress toward, and a wider crop that
moves the silhouette at frame rate. Zero measurements on our footage, and gated
on training that happens elsewhere.

### One rule for every environment file
**A change to one `.env` is a change to all of them.** Keys, section headers and
the order they appear in are kept identical across `.env.example` and every
local `.env`; only the *values* differ. Adding a setting to the example and not
to the machine that runs it is how a lever ends up existing in the code, being
documented, and doing nothing on the one box anybody uses — the same class of
silent gap as a model registry that drifted from the CLI.

Two things follow, and neither is optional:

- **Add the key everywhere, empty, in the same section.** An unset key that is
  present reads as "this exists and I have not set it". An absent key reads as
  nothing at all, and the difference is what someone scanning the file sees.
- **`.env.backup-*` is not a thing.** Backups were kept and went stale — one
  still held the RunPod configuration a whole migration ago — so they were
  deleted and `.gitignore` keeps them out. Git is the history; a file that looks
  like a config and is a fossil is worse than no file.

`tests/test_wiring.py` checks what can be checked from a clean checkout: the
example parses, holds no duplicate keys, and documents every variable the
pipeline reads. Parity with a local `.env` cannot be tested — it is gitignored
and absent in CI — so that half is a discipline, which is why it is written down
here rather than assumed.

### The three tiers, and why they are forced
`pipeline/services/tiers.py`. "Which models can be used on a call" had been three
implicit answers in three places — a speed comment in one registry, a refusal in
the stream loop, and a paragraph here — and an implicit rule drifts.

A tier is **derived from two declared facts**, never written down:

    live_capable    can it hold a frame deadline?
    needs_training  does it need a per-identity artifact before it runs?

| tier | live | training | registry | what it is |
|---|---|---|---|---|
| **LIVE** | yes | no | `swapper_models.py` | 8 general models, usable on a call today |
| **STUDIO** | no | no | `studio_swappers.py` | whole pipelines; RENDER and photo only |
| **TRAINED** | **yes** | **yes** | `identity_models.py` | one model per person, trained elsewhere |

**Two facts, not one enum, because the interesting tier is where they
disagree.** A DeepFaceLab model is *fast* — a small GAN at frame rate — and
unusable until someone has spent hours training it on one face. A speed ladder
would file it beside a diffusion model it has nothing in common with, and hide
the only question that matters about it: not "how fast" but "trained on whom".
`tiers.classify` **raises** for slow-and-trained, so a registry entry declaring
the one meaningless combination is caught at import rather than given a fourth
tier by accident.

**One registry per tier**, so the distinction is structural rather than a field
someone has to remember to check. Enforcement is a single point —
`ProcessingPipeline._clear_for_live`, called **before any model is warmed**, so
a session that cannot work is refused before the pod bills for loading weights.
All three registries answer it, so a model cannot reach a call by being added to
the wrong list, by having a speed comment edited, or through an environment
variable nobody checked. There is no default that lets an undeclared model
through: an unknown tier is refused.

The two refusals are worded differently on purpose, because they call for
opposite fixes — a STUDIO model is **the wrong tool for this job**, a TRAINED
model is **the right tool that does not exist yet for this person**.

### The TRAINED tier
The only registry that is **scanned rather than declared**: the others list
models that exist for everyone, this one lists whatever has been trained for
*these* people, which is a property of a directory. It is rescanned on every
call, because a `.dfm` arrives by being copied in and there is no restart
between training one and wanting it.

`IDENTITY_MODEL` **replaces** the swap model, and **no source photograph is
used at all** — the identity is in the weights, so embeddings, `source_blend`,
`identity_push` and the source guards are *inapplicable* rather than merely
unused. That is the whole reason it is a tier and not another name in
`swapper_models.py`. `guards.NO_SOURCE` correctly stands down for it; nothing
is missing.

It is the only thing here that moves the **silhouette at frame rate**: its
`dfl_whole_face` template is a wider crop than arcface, taking in jaw and
forehead, which every general model leaves at the target's. And it has no
generic-face manifold to regress toward — there is nothing for it to average
into but the one person.

Three runtime conventions differ from every other model here, and each produces
a plausible-looking bad face rather than an error: the tensor is **NHWC**, not
NCHW; it is **BGR**, not RGB; and the crop is **sharpened first** with the
unsharp mask DeepFaceLive trained against. The export returns three arrays and
the **middle** one is the face — reading the first gives a greyscale mask pasted
over the frame, which looks like a catastrophic model rather than a wiring
mistake. The crop size is read from the declared input shape, since these are
trained at 224 to 384 depending on who made them.

Unresolved by design: training is hours to days of GPU time per face, done
elsewhere. That is a product decision about onboarding, not a setting.

### Downloads are a stated policy now
`pipeline/services/downloads.py`, `MODEL_DOWNLOADS` — `selected` (default) /
`all` / `none` / an explicit list.

State the true position first, because it is less bad than it sounds: **nothing
bulk-downloads.** Weights fetch on first use, so a session pulls what its
configuration selects. The one real exception was `vast/startup.sh` fetching
**GFPGANv1.4.pth, 340 MB, on every deploy** — for the alternate restoration
backend, which is not the default and which most sessions never touch. That is
now conditional on it actually being selected.

What was missing is control, not throttling: a way to say "never download, this
volume is seeded" for a metered pod, and a way to pre-seed a chosen set rather
than discovering the cost mid-session. `.env.example` now lists **every model
with its exact size** — all eleven swap models is ~4.6 GB, and `blendswap_256`
alone is 1.6 GB. A refusal is never fatal: it lands on the same degradation path
a missing file already took, and names the variable that would allow it.

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

**On the live path that rule is absolute, and covers cases the guards do not.**
While a stream is running, a frame leaves the pipeline only if it was swapped, or
it is a previously swapped frame held unchanged. There is no third option — no
face detected, no source loaded, occlusion, a compositing failure all hold, and
if nothing has been swapped yet then nothing is emitted at all.

Two of those used to leak the raw camera. `check_frame` passes a frame with zero
faces on purpose — correctly, for batch — and the live path emitted it unswapped,
reasoning that someone stepping out of shot should not leave a stale face over an
empty chair. Sound reasoning, wrong conclusion: nothing distinguishes the two
cases that produce zero detections.

    stepped out of shot            -> 0 detections -> raw frame is an empty room
    light dropped, still sitting   -> 0 detections -> raw frame is their face

The pipeline cannot tell them apart, so the tie goes to the survivable outcome —
a frozen face reads as a network hiccup, a real one cannot be taken back. The
same held-frame path now covers a detected face with **no source loaded**
(`guards.NO_SOURCE`), which is the same exposure with a different cause: the
source failed to load, or was cleared mid-session.

**And more than one face.** `check_frame` already refuses a crowd, so on default
settings the frame never gets that far — but that guard is switchable, by
`guards` and by `guard_multi_face`, and on a live call the consequence of
switching it off is not a quality regression. Outside `many_faces` the detection
list is trimmed to one, so exactly one face is swapped and everyone else keeps
their real face — **including the operator**, if a bystander walks in closer to
the camera and becomes the largest face. So the live path decides this itself
rather than asking the guard config, and a stale `target_face_point` left over
from a photo job is not consulted: that is someone clicking a face in a still
they chose, not permission to swap one face out of two on a call. `many_faces` is
the one real exemption — when all of them are swapped, nobody is exposed.

Deliberately **not** gated on `guard_observe`. That mode exists so a calibration
run can see what a guard would have done while the swap still happens; with no
face there is no swap to let through, so honouring it would mean transmitting the
operator to measure a threshold. **Batch is untouched** — `_swap_frame_detail`
still passes frames through. `tests/test_live_exposure.py` pins all of it.

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

### Filters, backgrounds and effects — the last layers

Three decorative layers over the finished swap, in this order:

    swap  ->  background (replaces what is behind them)
          ->  filter     (regrades the picture)
          ->  effect     (draws on top of it)

A **background** (`desktop/backgrounds.py`) replaces the room — Blur, Blur+, and
five solid colours. A **filter** (`desktop/filters.py`) is a grade — Warm, Mono,
Noir. An **effect** (`desktop/effects.py`) is an overlay — Confetti, Snow,
Hearts, Bubbles, Sparkle.

**The order is not arrangement, it is correctness.** A filter regrades the whole
picture, so a background replaced *before* it is graded along with everything
else; grading first would leave an ungraded background behind a graded person,
which reads as pasted on — the same failure the compositor's colour stages exist
to prevent at the jaw. An effect comes last because it is meant to sit *on* the
picture rather than be part of it; grading confetti would tint it to match a
look it is supposed to be separate from.

Both are shown by one control. Pressing FILTERS shrinks the viewport and reveals
a **horizontal strip** of grades along the bottom and a **vertical rail** of
backgrounds down the right; HIDE gives the space back. APPLY sits beside HIDE
rather than at the end of the chips, since it acts on the whole panel and not on
any one chip.

**The rail used to hold the effects and now holds the backgrounds.** Effects were
never used, and a background is what someone on a video call actually reaches
for. `desktop/effects.py` is **unwired, not deleted**: `_decorate` still applies
an effect key, the bridge still exposes `effectList` / `selectEffect`, and the
module costs exactly zero while no key is set — so restoring a picker is a
one-place change rather than rebuilding a measured layer. Delete it later, with
evidence, or not at all.

Two properties do the work for all three:

- **Last, always.** A filter is applied after the swap has fully composited.
  Grading first would have `FaceCompositor` match the face to an already-graded
  frame and then grade it again; applied last, a filter cannot break the swap
  underneath it. The webcam frame sent *upstream* is deliberately ungraded for
  the same reason — only the local preview gets it.
- **Desktop-side, never the pipeline.** They need nothing the face models
  provide, so they must not compete for a latency budget the swap has not been
  measured against, and changing one should be a local variable rather than a
  round trip to a rented GPU.

**Picking a look engages the panel.** It did not, and that was the bug behind
"clicking any of the vertical options doesn't seem to work" — the chip
highlighted, `_background_key()` still returned `''` because `_filters_enabled`
was false, and nothing distinguished a pending look from a dead feature.
`Bridge._engage` now turns the panel on with the click that chose the look.

This file used to claim the preview showed a look while the call did not, and
that was never true: one accessor feeds the display, the virtual camera and a
saved photo, so all three are on or all three are off. That rule is kept rather
than the audition — an operator looking at a graded preview the far end is not
seeing is a worse failure than one who cannot audition. **ENABLE is therefore a
master switch, not a commit step**: it turns every layer off without discarding
the picks, and each list keeps its own `none`. Picking `none` for one layer does
not disengage, since turning one layer off says nothing about the other two.

Filters and backgrounds default **off**, and should stay off during the pod
session: that session exists to judge whether the swap reads as real, and
anything on top changes what is being looked at.

Measured at 960x540: filters worst 7.5ms (Soft), effects worst 2.8ms (Bubbles),
against a 33ms display tick — and **zero** when nothing is on, since the
undecorated path still lets Qt load the JPEG itself.

**Background is the expensive one: 19.5ms per frame at 640x360**, on a four-core
laptop, and it started at 42.9ms. Two things got it there, and both are worth
knowing before anyone "simplifies" them:

- **The composite is integer.** Converting both pictures to float32 and blending
  with a float alpha is seven full-frame float passes — 12.3ms. `cv2.multiply`
  on uint8 with a scale factor stays in 8-bit SIMD: **3.5ms**, maximum
  difference 2/255, which is inside the grain the compositor already added.
- **The matte is recomputed every other frame.** The model is 28ms of the
  original 43 and cannot be made cheaper: threading past two cores buys nothing
  (45.8 / 28.4 / 28.7 / 27.3ms at 1 / 2 / 4 / 8 threads) and its input edge is
  fixed by the export. Decimation is defensible *here* and not for the swap — a
  silhouette is slowly varying, the mask is already an EMA, and being one frame
  late costs a few pixels of background on a shoulder rather than a face lagging
  its own head. The bill is 66ms of matte lag at 15fps on the fastest movement;
  revisit `_MATTE_INTERVAL` if the edge is seen swimming on quick turns.

A dead net gives up rather than holding its last mask. `Segmenter` nulls its net
on any inference failure, so the failure is permanent, and holding would paint a
frozen silhouette over moving video for the rest of the session.

That last property is why **`_has_decoration()` has to name every layer**. A
layer left out of it does nothing at all whenever no other layer is on, then
starts working the moment one is: silent, no error, and reproducible only by
accident. Same class as the webp mimetype bug below.

That layout is not a style choice. The first version was a separate window with
its own preview, and it was wrong the moment the image tab was open — it showed
the live camera in a mode that has no live camera. A strip has nothing to
preview: whatever the mode was already showing is what a filter is judged
against, so the body is identical in every mode and only the panels move.
`filterStrip.reserved` and `backgroundRail.reserved` are the single numbers both
viewports read, so the body and the panels cannot disagree about the split.

Showing the strip persists across a media-tab switch — it is a preference, not
a detour.

#### What background replacement does and does not buy

It runs on the frame **that came back from the pod**, after the swap, before the
viewport and the virtual camera. Three consequences, and all three are why it is
placed there:

- **Every conferencing app gets the same behaviour.** Some provide a background
  of their own, some do not, and the ones that do disagree about quality. Doing
  it at the virtual camera is one answer everywhere instead of one per app —
  which is the whole reason to build it when Zoom already has its own.
- **The pipeline's measurements are untouched.** `--debug-frames` are written on
  the pod, before this stage exists, so `tools/compare_frames.py` still divides
  the face's statistics by a **real** background. Note what segmenting *before*
  the upload would have done instead: `compare_frames.py` takes "outside" as
  everything beyond a dilated face mask, so a blurred or flat background
  collapses `outside_hf` and `outside_sigma`, the face/frame detail ratio jumps
  from 0.584 to well above 1.0, and every reading taken afterwards is
  incomparable with every reading taken before.
- **It costs the swap nothing** — it spends the desktop's 33ms display tick, not
  the pipeline's 50ms frame deadline.

What it does **not** buy, and must not be sold as: the operator's real room still
travels to the pod, because the frame is segmented after it comes back. This is a
feature for the call, not a privacy measure, and it saves no uplink either.
Moving it ahead of the upload would make it both — and would break the
measurement property above, which is the trade to weigh if anyone proposes it.

**Blur is the forgiving mode and solid colour is the dangerous one**, which is
why the list is ordered that way and why the colours come last. A matte error
under blur puts a few pixels of a *blurred copy of the same scene* against a
sharp person: low contrast, close to invisible. The same error against a constant
colour is a high-contrast fringe, and hair is exactly where a cheap segmenter is
least certain. Judge it on footage by the right question — not "does the
background look good" but **"does the face still read real with it on"**, since
the matte edge is attached to the person's silhouette and a viewer will attribute
an artefact near the head to the swap.

**No new dependency, and one file that is not in the repository.** The desktop
deliberately loads no face model; `cv2.dnn` ships inside the opencv-python that
filters already require and reads ONNX directly, so this adds a **model file**
rather than a package. `python tools/fetch_segmentation_model.py` gets it —
OpenCV's own PPHumanSeg export, ~6 MB, **Apache-2.0**, which is the licence
question worth asking before something ships to paying customers. It verifies by
loading the file and running a frame through, because a truncated download, an
error page, or a **Git LFS pointer** under an `.onnx` name would all leave the
feature quietly dead; the last is not hypothetical, since `raw.githubusercontent`
serves the pointer and only `media.githubusercontent` serves the payload. Absent
the file the layer is a no-op that says once what is missing — the same way the
pipeline degrades without the occluder, rather than failing a live call over a
decorative stage.

`MODELS` in `backgrounds.py` is a registry, the same shape as
`swapper_models.py` and `enhancer_models.py` and for the same reason: **the model
owns the facts about itself.** Input edge, normalisation and channel order are
properties of the weights, not preferences — PPHumanSeg is 192px and wants
`[-1, 1]`, a MediaPipe-style export is 256px and wants `[0, 1]`, and feeding
either the other's convention yields a washed-out matte that reads as a weak
model rather than a wrong input. The output plane is read rather than assumed:
one-channel sigmoid and two-channel softmax are both handled, and guessing wrong
would invert the matte and composite the room over the person.

**One renderer per stream, and that is the load-bearing part.** `_decorate` runs
from the display timer *and* the webcam thread, over two genuinely different
videos. A matte has to be smoothed across frames or its boundary crawls — failure
mode 3, on the longest boundary in the picture — and smoothing is **state**. One
shared EMA would be advanced by both callers at the sum of their rates while
blending two unrelated pictures. `desktop/effects.py` dodges this by being a pure
function of the clock; a matte cannot be, so the state is separated by stream
instead: `_bg_display`, `_bg_webcam`, and a throwaway for a saved still, which
has nothing to smooth and must not disturb either.

Not covered: **RENDER**. A video is written pipeline-side, so the desktop never
holds those frames. Decorating a render needs either a pipeline stage or a local
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
if `libcudnn.so.9` will not load, and `vast/startup.sh` exits non-zero after
installing cuDNN if it still cannot. See vast/TROUBLESHOOTING.md section 5b —
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
`VAST_GEOLOCATIONS`, which exists because availability is the binding
constraint; it would trade "sometimes a slower card" for "sometimes no instance
at all", and on a paid session no instance is the worse failure. Each architecture pays
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
  `--aligned-size`, `--restore-size`, `--restore-min-face`, `--temporal-alpha`, `--color-strength`, `--texture-strength`, `--texture-band`,
  `--texture-relief`, `--texture-contrast`, `--no-enhance`,
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
  `flake8 pipeline.py pipeline desktop tests tools vast firebase`

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
- `pipeline/services/readings.py`: `Readings` — per-frame realism scalars, reported as distributions when a stream stops or a batch job finishes
- `pipeline/services/identity.py`: `IdentityProbe` — ArcFace similarity between the source and the output, measured per compositing stage. Shares the detector's own recognition model rather than loading a second copy, and re-frames aligned crops from whichever swapper template made them into the `arcface_112` framing recognition needs
- `pipeline/services/shape.py`: `ShapeProbe` — **the axis the cosine is blind to.** Whether the output took the source's head shape or kept the target's, from 106-point landmarks on three faces reduced to pure shape by fitting away the similarity transform. Borrows the detector's landmark model. See "Head shape" below

### Processing Pipeline
- `pipeline/processing/pipeline.py`: `ProcessingPipeline` orchestrator (batch & stream modes)
- `pipeline/processing/frame_processor.py`: `FrameProcessor` ABC + 4 implementations
- `pipeline/processing/compositor.py`: `FaceCompositor` aligned-space compositing
- `pipeline/processing/geometry.py`: FFHQ template, Umeyama similarity fit, the detail band's sigma — shared by the compositor and the texture extractor
- **[docs/PERFORMANCE_AUDIT.md](docs/PERFORMANCE_AUDIT.md)**: per-stage compositor profile, what was found wasteful and fixed, and a realism risk register for the optimisations that would *not* be free. Headline: `optimal` is 7.4ms of a 50ms budget with every realism layer on; **`production` is 39ms against a 33ms deadline before detection, swap, restoration or encode**, and no CPU-side work closes that
- `pipeline/processing/texture.py`: `SourceTexture` — skin detail extracted once per identity, reprojected per frame

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

### Vast.ai Deployment
- `vast/orchestrator.py`: CLI for managing GPU instances (offers, start, resume, stop, terminate, status, logs, run, push, pull)
- `vast/startup.sh`: Instance setup (ffmpeg, venv, pip, cuDNN, TLS certificate)
- `vast/TROUBLESHOOTING.md`: Every Vast API trap actually hit, plus the cuDNN one that outlived the provider change
- `VAST_DEPLOYMENT.md`: Setup and operation guide — account, keys, the loop, and
  the full `.env` reference. `tests/test_wiring.py` asserts it stays in step
  with the code
- **[docs/VAST_MIGRATION.md](docs/VAST_MIGRATION.md)**: why the move happened,
  measured against both APIs, and the four decisions it rests on

## Vast.ai Orchestrator

### Commands
```bash
python vast/orchestrator.py offers      # what is rentable, and why anything was refused
python vast/orchestrator.py start       # rent → ssh → startup.sh → pipeline → update .env
python vast/orchestrator.py resume      # start the stopped instance (VAST_INSTANCE_ID)
python vast/orchestrator.py stop        # stop it (disk survives, storage keeps billing)
python vast/orchestrator.py terminate   # destroy it (disk and models go too)
python vast/orchestrator.py status      # state, GPU, location, cost, uplink, address
python vast/orchestrator.py logs [n]    # tail the pipeline log
python vast/orchestrator.py run "cmd"   # one command on the instance, inside the venv
python vast/orchestrator.py push <local> [remote]
python vast/orchestrator.py pull <remote> [local]
```

### How It Works
- `start` always rents a new instance; `resume` starts an existing one, and
  **falls back to `start` only when the failure is about capacity** — falling
  back on any failure would rent a billing instance in response to a typo.
- **Offers are searched, not enumerated.** Every filter is server-side:
  geolocation, `dlperf`, VRAM, price, reliability, `inet_up`,
  `direct_port_count`, `compute_cap`, `gpu_arch`. RunPod could filter on none
  of the ones that matter here.
- **`VAST_PREFERRED_HOST` pins a host** for a stable IP and a warm disk; the
  filtered search runs whenever it has nothing rentable.
- **Search needs no API key.** `offers` works from a clean checkout with no
  account, which is the command someone runs to decide whether to open one.
- SSH is `runtype: ssh_direct` — a real sshd on the instance, so `exec_command`
  and SFTP both work. `pull` exists because of it.
- Only port 9000 is published, mapped to a random external port on a shared
  public IP.

### Critical API Notes
- **Two search filters fail silently**, returning HTTP 200 with zero offers:
  `gpu_name` needs spaces (`"RTX 4090"`, not `"RTX_4090"`), and `geolocation`
  matches the **country code** even though values read `"United Kingdom, GB"`.
- **An offer `id` is stable within a query but not across queries.** The same
  machine came back as `43933077` and `43933078` from two searches. Compare
  result sets on `machine_id`; never cache an `id` and rent it later.
- **Volumes are locked to one physical machine**, so there is no network-volume
  equivalent. The instance disk is the only copy of the venv and weights, and
  `terminate` destroys it.
- **Storage bills while stopped**, per host, up to $0.40/GB/month — several
  times RunPod's $0.07. `VAST_DISK` is a cost setting.
- **Bandwidth is billed**, per host (`inet_up_cost` / `inet_down_cost`). ~4% on
  top of the GPU at the `optimal` preset; `offers` prints it.
- `env` on create carries environment variables **and** docker `-p` flags in
  one object, with `"1"` as the value for a port entry.
- Under `ssh_direct` the image's entrypoint is replaced, so `onstart` (or, here,
  the orchestrator's own SSH session) is what launches anything.
- **Auto-stop**: the pipeline stops the instance after `VAST_MAX_UPTIME`
  minutes (default 120), sending `auto_stop_warning` 5 minutes before. It
  **stops** rather than destroys, so the models stay warm — which does not end
  storage billing. Works with no desktop connected.

### Transport
There is no TLS proxy. The instance generates a self-signed certificate once,
`startup.sh` prints the SHA-256 of its **DER** encoding, and the orchestrator
pins it into `.env` as `PHANTOM_TLS_FINGERPRINT`. A random `PHANTOM_API_TOKEN`
goes with it, required in the first frame.

This is not optional politeness: `desktop/controller.py` already speaks
`ws://host:port/ws`, so the naive path *works* — in cleartext, with the
operator's face in it. `tests/test_transport_security.py` pins both ends,
including that an unauthenticated client never joins the broadcast set.
