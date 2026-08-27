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

## Quick Commands

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
Two backends, chosen by `config.enhancer_model`:

- **`codeformer`** (default) — ONNX, runs on the onnxruntime session already
  required by the swapper, so it adds no dependency. Model downloads on first
  use. Exposes a **fidelity weight** (`enhancer_weight`): `0.0` restores hardest
  and hallucinates most, `1.0` stays closest to the input. This is the knob that
  matters for believability — GFPGAN v1.4 restores toward a beautified, poreless
  look with no way to dial it back, and that plastic skin is the strongest "this
  is AI" signal on a call.
- **`gfpgan`** — the previous backend, kept so the two can be compared on real
  footage. Needs torch + the `gfpgan` package. If the configured backend cannot
  load, the other is tried before restoration is disabled.

Both are trained on **FFHQ-framed 512x512 crops** and rely on features sitting
where FFHQ puts them, so `FaceCompositor` warps into FFHQ space around the
restore call rather than handing them the swapper's tighter arcface crop. FFHQ
framing is ~28% wider than arcface, so the crop given to the restorer is the real
frame in FFHQ framing with the swapped face composited over it — otherwise the
edges would be empty. Only what the swap covers survives the mask, so the real
face at the edges never reaches the output.

Geometry uses a closed-form Umeyama similarity fit (`estimate_similarity`), not
`cv2.estimateAffinePartial2D` — the OpenCV estimators are randomized and anything
that varies frame to frame feeds straight back into shimmer.

### Realism knobs (`FaceSwapConfig`)
| Field | Default | Effect |
|-------|---------|--------|
| `enhance` | `True` | Face restoration on/off |
| `enhancer_model` | `codeformer` | Restoration backend (`codeformer` or `gfpgan`) |
| `enhancer_weight` | `0.7` | CodeFormer fidelity: `0`=most restoration, `1`=closest to input |
| `enhance_strength` | `0.7` | How much of the restored face to keep. Full strength reads as AI; partial keeps believable imperfection |
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
| fp16 weights | `--fp16` / `FP16` | **Yes** | A/B on footage before shipping |
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
  `--aligned-size`, `--temporal-alpha`, `--color-strength`, `--no-enhance`,
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
- flake8 checks: E3, E4, F
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
