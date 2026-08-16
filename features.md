# Phantom — Features

Comprehensive feature reference for the Phantom face-swapping pipeline, desktop GUI, and cloud deployment system.

---

## Face Processing

The design target is **what a real video call looks like** — a face carrying
sensor noise, compression, and ordinary imperfection. Not a high-resolution
portrait, and not the poreless "beautified" look that reads as AI instantly.
Every stage below is tuned toward that, and several deliberately *add* back
imperfection that the models remove.

### Face Detection
- InsightFace FaceAnalysis (buffalo_l model) with configurable detection threshold (0.35)
- Single-face or multi-face detection modes
- Runs on **every frame**, so the swap is always warped with current landmarks
- No-face streak detection with warnings after 3 consecutive empty frames
- Thread-safe lazy initialization with execution provider selection (CUDA, CPU, ROCm, DML)

### Face Swapping
- ONNX-based inswapper_128 model
- Multiple source images with embedding averaging for improved likeness
- Pre-computed `.npy` embedding file support
- `swap_aligned()` returns the raw aligned crop plus its affine (`paste_back=False`),
  handing compositing to `FaceCompositor` instead of letting the model paste
- Falls back to InsightFace's own compositing if a build cannot supply the affine
- Model resolution priority: RunPod network volume → local `models/` → working directory

### Landmark Stabilization
- `LandmarkStabilizer` — EMA on `kps` (drives the swap warp) and
  `landmark_2d_106` (drives the mask), smoothed together so the two never disagree
- Replaces the previous CSRT/KCF/MOSSE correlation trackers. With detection on
  every frame a tracker only added latency and a stale warp
- Resets on face loss (3 consecutive misses) and on large centroid jumps, so
  re-acquisition does not interpolate from a stale position
- Bypassed in multi-face mode, where per-frame detection order is not stable

### Face Restoration
Two backends, selected by `enhancer_model`:

- **CodeFormer** (default) — ONNX, runs on the onnxruntime session the swapper
  already needs, so it adds no dependency. Model downloads on first use.
  Exposes a **fidelity weight**: `0.0` restores hardest and hallucinates most,
  `1.0` stays closest to the input.
- **GFPGAN** — the previous backend, kept for comparison on real footage.
  Needs torch + the `gfpgan` package, plus a torchvision ≥ 0.18 shim.

CodeFormer is the default because GFPGAN v1.4 restores toward a beautified,
poreless look with no way to dial it back. `enhance_strength` then blends only
part of the restored face back in, keeping some of the input's imperfection.

Both are trained on **FFHQ-framed 512×512 crops** and rely on features sitting
where FFHQ puts them, so the compositor warps into FFHQ space around the restore
call rather than handing them the swapper's tighter arcface crop.

### Compositing
Everything after the swap happens in **aligned face space**, not on whole frames
(`pipeline/processing/compositor.py`). Per face:

1. **Restore** — in FFHQ space, blended back at `enhance_strength`
2. **Temporal EMA** — on aligned pixels, gated by measured motion so it releases
   when the subject moves and cannot ghost
3. **Colour match** — LAB transfer sampled *inside the mask only*, ramped by
   colour distance so it never snaps on and off between frames
4. **Detail match** — high-frequency band scaled to the target's, correcting in
   both directions (the swap is softer before restoration, sharper after)
5. **Warp back** — into a region of interest, so cost scales with face size
6. **Composite** — soft alpha, feathered in both aligned and frame space
7. **Grain** — monochrome sensor noise matched to the surrounding frame

Geometry uses a closed-form Umeyama similarity fit, not
`cv2.estimateAffinePartial2D` — the OpenCV estimators are randomized, and
anything that varies frame to frame feeds straight back into shimmer.

### Masking
`FaceMasker` multiplies three terms in aligned space:

- **Landmark hull** — convex hull of the 106 landmarks InsightFace already
  computes, so it costs nothing extra and follows the real jawline rather than
  assuming an ellipse
- **Valid region** — the part of the crop that actually sampled real pixels, so
  faces near the frame edge do not bleed a black border into the composite
- **Occlusion** — optional DFL XSeg segmentation, so hands, microphones and hair
  crossing the face are not painted over with swapped skin

Degrades to hull + valid-region if the XSeg model is unavailable.

---

## Pipeline Modes

### Stream — Realtime
The primary mode, and where development is focused.

- Live webcam capture or network stream (RTSP/RTMP/HTTP)
- WebSocket push mode (desktop sends JPEG frames — used on RunPod, where the
  pod has no camera)
- Processing chain: Detect → Stabilize → Swap → Composite → Emit
- Frame warmup period (configurable, default 5 frames)
- Per-stage timing diagnostics at `--log-level debug`
- Frame drop detection and reporting
- Parallel model warm-up on start (detection, swap, occluder, restoration)

### Batch — Image
- Single image face swap with source embedding
- Shares the **same** compositing path as stream mode, so batch and realtime
  output cannot drift apart. Temporal smoothing is a no-op with no previous frame
- Output to file

### Batch — Video
> **Not yet implemented.** `ProcessingPipeline._process_target_batch()` handles
> images only and reports an error for video. The FFmpeg building blocks all
> exist in `pipeline/io/ffmpeg.py` (`extract_frames`, `create_video`,
> `restore_audio`, `clean_temp`) but are not yet wired into the pipeline.
>
> Affects: desktop VIDEO mode (fully built UI-side, errors on start), the CLI
> `-t <video>` path, and the CI end-to-end test.
>
> Planned once the live-call path meets its quality target. Because batch reuses
> the same compositor, most of the work is frame iteration and audio/FPS
> restoration rather than new image processing.

Settings already in place for it: audio preservation, FPS preservation, encoder
selection (libx264, libx265, libvpx-vp9), CRF quality (0–51, default 18).

---

## Quality Presets

Three presets trade latency against realism. Defined once in
`pipeline/api/schema.py::PRESETS`, applied via `FaceSwapConfig.apply_preset()`.

A preset picks **how much compute to spend. It does not change how the face
looks.** `enhancer_weight` and `enhance_strength` are what decide whether the
output reads as a real call or as AI, and neither costs anything to compute —
the weight is a scalar model input, the strength is one blend. Varying them per
preset only meant "production" restored hardest and so looked *most* synthetic
while presenting itself as the best option. They are identical everywhere now,
and an operator choosing a preset for their GPU cannot change the look by
accident.

| Cost                 | Fast    | Optimal (default) | Production |
|----------------------|---------|-------------------|------------|
| Capture resolution   | 480×270 | 640×360           | 960×540    |
| Frame rate           | 15 fps  | 20 fps            | 30 fps     |
| JPEG quality         | 60      | 70                | 85         |
| Detector input       | 320     | 448               | 640        |
| Compositing ceiling  | 192     | 256               | 320        |
| Occlusion masking    | Off     | On                | On         |

| Stability (scales with frame rate) | Fast | Optimal | Production |
|------------------------------------|------|---------|------------|
| Landmark EMA (alpha)               | 0.7  | 0.6     | 0.5        |
| Temporal EMA                       | 0.7  | 0.6     | 0.5        |
| Buffer size                        | 3    | 4       | 5          |
| Warmup frames                      | 3    | 5       | 5          |

| Look (identical by design) | all presets |
|----------------------------|-------------|
| Fidelity weight            | 0.7         |
| Restore strength           | 0.7         |
| Grain matching             | On          |

**Fast**: quarter the detector pixels, cheaper compositing, no occlusion pass. For low-powered GPUs.
**Optimal**: balanced. Default.
**Production**: largest detector and compositing ceiling, heaviest smoothing.

Capture settings live in `pipeline/api/schema.py::PRESETS` and are read by both
the pipeline's own `VideoCapture` loop and the desktop's webcam thread, so local
and push mode capture identically.

Note the capture resolutions are deliberately modest. The target is a normal
video call, and a 1080p-sharp face on a call is itself a tell.

Presets deliberately **do not** set `enhance` or `color_correction` — both have
explicit toggles in the desktop header, and a preset must not silently undo
something the operator just clicked.

Changing quality restarts the webcam capture device to apply the new
resolution/fps.

---

## Realism Knobs

| Field | Default | Effect |
|-------|---------|--------|
| `enhance` | `True` | Face restoration on/off |
| `enhancer_model` | `codeformer` | Backend (`codeformer` or `gfpgan`) |
| `enhancer_weight` | `0.7` | CodeFormer fidelity: `0`=most restoration, `1`=closest to input |
| `enhance_strength` | `0.7` | How much of the restored face to keep |
| `aligned_size` | `256` | **Ceiling** on compositing resolution (clamped 128–512); actual size follows the face's size in frame |
| `temporal_alpha` | `0.6` | EMA on aligned pixels, kills shimmer (`1.0` disables) |
| `color_correction` | `True` | LAB transfer, sampled inside the mask |
| `color_strength` | `1.0` | Scales that transfer |
| `grain` | `True` | Matches sensor noise on the composited face |
| `occluder` | `True` | XSeg mask so hands/mics are not overpainted |

Three ways to set them:

- **Quality preset** — the desktop dropdown; see the table above.
- **CLI / env** — `--enhancer-model`, `--enhancer-weight`, `--enhance-strength`,
  `--aligned-size`, `--temporal-alpha`, `--color-strength`, `--no-enhance`,
  `--no-grain`, `--no-occluder`. Each also reads an env var
  (`ENHANCER_MODEL`, `ENHANCER_WEIGHT`, …) since the pod is configured via `.env`.
  Precedence: preset first, then CLI/env overrides.
- **`set_realism` API command** — `{"action": "set_realism", "values": {...}}`,
  or `controller.set_realism(enhancer_weight=0.5)`. Validates and clamps;
  unknown fields are reported back rather than silently ignored.

> These are not yet exposed in the desktop UI — live A/B tuning currently
> requires the API or a Python REPL.

Models (`codeformer.onnx`, `dfl_xseg.onnx`) download on first use to
`/workspace/models/` or `pipeline/models/`. If one is unavailable the pipeline
degrades — masking falls back to landmark hull + valid-region, restoration falls
back to the other backend or off — rather than failing.

---

## WebSocket API

### Architecture
- Single port (9000) for all communication
- Binary frames: JPEG-encoded video frames pushed to all connected clients
- Text frames: JSON commands (client → server) and events (server → client)
- Ping/pong heartbeat every 30s with 120s timeout
- Decoupled frame broadcast queue (slow clients never stall the pipeline)

### Commands
| Command | Description |
|---------|-------------|
| `set_source` | Set single source face image |
| `set_source_paths` | Set multiple sources for embedding averaging |
| `set_target` | Set target image or video |
| `set_output` | Set output file path |
| `set_quality` | Apply quality preset (fast/optimal/production) |
| `set_alpha` | Set landmark EMA smoothing factor |
| `set_enhance` | Toggle face restoration on/off |
| `set_realism` | Set any realism knob(s) — validated and clamped |
| `set_color_correction` | Toggle LAB colour transfer |
| `set_preprocessing` | Toggle input normalization (lighting, white balance) |
| `set_blend` | Legacy blend ratio — accepted, no longer read |
| `set_input_url` | Set network stream URL |
| `upload_source` | Send a source image as binary (used for remote pods) |
| `get_state` | Read current config back, for UI re-sync after reconnect |
| `cleanup_session` | Clear source/target/output and temporary session files |
| `set_keep_fps` | Preserve original FPS |
| `set_keep_audio` | Preserve original audio |
| `set_many_faces` | Toggle multi-face mode |
| `start` | Start batch processing |
| `start_stream` | Start realtime stream |
| `stop` / `stop_stream` | Stop processing |
| `create_embedding` | Generate face embedding from images |
| `keep_alive` | Extend auto-stop deadline |
| `shutdown` | Shutdown the pod/server |
| `health` | Health check → `{"status": "healthy", "uptime": <seconds>}` |

### Events
| Event | Description |
|-------|-------------|
| `pipeline_started` / `pipeline_stopped` | Pipeline lifecycle |
| `status` | Status message updates |
| `detection` | Face detection results per frame |
| `face_lost` | No face detected |
| `drop_rate` | Frame drop statistics |
| `auto_stop_warning` | Minutes remaining before auto-stop |
| `auto_stop` | Pod is being stopped |

---

## Desktop GUI

### Three Modes
- **LIVE**: Realtime webcam/stream processing with live preview
- **VIDEO**: Batch video processing with file selection
- **IMAGE**: Batch image processing with file selection

### LIVE Mode Controls
- Webcam index selector
- Quality preset dropdown (fast / optimal / production)
- VCAM toggle — route processed frames to virtual camera
- Enhance toggle — enable/disable face restoration in real time
- Color correction toggle — LAB transfer for cross-skin-tone swaps
- Preprocessing toggle — input lighting / white-balance normalization
- Start / Stop button
- Live processed frame display

### Batch Mode Controls (VIDEO / IMAGE)
- Target file selector with thumbnail preview
- Output path selector with auto-naming
- Process / Stop button

### Source Management
- Select source face images (one or multiple)
- Source thumbnail with clear button
- Embedding progress indicator
- Multiple image averaging for improved accuracy

### Connection & Status
- Live connection indicator (green/red badge)
- Status message display
- Auto-reconnect on connection loss

### Auto-Stop Warning Dialog
- Countdown overlay showing minutes remaining
- **Extend** button — resets the auto-stop timer
- **Dismiss** button — acknowledges without extending
- Works with RunPod auto-stop billing protection

### Audio & Voice
- Real-time audio capture with timestamped PCM chunks
- Jitter buffer for audio/video synchronization
- Voice transformation presets: Female (+4 semitones), Male (-3.5), Child (+6), Deep (-5)
- Parselmouth-based pitch and formant shifting
- Graceful disable if voice libraries unavailable

---

## RunPod Cloud Deployment

### Commands
```
python runpod/orchestrator.py start        # Deploy fresh GPU pod
python runpod/orchestrator.py resume       # Resume stopped pod
python runpod/orchestrator.py stop         # Pause pod (volume preserved)
python runpod/orchestrator.py terminate    # Delete pod (network volume survives)
python runpod/orchestrator.py status       # Show pod state, GPU, cost, WebSocket URL
python runpod/orchestrator.py gpus         # List GPUs with VRAM, pricing, eligibility
python runpod/orchestrator.py datacenters  # List all datacenters
```

### Deployment Modes
- **SSH** (development): Clones repo, installs dependencies, starts pipeline in tmux
- **Docker** (production): Custom image with pipeline baked in, auto-starts

### GPU Auto-Discovery
- Queries RunPod GraphQL for all available GPU types
- Filters by minimum VRAM (`RUNPOD_MIN_VRAM`, default 16 GB)
- Filters by maximum hourly price (`RUNPOD_MAX_PRICE`, default $1.00)
- Sorts by cheapest first, tries until one succeeds
- Manual override via `RUNPOD_GPU_TYPES` (comma-separated display names)

### Multi-Datacenter Fallback
- Format: `RUNPOD_DATACENTERS=DC1:vol1,DC2:vol2`
- Each datacenter paired with its own network volume (volumes are datacenter-local)
- Tries all eligible GPUs in datacenter 1 first, then datacenter 2, etc.
- Network volumes persist models and venv across pod restarts

### Auto-Stop (Billing Protection)
- `RUNPOD_MAX_UPTIME`: Stop pod after N minutes (default 120, 0 = disabled)
- `RUNPOD_STOP_WARNING`: Warning N minutes before stop (default 5)
- Background timer runs in the pipeline server — works even without a desktop connected
- Desktop shows warning dialog with extend option
- Calls `runpod.stop_pod()` on expiry (pod can be resumed later)

### Networking
- WebSocket: RunPod proxy — `wss://{pod_id}-9000.proxy.runpod.net/ws`
- SSH: RunPod proxy — `{podHostId}@ssh.runpod.io`
- Only port 9000/tcp exposed (avoids JupyterLab initialization on 8888)

---

## Configuration

### Observable Config
- `FaceSwapConfig` dataclass with `set()` method and `on_change()` callbacks
- Field-level change notifications trigger pipeline rebuilds as needed
- Environment variable loading via python-dotenv

### Key Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `EXECUTION_PROVIDER` | GPU backend (cuda, cpu, rocm, dml) | cuda |
| `API_PORT` | WebSocket server port | 9000 |
| `LOG_LEVEL` | Logging verbosity | info |
| `PHANTOM_API_URL` | Desktop → pipeline WebSocket URL | ws://localhost:9000/ws |
| `ENHANCER_MODEL` | Restoration backend | codeformer |
| `ENHANCER_WEIGHT` | CodeFormer fidelity (0–1) | preset |
| `ENHANCE_STRENGTH` | How much restored face to keep (0–1) | preset |
| `ALIGNED_SIZE` | Compositing working resolution | preset |
| `TEMPORAL_ALPHA` | EMA on composited pixels | preset |
| `COLOR_STRENGTH` | Scales the LAB transfer | 1.0 |
| `ENHANCE` / `GRAIN` / `OCCLUDER` | Feature toggles | on |

### CLI Arguments
```
python pipeline.py -s <source> -t <target> -o <output>  # Batch (image only)
python pipeline.py --stream                             # Realtime mode
python pipeline.py --execution-provider cuda            # GPU selection
python pipeline.py --quality production                 # Preset selection
python pipeline.py --log-level debug                    # Per-stage timings

# Realism tuning (override the preset)
python pipeline.py --stream --enhancer-weight 0.5 --enhance-strength 0.6
python pipeline.py --stream --enhancer-model gfpgan --aligned-size 320
python pipeline.py --stream --no-grain --no-occluder
```

---

## Event System

- Lightweight pub/sub `EventBus` with string-identified events
- ThreadPoolExecutor-based async dispatch (4 workers, non-blocking)
- Used for all inter-module communication (no direct function calls between modules)
- Pipeline → EventBus → WebSocket Server → Desktop

---

## Performance

- Single-threaded CUDA mode (`OMP_NUM_THREADS=1`) for optimal GPU utilization
- Lazy model loading with warm-up on first pipeline start
- Decoupled frame broadcast thread (network I/O never blocks processing)
- GC threshold tuning to avoid allocation freezes during processing
- Frame drop rate tracking and reporting
- Capture timestamp tracking for end-to-end latency analysis
- RTT-based adaptive playout delay for audio/video sync

---

## Supported Formats

### Input
- **Images**: JPG, PNG, BMP, TIFF
- **Video**: MP4, AVI, MKV, MOV, WebM
- **Streams**: RTSP, RTMP, HTTP, webcam
- **Embeddings**: `.npy` pre-computed face embeddings

### Output
- **Video encoders**: libx264 (H.264), libx265 (H.265), libvpx-vp9 (VP9)
- **Image**: JPG, PNG
- **Stream**: WebSocket binary frames (JPEG), virtual camera (DirectShow)

---

## How It Works — Visual Flows

### GPU Deployment

What happens when you run `python runpod/orchestrator.py start`:

```
orchestrator.py start
│
├─ Load .env, verify API key
│
├─ RUNPOD_POD_ID already set?
│  ├─ Yes → "Deploy NEW pod? [y/N]"
│  │         ├─ No  → abort
│  │         └─ Yes → continue
│  └─ No → continue
│
├─ Read deploy mode (ssh or docker)
│
├─ FIND A GPU
│  │
│  ├─ Parse datacenters (each paired with its network volume)
│  │   1. EU-RO-1 ←→ volume z8now7p5ts
│  │   2. US-TX-3 ←→ volume abc123
│  │      (volumes are datacenter-local, so each DC needs its own)
│  │
│  ├─ Build GPU candidate list
│  │   RUNPOD_GPU_TYPES set?
│  │   ├─ Yes → use those exact GPUs in order
│  │   └─ No  → query RunPod API for all GPUs
│  │            filter: ≥16GB VRAM, ≤$1/hr
│  │            sort: cheapest first
│  │            e.g. [RTX 4000 $0.38, RTX 4090 $0.69, ...]
│  │
│  └─ Try datacenter × GPU (datacenter is outer loop)
│
│      EU-RO-1 (volume: z8now7p5ts)
│      ├─ RTX 4000  → unavailable
│      ├─ RTX 4090  → unavailable
│      ├─ RTX A4500 → created ✓ → skip to WAIT
│      └─ (all fail → fall through to next datacenter)
│
│      US-TX-3 (volume: abc123)
│      ├─ RTX 4000  → unavailable
│      ├─ RTX 4090  → created ✓ → skip to WAIT
│      └─ (all fail → exit with error ✗)
│
├─ WAIT FOR POD
│  │
│  └─ Poll every 3s until status = RUNNING (up to 5 min)
│     then resolve SSH address + WebSocket address
│
├─ SSH SETUP (ssh mode only)
│  │
│  ├─ Wait for SSH port to accept connections
│  ├─ Connect with key (~/.ssh/id_ed25519)
│  ├─ Open interactive shell (RunPod drops exec_command)
│  │
│  ├─ /workspace/Phantom exists?
│  │   ├─ No  → git clone repo
│  │   └─ Yes → skip (already deployed before)
│  │
│  ├─ Run startup.sh (ffmpeg, venv, pip install)
│  ├─ Kill any old pipeline process
│  └─ Start pipeline in background (nohup)
│
├─ WAIT FOR PIPELINE HEALTH
│  │
│  └─ WebSocket → {"action":"health"}
│     wait for → {"status":"healthy"} (up to 2 min)
│
├─ UPDATE .env
│  ├─ RUNPOD_POD_ID = <new pod id>
│  └─ PHANTOM_API_URL = wss://<pod>-9000.proxy.runpod.net/ws
│
└─ DONE — "python desktop.py" to connect
   Auto-stop timer now running (2hr, 5min warning)
```

---

### Frame Processing Pipeline

How each frame flows through the realtime processing chain:

```
Webcam / Network Stream / Desktop Push
│
▼
┌──────────────────────────────────────────────────────┐
│  ProcessingPipeline                                  │
│                                                      │
│  frame arrives                                       │
│  │                                                   │
│  ├─ Warmup period? (first N frames)                  │
│  │   └─ Yes → skip, do not emit                      │
│  │                                                   │
│  ├─ PreprocessingProcessor (optional)                │
│  │   normalize lighting / white balance              │
│  │                                                   │
│  ├─ DetectionProcessor — EVERY frame                 │
│  │   run InsightFace on full frame                   │
│  │   └─ No face → stabilizer.mark_missing()          │
│  │               (resets state after 3 misses)       │
│  │                                                   │
│  ├─ LandmarkStabilizer  (single-face only)           │
│  │   EMA on kps + landmark_2d_106 together           │
│  │   resets on large centroid jump                   │
│  │                                                   │
│  ├─ SwappingProcessor.swap_aligned()                 │
│  │   ONNX inswapper_128, paste_back=False            │
│  │   → returns (aligned crop, affine)                │
│  │                                                   │
│  └─ FaceCompositor.composite()  ── aligned space ──┐ │
│      1. restore   CodeFormer in FFHQ space,        │ │
│                   blended at enhance_strength      │ │
│      2. temporal  EMA on pixels, released on motion│ │
│      3. mask      hull × valid-region × XSeg       │ │
│      4. colour    LAB, sampled inside mask         │ │
│      5. detail    high-frequency band matched      │ │
│      6. warp back into ROI, soft-alpha composite   │ │
│      7. grain     sensor noise matched to frame    │ │
│                                                  ──┘ │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
         EventBus.emit(FRAME_READY)
               │
               ▼
         WebSocket Server
         encode as JPEG → push binary frame to all clients
```

Temporal continuity comes from EMA — on landmarks and on aligned pixels — not
from a correlation tracker carrying a stale face forward. Both release under
motion and reset on face loss or source change. Both are bypassed when
`many_faces` is set, since per-frame detection order is not stable.

---

### WebSocket Protocol

How the desktop and pipeline communicate over a single port:

```
Desktop (client)                    Pipeline (server :9000)
    │                                       │
    ├──── WebSocket connect ───────────────►│
    │     ws://localhost:9000/ws             │
    │     or wss://<pod>-9000.proxy.../ws   │
    │                                       │
    │                          ◄─── TEXT ───┤  {"event":"status","message":"ready"}
    │                                       │
    │                    COMMANDS (JSON text frames)
    │                    ─────────────────────────
    ├──── TEXT ────────────────────────────►│  {"action":"set_source","path":"/img.jpg"}
    │                          ◄─── TEXT ───┤  {"action":"set_source","success":true}
    │                                       │
    ├──── TEXT ────────────────────────────►│  {"action":"start_stream"}
    │                          ◄─── TEXT ───┤  {"event":"pipeline_started"}
    │                                       │
    │                    FRAMES (binary frames)
    │                    ─────────────────────────
    │                          ◄── BINARY ──┤  [JPEG bytes] ← pushed every frame
    │                          ◄── BINARY ──┤  [JPEG bytes]
    │                          ◄── BINARY ──┤  [JPEG bytes]
    │                                       │
    │                    EVENTS (JSON text, interleaved with frames)
    │                    ─────────────────────────
    │                          ◄─── TEXT ───┤  {"event":"detection","faces":1}
    │                          ◄─── TEXT ───┤  {"event":"face_lost"}
    │                          ◄─── TEXT ───┤  {"event":"drop_rate","rate":0.02}
    │                                       │
    │                    HEARTBEAT
    │                    ─────────────────────────
    │                          ◄─── PING ───┤  every 30s
    ├──── PONG ───────────────────────────►│
    │                                       │
    │                    ENHANCEMENT TOGGLE
    │                    ─────────────────────────
    ├──── TEXT ────────────────────────────►│  {"action":"set_enhance","value":false}
    │                          ◄─── TEXT ───┤  {"action":"set_enhance","success":true}
    │                                       │
    │                    STOP
    │                    ─────────────────────────
    ├──── TEXT ────────────────────────────►│  {"action":"stop_stream"}
    │                          ◄─── TEXT ───┤  {"event":"pipeline_stopped"}
    │                                       │
```

---

### Auto-Stop Timer

Billing protection flow — works even with no desktop connected:

```
Pod starts
│
├─ RUNPOD_MAX_UPTIME = 120 min?
│  ├─ 0 → timer disabled, no auto-stop
│  └─ >0 → start background timer thread
│
│  ┌─────────────────────────────────────────────┐
│  │  Auto-Stop Timer (checks every 10s)         │
│  │                                              │
│  │  deadline = now + 120 minutes                │
│  │                                              │
│  │  every 10s:                                  │
│  │  │                                           │
│  │  ├─ time remaining > warning threshold?      │
│  │  │   └─ Yes → keep waiting                   │
│  │  │                                           │
│  │  ├─ time remaining ≤ 5 min?                  │
│  │  │   └─ broadcast auto_stop_warning          │
│  │  │      ┌──────────────────────────────┐     │
│  │  │      │  Desktop (if connected)      │     │
│  │  │      │  ┌────────────────────────┐  │     │
│  │  │      │  │  ⚠ Auto-stop in 5 min  │  │     │
│  │  │      │  │                        │  │     │
│  │  │      │  │  [Extend]  [Dismiss]   │  │     │
│  │  │      │  └────────────────────────┘  │     │
│  │  │      │       │                      │     │
│  │  │      │       ├─ Extend clicked      │     │
│  │  │      │       │  send keep_alive ────────► │
│  │  │      │       │  deadline = now + 120 min  │
│  │  │      │       │  timer resets ✓            │
│  │  │      │       │                      │     │
│  │  │      │       └─ Dismiss / no desktop│     │
│  │  │      │          timer keeps ticking │     │
│  │  │      └──────────────────────────────┘     │
│  │  │                                           │
│  │  └─ deadline reached?                        │
│  │      └─ Yes → broadcast auto_stop event      │
│  │              call runpod.stop_pod()           │
│  │              pod pauses (can resume later)    │
│  │                                              │
│  └──────────────────────────────────────────────┘
│
```

---

### Desktop ↔ Pipeline Connection

Startup, reconnection, and event flow:

```
python desktop.py
│
├─ Load .env → read PHANTOM_API_URL
│  (local: ws://localhost:9000/ws)
│  (cloud: wss://<pod>-9000.proxy.runpod.net/ws)
│
├─ Launch QML UI
│  ├─ Connection badge: red (disconnected)
│  ├─ Controls disabled
│  └─ Waiting for pipeline...
│
├─ WebSocket Client (background thread)
│  │
│  ├─ Connect to PHANTOM_API_URL
│  │   ├─ Success → badge turns green
│  │   └─ Fail    → retry with backoff (3 attempts)
│  │               1s → 2s → 4s
│  │
│  ├─ CONNECTION ESTABLISHED
│  │   │
│  │   ├─ User selects source images
│  │   │   bridge → set_source_paths → pipeline
│  │   │   pipeline → embedding_ready → bridge
│  │   │   thumbnail + label update in UI
│  │   │
│  │   ├─ User clicks START (live mode)
│  │   │   bridge → start_stream → pipeline
│  │   │   bridge → set_enhance(current state) → pipeline
│  │   │   pipeline → pipeline_started → bridge
│  │   │   │
│  │   │   │  ┌─── Frame loop ──────────────┐
│  │   │   │  │ pipeline pushes JPEG binary  │
│  │   │   │  │ bridge updates live display  │
│  │   │   │  │ pipeline pushes JSON events  │
│  │   │   │  │ bridge updates status/badges │
│  │   │   │  └─────────────────────────────┘
│  │   │   │
│  │   │   ├─ User toggles ENHANCE
│  │   │   │   bridge → set_enhance → pipeline
│  │   │   │   takes effect on next frame
│  │   │   │
│  │   │   ├─ User changes quality preset
│  │   │   │   bridge → set_quality → pipeline
│  │   │   │   pipeline rebuilds processors
│  │   │   │   webcam restarts with new resolution/fps
│  │   │   │
│  │   │   └─ User clicks STOP
│  │   │       bridge → stop_stream → pipeline
│  │   │       pipeline → pipeline_stopped → bridge
│  │   │
│  │   └─ User starts batch (video/image mode)
│  │       bridge → set_target + set_output + start → pipeline
│  │       pipeline → status updates → bridge (progress)
│  │       pipeline → batch_complete → bridge
│  │
│  └─ CONNECTION LOST
│     badge turns red
│     auto-reconnect with backoff
│     on reconnect → re-sync state
│
```
