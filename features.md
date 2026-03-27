# Phantom — Features

Comprehensive feature reference for the Phantom face-swapping pipeline, desktop GUI, and cloud deployment system.

---

## Face Processing

### Face Detection
- InsightFace FaceAnalysis (buffalo_l model) with configurable detection threshold (0.35)
- Single-face or multi-face detection modes
- No-face streak detection with warnings after 3 consecutive empty frames
- Thread-safe lazy initialization with execution provider selection (CUDA, CPU, ROCm, DML)

### Face Swapping
- ONNX-based inswapper_128 model
- Multiple source images with embedding averaging for improved likeness
- Pre-computed `.npy` embedding file support
- Model resolution priority: RunPod network volume → local `models/` → working directory

### Face Tracking
- Three tracker algorithms: CSRT (accurate), KCF (fast), MOSSE (ultra-fast)
- Exponential Moving Average (EMA) keypoint smoothing with configurable alpha
- Automatic redetection on interval or tracker failure
- Fallback to raw detection when tracker is unavailable

### Face Enhancement
- GFPGAN-based face restoration (synchronous, GPU-optimized)
- Independent toggle — enable/disable at any time from desktop or API
- Graceful fallback if GFPGAN or model files are unavailable
- torchvision >= 0.18 compatibility shim

### Blending
- Configurable face blend ratio (0–1)
- Luminance-adaptive blending for natural lighting match

---

## Pipeline Modes

### Batch — Image
- Single image face swap with source embedding
- Optional enhancement post-processing
- Output to file

### Batch — Video
- Full video processing with face detection per frame
- Audio preservation from source
- FPS preservation from source
- Video encoder selection: libx264, libx265, libvpx-vp9
- CRF quality control (0–51, default 18)
- FFmpeg hardware acceleration (`-hwaccel auto`)

### Stream — Realtime
- Live webcam capture or network stream (RTSP/RTMP/HTTP)
- WebSocket push mode (desktop sends JPEG frames)
- Processing chain: Detect → Track → Swap → Enhance → Emit
- Frame warmup period (configurable, default 5 frames)
- Per-stage timing diagnostics (detect, track, swap, enhance, total)
- Frame drop detection and reporting

---

## Quality Presets

Three presets control tracking, smoothing, and blending. Enhancement is independent.

| Setting            | Fast  | Optimal (default) | Production |
|--------------------|-------|--------------------|------------|
| Tracker            | KCF   | CSRT               | CSRT       |
| Alpha (smoothing)  | 0.7   | 0.6                | 0.5        |
| Blend              | 0.65  | 0.65               | 0.65       |
| Luminance Blend    | No    | Yes                | Yes        |
| Buffer Size        | 3     | 4                  | 5          |
| Redetect Interval  | 30    | 30                 | 20         |
| Warmup Frames      | 3     | 5                  | 5          |

**Fast**: Lower latency, less smoothing. Best for testing or low-powered GPUs.
**Optimal**: Balanced quality and performance. Default for most use cases.
**Production**: Maximum smoothing and stability. Best for final output or recording.

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
| `set_blend` | Set face blend ratio |
| `set_alpha` | Set EMA smoothing factor |
| `set_enhance` | Toggle face enhancement on/off |
| `set_input_url` | Set network stream URL |
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
- Enhance toggle — enable/disable GFPGAN enhancement in real time
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

### CLI Arguments
```
python pipeline.py -s <source> -t <target> -o <output>   # Batch mode
python pipeline.py --stream                                # Realtime mode
python pipeline.py --execution-provider cuda               # GPU selection
python pipeline.py --quality production                    # Preset selection
python pipeline.py --tracker csrt --alpha 0.6 --blend 0.65 # Fine-tuning
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
┌──────────────────────────────────────────────────┐
│  ProcessingPipeline                              │
│                                                  │
│  frame arrives                                   │
│  │                                               │
│  ├─ Warmup period? (first N frames)              │
│  │   └─ Yes → skip processing, emit raw frame    │
│  │                                               │
│  ├─ Time to redetect? (every 20-30 frames)       │
│  │   ├─ Yes → DetectionProcessor                 │
│  │   │        run InsightFace on full frame       │
│  │   │        ├─ Face found → update tracker      │
│  │   │        └─ No face   → emit face_lost       │
│  │   │                                            │
│  │   └─ No  → TrackingProcessor                  │
│  │            use tracker (CSRT/KCF/MOSSE)        │
│  │            predict face position from motion    │
│  │            apply EMA smoothing to keypoints     │
│  │            ├─ Tracker OK → use predicted bbox   │
│  │            └─ Tracker lost → force redetect     │
│  │                                               │
│  ├─ SwappingProcessor                            │
│  │   load source embedding (single or averaged)   │
│  │   run ONNX inswapper_128 model                 │
│  │   blend swapped face onto frame                │
│  │                                               │
│  ├─ Enhancement enabled?                         │
│  │   ├─ Yes → EnhancementProcessor               │
│  │   │        GFPGAN restore (synchronous, ~10ms) │
│  │   └─ No  → skip                               │
│  │                                               │
│  └─ BlendingProcessor                            │
│     apply blend ratio + luminance matching         │
│                                                  │
└──────────────┬───────────────────────────────────┘
               │
               ▼
         EventBus.emit(FRAME_READY)
               │
               ▼
         WebSocket Server
         encode as JPEG → push binary frame to all clients
```

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
