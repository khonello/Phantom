# Architecture Overview

## System Design

Phantom is a modern, event-driven face-swapping application refactored for composability and testability.

### 1. Pipeline Engine (`pipeline/`)

**Event-driven, service-oriented processing:**
- Detects faces using InsightFace (FaceDetector service)
- Performs face swapping with ONNX models (FaceSwapper service)
- Runs in real-time or batch mode via ProcessingPipeline
- Communicates via WebSocket on port 9000 (async, no polling)

**Core Layers:**

**Configuration & Infrastructure:**
- `pipeline/config.py` — Observable FaceSwapConfig (no global state)
- `pipeline/events.py` — EventBus pub/sub system (inter-module communication)
- `pipeline/logging.py` — Structured logging with event emission
- `pipeline/types.py` — Typed dataclasses (Detection, VideoProperties, etc.)

**Services (ML/CV):**
- `pipeline/services/face_detection.py` — InsightFace wrapper
- `pipeline/services/face_swapping.py` — ONNX face swap model
- `pipeline/services/enhancement.py` — Face restoration (CodeFormer / GFPGAN)
- `pipeline/services/masking.py` — Landmark hull + valid region + XSeg occlusion
- `pipeline/services/face_tracking.py` — LandmarkStabilizer (EMA on landmarks)
- `pipeline/services/database.py` — Embedding cache & averaging

**Processing Pipeline:**
- `pipeline/processing/pipeline.py` — Main orchestrator (batch & stream modes)
- `pipeline/processing/frame_processor.py` — Composable processor chain
- `pipeline/processing/compositor.py` — FaceCompositor (aligned-space compositing)
- `pipeline/processing/geometry.py` — FFHQ template, Umeyama similarity fit, detail-band sigma
- `pipeline/processing/texture.py` — SourceTexture (skin detail, extracted once per identity)
- `pipeline/processing/async_processor.py` — Background processing wrapper

**I/O Layer:**
- `pipeline/io/capture.py` — Abstract input sources (webcam, file, network)
- `pipeline/io/output.py` — Abstract output sinks (file, HTTP, WebSocket)
- `pipeline/io/ffmpeg.py` — FFmpeg utilities

**API & Control:**
- `pipeline/api/server.py` — WebSocket API server (replaces HTTP control)
- `pipeline/api/handlers.py` — Type-safe command handlers
- `pipeline/api/schema.py` — Message types and constants

**Entry Points:**
- `pipeline/core.py` — CLI argument parsing and orchestration (~100 lines)
- `pipeline/stream.py` — Stream mode convenience wrapper (~57 lines)

### 2. Desktop GUI (`desktop/`)

A modern interface that:
- Provides user controls (source selection, quality presets, start/stop)
- Displays live preview via WebSocket events
- Sends commands via WebSocket (no polling)
- Manages virtual camera output for video calls

**Key modules:**
- `desktop/ui.py` — CustomTkinter interface
- `desktop/bridge.py` — Signal mapping and WebSocket event subscription
- `desktop/controller.py` — WebSocket client for pipeline API

## Data Flow (Event-Driven)

```
┌──────────────────────────────┐
│    Desktop GUI               │
│  - User selects source       │
│  - Sends command via WS      │
└──────────────┬───────────────┘
               │ WebSocket command
               ▼
    ┌──────────────────────────┐
    │  WebSocketAPIServer      │
    │  - Routes commands       │
    │  - Broadcasts events     │
    └──────────┬───────────────┘
               │
    ┌──────────▼───────────────┐
    │  ProcessingPipeline      │
    │  - Orchestrates services │
    │  - Emits FRAME_READY,    │
    │    DETECTION, etc.       │
    └──────────┬───────────────┘
               │ EventBus (pub/sub)
    ┌──────────▼───────────────┐
    │  frame_processor chain    │
    │  - Preprocessing          │
    │  - Detection (every frame)│
    │  - Landmark stabilization │
    │  - Swapping (aligned)     │
    │  - Compositing            │
    └──────────┬───────────────┘
               │
    ┌──────────▼───────────────┐
    │  Events back to server   │
    │  → WebSocket broadcast   │
    │  → Desktop receives      │
    └──────────────────────────┘

    ┌──────────────────────────┐
    │  Input Sources           │
    │  - Webcam                │
    │  - File                  │
    │  - Network Stream        │
    └──────────────────────────┘
```

## Key Design Principles

**1. Observable Configuration**
- Single source of truth: `CONFIG` dataclass
- Changes propagate via `CONFIG.on_change()` callbacks
- No hidden mutable state

**2. Event-Driven Communication**
- Inter-module communication via `EventBus`
- Decoupled publishers and subscribers
- No direct module imports for side effects

**3. Composable Processing**
- `FrameProcessor` ABC with 5 implementations
- Chain without modification or side effects
- Easy to add new processors (color correction, face morphing, etc.)

**4. Service-Oriented**
- Each service has one responsibility
- Pure interfaces (deterministic, testable)
- Lazy initialization (models load on demand)

**5. Type-Safe API**
- All commands and responses are typed dataclasses
- mypy strict mode enforced
- Validation at system boundaries

## Virtual Camera Integration

### Why pyvirtualcam?

Video call applications (Skype, Zoom, Teams) only accept system camera devices. They can't:
- Consume RTMP streams
- Display raw frame buffers
- Connect to arbitrary sockets

To inject processed video into a video call, we need a **virtual camera driver** — a kernel/OS-level service that registers a fake camera device.

### Architecture

```
┌─────────────────────────────┐
│   Video Call App (Skype)    │
│  - Enumerates cameras       │
│  - Selects virtual camera   │
│  - Reads frames             │
└──────────────┬──────────────┘
               │
         Camera Device
        (OS-level virtual camera)
               ▲
               │
         pyvirtualcam
      (Python library)
               ▲
               │
  ┌────────────┴──────────────┐
  │   Pipeline + Desktop      │
  │  - Processes frames       │
  │  - Decodes JPEG → BGR     │
  │  - Writes to virtual cam  │
  └──────────────────────────┘
```

### Supported Drivers

`pyvirtualcam` is a Python library that **bridges** your code to virtual camera drivers:

**Windows:**
- `obs` — OBS Virtual Camera (recommended, free)
- `unitycapture` — Unity Capture (Windows-specific)

**macOS:**
- `obs` — OBS Virtual Camera

**Linux:**
- `v4l2loopback` — Kernel-based virtual camera

**Why drivers are separate from pyvirtualcam:**
- `pyvirtualcam` is just the interface library
- The actual driver must be installed separately (e.g., OBS installer)
- `pyvirtualcam` communicates with the installed driver
- This design allows users to choose their preferred driver without bloat

### Desktop Implementation

In `desktop/bridge.py`:

1. **WebSocket receiver** gets JPEG frames from pipeline
2. **Frame decoder** converts JPEG → BGR numpy array
3. **Virtual camera thread** runs `pyvirtualcam.Camera()` context
4. **Frame queue** buffers decoded frames (drop oldest if full)
5. **Camera.send()** writes frames to the virtual camera driver

**Code flow:**
```python
# WebSocket thread receives JPEG
data = ws.recv()  # Binary JPEG bytes

# Push to virtual camera (if active)
if self._virtual_cam_active:
    self._push_to_vcam(data)

# _push_to_vcam decodes and queues
frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)  # BGR
self._vcam_queue.put_nowait(frame)

# Virtual camera thread sends to driver
with pyvirtualcam.Camera(backend='obs') as cam:
    while not stop:
        frame = self._vcam_queue.get()
        cam.send(frame)
        cam.sleep_until_next_frame()
```

## Performance Considerations

### Real-time Processing (Stream Mode)

- **Input**: Webcam, network source (OpenCV), or JPEG frames pushed over WebSocket
- **Face detection**: Every frame — no correlation tracker, no stale geometry
- **Stabilization**: EMA on landmarks (warp) and on aligned pixels (shimmer)
- **Compositing**: Aligned-space — masking, colour, detail, grain
- **Restoration**: Optional, in FFHQ space, blended back at partial strength
- **Output**: 480x270 to 960x540, 15–30 FPS, set by quality preset

### Bottlenecks

1. **Face detection** (InsightFace) — runs every frame, most expensive operation
2. **Face restoration** (CodeFormer/GFPGAN) — second most expensive
3. **Face swapping** (ONNX inference) — depends on GPU availability
4. **Occlusion masking** (XSeg) — one extra ONNX pass per frame
5. **WebSocket broadcast** — can drop frames if network is slow
6. **Virtual camera** — minimal overhead (frame copy only)

### Optimization Strategies

- **GPU acceleration**: Use CUDA/CoreML for 5-10x speedup
- **Aligned-space work**: Compositing happens at 192–320px, not frame size
- **ROI warp-back**: Cost scales with face size, not frame size
- **Parallel warm-up**: Detection, swap, occluder and restoration models load
  concurrently on start, halving startup time
- **Capture resolution**: Set by preset. Modest resolutions are both cheaper and
  closer to the target look
- **Preset trade-offs**: `fast` drops the occluder pass and composites at 192px

Detection is deliberately *not* skipped or intervalled. A correlation tracker
between detections warps the swap with stale landmarks, which costs more in
quality than it saves in latency.

## Thread Safety

### Concurrent Components

- **Capture thread** (InputSource): Reads frames from webcam/file/network
- **Processing thread** (ProcessingPipeline): Runs frame processor chain
- **Model warm-up pool**: Loads the four ONNX models in parallel at stream start
- **WebSocket server thread**: Receives commands, broadcasts events
- **Event bus** (EventBus): Pub/sub dispatch (synchronous, in calling thread)
- **Desktop bridge**: Subscribes to events, updates UI

### Synchronization

- **Observable config** (`CONFIG`): Uses listeners instead of locks
  - `CONFIG.on_change(callback)` — fires on any field update
  - All services and processors subscribe to config changes
- **Event bus** (`BUS`): Synchronous pub/sub (no race conditions)
  - Emitters and listeners same thread (no cross-thread calls)
  - Event constants prevent string typos: `FRAME_READY`, `DETECTION`, etc.
- **Stop events** (`CONFIG.shutdown_event`): `threading.Event` for graceful shutdown
- **Frame queues**: `AsyncProcessor` uses size-1 queue (drop-newest on overflow)

## Caching & Memory

- **Face embeddings** (FaceDatabase): Cached in memory (~256KB per embedding)
  - Loaded from `.npy` files or computed from images
  - Averaged across multiple source images
  - Reused across frames (not recomputed)
- **Frame buffers** (AsyncProcessor): Size-1 queue
  - Drops frame if processor thread hasn't caught up
  - Keeps latest, discards oldest on overflow
- **Model weights**: Lazy loaded, cached in memory or CUDA
  - InsightFace (det_size=480) — ~200MB
  - Inswapper_128 (ONNX) — ~350MB
  - CodeFormer (ONNX, optional) — ~360MB
  - DFL XSeg (ONNX, optional) — ~50MB
  - GFPGAN (torch, optional) — ~350MB
- **Memory limits** (`CONFIG.max_memory`):
  - Set via `--max-memory` flag or GUI
  - Enforced per `limit_resources()` on startup
  - Prevents OOM on GPU with multiple processes

See [DEVELOPMENT.md](DEVELOPMENT.md) for profiling and optimization.
