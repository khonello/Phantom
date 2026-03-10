# Architecture Overview

## System Design

roop-cam is a face-swapping application with two main components:

### 1. Pipeline Engine (`pipeline/`)

The headless processing engine that:
- Detects faces using InsightFace
- Performs face swapping with ONNX models
- Runs in real-time or batch mode
- Communicates via HTTP on port 9000

**Key modules:**
- `pipeline/core.py` — Argument parsing and entry point
- `pipeline/globals.py` — Shared application state
- `pipeline/face_analyser.py` — Face detection (InsightFace)
- `pipeline/predicter.py` — High-level prediction API
- `pipeline/processors/frame/` — Frame processing (face swap, enhancement)
- `pipeline/stream.py` — Real-time webcam processing loop
- `pipeline/control.py` — HTTP control server and frame serving
- `pipeline/utilities.py` — Video/image utilities (FFmpeg, frame extraction)

### 2. Desktop GUI (`desktop/`)

A Qt/QML interface that:
- Provides user controls (source selection, quality, start/stop)
- Displays live preview via WebSocket
- Sends webcam frames to the pipeline
- Manages virtual camera output for video calls
- Communicates with pipeline via HTTP + WebSocket

**Key modules:**
- `desktop/main.qml` — UI layout and interactions
- `desktop/bridge.py` — Qt/Python integration and pipeline communication
- `desktop/controller.py` — HTTP client for pipeline API

## Data Flow

```
┌──────────────────────────────────────────────┐
│          Desktop GUI (Qt/QML)                │
│  - User selects face image                   │
│  - Clicks START                              │
└─────────────────┬──────────────────────────┬─┘
                  │                          │
          HTTP POST               WebSocket JPEG
       /control, /status          (binary frames)
                  │                          │
                  ▼                          ▼
         ┌─────────────────────────────────────┐
         │    Pipeline Engine (Python)         │
         │  - Face detection (InsightFace)     │
         │  - Face swapping (ONNX)             │
         │  - Real-time processing loop        │
         │  - WebSocket frame broadcast        │
         └─────────────────────────────────────┘
                  ▲
                  │
           Webcam / Input URL
         (OpenCV VideoCapture)
```

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

- **Input**: Webcam or network source (OpenCV)
- **Face detection**: Every 30 frames (configurable)
- **Face tracking**: Smooth motion estimation between detections
- **Blending**: Luminance-adaptive blend for seamless integration
- **Enhancement**: Optional GFPGAN post-processing (every N frames)
- **Output**: 960x540 @ 30 FPS (adjustable)

### Bottlenecks

1. **Face detection** (InsightFace) — most expensive operation
2. **Face swapping** (ONNX inference) — depends on GPU availability
3. **WebSocket broadcast** — can drop frames if network is slow
4. **Virtual camera** — minimal overhead (frame copy only)

### Optimization Strategies

- **Detection interval**: Skip detection on non-keyframes (cached tracking)
- **Enhancement interval**: Only enhance every N frames
- **Tracker**: Use lightweight trackers (MOSSE, KCF) to reduce detection frequency
- **GPU acceleration**: Use CUDA/CoreML for 5-10x speedup
- **Resolution**: 960x540 chosen for balance (not too heavy, not too light)

## Thread Safety

### Concurrent Components

- **Webcam thread**: Reads frames, broadcasts to pipeline
- **Enhancement thread**: Async face enhancement queue
- **WebSocket thread**: Receives processed frames, writes to virtual camera
- **Stream processing thread**: Core face swapping loop
- **HTTP server thread**: Handles control requests
- **Virtual camera thread**: Writes to camera driver

### Synchronization

- **Global state** (`pipeline/globals.py`): Shared settings (not locked, assumed atomic)
- **Frame queues** (`queue.Queue`): Thread-safe, dropping oldest on overflow
- **Stop events** (`threading.Event`): Safe for cross-thread signaling
- **Status updates** (`QTimer.singleShot`): Main thread delivery via Qt event loop

## Caching & Memory

- **Face embeddings** (source): Cached in memory (~256KB)
- **Frame buffers**: Rolling buffers with max size = 3 frames
- **Model weights**: CUDA memory (if available) or RAM
- **GC tuning**: Thresholds raised to reduce pauses during frame allocation

See [DEVELOPMENT.md](DEVELOPMENT.md) for debugging and profiling.
