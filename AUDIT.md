# Phantom Project Audit — 2026-03-13

## Executive Summary

Phase 7 architecture migration is ~95% complete but **currently broken**. The new services/processor architecture is solid. However, old processor files were modified without being migrated, leaving broken imports and missing function references that will prevent startup. Additionally, the API server — despite being named `WebSocketAPIServer` — is pure HTTP polling, which is fundamentally incompatible with the low-latency goals of a real-time face-swapping application.

**Status: BLOCKED — app cannot run as-is. Real-time streaming requires WebSocket replacement of HTTP server.**

---

## Deployment Modes

This audit covers fixes for **two distinct deployment scenarios**:

### Mode 1: Local Development
- **Setup**: Pipeline (`pipeline.py`) and Desktop (`desktop.py`) both run on your local machine
- **Connection**: Desktop connects to `ws://localhost:9000` (local WebSocket)
- **Model caching**: Models cached at `~/.insightface/models/` (standard insightface location, no special handling)
- **Steps to follow**: Steps 1–6 (all critical + performance fixes)
- **See**: Step 4A (Local/Development Deployment)

### Mode 2: RunPod Cloud Deployment
- **Setup**: Desktop runs on your local machine; Pipeline runs on a RunPod GPU pod
- **Connection**: Desktop connects to `wss://<pod-url>:9000` (remote WebSocket over HTTPS)
- **Model caching**: Models cached via RunPod Network Volume at `/workspace/models/` (persistent across pod restarts)
- **Steps to follow**: Steps 1–6 (all critical + performance fixes) **PLUS** Step 4B (RunPod-specific tasks)
- **See**: Step 4B (RunPod Cloud Deployment) + `RUNPOD_DEPLOYMENT.md`

**Key difference**: Local uses filesystem caching, RunPod uses Network Volume. `pipeline/services/face_detection.py` handles both automatically if you follow Step 4B.

---

## Remote Deployment Architecture (RunPod Only: Desktop on Local Machine → RunPod Pipeline)

This section applies **only to RunPod cloud deployment**. For local development (pipeline and desktop on the same machine), see Step 4A.

The updated deployment model sends source/target files from the local desktop to RunPod GPUs over WebSocket:

```
Local Desktop (Qt/PySide6)
  ↓ WebSocket connection to RunPod pod URL
  ↓ sends source.jpg + target.mp4 as binary over WS
RunPod WebSocket API Server (port 9000)
  ↓ receives PUT /source, PUT /target commands
  ↓ streams processing to GPU
ProcessingPipeline (CUDA GPU on RunPod)
  ↓ emits FRAME_READY events with JPEG frames
  ↓ EventBus broadcasts to handlers
WebSocketAPIServer
  ↓ pushes JPEG frames + JSON status events over WS
  ↓ (no polling, true push architecture)
Local Desktop (receives frames in real time)
  ↓ displays in UI + saves final output to local disk
```

**Key differences from local operation:**
- Desktop connects remotely to RunPod pod URL instead of `localhost:9000`
- Source/target are uploaded as binary payload (not file paths on local disk)
- Frame delivery must be fast (JPEG, not PNG) to compensate for network latency
- Pod IP is ephemeral — reconnection logic needed if pod restarts
- Network timeouts are possible during processing — WebSocket ping/pong heartbeat required
- Model caching via RunPod Network Volume (~200MB InsightFace models persist across pod restarts, ~3 min download on first run only)

---

## Critical Issues (Application will not start)

### 1. Wrong import path in old processor files
**Files affected:**
- `pipeline/processors/frame/face_enhancer.py:10`
- `pipeline/processors/frame/face_swapper.py:11`
- `pipeline/face_analyser.py:11`

All three import `Frame` and `Face` from `pipeline.typing` (old module) instead of `pipeline.types` (new module).

```python
# Current (broken)
from pipeline.typing import Frame, Face

# Should be
from pipeline.types import Frame, Face
```

### 2. Missing function `update_status`
**Files affected:**
- `pipeline/processors/frame/face_swapper.py:9` — `from pipeline.core import update_status`
- `pipeline/processors/frame/face_enhancer.py:8` — `from pipeline.core import update_status`

`update_status()` does not exist in `pipeline/core.py`. The new equivalent is `emit_status()` in `pipeline/logging.py`.

```python
# Current (broken)
from pipeline.core import update_status

# Should be
from pipeline.logging import emit_status
```

All call sites (`face_swapper.py` lines 44, 76, 82, 85, 93, 97, 100, 103; `face_enhancer.py` lines 38-39) must change `update_status(...)` → `emit_status(...)`.

---

## High Priority Issues

### 3. Two parallel, conflicting processor systems
- **OLD**: `pipeline/processors/frame/face_swapper.py`, `face_enhancer.py`, `core.py` — function-based, globals-dependent
- **NEW**: `pipeline/processing/frame_processor.py` — OOP, ABC-based, service-oriented (`SwappingProcessor`, `EnhancementProcessor`, etc.)

The old files are no longer integrated into the new `ProcessingPipeline`. They are dead code that causes import errors. They should be removed or fully rewritten to match the new architecture.

### 4. Duplicate face analysis module
- **OLD**: `pipeline/face_analyser.py` — exposes a global `FACE_ANALYSER` singleton
- **NEW**: `pipeline/services/face_detection.py` — `FaceDetector` service class

Old module duplicates new service functionality. It is still imported by the old processors, contributing to the broken import chain.

### 5. Two type modules (inconsistent)
- **OLD**: `pipeline/typing.py` — basic re-exports from insightface
- **NEW**: `pipeline/types.py` — enhanced dataclasses (`Bbox`, `Detection`, `VideoProperties`, `SwapResult`)

`pipeline/typing.py` is deprecated but still referenced by old files. `pipeline/types.py` defines a `Detection` dataclass that does not exist in the old module, causing type mismatches when old code interfaces with new services.

---

## Medium Priority Issues

### 6. Global state in API handlers
**File**: `pipeline/api/handlers.py:25-40`

```python
_pipeline: Optional[ProcessingPipeline] = None
_shutdown_event: Optional[threading.Event] = None
```

This violates the "no global state" principle in CLAUDE.md. Handlers cannot be unit tested in isolation; multiple pipeline instances would conflict. Should use dependency injection instead.

### 7. No real WebSocket — HTTP polling kills low-latency goals (CRITICAL for real-time)
**File**: `pipeline/api/server.py`

`WebSocketAPIServer` is a **misleading name**. It is a plain HTTP server (`http.server.HTTPServer`). The desktop must repeatedly poll `GET /frame` to receive frames. This is fundamentally broken for low-latency real-time streaming:

| Problem | Impact |
|---------|--------|
| Full HTTP request/response per frame | ~5-20ms overhead per frame on top of processing |
| No push — client doesn't know when frame is ready | Polling interval creates artificial lag |
| PNG encoding per frame (`cv2.imencode('.png', ...)`) | PNG is slow and large; JPEG is 3-5x faster at same quality |
| No persistent connection | Connection setup cost on every frame request |
| Events not pushed to desktop | Desktop can't react to FRAME_READY, DETECTION, STATUS_CHANGED in real time |

**What is needed**: Replace `http.server.HTTPServer` with a real WebSocket server (e.g. `websockets` library or `aiohttp`). The `EventBus` is already set up to emit `FRAME_READY` events — the server just needs to push those over a persistent WebSocket connection instead of waiting to be polled.

**Recommended implementation**:
- Use `websockets` (async) or `simple-websocket-server` for the server
- On `FRAME_READY` event: encode frame as **JPEG** (not PNG) and push binary frame to all connected clients
- Push `STATUS_CHANGED`, `DETECTION` events as JSON text frames
- `POST /control` can remain HTTP or move to WebSocket commands

**DEPLOYMENT.md** already documents WebSocket endpoints — the docs are correct, the implementation is wrong.

### 8. Type mismatch between old and new face types
New services (`FaceDetector`) return `List[Detection]` (new dataclass). Old processors expect raw `insightface.app.common.Face` objects. This causes `AttributeError` at runtime when old code interfaces with new services.

Three incompatible face representations exist:
- Raw `insightface.app.common.Face`
- `pipeline.types.Detection` (new dataclass)
- `types.SimpleNamespace(normed_embedding=...)` used in `pipeline/services/database.py`

---

## Low Priority / Technical Debt

### 9. `pipeline/utilities.py` duplicates `pipeline/io/ffmpeg.py`
FFmpeg utilities were moved to the new I/O layer, but the old `utilities.py` still exists with overlapping functions. It should be removed after confirming no remaining references.

### 10. Dead code in `pipeline/core.py:26`
```python
del torch  # inside ROCm conditional
```
`del torch` inside an `if` block that only runs when ROCm is detected. Fragile and unnecessary.

### 11. `pipeline/ws_server.py` and `pipeline/capturer.py` exist but are unused
These appear to be pre-migration files that were superseded by the new architecture. They should be removed.

---

## Migration Completeness

| Component | Status |
|-----------|--------|
| `pipeline/config.py` — FaceSwapConfig | ✅ Complete |
| `pipeline/events.py` — EventBus | ✅ Complete |
| `pipeline/types.py` — Type definitions | ✅ Complete |
| `pipeline/logging.py` — Structured logging | ✅ Complete |
| `pipeline/services/*.py` — ML/CV services | ✅ Complete |
| `pipeline/processing/frame_processor.py` — Processor ABC | ✅ Complete |
| `pipeline/processing/pipeline.py` — Orchestrator | ✅ Complete |
| `pipeline/processing/async_processor.py` | ✅ Complete |
| `pipeline/api/schema.py` | ✅ Complete |
| `pipeline/api/handlers.py` | ✅ Complete (design smell) |
| `pipeline/api/server.py` | ✅ Complete (misleadingly named) |
| `pipeline/io/capture.py`, `output.py`, `ffmpeg.py` | ✅ Complete |
| `pipeline/processors/frame/face_swapper.py` | ❌ Broken (old patterns) |
| `pipeline/processors/frame/face_enhancer.py` | ❌ Broken (old patterns) |
| `pipeline/processors/frame/core.py` | ❌ Orphaned |
| `pipeline/face_analyser.py` | ❌ Duplicate of services/face_detection.py |
| `pipeline/typing.py` | ❌ Deprecated, still referenced |
| `pipeline/utilities.py` | ❌ Partially duplicated by io/ffmpeg.py |
| `pipeline/ws_server.py` | ❌ Unused |
| `pipeline/capturer.py` | ❌ Unused |

---

## Recommended Fix Order

### Step 1 — Unblock app startup (Critical)
- [ ] Fix imports in `face_enhancer.py`, `face_swapper.py`, `face_analyser.py`: change `pipeline.typing` → `pipeline.types`
- [ ] Fix function references: change `update_status` → `emit_status` in old processor files
- [ ] Verify app starts without `ImportError`

### Step 2 — Remove orphaned old code (High)
- [ ] Delete `pipeline/processors/frame/face_swapper.py` (superseded by `SwappingProcessor`)
- [ ] Delete `pipeline/processors/frame/face_enhancer.py` (superseded by `EnhancementProcessor`)
- [ ] Delete `pipeline/processors/frame/core.py` (orphaned)
- [ ] Delete or deprecate `pipeline/face_analyser.py` (superseded by `FaceDetector`)
- [ ] Delete `pipeline/typing.py` after removing all references
- [ ] Delete `pipeline/ws_server.py` and `pipeline/capturer.py` (unused)

### Step 3 — Implement real WebSocket server (Critical for low latency)
- [ ] Replace `http.server.HTTPServer` in `server.py` with `websockets` or `aiohttp` WebSocket server
- [ ] Push JPEG-encoded frames over WebSocket on `FRAME_READY` event (drop PNG encoding)
- [ ] Push `STATUS_CHANGED` and `DETECTION` events as JSON text frames to all connected clients
- [ ] Update `desktop/bridge.py` / `desktop/controller.py` to connect via WebSocket and receive push frames
- [ ] Keep or migrate `POST /control` to WebSocket command messages
- [ ] Replace global `_pipeline`/`_shutdown_event` in `handlers.py` with dependency injection

### Step 4A — Local/Development Deployment

**Core pipeline setup (both local & remote):**
- [ ] Add root `requirements.txt` (or rename `requirements-pipeline-gpu.txt` → `requirements.txt`) — DEPLOYMENT.md install step currently fails
- [ ] Remove desktop GUI packages (`customtkinter`, `tk`) from server requirements — unnecessary on headless Linux
- [ ] Add `--stream` CLI flag to `pipeline/core.py` argument parser
- [ ] Add `--log-level` CLI flag to `pipeline/core.py` argument parser (or alias `--log-level` → existing `log_level` CONFIG field)
- [ ] Add `/health` endpoint to `server.py` returning `{"status": "healthy", "uptime": <seconds>}`
- [ ] Add `.env` file loading (e.g. `python-dotenv`) so `EXECUTION_PROVIDER`, `API_PORT`, etc. are actually read by the app

**Local development setup:**
- [ ] Update `DEPLOYMENT.md` for local operation:
  - Document running both `pipeline.py` and `desktop.py` on same machine
  - Default to `ws://localhost:9000/ws` (no URL input needed)
  - Local models cache at `~/.insightface/models/` (standard insightface location, no special handling needed)
  - First run: ~3 min model download, subsequent runs use cache
- [ ] Update `desktop/controller.py`:
  - For local mode: connect to `ws://localhost:9000/ws` by default
  - Optional override via `PHANTOM_API_URL` env var for testing remote connections

### Step 4B — RunPod Cloud Deployment

**RunPod-specific setup (source/target streaming from local desktop to cloud GPU):**
- [ ] Create `RUNPOD_DEPLOYMENT.md` (separate from local `DEPLOYMENT.md`):
  - Step-by-step pod creation in RunPod UI
  - Create/mount Network Volume for persistent model cache
  - Expose TCP 9000 (WebSocket) at pod creation
  - Cost estimates and GPU tier recommendations (RTX 4090 vs. A100 vs. L40S)
  - Troubleshooting network issues

- [ ] Create `runpod/startup.sh` — pod initialization script:
  - Install FFmpeg (`apt-get install ffmpeg`)
  - Check CUDA availability (`torch.cuda.is_available()`) and log GPU name
  - Create `/workspace/models` directory if not present
  - Pre-warm models during pod init (optional, for faster first connection)

- [ ] Update `pipeline/services/face_detection.py` for Network Volume caching:
  - Check `/workspace/models/insightface/` first (Network Volume path)
  - Fall back to `~/.insightface/models/` (pod ephemeral storage)
  - Add startup log: "Using cached models from Network Volume" or "Downloading models to Network Volume (first run, ~3 min)"
  - **Do NOT modify for local operation** — local uses default `~/.insightface/models/` only

- [ ] Update `desktop/controller.py` for remote RunPod connections:
  - Prompt user to enter RunPod pod URL (or read `PHANTOM_API_URL` env var)
  - Parse pod URL → construct `wss://<pod-url>:9000/ws` (WebSocket over HTTPS)
  - Add connection timeout (30s) and retry logic (exponential backoff, max 3 retries)
  - Display connection status in UI ("Connecting to RunPod...", "Connected", "Disconnected")
  - **Only for remote** — local mode bypasses URL input

- [ ] Validate remote input/output flow (local desktop ↔ RunPod pipeline):
  - Desktop sends source image/video as binary over WebSocket `PUT /source` command
  - Desktop sends target image/video as binary over WebSocket `PUT /target` command
  - Pipeline processes on GPU and streams JPEG frames via `FRAME_READY` events
  - Desktop receives frames in real time, saves final output to local disk
  - Document maximum file sizes (e.g., videos > 1GB should be chunked or streamed)

- [ ] Add cloud-specific error handling and monitoring:
  - Network timeout during long processing: add WebSocket heartbeat (ping/pong every 30s)
  - Pod out-of-memory: emit `OOM` event with degradation options (skip enhancement, reduce resolution)
  - GPU out-of-memory during swap: emit `OOM` event, desktop retries with lower resolution
  - Connection drop: auto-reconnect with exponential backoff
  - Document recommended GPU tiers based on video resolution/fps

- [ ] Fix `DEPLOYMENT.md` documentation for both modes:
  - Add section "Local Development" (connect to `localhost:9000`)
  - Add section "RunPod Cloud Deployment" (reference `RUNPOD_DEPLOYMENT.md`)
  - Clarify which instructions apply to which deployment mode

### Step 5 — Latency fixes (Performance)
- [ ] `pipeline.py`: Add `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` after VideoCapture creation (L6 — trivial, 30–100ms saved)
- [ ] `server.py:270`: Change `.png` → `.jpg` in `cv2.imencode` (L2 partial — 3–5× encoding speedup)
- [ ] `face_detection.py`: Call `_ensure_loaded()` eagerly in `__init__` (L8 — eliminates 1–3s first-frame spike)
- [ ] `async_processor.py:117`: Change `timeout=0.1` → `timeout=0.01` (L3 partial — 90ms worst-case queue wait)
- [ ] `pipeline/processing/pipeline.py`: Implement `warmup_frames` skip logic (L13 — drop first N frames)
- [ ] `frame_processor.py:423`: Cache original face luminance, compute only swapped (L7 — 10–20ms/frame)
- [ ] `async_processor.py:47`: Increase queue size to 2–3 frames (L3 — reduces silent drops)
- [ ] `output.py:143,154`: Implement double-buffering / atomic pointer swap to eliminate frame copies (L4)
- [ ] `frame_processor.py:171`: Move periodic redetection to background thread (L5 — eliminates 100–300ms spikes)
- [ ] `events.py:59`: Dispatch event handlers via thread pool, not inline (L1 — biggest single latency win)
- [ ] `frame_processor.py:315`: Run enhancement on every frame async, use last enhanced frame for temporal consistency (L11)

### Step 6 — Cleanup (Low)
- [ ] Remove `pipeline/utilities.py` after verifying no remaining references
- [ ] Remove dead `del torch` in `pipeline/core.py:26`
- [ ] Audit `desktop/bridge.py` — verify it uses new event API correctly

---

## Files to Delete (after Step 1 verified working)

```
pipeline/processors/frame/face_swapper.py
pipeline/processors/frame/face_enhancer.py
pipeline/processors/frame/core.py
pipeline/face_analyser.py
pipeline/typing.py
pipeline/ws_server.py
pipeline/capturer.py
```

---

## Files Requiring Edits

| File | Change Required | Applies To | Step |
|------|----------------|-----------|------|
| `pipeline/processors/frame/face_enhancer.py` | Fix imports | Local & RunPod | 1 |
| `pipeline/processors/frame/face_swapper.py` | Fix imports + function refs | Local & RunPod | 1 |
| `pipeline/face_analyser.py` | Fix import path | Local & RunPod | 1 |
| `pipeline/api/server.py` | Replace HTTP with real WebSocket; add `/health` endpoint | Local & RunPod | 3 |
| `pipeline/api/server.py` | Add WebSocket heartbeat (ping/pong); add OOM error event; change `.png` → `.jpg` in imencode | Local & RunPod | 4, 5 |
| `pipeline/api/handlers.py` | Replace globals with DI | Local & RunPod | 3 |
| `desktop/bridge.py` | Update to receive WebSocket push frames | Local & RunPod | 3 |
| `desktop/controller.py` | Connect to WebSocket; local: default to `localhost:9000`; RunPod: accept pod URL input or `PHANTOM_API_URL` env var; add remote timeout/retry logic | Local & RunPod | 3, 4B |
| `pipeline/core.py` | Add `--stream` and `--log-level` flags | Local & RunPod | 4A |
| `pipeline/core.py` | Add CUDA availability check at startup (log GPU name) | Local & RunPod | 4A |
| `requirements-pipeline-gpu.txt` | Remove desktop GUI packages (`customtkinter`, `tk`) | Local & RunPod | 4A |
| `DEPLOYMENT.md` | Split into Local & RunPod sections; update flags; document localhost:9000 for local, reference RUNPOD_DEPLOYMENT.md for cloud | Local & RunPod | 4A, 4B |
| `pipeline/processing/pipeline.py` | Add `cap.set(CAP_PROP_BUFFERSIZE, 1)`; implement `warmup_frames` skip logic | Local & RunPod | 4A, 5 |
| `pipeline/services/face_detection.py` | Eager model loading in `__init__` (local & RunPod); Check `/workspace/models/` first for RunPod Network Volume cache (RunPod only) | Local & RunPod | 4B, 5 |
| `pipeline/processing/async_processor.py` | Fix queue size to 2–3 frames; reduce timeout from 0.1s to 0.01s | Local & RunPod | 5 |
| `pipeline/processing/frame_processor.py` | Cache luminance; move redetection to background thread; async enhancement consistency | Local & RunPod | 5 |
| `pipeline/events.py` | Thread pool handler dispatch; add heartbeat/ping-pong for WebSocket (RunPod uses it, local optional) | Local & RunPod | 5 |
| `pipeline/io/output.py` | Double-buffering to eliminate frame copies | Local & RunPod | 5 |

## New Files Required

| File | Purpose | Applies To | Step |
|------|---------|-----------|------|
| `requirements.txt` | Root requirements file for deployment install | Local & RunPod | 4A |
| `.env.example` | Example env file for optional settings: `PHANTOM_API_URL` (RunPod pod URL), `EXECUTION_PROVIDER` (cpu/cuda/rocm/dml), `API_PORT` (default 9000), `LOG_LEVEL` | Local & RunPod | 4A |
| `runpod/startup.sh` | RunPod-only: pod initialization script (install FFmpeg, check CUDA, create /workspace/models dir, expose port 9000) | RunPod only | 4B |
| `RUNPOD_DEPLOYMENT.md` | RunPod-only: dedicated deployment guide (pod creation steps, Network Volume mounting, port 9000 exposure, first-run setup, troubleshooting, cost estimates, GPU tier recommendations) | RunPod only | 4B |

## Low-Latency Requirements

For real-time face swapping, the entire frame delivery path must be push-based and minimal:

```
ProcessingPipeline → FRAME_READY event → WebSocket server → push JPEG binary → desktop client → display
```

Target: **< 16ms frame delivery** (60fps budget) from pipeline output to desktop display.

- Frame encoding: **JPEG quality 85** (not PNG) — ~3x smaller, ~5x faster encode
- Transport: **WebSocket binary frames** (not HTTP polling) — persistent, zero connection overhead
- Events: **WebSocket JSON push** — desktop reacts immediately to status/detection changes
- No polling anywhere in the stack

---

## Latency Bottleneck Analysis

Beyond the HTTP/WebSocket issue, a full audit of the hot path reveals many additional latency sources. The per-frame worst-case budget at 30fps is 33ms; the current stack can take 255–855ms.

### Per-Frame Latency Breakdown (Worst Case)

| Stage | Estimated Cost |
|-------|---------------|
| Capture (OpenCV buffer) | 10–20ms |
| Face detection (ONNX) | 50–150ms |
| Face tracking (OpenCV) | 5–20ms |
| Face swapping (ONNX) | 50–200ms |
| Luminance blend | 10–20ms |
| Enhancement (async) | 100–300ms (if queue backed up) |
| PNG encoding on event | 20–100ms |
| HTTP response overhead | 5–10ms |
| Desktop QImage + display | 5–15ms |
| **Total worst case** | **255–855ms** |

---

### L1 — Synchronous Event Bus Blocks Pipeline
**File**: `pipeline/events.py:59-61`
**Impact**: 50–100ms added per frame

`EventBus.emit()` calls all handlers sequentially and synchronously. When `FRAME_READY` is emitted, the pipeline thread stalls while every handler runs — including PNG encoding in `server.py` and any desktop bridge logic. The pipeline cannot process the next frame until every handler completes.

```python
# events.py:59-61 — BLOCKS until all handlers finish
for handler in self._handlers.get(event, []):
    handler(**data)
```

**Fix**: Dispatch handlers via a thread pool (`concurrent.futures.ThreadPoolExecutor`) or use a dedicated handler queue so the emitting thread is never blocked.

---

### L2 — PNG Encoding on the Event Hot Path
**File**: `pipeline/api/server.py:266-275`
**Impact**: 20–100ms per frame

`_on_frame_ready()` encodes the frame to PNG synchronously during the event callback. PNG compression is expensive. This runs inside `emit()`, so the main pipeline thread pays the full encoding cost.

```python
def _on_frame_ready(self, frame: Any, seq: int) -> None:
    success, png_data = cv2.imencode('.png', frame)  # expensive, blocking
    if success:
        self._set_frame(png_data.tobytes())
```

**Fix**: Move encoding off the hot path. Store raw frame in frame buffer; encode to JPEG in a background thread or only when a client requests the frame. Switch `.png` → `.jpg` immediately for a 3–5× speedup even while async encoding is being implemented.

---

### L3 — AsyncProcessor Queue Too Small + Wrong Timeout
**File**: `pipeline/processing/async_processor.py:47-48, 117`
**Impact**: 10–30ms latency, silent frame drops

Both input and output queues have `maxsize=1`. The worker thread calls `queue.get(timeout=0.1)` — a 100ms timeout that adds worst-case 100ms latency on an empty queue. When enhancement is slower than frame arrival, frames are silently dropped with no backpressure feedback to the pipeline.

```python
self._input_queue: queue.Queue = queue.Queue(maxsize=1)   # too small
self._output_queue: queue.Queue = queue.Queue(maxsize=1)  # too small
seq, frame = self._input_queue.get(timeout=0.1)           # 100ms max wait
```

**Fix**: Increase queue size to 2–3 frames. Reduce timeout to 5–10ms. Add a drop counter that is emitted as a metric event so the pipeline can adapt.

---

### L4 — Frame Copies on Every Write and Read
**Files**: `pipeline/io/output.py:143, 154, 203`; `desktop/bridge.py:40-42`
**Impact**: 5–15ms per frame

Every output sink calls `frame.copy()` on write and again on read. The desktop bridge additionally calls `np.ascontiguousarray()` then `.copy()` on the QImage. A 1080p frame is ~6MB; copying it 3–4 times per pipeline tick is measurable.

```python
# output.py:143
self._latest_frame = frame.copy()   # copy 1
# output.py:154
return self._latest_frame.copy()    # copy 2
# bridge.py:42
qimg = qimg.copy()                  # copy 3
```

**Fix**: Use double-buffering or a lock-free ring buffer. The reader can reference the same allocation as long as the writer atomically swaps the pointer. Only copy when a frame must be sent over the wire.

---

### L5 — Periodic Full Face Redetection Causes Latency Spikes
**File**: `pipeline/processing/frame_processor.py:171-174`
**Impact**: 100–300ms spike every 30 frames

`TrackingProcessor` runs a full face detection call every `redetect_interval` frames (default 30). At 30fps that is a detection spike once per second. Detection takes 50–150ms and runs synchronously on the pipeline thread.

```python
if self._frame_count % self.redetect_interval == 0:
    det = self.detector.detect_one(frame)  # 50-150ms, synchronous
```

**Fix**: Move redetection to a background thread. The tracker continues with the last known face while detection runs asynchronously. Swap in the new detection result when it returns.

---

### L6 — OpenCV Capture Buffer Not Set
**File**: `pipeline/processing/pipeline.py:196-200`
**Impact**: 30–100ms of stale frames

`VideoCapture` is created without setting `CAP_PROP_BUFFERSIZE`. The default buffer is 1–3 frames. On slow hardware it can be larger, meaning the pipeline processes frames that are already 100ms old before any processing begins.

**Fix**: Add one line immediately after capture creation:
```python
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```
This is already done in `desktop/bridge.py:460` for the desktop path — the pipeline path is missing it.

---

### L7 — Luminance LAB Conversion on Every Swap
**File**: `pipeline/processing/frame_processor.py:423-428`
**Impact**: 10–20ms per swapped frame

`_luminance_adaptive_blend()` converts two face ROI crops to LAB color space on every frame to measure luminance for adaptive blending.

```python
orig_lum = float(cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0].mean())
swap_lum = float(cv2.cvtColor(swapped[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0].mean())
```

**Fix**: Cache the original face's luminance from the detection step. Only compute swapped luminance per frame (one conversion instead of two). Alternatively compute luminance directly from BGR channel means (weighted sum), avoiding color space conversion entirely.

---

### L8 — Lazy Model Loading on First Frame
**File**: `pipeline/services/face_detection.py:43-57`
**Impact**: 1–3 second spike on first frame

`FaceDetector` does not load the InsightFace model until the first call to `detect()`. Model loading takes 1–3 seconds and holds a lock, blocking any concurrent detection calls.

```python
if self._analyser is None:
    with self._lock:
        if self._analyser is None:
            self._analyser = insightface.app.FaceAnalysis(...)
            self._analyser.prepare(ctx_id=0, det_size=(640, 640))
```

**Fix**: Call `_ensure_loaded()` eagerly in `__init__()` or in a background thread during startup warmup, so the first real frame does not pay the loading cost.

---

### L9 — No Frame Skipping Under Load
**File**: `pipeline/processing/pipeline.py`
**Impact**: Cascading delay when processing can't keep up

The stream loop processes every frame regardless of whether the pipeline is falling behind. If detection takes 200ms and frames arrive every 33ms, the queue depth grows unboundedly and displayed frames become increasingly stale.

**Fix**: Measure per-frame processing time. If processing time exceeds frame interval, skip every Nth input frame. Signal adaptive skipping through `CONFIG` so the UI can show a "performance degraded" indicator.

---

### L10 — Status Polling Every 2 Seconds in Desktop
**File**: `desktop/bridge.py:198-201`
**Impact**: Up to 2000ms UI lag for status changes

The desktop polls `GET /status` on a 2-second timer. Pipeline state changes (embedding ready, source changed, errors) take up to 2 seconds to appear in the UI.

**Fix**: Once WebSocket is implemented (see Step 3), remove the polling timer entirely. `STATUS_CHANGED` events are already emitted on the bus — just push them to the desktop client.

---

### L11 — Enhancement Runs Every N Frames Causing Flicker
**File**: `pipeline/processing/frame_processor.py:315-323`
**Impact**: Visual quality flicker

`EnhancementProcessor` skips enhancement every N frames (default 5). Frames alternate between enhanced and non-enhanced quality, causing visible quality jitter in the output stream.

**Fix**: Always run enhancement asynchronously on every frame. Use the async processor to absorb the cost. If the enhancer can't keep up, show the last enhanced frame instead of the non-enhanced raw frame (temporal consistency).

---

### L12 — GIL Contention Across All ML Threads
**Files**: `pipeline/services/face_detection.py`, `face_swapping.py`, `enhancement.py`
**Impact**: 5–20ms effective serialization

InsightFace, ONNX runtime, and GFPGAN all acquire the Python GIL during inference even when running on separate threads. Multiple ML calls cannot truly parallelize.

**Fix**: Use ONNX runtime's built-in thread pool (`session_options.intra_op_num_threads`) and set execution mode to `ORT_PARALLEL`. For InsightFace, pre-run warmup to populate ONNX session caches. This won't fully remove GIL contention but reduces idle waiting.

---

### L13 — `warmup_frames` Config Field Is Defined but Never Used
**File**: `pipeline/config.py:50`; `pipeline/processing/pipeline.py`

`FaceSwapConfig` defines `warmup_frames: int = 5` but the pipeline never reads it. The first N frames hit cold model caches and tracker initialization, producing degraded output.

**Fix**: In `run_stream()`, drop (don't emit) the first `CONFIG.warmup_frames` frames. This avoids sending low-quality or glitched output to the desktop during warmup.

---

### Latency Fix Priority Table

| # | Issue | File | Impact | Difficulty |
|---|-------|------|--------|------------|
| L1 | Synchronous event bus | `events.py:59` | 50–100ms/frame | Hard |
| L2 | PNG encoding on hot path | `server.py:266` | 20–100ms/frame | Medium |
| L3 | AsyncProcessor queue/timeout | `async_processor.py:47` | 10–30ms + drops | Medium |
| L4 | Frame copies on write+read | `output.py:143,154` | 5–15ms/frame | Medium |
| L5 | Synchronous redetection | `frame_processor.py:171` | 100–300ms spike | Hard |
| L6 | OpenCV buffer not set | `pipeline.py:196` | 30–100ms | Trivial |
| L7 | LAB luminance both crops | `frame_processor.py:423` | 10–20ms/frame | Easy |
| L8 | Lazy model loading | `face_detection.py:49` | 1–3s first frame | Easy |
| L9 | No frame skipping | `pipeline.py` | Cascading delay | Medium |
| L10 | Status polling 2s interval | `bridge.py:198` | 2000ms UI lag | Easy (after WS) |
| L11 | Enhancement every N frames | `frame_processor.py:315` | Visual flicker | Medium |
| L12 | GIL across ML threads | Multiple services | 5–20ms serial | Hard |
| L13 | warmup_frames unused | `pipeline.py` | First-frame glitch | Easy |

### Recommended Immediate Wins (Trivial/Easy fixes, high impact)

1. **`pipeline.py`**: Add `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` after VideoCapture creation — one line, 30–100ms saved (L6)
2. **`server.py:270`**: Change `.png` → `.jpg` in `cv2.imencode` — one character, 3–5× encoding speedup (L2 partial)
3. **`face_detection.py`**: Call `_ensure_loaded()` in `__init__` — eliminates first-frame 1–3s spike (L8)
4. **`async_processor.py:117`**: Change `timeout=0.1` → `timeout=0.01` — reduces worst-case queue wait 10× (L3 partial)
5. **`pipeline/processing/pipeline.py`**: Implement `warmup_frames` skip logic — eliminates glitchy first frames (L13)
