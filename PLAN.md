# Phantom Remediation Plan

Based on the full audit (AUDIT.md, 2026-03-13), this plan addresses every issue in 6 phases, ordered by dependency and priority.

**Status**: All 6 phases complete as of 2026-03-14.

---

## Phase 1: Unblock Startup ✅ COMPLETE (Critical — App Cannot Run)

**Goal**: Fix broken imports so the application starts without errors.

**Issues addressed**: Audit #1, #2

### Task 1.1 — Fix import paths in old processor files
- `pipeline/processors/frame/face_enhancer.py:10` — change `from pipeline.typing import Frame, Face` → `from pipeline.types import Frame, Face`
- `pipeline/processors/frame/face_swapper.py:11` — change `from pipeline.typing import Frame, Face` → `from pipeline.types import Frame, Face`
- `pipeline/face_analyser.py:11` — change `from pipeline.typing import Frame, Face` → `from pipeline.types import Frame, Face`

### Task 1.2 — Fix missing `update_status` function references
- `pipeline/processors/frame/face_swapper.py:9` — change `from pipeline.core import update_status` → `from pipeline.logging import emit_status`
- `pipeline/processors/frame/face_enhancer.py:8` — change `from pipeline.core import update_status` → `from pipeline.logging import emit_status`
- Replace all `update_status(...)` call sites with `emit_status(...)`:
  - `face_swapper.py` lines 44, 76, 82, 85, 93, 97, 100, 103
  - `face_enhancer.py` lines 38-39

### Task 1.3 — Verify startup
- Run `python pipeline.py --help` to confirm no ImportError
- Run `python -c "from pipeline.processors.frame.face_swapper import *"` to verify imports

**Estimated scope**: 3 files, ~15 line changes

---

## Phase 2: Remove Dead Code ✅ COMPLETE (High Priority)

**Goal**: Delete orphaned old-architecture files that duplicate new services and cause confusion.

**Issues addressed**: Audit #3, #4, #5, #9, #11

**Dependency**: Phase 1 must be complete first (we fix imports before deleting files, so we can verify nothing else depends on them).

### Task 2.1 — Audit remaining references to old modules
Before deleting, grep the entire codebase for imports of:
- `pipeline.processors.frame.face_swapper`
- `pipeline.processors.frame.face_enhancer`
- `pipeline.processors.frame.core`
- `pipeline.face_analyser`
- `pipeline.typing`
- `pipeline.utilities`
- `pipeline.ws_server`
- `pipeline.capturer`

If any live code still imports these, migrate those references to the new equivalents first.

### Task 2.2 — Delete superseded processor files
- Delete `pipeline/processors/frame/face_swapper.py` (replaced by `pipeline/processing/frame_processor.py::SwappingProcessor`)
- Delete `pipeline/processors/frame/face_enhancer.py` (replaced by `pipeline/processing/frame_processor.py::EnhancementProcessor`)
- Delete `pipeline/processors/frame/core.py` (orphaned, no new equivalent needed)
- Delete `pipeline/processors/frame/__init__.py` if it exists and is now empty
- Delete `pipeline/processors/` directory if fully empty

### Task 2.3 — Delete duplicate modules
- Delete `pipeline/face_analyser.py` (replaced by `pipeline/services/face_detection.py::FaceDetector`)
- Delete `pipeline/typing.py` (replaced by `pipeline/types.py`)

### Task 2.4 — Delete unused pre-migration files
- Delete `pipeline/ws_server.py` (unused, superseded by new API server)
- Delete `pipeline/capturer.py` (unused, superseded by `pipeline/io/capture.py`)

### Task 2.5 — Delete duplicate utilities
- Verify `pipeline/utilities.py` functions are fully covered by `pipeline/io/ffmpeg.py`
- If any unique functions remain, migrate them to the appropriate new module
- Delete `pipeline/utilities.py`

### Task 2.6 — Clean up dead code in core
- Remove `del torch` at `pipeline/core.py:26` (fragile, unnecessary)

### Task 2.7 — Verify after cleanup
- Run `python pipeline.py --help` — no ImportError
- Run `mypy pipeline.py pipeline desktop` — no missing module errors
- Run `flake8 pipeline.py pipeline desktop` — no undefined name errors

**Estimated scope**: 7 files deleted, 1 file edited

---

## Phase 3: Real WebSocket Server ✅ COMPLETE (Critical for Real-Time)

**Goal**: Replace the HTTP-based API server with a true WebSocket server for push-based frame delivery and event streaming. This is the single most impactful architectural change.

**Issues addressed**: Audit #7, #6, #8, #10 (L10)

**Dependency**: Phase 2 complete (dead code removed so we're only working with the new architecture).

### Task 3.1 — Replace HTTP server with WebSocket server
**File**: `pipeline/api/server.py`

- Replace `http.server.HTTPServer` with `websockets` library (already in requirements)
- Server listens on port 9000 (single port, not split 9000/9001)
- Accept WebSocket connections at `ws://host:9000/ws`
- Protocol:
  - **Text frames**: JSON messages for commands and events
  - **Binary frames**: JPEG-encoded video frames
- On `FRAME_READY` event from EventBus: encode frame as JPEG (quality 85), push binary to all connected clients
- On `STATUS_CHANGED`, `DETECTION` events: push JSON text frame to all connected clients
- Receive commands as JSON text frames: `{"action": "set_source", "data": {...}}`
- Add `/health` HTTP endpoint returning `{"status": "healthy", "uptime": <seconds>}` (can be a simple HTTP fallback or a WebSocket message)

### Task 3.2 — Remove global state from handlers
**File**: `pipeline/api/handlers.py`

- Remove module-level `_pipeline` and `_shutdown_event` globals (lines 25-40)
- Create a `HandlerContext` dataclass or pass pipeline/shutdown references via constructor
- Handlers receive context through dependency injection, not module globals
- This enables unit testing handlers in isolation

### Task 3.3 — Unify face type representations
**Files**: `pipeline/types.py`, `pipeline/services/database.py`

Three incompatible face types currently exist:
1. Raw `insightface.app.common.Face`
2. `pipeline.types.Detection` dataclass
3. `types.SimpleNamespace(normed_embedding=...)` in database.py

Resolution:
- `Detection` dataclass becomes the single canonical type
- `FaceDetector` returns `List[Detection]` (already does)
- `FaceDatabase` accepts/returns `Detection` objects (update `SimpleNamespace` usage)
- Add a `Detection.from_insightface(face)` class method if raw conversion is needed anywhere

### Task 3.4 — Update desktop to use WebSocket push
**Files**: `desktop/bridge.py`, `desktop/controller.py`

- `controller.py`: Replace HTTP client with WebSocket client (`websockets` library)
  - Single persistent connection to `ws://host:9000/ws`
  - Send commands as JSON text frames
  - Receive events and frames over the same connection
- `bridge.py`:
  - Remove 2-second status polling timer (line 201) — status now pushed via WebSocket
  - Remove HTTP `/frame` polling — frames now pushed as binary WebSocket messages
  - Keep existing WebSocket frame receiver logic (port 9001) as reference, but consolidate to single port 9000
  - Handle reconnection on disconnect (exponential backoff)

### Task 3.5 — Verify WebSocket integration
- Start pipeline: `python pipeline.py --stream`
- Connect desktop: `python desktop.py`
- Verify: frames display in real-time without polling
- Verify: status changes appear immediately (not 2s delayed)
- Verify: commands (set_source, start_stream, etc.) work over WebSocket

**Estimated scope**: 4 files heavily modified, ~400-600 lines changed

---

## Phase 4: Deployment & Configuration ✅ COMPLETE

**Goal**: Fix deployment infrastructure for both local and RunPod modes.

**Issues addressed**: Audit Step 4A, Step 4B

**Dependency**: Phase 3 complete (WebSocket server is the foundation for both deployment modes).

### Phase 4A — Local Development

#### Task 4A.1 — Fix requirements
- Create root `requirements.txt` that references the appropriate sub-requirements or consolidates them
- Remove desktop GUI packages (`customtkinter`, `tk`) from pipeline requirements (they belong in `desktop/requirements.txt` only)

#### Task 4A.2 — Add CLI flags
**File**: `pipeline/core.py`

- Add `--stream` flag to argument parser (start in stream mode)
- Add `--log-level` flag (maps to `CONFIG.log_level`)
- Add CUDA availability check at startup (log GPU name if available)

#### Task 4A.3 — Add environment variable support
- Add `python-dotenv` to requirements
- Create `.env.example` with documented variables:
  ```
  EXECUTION_PROVIDER=cuda
  API_PORT=9000
  LOG_LEVEL=info
  PHANTOM_API_URL=ws://localhost:9000/ws
  ```
- Load `.env` in `pipeline/core.py` at startup (before config initialization)

#### Task 4A.4 — Add health endpoint
**File**: `pipeline/api/server.py`

- Add `/health` endpoint returning `{"status": "healthy", "uptime": <seconds>}`
- Can be HTTP GET on the same port or a WebSocket command

#### Task 4A.5 — Update local deployment docs
**File**: `DEPLOYMENT.md`

- Add "Local Development" section
- Document running `pipeline.py` and `desktop.py` on same machine
- Default connection: `ws://localhost:9000/ws`
- Model cache: `~/.insightface/models/` (standard, no special handling)

### Phase 4B — RunPod Cloud Deployment

#### Task 4B.1 — Create RunPod deployment guide
- Create `RUNPOD_DEPLOYMENT.md` with:
  - Step-by-step pod creation in RunPod UI
  - Network Volume setup for persistent model cache
  - Port 9000 exposure
  - GPU tier recommendations and cost estimates
  - Troubleshooting section

#### Task 4B.2 — Create pod startup script
- Create `runpod/startup.sh`:
  - Install FFmpeg
  - Check CUDA availability, log GPU name
  - Create `/workspace/models` if not present
  - Optional model pre-warming

#### Task 4B.3 — Network Volume model caching
**File**: `pipeline/services/face_detection.py`

- Check `/workspace/models/insightface/` first (RunPod Network Volume)
- Fall back to `~/.insightface/models/` (default)
- Log which cache path is being used

#### Task 4B.4 — Remote connection support
**File**: `desktop/controller.py`

- Support `PHANTOM_API_URL` env var for remote connections
- Parse RunPod pod URL → construct `wss://<pod-url>:9000/ws`
- Add connection timeout (30s) and retry with exponential backoff (max 3 retries)
- Display connection status in UI

#### Task 4B.5 — Binary file transfer over WebSocket
- Desktop sends source image as binary WebSocket message with metadata header
- Desktop sends target video as binary WebSocket message (or chunked for large files)
- Pipeline receives, saves to temp path, processes
- Document maximum recommended file sizes

#### Task 4B.6 — Cloud error handling
**File**: `pipeline/api/server.py`

- WebSocket heartbeat: ping/pong every 30s
- OOM event: emit `OOM` event with degradation options
- Connection drop: auto-reconnect with exponential backoff
- GPU OOM during swap: retry with lower resolution

**Estimated scope**: 5-6 files modified, 2-3 new files, ~500-800 lines

---

## Phase 5: Latency Optimization ✅ COMPLETE (Performance)

**Goal**: Reduce per-frame latency from 255-855ms worst case toward the 33ms budget (30fps) or 16ms budget (60fps).

**Issues addressed**: Audit L1-L13

**Dependency**: Phase 3 complete (WebSocket server eliminates HTTP overhead). Some items can be done in parallel.

### Tier 1 — Trivial/Easy Wins (do first, high impact, low risk)

#### Task 5.1 — Set OpenCV capture buffer size (L6)
**File**: `pipeline/processing/pipeline.py`

Add after VideoCapture creation:
```python
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```
**Impact**: 30-100ms saved. One line.

#### Task 5.2 — Switch PNG to JPEG encoding (L2 partial)
**File**: `pipeline/api/server.py`

Change `cv2.imencode('.png', frame)` → `cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])`

**Impact**: 3-5x encoding speedup (20-100ms → 5-20ms).

*Note: If Phase 3 is complete, this is already done. Include here as a checkpoint.*

#### Task 5.3 — Eager model loading (L8)
**File**: `pipeline/services/face_detection.py`

Call `_ensure_loaded()` in `__init__()` (or in background thread during startup).

**Impact**: Eliminates 1-3s first-frame spike.

#### Task 5.4 — Reduce async processor timeout (L3 partial)
**File**: `pipeline/processing/async_processor.py:117`

Change `timeout=0.1` → `timeout=0.01`.

**Impact**: 90ms worst-case queue wait eliminated.

#### Task 5.5 — Implement warmup_frames skip (L13)
**File**: `pipeline/processing/pipeline.py`

In `run_stream()`, drop the first `CONFIG.warmup_frames` frames (don't emit `FRAME_READY` for them).

**Impact**: Eliminates glitchy first frames during model warmup.

### Tier 2 — Medium Difficulty, Significant Impact

#### Task 5.6 — Increase async processor queue size (L3)
**File**: `pipeline/processing/async_processor.py:47-48`

Change `maxsize=1` → `maxsize=3` for both input and output queues. Add a drop counter metric.

**Impact**: Reduces silent frame drops, provides backpressure visibility.

#### Task 5.7 — Cache luminance computation (L7)
**File**: `pipeline/processing/frame_processor.py:423-428`

Cache the original face's LAB luminance from the detection step. Only compute swapped luminance per frame. Alternatively, compute luminance from BGR channel weighted sum (no color space conversion).

**Impact**: 10-20ms/frame saved.

#### Task 5.8 — Frame skipping under load (L9)
**File**: `pipeline/processing/pipeline.py`

Measure per-frame processing time. If processing time exceeds frame interval, skip every Nth input frame. Signal via CONFIG so UI can show degraded performance indicator.

**Impact**: Prevents cascading delay when pipeline can't keep up.

#### Task 5.9 — Double-buffering for frame output (L4)
**Files**: `pipeline/io/output.py:143,154`

Replace `frame.copy()` on write and read with atomic pointer swap (double buffer). Only copy when frame must be serialized for transmission.

**Impact**: 5-15ms/frame saved (eliminates 2-3 unnecessary 6MB copies per frame).

#### Task 5.10 — Enhancement every frame with temporal consistency (L11)
**File**: `pipeline/processing/frame_processor.py:315-323`

Always run enhancement asynchronously. If enhancer can't keep up, display last enhanced frame instead of raw (temporal consistency over quality jitter).

**Impact**: Eliminates visual flicker between enhanced/non-enhanced frames.

### Tier 3 — Hard, High Impact

#### Task 5.11 — Async event bus dispatch (L1)
**File**: `pipeline/events.py:59-61`

Replace synchronous handler iteration with `concurrent.futures.ThreadPoolExecutor` dispatch. Emitting thread is never blocked by handler execution.

**Impact**: 50-100ms/frame saved. Biggest single latency win.

#### Task 5.12 — Background redetection (L5)
**File**: `pipeline/processing/frame_processor.py:171-174`

Move periodic face redetection to a background thread. Tracker continues with last known face while detection runs async. Swap in new detection when ready.

**Impact**: Eliminates 100-300ms spikes every 30 frames.

#### Task 5.13 — ONNX thread pool optimization (L12)
**Files**: `pipeline/services/face_detection.py`, `face_swapping.py`, `enhancement.py`

Set `session_options.intra_op_num_threads` and execution mode to `ORT_PARALLEL`. Pre-run warmup to populate ONNX session caches.

**Impact**: 5-20ms reduction in GIL contention (partial fix, full fix requires multiprocessing).

**Estimated scope**: ~13 tasks across 8 files, ~300-500 lines changed

---

## Phase 6: Final Cleanup ✅ COMPLETE

**Goal**: Remove remaining technical debt and verify overall system health.

**Issues addressed**: Audit Step 6, remaining items

**Dependency**: All prior phases complete.

### Task 6.1 — Verify utilities.py removal
If not already deleted in Phase 2, confirm `pipeline/utilities.py` has no remaining references and delete it.

### Task 6.2 — Audit desktop/bridge.py
Verify bridge.py uses the new event API correctly after Phase 3 WebSocket changes. Check:
- No remaining HTTP polling code
- WebSocket reconnection works
- Frame buffer thread safety is maintained
- Virtual camera output still works

### Task 6.3 — Update CLAUDE.md
Update architecture documentation to reflect:
- WebSocket server (not HTTP)
- Single port 9000 (not split ports)
- Removed files list
- Updated data flow diagram

### Task 6.4 — Run full test suite
- `mypy pipeline.py pipeline desktop` — all type checks pass
- `flake8 pipeline.py pipeline desktop` — all lint checks pass
- `python pipeline.py -s=.github/examples/source.jpg -t=.github/examples/target.mp4 -o=.github/examples/output.mp4` — batch mode works
- Manual stream mode test with desktop GUI

### Task 6.5 — Archive or delete AUDIT.md
Once all issues are resolved, either delete AUDIT.md or move it to a `docs/archived/` directory with a completion note.

**Estimated scope**: Documentation updates, verification, minimal code changes

---

## Phase Summary

| Phase | Focus | Issues Addressed | Priority | Est. Files Changed |
|-------|-------|-----------------|----------|-------------------|
| 1 | Unblock Startup | #1, #2 | Critical | 3 |
| 2 | Remove Dead Code | #3, #4, #5, #9, #11 | High | 7 deleted, 1 edited |
| 3 | WebSocket Server | #7, #6, #8, L10 | Critical | 4 |
| 4 | Deployment & Config | Step 4A, 4B | Medium | 5-6 modified, 2-3 new |
| 5 | Latency Optimization | L1-L13 | Medium-High | 8 |
| 6 | Final Cleanup | Step 6 | Low | 2-3 |

### Dependency Graph

```
Phase 1 (startup fixes)
  └→ Phase 2 (dead code removal)
       └→ Phase 3 (WebSocket server)  ← most impactful
            ├→ Phase 4A (local deployment)
            ├→ Phase 4B (RunPod deployment)
            └→ Phase 5 (latency optimization)
                 └→ Phase 6 (cleanup & verification)
```

Phases 4A, 4B, and 5 can be worked on in parallel once Phase 3 is complete. Within Phase 5, Tier 1 tasks are independent and can all be done simultaneously.
