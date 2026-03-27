# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Phantom is a modern, composable face-swapping application for videos and images. It uses deep learning models (ONNX-based face detection and swapping via InsightFace) to replace faces in media with high quality.

**Architecture**: Clean, event-driven, service-oriented design with unified ProcessingPipeline for both batch and realtime modes. No global state.

**Two entry points**:
- `pipeline.py` (headless engine): Supports batch mode (`-s <source> -t <target> -o <output>`) and realtime stream mode
- `desktop.py` (GUI controller): Qt/PySide6 interface, communicates with pipeline via WebSocket API on port 9000

## Quick Commands

### Running
- **Pipeline engine**: `python pipeline.py`
- **Desktop GUI**: `python desktop.py`
- **CLI batch mode**: `python pipeline.py -s <source_image> -t <target_video> -o <output_path>`
- **With CUDA**: `python pipeline.py --execution-provider cuda`

### Development
- **Lint**: `flake8 pipeline.py pipeline desktop`
- **Type check**: `mypy pipeline.py pipeline desktop`
- **Test**: `python pipeline.py -s=.github/examples/source.jpg -t=.github/examples/target.mp4 -o=.github/examples/output.mp4`

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
- **pipeline/services/enhancement.py**: `Enhancer` (GFPGAN face enhancement)
- **pipeline/services/face_tracking.py**: `FaceTrackerState` (OpenCV tracker wrapper)
- **pipeline/services/database.py**: `FaceDatabase` (embedding cache & averaging)

**Processing Pipeline:**
- **pipeline/processing/frame_processor.py**: `FrameProcessor` ABC + implementations
  - `DetectionProcessor`, `TrackingProcessor`, `SwappingProcessor`, `EnhancementProcessor`, `BlendingProcessor`
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
3. **Batch mode**: `ProcessingPipeline.run_batch()` → detects faces → swaps → enhances (if enabled) → outputs
4. **Stream mode**: `ProcessingPipeline.run_stream()` → captures frames → detects/tracks → swaps → enhances (if enabled, synchronous) → emits `FRAME_READY` event
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

|                        | Fast              | Optimal (default) | Production        |
|------------------------|-------------------|--------------------|-------------------|
| **Capture resolution** | 480x270           | 640x360            | 960x540           |
| **Frame rate**         | 15 fps            | 20 fps             | 30 fps            |
| **JPEG quality**       | 60                | 70                 | 85                |
| **Tracker**            | KCF (fast)        | CSRT (accurate)    | CSRT (accurate)   |
| **Alpha smoothing**    | 0.7               | 0.6                | 0.5               |
| **Luminance blend**    | Off               | On                 | On                |
| **Redetect interval**  | Every 30 frames   | Every 30 frames    | Every 20 frames   |

Changing quality restarts the webcam capture device to apply new resolution/fps.

### Enhancement Toggle
GFPGAN face enhancement is controlled by `config.enhance` (bool, default `True`) — independent of quality presets. Toggled from the desktop header via the ENHANCE button or `set_enhance` API command. Enhancement runs **synchronously** in the frame processing loop on GPU; no frames are dropped or skipped.

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
- Run before commit: `mypy pipeline.py pipeline desktop` and `flake8 pipeline.py pipeline desktop`

## Dependencies & Environment

### Runtime
- **Python**: 3.9+ (required for type annotations)
- **Deep Learning**: `torch`, `onnxruntime`, `tensorflow`, `insightface`
- **Computer Vision**: `opencv-python`, `pillow`
- **Enhancement**: `gfpgan` (optional, graceful fallback if missing)
- **GUI**: `customtkinter` (for desktop.py)
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
- Write complete type annotations; run `mypy pipeline.py pipeline desktop` locally
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
- `pipeline/services/enhancement.py`: `Enhancer` GFPGAN optional enhancement
- `pipeline/services/database.py`: `FaceDatabase` embedding cache & averaging

### Processing Pipeline
- `pipeline/processing/pipeline.py`: `ProcessingPipeline` orchestrator (batch & stream modes)
- `pipeline/processing/frame_processor.py`: `FrameProcessor` ABC + 5 implementations

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
- **GPU auto-discovery**: By default, queries RunPod API for GPUs matching `RUNPOD_MIN_VRAM` (default 16GB) and `RUNPOD_MAX_PRICE` (default $1.00/hr), tries cheapest first. Set `RUNPOD_GPU_TYPES` to override with specific GPUs.
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
