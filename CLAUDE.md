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
- **Basic**: `pip install -r requirements.txt`
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
- **pipeline/processing/async_processor.py**: `AsyncProcessor` (background thread wrapper)
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
- **pipeline/api/handlers.py**: Type-safe command handlers; `HandlerContext` dataclass (no globals)
- **pipeline/api/schema.py**: Message types, command/event constants

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
3. **Batch mode**: `ProcessingPipeline.run_batch()` → detects faces → swaps → enhances → outputs
4. **Stream mode**: `ProcessingPipeline.run_stream()` → captures frames → detects/tracks → swaps → async enhancement → emits `FRAME_READY` event
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
- `pipeline/processing/async_processor.py`: Background processing wrapper

### I/O & API
- `pipeline/io/capture.py`: Input sources (webcam, file, network)
- `pipeline/io/output.py`: Output sinks (file, HTTP, WebSocket)
- `pipeline/api/server.py`: WebSocket API server (replaces HTTP control)
- `pipeline/api/handlers.py`: Command dispatching & business logic

### Entry Points & Config
- `pipeline/core.py`: CLI argument parsing, headless orchestration
- `pipeline/stream.py`: Stream mode convenience wrapper
- `.flake8`: Linting configuration (E3, E4, F only)
- `mypy.ini`: Type checking (strict mode)
- `.github/workflows/ci.yml`: CI pipeline (mypy → flake8 → test)
