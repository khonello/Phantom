# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
roop-cam is a face-swapping application for videos and images. It uses deep learning models (ONNX-based face detection and swapping via InsightFace) to replace faces in media with high quality. Two entry points: `pipeline.py` (headless engine) and `desktop/` (GUI controller), communicating via HTTP on port 9000.

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

### Core Modules
- **pipeline/core.py**: Main entry point, argument parsing, execution flow control
- **pipeline/globals.py**: Shared application state (settings, paths, configuration)
- **pipeline/predicter.py**: High-level prediction API (`predict_image()`, `predict_video()`)
- **pipeline/face_analyser.py**: Face detection and analysis using InsightFace
- **pipeline/capturer.py**: Video frame extraction and capture handling
- **pipeline/utilities.py**: Video/image utilities (FFmpeg integration, frame extraction, video creation)
- **pipeline/stream.py**: Webcam capture loop, face swap pipeline, frame serving
- **pipeline/control.py**: HTTP control server (`GET /status`, `GET /frame`, `POST /control`)
- **pipeline/api/schema.py**: `PRESETS` (quality presets) and `COMMANDS` (control API schema)
- **pipeline/processors/frame/**: Frame processing pipeline
  - `core.py`: Processor module loading and orchestration
  - `face_swapper.py`: ONNX-based face swap operation
  - `face_enhancer.py`: Face enhancement post-processing
- **desktop/ui.py**: CustomTkinter GUI (main window, file selection, settings UI)
- **desktop/controller.py**: HTTP client (`PipelineClient`) for communicating with the pipeline

### Data Flow
1. `pipeline.py` → `pipeline/core.py` parses args → sets `pipeline/globals` state
2. Control server starts on port 9000 (always)
3. Batch mode: `predicter.predict_video()` → frame processors → output
4. Stream mode: `pipeline/stream.py` → webcam loop → face swap → `_latest_frame` served via `/frame`
5. Desktop polls `GET /frame` every 33ms for live preview

### Two Entry Points
- **pipeline.py**: Always headless engine; starts control server + stream pipeline or batch job
- **desktop.py**: Always GUI; connects to pipeline via HTTP, never processes frames itself

## Code Style & Standards

### Functional Programming Only
- **No OOP**: Classes are only used for framework requirements (CustomTkinter). Avoid creating new classes.
- Use functional composition for logic flow.

### Naming & Comments
- Use clear, self-documenting names.
- Comments only for non-obvious logic.
- Avoid verbose docstrings; let code speak for itself.

### Type Checking
- Strict mypy enabled (`disallow_untyped_defs = True`, `disallow_any_generics = True`)
- All functions must have complete type annotations
- `ignore_missing_imports = True` allows third-party stubs to be optional

### Linting
- flake8 checks: E3, E4, F
- Exception: `pipeline/core.py` ignores E402 (imports after code) for performance-critical initialization

## Dependencies & Environment
- Python 3.9+
- Core AI: `torch`, `onnxruntime`, `tensorflow`, `insightface`
- CV: `opencv-python`, `pillow`, `gfpgan`
- GUI: `customtkinter`
- Video: FFmpeg (external tool, not pip)
- Platform-specific: CUDA variants for torch/onnxruntime on Linux/Windows, M1 silicon support for macOS

## PR Guidelines (from CONTRIBUTING.md)
- One PR per feature/fix; consult before major changes
- Prioritize bug fixes over features
- Proper testing required before submission
- Resolve CI pipeline failures
- **Don't**: introduce OOP, ignore requirements, massive code changes, proof-of-concepts, undocumented APIs

## Key Files
- `.flake8`: Linting config
- `mypy.ini`: Type checking config
- `.github/workflows/ci.yml`: CI pipeline (lint → mypy → flake8 → test with example files)
- `pipeline/ui.json`: CustomTkinter theme configuration
