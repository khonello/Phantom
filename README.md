# Phantom — Real-Time Face Swapping

Replace faces in videos and images with a single reference image. No dataset, no training required.

**Modern Architecture**: Event-driven, service-oriented design with WebSocket API, composable frame processors, and zero global state.

**Design target:** what a real video call actually looks like — a face carrying
sensor noise, compression and ordinary imperfection. Not a high-resolution
portrait, and not the poreless "beautified" look that reads as AI instantly.

**Key Features:**
- Real-time webcam processing, on a local GPU or a remote RunPod pod
- Single-face and multi-face swapping
- Aligned-space compositing: occlusion-aware masking, colour and detail
  matching, temporal smoothing, sensor-matched grain
- Face restoration with a tunable fidelity weight (CodeFormer, or GFPGAN)
- Configurable quality presets (fast/optimal/production)
- Batch image processing (video batch is not yet implemented — see below)
- WebSocket API for integration

![demo-gif](demo.gif)

## Installation

**Requirements:**
- Python 3.9+
- FFmpeg (for video processing)
- CUDA (optional, for GPU acceleration)

**Quick Start:**

```bash
# Clone the repository
git clone <repo-url>
cd Phantom

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies (CPU)
pip install -r requirements-pipeline-cpu.txt
# Or for GPU (CUDA):
pip install -r requirements-pipeline-gpu.txt

# Run headless engine
python pipeline.py

# Or run with GPU (if available)
python pipeline.py --execution-provider cuda
```

**For Desktop GUI:**
```bash
python desktop.py
```

See [Installation Guide](docs/INSTALLATION.md) for detailed instructions including GPU setup.

## Usage

### Batch Mode (CLI)

Process a video in one command:

```bash
python pipeline.py \
  -s <source_image> \
  -t <target_video> \
  -o <output_path>

# Example:
python pipeline.py \
  -s face.jpg \
  -t video.mp4 \
  -o output.mp4
```

### Stream Mode (Real-Time Webcam)

```bash
# Start pipeline engine
python pipeline.py --input-url 0  # Use webcam 0

# In another terminal, open desktop GUI
python desktop.py
```

Select a source face image, click "Live", and the preview will show real-time face-swapped video.

### Desktop GUI

```bash
python desktop.py
```

The GUI supports LIVE, VIDEO and IMAGE modes with:
- Source image selection (single or multiple, averaged into one embedding)
- Quality presets (fast/optimal/production)
- Enhance, colour-correction and preprocessing toggles
- Virtual camera output
- Live preview, audio capture and voice transformation

> VIDEO mode is built UI-side but the pipeline does not yet process video —
> see [Status](#status).

## Command-Line Options

```
usage: pipeline.py [-h] [-s SOURCE_PATH] [-t TARGET_PATH] [-o OUTPUT_PATH]
                   [--keep-fps] [--keep-audio] [--keep-frames] [--many-faces]
                   [--video-encoder {libx264,libx265,libvpx-vp9}]
                   [--video-quality [0-51]] [--max-memory MAX_MEMORY]
                   [--execution-provider {cpu,cuda,rocm,dml}]
                   [--execution-threads EXECUTION_THREADS]
                   [--quality {fast,optimal,production}] [--alpha ALPHA]
                   [--enhancer-model {codeformer,gfpgan}]
                   [--enhancer-weight W] [--enhance-strength S]
                   [--aligned-size N] [--temporal-alpha A]
                   [--color-strength C]
                   [--no-enhance] [--no-grain] [--no-occluder]
                   [--stream] [--log-level {debug,info,warning,error}]
                   [--input-url INPUT_URL] [--control-port PORT]
                   [-v]

options:
  -s, --source              Source image or embedding (.npy file)
  -t, --target              Target image or video
  -o, --output              Output file or directory
  --keep-fps                Preserve original frame rate
  --keep-audio              Preserve original audio
  --keep-frames             Keep temporary extracted frames
  --many-faces              Process all faces (not just largest)
  --video-encoder           Encoder: libx264 (default), libx265, libvpx-vp9
  --video-quality           Quality 0-51 (default 18, lower=better)
  --max-memory              Max RAM in GB (default 16)
  --execution-provider      GPU provider: cpu, cuda, rocm, dml
  --execution-threads       Worker threads (default 8)
  --quality                 Preset: fast, optimal (default), production
  --alpha                   Landmark EMA 0.0-1.0 (0.0=max smoothing, 1.0=off)
  --enhancer-model          Restoration backend: codeformer (default), gfpgan
  --enhancer-weight         CodeFormer fidelity: 0=most restoration, 1=closest to input
  --enhance-strength        How much of the restored face to keep (0.0-1.0)
  --aligned-size            Ceiling on compositing resolution (128-512)
  --temporal-alpha          EMA on composited pixels (1.0=off)
  --color-strength          Scales the LAB colour transfer (0.0-1.0)
  --no-enhance              Disable face restoration
  --no-grain                Disable sensor-noise matching
  --no-occluder             Disable occlusion masking
  --stream                  Start in realtime stream mode
  --log-level               debug, info (default), warning, error
  --input-url               Network stream URL (RTSP/RTMP/HTTP)
  --control-port            API server port (default 9000)
  -v, --version             Show version
```

Every realism flag also reads an environment variable (`ENHANCER_MODEL`,
`ENHANCER_WEIGHT`, `ENHANCE_STRENGTH`, `ALIGNED_SIZE`, `TEMPORAL_ALPHA`,
`COLOR_STRENGTH`, `ENHANCE`, `GRAIN`, `OCCLUDER`), since a RunPod pod is
configured through `.env` rather than a command line. Precedence: quality preset
first, then CLI/env overrides.

See [Usage Guide](docs/USAGE.md) for detailed examples and advanced options.

## Status

Development is focused on the **live call** path. Batch follows once that meets
its quality target, and reuses the same compositor.

| Area | State |
|------|-------|
| Realtime stream (webcam, network, WebSocket push) | Working |
| Aligned-space compositing, masking, restoration | Working |
| RunPod deployment, auto-stop, orchestrator | Working |
| Desktop LIVE mode, audio, voice, virtual camera | Working |
| Batch — image | Working |
| **Batch — video** | **Not implemented** |
| Realism knobs in the desktop UI | Not exposed (API only) |
| RTMP output sink | Placeholder |
| Automated tests | None |

Video batch is stubbed at `ProcessingPipeline._process_target_batch()`. The
FFmpeg building blocks exist in `pipeline/io/ffmpeg.py`; they are not yet wired
in. This also means the CI end-to-end test and the desktop VIDEO mode do not
currently pass.

## Development

### Project Structure

See [CLAUDE.md](CLAUDE.md) for:
- Architecture documentation
- Code style guidelines
- Type checking and linting requirements
- PR guidelines and contribution standards

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Setting up development environment
- Running tests
- Extending with new processors or services
- Performance profiling

### Type Safety & Code Quality

```bash
# Type checking (strict mode)
mypy pipeline desktop

# Linting
flake8 pipeline.py pipeline desktop

# Run test
python pipeline.py -s=.github/examples/source.jpg -t=.github/examples/target.mp4 -o=/tmp/test.mp4
```

## Disclaimer

This software is designed for artistic and productive use cases. Users are responsible for:
- Obtaining consent from individuals whose faces are used
- Complying with local laws and regulations
- Clearly disclosing deepfake content when shared online

The developers are committed to ethical use and will comply with takedown requests.

## Credits

Built with:
- [InsightFace](https://github.com/deepinsight/insightface) — Face detection and analysis
- [ONNX Runtime](https://onnxruntime.ai/) — Model inference
- [FFmpeg](https://ffmpeg.org/) — Video encoding/decoding
- [OpenCV](https://opencv.org/) — Computer vision utilities
- [CodeFormer](https://github.com/sczhou/CodeFormer) — Face restoration (default backend)
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face restoration (alternate backend)
- [DFL XSeg](https://github.com/iperov/DeepFaceLab) — Occlusion segmentation, via
  [facefusion-assets](https://github.com/facefusion/facefusion-assets)
