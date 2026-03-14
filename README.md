# Phantom — Real-Time Face Swapping

Replace faces in videos and images with a single reference image. No dataset, no training required.

**Modern Architecture**: Event-driven, service-oriented design with WebSocket API, composable frame processors, and zero global state.

**Key Features:**
- Single-face and multi-face swapping
- Real-time webcam processing
- Batch video processing
- Face enhancement (optional GFPGAN)
- Face tracking for temporal consistency
- Configurable quality presets (fast/optimal/production)
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

The GUI supports both batch and stream modes with:
- Source image selection
- Quality presets (fast/optimal/production)
- Blend controls
- Tracker selection
- Live preview and frame serving

## Command-Line Options

```
usage: pipeline.py [-h] [-s SOURCE_PATH] [-t TARGET_PATH] [-o OUTPUT_PATH]
                   [--keep-fps] [--keep-audio] [--keep-frames] [--many-faces]
                   [--video-encoder {libx264,libx265,libvpx-vp9}]
                   [--video-quality [0-51]] [--max-memory MAX_MEMORY]
                   [--execution-provider {cpu,cuda,rocm,dml}]
                   [--execution-threads EXECUTION_THREADS]
                   [--quality {fast,optimal,production}]
                   [--tracker {csrt,kcf,mosse}]
                   [--alpha ALPHA] [--blend BLEND]
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
  --tracker                 Face tracking: csrt (default), kcf, mosse
  --alpha                   EMA smoothing 0.0-1.0 (default 0.6)
  --blend                   Swap blend 0.0-1.0 (default 0.65)
  --input-url               Network stream URL (RTSP/RTMP/HTTP)
  --control-port            API server port (default 9000)
  -v, --version             Show version
```

See [Usage Guide](docs/USAGE.md) for detailed examples and advanced options.

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
mypy pipeline.py pipeline desktop

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
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face enhancement (optional)
