# Installation Guide

This guide covers platform-specific installation for roop-cam on Windows, macOS, and Linux with CPU and GPU options.

## Prerequisites

- **Python 3.9** or higher
- **FFmpeg** (external tool, not installed via pip)
- **Git** (for cloning the repository)

### Installing FFmpeg

#### Windows
- Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use Chocolatey:
  ```bash
  choco install ffmpeg
  ```
- Or using Windows Package Manager:
  ```bash
  winget install FFmpeg
  ```

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install ffmpeg
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/s0md3v/roop-cam.git
cd roop-cam
```

### 2. Create Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

**CPU-only (recommended for beginners):**
```bash
pip install -r requirements-ci.txt
```

**With CUDA 11.8 (NVIDIA GPU, faster):**
```bash
pip install -r requirements.txt
```

### 4. Install Virtual Camera Driver (for webcam mode)

To use the desktop GUI's virtual camera feature (for Skype, Zoom, Teams, etc.), install a virtual camera driver:

#### Windows
Install **OBS Virtual Camera** (easiest):
1. Download [OBS Studio](https://obsproject.com/) (free, open-source)
2. Run the installer — this installs the OBS Virtual Camera driver
3. You don't need to run OBS itself; the driver is all that's needed

**Alternative (Windows only):** Install [Unity Capture](https://github.com/scythe-studio/unity-capture) if you prefer

#### macOS
Install **OBS Virtual Camera**:
1. Download [OBS Studio](https://obsproject.com/)
2. Install and run it once to set up the virtual camera driver
3. Driver will persist after closing OBS

#### Linux
Virtual camera uses `v4l2loopback` (usually pre-installed on most distributions):
```bash
# Ubuntu/Debian
sudo apt-get install v4l2loopback-dkms v4l2loopback-utils
```

**Note:** Desktop GUI requires pyvirtualcam, which is included in `requirements-ci.txt` and `requirements.txt`. If upgrading, ensure it's installed:
```bash
pip install pyvirtualcam
```

### 5. Download Models

On first run, roop-cam automatically downloads ~300MB of models:
- `inswapper_128.onnx` (face swapper, ~512MB) from Hugging Face
- `GFPGANv1.4.pth` (face enhancer, optional)
- `buffalo_l` (InsightFace face detection, auto-downloaded by InsightFace library)

These are cached in `models/` directory for subsequent runs.

## GPU Setup (Optional)

### NVIDIA CUDA

**Windows/Linux with NVIDIA GPU:**
1. Install [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [cuDNN 8.x](https://developer.nvidia.com/cudnn)
2. Verify installation:
   ```bash
   nvcc --version
   ```
3. Install requirements with CUDA support:
   ```bash
   pip install -r requirements.txt
   ```
4. Run with CUDA provider:
   ```bash
   python pipeline.py --execution-provider cuda
   ```

### AMD ROCm

**Linux/Windows with AMD GPU:**
1. Install [ROCm](https://rocmdocs.amd.com/en/latest/deploy/linux/index.html)
2. Modify `requirements.txt` to use ROCm PyTorch:
   ```bash
   pip uninstall torch -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   ```
3. Run with default execution provider (ROCm auto-detected):
   ```bash
   python pipeline.py
   ```

### Intel Arc GPU (Windows)

**Windows with Intel Arc GPU:**
1. Install [Intel GPU Drivers](https://www.intel.com/content/www/us/en/download/19867/intel-arc-graphics-drivers.html)
2. Install DirectML provider:
   ```bash
   pip install onnxruntime-directml
   ```
3. Run with DirectML:
   ```bash
   python pipeline.py --execution-provider dml
   ```

### macOS (Apple Silicon M1/M2/M3)

**macOS with Apple Silicon:**
1. Install PyTorch with CoreML support:
   ```bash
   pip uninstall torch -y
   pip install torch torchvision torchaudio
   ```
2. Run with CoreML provider (auto-detected):
   ```bash
   python pipeline.py --execution-provider coreml
   ```

## Known Issues

### Protobuf DecodeError

If you see `google.protobuf.message.DecodeError: Error parsing message`, this is caused by protobuf's C++ implementation failing on large ONNX models. `pipeline.py` handles this automatically by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. If running custom scripts, set this environment variable before importing roop modules. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for details.

## Verification

Test your installation:

```bash
python pipeline.py -s .github/examples/source.jpg -t .github/examples/target.mp4 -o output.mp4
```

This runs a CLI test on example files. If successful, you're ready to use roop-cam!

## Quick Start

### GUI Mode (Recommended for Beginners)
```bash
python pipeline.py
```

### CLI Mode (Scripting/Automation)
```bash
python pipeline.py -s face.jpg -t video.mp4 -o output.mp4
```

### Webcam Mode
1. Launch GUI: `python pipeline.py`
2. Select a face image
3. Click "Live" button
4. Wait 10-30 seconds for preview to appear

## Troubleshooting

**"FFmpeg not found":** Ensure FFmpeg is in your PATH:
- Windows: Add to System Environment Variables
- macOS/Linux: Run `which ffmpeg` to verify installation

**"Model download failed":** Check internet connection. Models are downloaded on first run and cached.

**"CUDA not detected":** Run `python pipeline.py --execution-provider cpu` to verify CPU mode works. Then check NVIDIA driver and CUDA installation.

**"Out of memory":** Reduce `--max-memory` or use CPU mode. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for details.

For more help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
