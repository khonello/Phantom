# Troubleshooting Guide

Common issues and solutions for roop-cam.

## Installation & Setup

### "FFmpeg not found" error

**Cause:** FFmpeg is not installed or not in PATH.

**Solution:**
1. Verify FFmpeg installation:
   ```bash
   ffmpeg -version
   ```
2. If not found, install it:
   - **Windows:** Chocolatey: `choco install ffmpeg` or download from ffmpeg.org
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt-get install ffmpeg`
3. Add to PATH if installed but not found:
   - **Windows:** System Properties → Environment Variables → PATH → Add FFmpeg bin directory
   - **macOS/Linux:** Usually auto-added by package manager

### "Python version not supported"

**Cause:** Python < 3.9

**Solution:**
```bash
python --version
```

If < 3.9, install Python 3.9+ from python.org or your package manager.

### "ModuleNotFoundError: No module named 'torch'"

**Cause:** Dependencies not installed.

**Solution:**
1. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (or `requirements-ci.txt` for CPU-only)

### "pip: command not found"

**Cause:** Virtual environment not activated.

**Solution:**
```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

## Model Download Issues

### "Failed to download model" or "Connection timeout"

**Cause:** Network issue or server unavailable.

**Solution:**
1. Check internet connection:
   ```bash
   ping huggingface.co
   ```
2. Manually download and place models:
   - Download `inswapper_128.onnx` from Hugging Face or provided Google Drive link
   - Place in `models/inswapper_128.onnx` (root directory)
3. Retry processing

### "OSError: [Errno 28] No space left on device"

**Cause:** Insufficient disk space for models (~300MB) and temp frames.

**Solution:**
1. Free disk space:
   ```bash
   # On Linux/macOS
   df -h
   # Remove large files or clean up
   ```
2. Process shorter videos or lower resolution to reduce temp file size
3. Use `--keep-frames` sparingly (removes temp frames after processing)

### "google.protobuf.message.DecodeError: Error parsing message"

**Cause:** Protobuf C++ implementation fails to parse the large inswapper ONNX model (~528MB), especially after other models are already loaded.

**Solution:**
This is handled automatically in `pipeline.py`. If you encounter this error running scripts directly, set the environment variable before running:

**PowerShell (Windows):**
```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
python pipeline.py -s face.jpg -t video.mp4 -o output.mp4
```

**Bash (macOS/Linux):**
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python pipeline.py -s face.jpg -t video.mp4 -o output.mp4
```

**Note:** `pipeline.py` sets this automatically. If you import `roop` modules directly in custom scripts, set `os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'` before any imports.

## GPU & Hardware

### "CUDA not found" or "No GPU detected"

**Cause:** CUDA drivers/toolkit not installed or not detected by PyTorch.

**Solution:**
1. Verify NVIDIA GPU exists:
   ```bash
   # Windows PowerShell
   Get-PnpDevice | Where-Object { $_.Name -match "NVIDIA" }
   # Linux
   lspci | grep -i nvidia
   ```
2. Install NVIDIA drivers: nvidia.com/Download/driverDetails.html
3. Install CUDA 11.8: developer.nvidia.com/cuda-11-8-0-download-archive
4. Reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall torch -y
   pip install -r requirements.txt
   ```
5. Verify CUDA availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### "CUDA out of memory"

**Cause:** GPU memory exhausted during processing.

**Solution:**
1. Reduce execution threads:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --execution-threads 2
   ```
2. Reduce max memory:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --max-memory 2
   ```
3. Process lower resolution video:
   ```bash
   ffmpeg -i video.mp4 -vf scale=1280:720 video_720p.mp4
   python pipeline.py -s face.jpg -t video_720p.mp4 -o output.mp4
   ```
4. Use CPU mode temporarily:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --execution-provider cpu
   ```

### "DirectML not available" (Windows)

**Cause:** DirectML provider not installed or GPU drivers missing.

**Solution:**
1. Install DirectML:
   ```bash
   pip install onnxruntime-directml
   ```
2. Update GPU drivers from manufacturer
3. Fall back to CPU:
   ```bash
   python pipeline.py --execution-provider cpu
   ```

### "CoreML not available" (macOS)

**Cause:** PyTorch compiled without CoreML support.

**Solution:**
1. Reinstall PyTorch with native macOS build:
   ```bash
   pip uninstall torch -y
   pip install torch torchvision torchaudio
   ```
2. CoreML should auto-detect on Apple Silicon (M1/M2/M3)

## Processing & Output

### No output file generated

**Cause:** Processing failed or output path not writable.

**Solution:**
1. Check console for error messages
2. Verify output directory exists:
   ```bash
   mkdir -p output_directory
   ```
3. Use absolute path:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o /absolute/path/output.mp4
   ```
4. Check temp directory for frames:
   ```bash
   ls temp/
   ```

### "Face not detected"

**Cause:** Source or target image has no clear face or poor lighting.

**Solution:**
1. Use a clear, front-facing face image as source
2. Ensure target video has good lighting
3. Try with different source image
4. Use `--many-faces` if multiple faces in target:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --many-faces
   ```

### "Output video is corrupted or won't play"

**Cause:** Encoding error or incomplete write.

**Solution:**
1. Try different video encoder:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --video-encoder libx265
   ```
2. Try different quality setting:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --video-quality 28
   ```
3. Check temp frames were created:
   ```bash
   ls temp/video/ | wc -l
   ```
4. Use `--keep-frames` to preserve temp files for manual inspection:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --keep-frames
   ```

### Audio is missing from output

**Cause:** Audio not restored after video encoding.

**Solution:**
1. Ensure source video has audio:
   ```bash
   ffprobe -show_streams video.mp4
   ```
2. Use `--keep-audio` flag (enabled by default):
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --keep-audio
   ```
3. Manually restore audio:
   ```bash
   ffmpeg -i output.mp4 -i video.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output_with_audio.mp4
   ```

### Output video is very slow/has wrong FPS

**Cause:** FPS not preserved during processing.

**Solution:**
1. Use `--keep-fps` flag (enabled by default):
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --keep-fps
   ```
2. Check original FPS:
   ```bash
   ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1:noval=1 video.mp4
   ```

### Processing takes a very long time

**Cause:** CPU-only mode, low threading, or high-resolution video.

**Solution:**
1. Use GPU acceleration (if available):
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --execution-provider cuda
   ```
2. Increase execution threads:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --execution-threads 16
   ```
3. Reduce video resolution:
   ```bash
   ffmpeg -i video.mp4 -vf scale=640:480 video_smaller.mp4
   python pipeline.py -s face.jpg -t video_smaller.mp4 -o output.mp4
   ```
4. Skip face enhancement:
   ```bash
   python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --frame-processor face_swapper
   ```

## Virtual Camera (Video Call Integration)

### "pyvirtualcam not installed"

**Cause:** `pyvirtualcam` library not installed.

**Solution:**
```bash
pip install pyvirtualcam
```

This is included in `requirements.txt` and `requirements-ci.txt`. If upgrading an existing installation, manually install it.

### "Virtual camera driver not found" or "Backend not available"

**Cause:** Virtual camera driver not installed on system, or wrong backend selected.

**Solution:**

**Windows:**
1. Verify OBS Virtual Camera is installed:
   - Download [OBS Studio](https://obsproject.com/)
   - Run the installer (this installs the virtual camera driver)
   - You don't need to run OBS itself
2. Restart the desktop GUI
3. If still not found, try "Unity Capture" backend instead:
   - Download [Unity Capture](https://github.com/scythe-studio/unity-capture)
   - Install and restart

**macOS:**
1. Install [OBS Studio](https://obsproject.com/)
2. Run OBS once (even for a few seconds) — this activates the virtual camera extension
3. Close OBS and try roop-cam again

**Linux:**
1. Install v4l2loopback:
   ```bash
   sudo apt-get install v4l2loopback-dkms v4l2loopback-utils
   ```
2. Load the module:
   ```bash
   sudo modprobe v4l2loopback
   ```
3. Verify it's loaded:
   ```bash
   ls /dev/video*
   ```

### "Virtual camera active but not showing in app camera list"

**Cause:** Virtual camera device registered but app hasn't refreshed camera list.

**Solution:**
1. Restart the video call application (Skype, Zoom, Teams, etc.)
2. Go to app **Settings** → **Camera**
3. Look for "OBS Virtual Camera", "Unity Capture", or `/dev/video2+` (Linux)
4. If still not visible, restart the computer

### "Virtual camera shows but video is black/frozen"

**Cause:** Pipeline not running, or no face detected.

**Solution:**
1. Verify pipeline is running (status shows "pipeline connected · processing")
2. Verify a face was selected (status shows "face set: ...")
3. In the GUI preview, check if the processed video is showing
4. If preview is black, the source image may not have a detectable face — try a different image
5. Check console for errors: `[DESKTOP.BRIDGE]` messages

### "Video is extremely laggy or drops frames"

**Cause:** Processing too slow for real-time, or GPU not being used.

**Solution:**
1. Reduce quality:
   - In GUI: Set **Quality** dropdown to "fast"
   - CLI: Add `--quality fast`
2. Enable GPU acceleration:
   - If NVIDIA: `python pipeline.py --execution-provider cuda`
   - If AMD: Install ROCm and PyTorch with ROCm support
   - If Apple Silicon: GPU auto-detected
3. Reduce source resolution (if using network stream):
   ```bash
   ffmpeg -i input.mp4 -vf scale=1280:720 input_720p.mp4
   ```
4. Check task manager (Windows) or Activity Monitor (macOS) for CPU/GPU usage

### "Audio not working in video call"

**Cause:** Virtual camera driver only handles video, not audio.

**Solution:**
- Audio comes from your microphone (a separate device)
- Ensure microphone is selected in the video call app **Settings** → **Microphone**
- Test microphone separately to verify it works
- Virtual camera is video-only by design

## Webcam Mode

### "Webcam not detected"

**Cause:** Camera not connected or driver missing.

**Solution:**
1. Check camera is connected
2. Verify other apps can access camera (e.g., system camera app)
3. Update camera drivers
4. Restart application

### "Webcam preview is black or frozen"

**Cause:** Camera access denied or hardware issue.

**Solution:**
1. Grant camera permissions to Python
   - **Windows:** Check privacy settings → Camera → Allow apps
   - **macOS:** System Preferences → Security & Privacy → Camera
   - **Linux:** Check /dev/video* permissions
2. Restart application
3. Try system camera app to verify hardware works

### "Preview takes 30+ seconds to appear"

**Cause:** Models loading and GPU initialization.

**Solution:**
- First launch always takes time (model download + initialization)
- Subsequent launches faster
- Ensure GPU is available if expecting acceleration

## GUI Issues

### "Window not responding" or frozen

**Cause:** Long processing operation blocking UI thread.

**Solution:**
1. Wait (processing ongoing in background)
2. Monitor temp directory for new frames:
   ```bash
   ls -la temp/video/ | tail -10
   ```
3. If truly frozen, kill and restart:
   ```bash
   python pipeline.py
   ```

### "UI elements not clickable"

**Cause:** File dialogs or processing dialog blocking input.

**Solution:**
1. Complete or close current dialog
2. Click "Cancel" if dialog is stuck
3. Restart application

## Developer Issues

### "mypy type checking failures"

**Cause:** Type annotation errors in code.

**Solution:**
1. Run mypy to see errors:
   ```bash
   mypy pipeline.py roop
   ```
2. Add type annotations to function signatures
3. See docs/DEVELOPMENT.md for type checking guide

### "flake8 linting errors"

**Cause:** Code style violations.

**Solution:**
1. Run flake8:
   ```bash
   flake8 pipeline.py roop
   ```
2. Fix issues (most auto-fixable):
   ```bash
   autopep8 --in-place --aggressive file.py
   ```
3. See .flake8 for configuration

## Getting Help

1. Check this troubleshooting guide
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Search existing GitHub issues
4. Create new issue with:
   - OS and Python version
   - `python pipeline.py --help` output
   - Complete error message and traceback
   - Minimal reproducible example
