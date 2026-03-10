# Usage Guide

This guide covers how to use roop-cam in GUI mode and CLI mode.

## GUI Mode (Desktop)

### Starting the GUI

```bash
python desktop.py
```

This launches the Phantom desktop application.

### Workflow

#### 1. Select a Face Source

1. Click **"Select Source Images"** in the left sidebar
2. Choose one or more images containing a face:
   - Single image: used directly as the face to swap
   - Multiple images: averaged into a single face embedding for better accuracy
3. Status will show: `face set: filename.jpg` or `creating embedding from N images...`

#### 2. Choose Input Source

Three options available in the **"Quality"** dropdown:

- **Live Webcam** (default): Your system camera
- **Network Stream**: External video source (RTMP, HTTP, etc.) — set via input URL
- **Webcam Index**: Change if you have multiple cameras (0 = default, 1 = second camera, etc.)

#### 3. Set Processing Quality

Click the **"Quality"** dropdown to choose:
- **fast**: Lowest quality, fastest processing
- **optimal**: Balanced quality and speed (recommended)
- **production**: Highest quality, slowest processing

#### 4. Start the Pipeline

Click **"START"** to begin real-time face swapping:
- Pipeline initializes face detection and models (~10-30 seconds)
- Preview appears in the main viewport
- You'll see your webcam (unprocessed) in a PiP window (top-right)

#### 5. (Optional) Stream to a Virtual Camera

To use the processed output in Skype, Zoom, Teams, or other video call apps:

1. Ensure a virtual camera driver is installed (see [INSTALLATION.md](INSTALLATION.md))
2. Click **"PLATFORM"** dropdown to select:
   - **OBS Virtual Camera** (recommended, Windows/macOS/Linux)
   - **Unity Capture** (Windows only)
3. Click **"VIRTUAL CAM"** to enable
4. Status shows: `virtual camera active · device_name`
5. In your video call app, select the virtual camera as your camera source
6. Click **"VCAM ON"** to disable when done

**Why this workflow?** The virtual camera driver (OBS, Unity Capture, v4l2loopback) is a system-level service that presents a fake camera device. roop-cam writes processed frames to this device via `pyvirtualcam`. Your video call app then selects it from the camera picker like any other USB camera.

### Advanced Controls

#### Fine-tuning (Stream Controls in Development)

Settings for stream quality can be adjusted (alpha, blend, enhancement interval). These are typically accessed via the control panel in development mode.

### Stopping the Pipeline

Click **"STOP"** to:
- Stop webcam processing
- Disconnect virtual camera (if active)
- Release GPU/CPU resources

## CLI Mode (Batch Processing)

For non-interactive batch processing of video files:

```bash
python pipeline.py -s face.jpg -t video.mp4 -o output.mp4
```

### Common Options

```bash
-s, --source      Path to face image (required)
-t, --target      Path to target video or image (required)
-o, --output      Output file path (required)
--keep-fps        Preserve original video FPS (default: enabled)
--keep-audio      Preserve original audio (default: enabled)
--keep-frames     Don't delete temp frames after processing
--many-faces      Swap multiple faces in target
--quality         Processing quality: fast, optimal, production
--execution-provider  cpu, cuda, coreml, dml (default: auto-detect)
--execution-threads   Number of parallel threads
--max-memory      GPU memory limit in GB
```

### Examples

**Single face swap:**
```bash
python pipeline.py -s my_face.jpg -t video.mp4 -o output.mp4
```

**Multiple faces in video:**
```bash
python pipeline.py -s my_face.jpg -t video.mp4 -o output.mp4 --many-faces
```

**High quality (slower):**
```bash
python pipeline.py -s my_face.jpg -t video.mp4 -o output.mp4 --quality production
```

**GPU acceleration (NVIDIA):**
```bash
python pipeline.py -s my_face.jpg -t video.mp4 -o output.mp4 --execution-provider cuda
```

**CPU only:**
```bash
python pipeline.py -s my_face.jpg -t video.mp4 -o output.mp4 --execution-provider cpu
```

## Video Call Integration (Skype, Zoom, Teams)

### Requirements

1. **Virtual Camera Driver** installed (OBS Virtual Camera recommended)
2. **pyvirtualcam** Python library (included in dependencies)
3. Video call app (Skype, Zoom, Microsoft Teams, etc.)

### Setup Steps

1. Start roop-cam GUI: `python desktop.py`
2. Select a face source image
3. Click **"START"** to begin processing
4. Click **"VIRTUAL CAM"** to enable virtual camera output
5. Open your video call app (Skype, Zoom, etc.)
6. Go to **Settings** → **Camera** and select the virtual camera:
   - **Windows/macOS**: Look for "OBS Virtual Camera" or "Unity Capture"
   - **Linux**: Look for the v4l2loopback device (usually `/dev/video2` or higher)
7. The processed video feed will now appear in your call

### Why Virtual Cameras?

Video call applications don't accept network streams (RTMP) or raw frame feeds. They only work with system camera devices. A virtual camera driver:
- Registers a fake camera device with the OS
- Allows applications to select it from their camera picker
- Receives frames via `pyvirtualcam` from the processing pipeline
- Presents them to the video call app as if from a real USB camera

This is the same technology used by OBS for streaming webcam replacements.

### Troubleshooting

**"Virtual camera not showing in app camera list"**
- Ensure driver is installed (run OBS installer)
- Restart the video call app
- Check TROUBLESHOOTING.md for platform-specific issues

**"Slow or laggy video in call"**
- Reduce quality setting to "fast"
- Check CPU/GPU usage (may be CPU-bound)
- Lower resolution or use GPU acceleration

**"Audio not working in call"**
- Virtual cameras handle video only
- Audio comes from your microphone (separate device)
- Ensure microphone is selected in call app settings

## Tips & Best Practices

1. **Face Selection**: Use a clear, front-facing photo with good lighting
2. **Multiple Faces**: If using multiple source images, ensure they're all the same person
3. **Performance**: Start with "optimal" quality; adjust if it's too slow
4. **GPU**: Enable CUDA/CoreML for 5-10x speedup if available
5. **Video Calls**: Keep "fast" quality for smooth real-time performance
