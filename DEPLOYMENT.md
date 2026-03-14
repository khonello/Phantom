# Phantom Face-Swap Deployment Guide

## RunPod.io Deployment

This guide covers deploying Phantom to RunPod.io with WebSocket-based real-time communication for low-latency face-swapping.

### System Requirements

**RunPod Instance Specs:**
- **GPU**: NVIDIA A40, A100, or RTX 4090 (RTX 4090 recommended for fastest inference)
- **vCPU**: 8+ cores
- **RAM**: 32GB minimum (64GB recommended)
- **Storage**: 100GB+ (for models and I/O)
- **Base Image**: `runpod/pytorch:2.0.1-py3.10-cuda-11.8.0`

**FFmpeg Required**: Install during setup for video encoding/decoding

### Architecture Overview

```
Client (Desktop/Web)
    ↓
WebSocket Connection (port 9000)
    ↓
Phantom Pipeline Server
    ├─ API Server (WebSocket)
    ├─ Processing Pipeline (batch/stream)
    └─ ML Services (detection, swap, enhancement)
    ↓
GPU (CUDA acceleration)
```

### Network Ports

| Port | Protocol | Purpose | Direction |
|------|----------|---------|-----------|
| **9000** | WebSocket | Main API (commands + frame streaming) | Bidirectional |
| **9001** | HTTP | Health check / status endpoint | Server→Client |
| **9002** | HTTP | Metrics/monitoring (optional) | Server→Client |

### WebSocket Endpoints

**Base URL:** `ws://runpod-public-ip:9000`

#### Commands (Client → Server)

Send JSON command via WebSocket:
```json
{
  "type": "command",
  "action": "set_source",
  "data": {"path": "/workspace/source.jpg"}
}
```

**Supported Actions:**
- `set_source` - Set source image
- `set_source_paths` - Set multiple sources (for averaging)
- `set_target` - Set target image/video
- `set_output` - Set output file path
- `start` - Begin batch processing
- `start_stream` - Begin stream mode
- `stop` - Stop pipeline
- `set_quality` - Set preset (fast/optimal/production)
- `set_blend` - Set blend factor (0.0-1.0)
- `set_alpha` - Set alpha smoothing (0.0-1.0)
- `set_input_url` - Set network stream URL (RTSP/RTMP)
- `create_embedding` - Create face embedding from sources
- `cleanup_session` - Clear session data
- `shutdown` - Graceful shutdown

#### Events (Server → Client)

Server pushes JSON events via WebSocket:
```json
{
  "type": "event",
  "event": "FRAME_READY",
  "data": {
    "frame_number": 42,
    "timestamp": 1234567890.5,
    "width": 1920,
    "height": 1080
  }
}
```

**Event Types:**
- `FRAME_READY` - Frame ready for display (includes metadata)
- `DETECTION` - Face detection results
- `STATUS_CHANGED` - Pipeline status update
- `PIPELINE_STARTED` - Pipeline initialization complete
- `PIPELINE_STOPPED` - Pipeline shutdown
- `ERROR` - Error event with message

#### Frame Streaming

After `FRAME_READY` event, retrieve frame via HTTP:
```
GET http://runpod-public-ip:9001/frame
Content-Type: image/png
```

Or retrieve directly via WebSocket binary message (if implemented).

### HTTP Endpoints (Fallback)

**Base URL:** `http://runpod-public-ip:9001`

- `GET /status` - Get current pipeline status
  ```json
  {
    "running": false,
    "message": "Idle",
    "embedding_ready": false,
    "source_path": null,
    "target_path": null,
    "output_path": null
  }
  ```

- `GET /frame` - Get current frame as PNG
  - Returns: PNG image data or 204 No Content

- `GET /health` - Health check
  ```json
  {"status": "healthy", "uptime": 3600}
  ```

### Installation & Setup

#### 1. Create RunPod Pod

```bash
# Connect to RunPod via SSH
ssh root@your-runpod-ip
```

#### 2. Clone Repository

```bash
cd /workspace
git clone https://github.com/yourusername/phantom.git
cd phantom
```

#### 3. Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y ffmpeg

# Install Python dependencies
pip install -r requirements-pipeline-gpu.txt

# Optional: Install CI/testing dependencies
pip install -r requirements-ci.txt
```

#### 4. Download Models

Models auto-download on first run, but you can pre-download:

```bash
python -c "
from pipeline.services.face_detection import FaceDetector
from pipeline.services.face_swapping import FaceSwapper
detector = FaceDetector()
swapper = FaceSwapper()
print('Models ready')
"
```

#### 5. Configure Environment

Create `.env` file:
```bash
# Pipeline Configuration
EXECUTION_PROVIDER=cuda
BLEND_FACTOR=0.65
ALPHA_SMOOTHING=0.6
QUALITY_PRESET=optimal

# API Server Configuration
API_PORT=9000
HTTP_PORT=9001
API_HOST=0.0.0.0

# Logging
LOG_LEVEL=info
SCOPE=PHANTOM
```

### Running the Server

#### Start Pipeline Server (Headless)

```bash
cd /workspace/phantom
python pipeline.py
```

The server will:
- Start WebSocket API on port 9000
- Start HTTP fallback on port 9001
- Load ML models (first run takes ~2-3 minutes)
- Listen for client connections

#### With CUDA Acceleration

```bash
python pipeline.py --execution-provider cuda
```

#### With Custom Port

```bash
python pipeline.py --api-port 9000
```

### Client Connection Examples

#### JavaScript/TypeScript

```javascript
const ws = new WebSocket('ws://runpod-ip:9000');

ws.onopen = () => {
  // Set source
  ws.send(JSON.stringify({
    type: 'command',
    action: 'set_source',
    data: { path: '/workspace/source.jpg' }
  }));

  // Start batch processing
  ws.send(JSON.stringify({
    type: 'command',
    action: 'start',
    data: {}
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'event' && message.event === 'FRAME_READY') {
    // Fetch frame
    fetch(`http://runpod-ip:9001/frame`)
      .then(r => r.blob())
      .then(blob => {
        // Display frame
        const url = URL.createObjectURL(blob);
        document.getElementById('preview').src = url;
      });
  }
};

ws.onerror = (error) => console.error('WebSocket error:', error);
```

#### Python

```python
import asyncio
import websockets
import json

async def connect():
    uri = "ws://runpod-ip:9000"
    async with websockets.connect(uri) as websocket:
        # Set source
        await websocket.send(json.dumps({
            "type": "command",
            "action": "set_source",
            "data": {"path": "/workspace/source.jpg"}
        }))

        # Listen for events
        async for message in websocket:
            event = json.loads(message)
            print(f"Event: {event['event']}")

asyncio.run(connect())
```

### Batch Processing (CLI)

For non-real-time batch jobs:

```bash
python pipeline.py \
  -s /workspace/source.jpg \
  -t /workspace/target.mp4 \
  -o /workspace/output.mp4 \
  --execution-provider cuda
```

### Streaming Mode (Webcam/RTSP)

```bash
# Start stream mode
python pipeline.py --stream

# Then send via WebSocket:
# action: "set_input_url"
# data: { "url": "rtsp://camera-ip/stream" }
```

### Monitoring & Debugging

#### View Logs

```bash
# Tail logs (if using structured logging)
tail -f /workspace/phantom/logs/phantom.log

# Run with verbose logging
python pipeline.py --log-level debug
```

#### Check WebSocket Status

```bash
# From another terminal
python -c "
import requests
import json
r = requests.get('http://localhost:9001/status')
print(json.dumps(r.json(), indent=2))
"
```

#### Monitor GPU Usage

```bash
nvidia-smi -l 1  # Update every 1 second
```

### Performance Tuning

#### Quality Presets

```
fast:       Skip enhancement, use fast models (30-50 FPS)
optimal:    Standard quality (10-20 FPS)
production: Full enhancement with refinement (5-10 FPS)
```

Set via WebSocket:
```json
{"type": "command", "action": "set_quality", "data": {"preset": "optimal"}}
```

#### GPU Memory

If OOM errors occur:
1. Reduce batch size (lower resolution)
2. Use `fast` quality preset
3. Reduce blend iterations

#### Network Optimization

- Use same region for RunPod instance and client
- Keep WebSocket connection persistent (don't reconnect frequently)
- Compress frame data for transmission (optional)

### Deployment Best Practices

1. **Security**: Run behind authenticated proxy (RunPod provides URL authentication)
2. **Persistence**: Store models in `/workspace` (persists across pod restarts)
3. **Logging**: Use structured logging with timestamps for debugging
4. **Health Checks**: Implement `/health` endpoint polling
5. **Graceful Shutdown**: Use `shutdown` command before stopping pod
6. **Model Caching**: Keep models downloaded to avoid re-download on restart

### Troubleshooting

**Issue: WebSocket connection refused**
```
Solution: Check firewall, ensure port 9000 is open
runpod-pod$ netstat -tlnp | grep 9000
```

**Issue: CUDA out of memory**
```
Solution: Reduce quality preset or use CPU
python pipeline.py --execution-provider cpu
```

**Issue: Slow frame detection**
```
Solution: Check GPU utilization (nvidia-smi), increase batch size, profile with:
python -m cProfile -s cumulative pipeline.py
```

**Issue: Models not downloading**
```
Solution: Download manually during setup:
python -c "from pipeline.services.face_detection import FaceDetector; FaceDetector()"
```

### Scaling

For multiple concurrent sessions:
- Use separate pipeline instances on different GPU contexts
- Load balance via reverse proxy (nginx)
- Monitor GPU/memory per pod
- Consider pod auto-scaling via RunPod API

### Cost Optimization (RunPod)

- Use **Spot Instances** for 70% cost savings (if latency-tolerant)
- **Auto-start**: Set pod to auto-start on demand
- **Spot Bid**: Set max bid to 2-3x base hourly rate
- **Batch Jobs**: Use cheapest A40 pods for non-real-time processing

### Production Checklist

- [ ] Models downloaded and cached
- [ ] GPU acceleration enabled and tested
- [ ] WebSocket connection stable
- [ ] Frame quality acceptable at target latency
- [ ] Error handling and logging configured
- [ ] Health check endpoint monitoring
- [ ] Security (authentication/authorization)
- [ ] Load testing completed
- [ ] Monitoring/alerting set up
- [ ] Graceful shutdown procedure documented

---

## Local Development

This section covers running both the pipeline engine and the desktop GUI on the same machine during development.

### Prerequisites

- Python 3.9+
- FFmpeg installed and on PATH
- (Optional) CUDA-capable GPU for hardware acceleration

### Quick Start

#### 1. Install Dependencies

```bash
# CPU mode (no GPU)
pip install -r requirements-pipeline-cpu.txt

# GPU mode (CUDA)
pip install -r requirements-pipeline-gpu.txt

# Desktop GUI additionally needs:
pip install PySide6 pyvirtualcam
```

#### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to set LOG_LEVEL, EXECUTION_PROVIDER, etc.
```

#### 3. Run the Pipeline Engine

```bash
# Server-only mode (wait for commands from desktop or WebSocket client)
python pipeline.py

# Stream mode (start realtime face-swap immediately)
python pipeline.py --stream

# Batch mode (process a file and exit)
python pipeline.py -s source.jpg -t target.mp4 -o output.mp4

# With CUDA
python pipeline.py --execution-provider cuda

# Custom port
python pipeline.py --control-port 9000
```

The pipeline always listens on `ws://localhost:9000/ws` (or the configured port).

#### 4. Run the Desktop GUI

In a separate terminal:

```bash
python desktop.py
```

The GUI connects to the pipeline at `ws://localhost:9000/ws` by default.
Override with `PHANTOM_API_URL` env var for remote pipelines:

```bash
PHANTOM_API_URL=ws://192.168.1.50:9000/ws python desktop.py
```

### Default Connection

| Component | Address |
|-----------|---------|
| WebSocket API | `ws://localhost:9000/ws` |
| Frame streaming | Pushed as binary over same WebSocket |
| Status events | Pushed as JSON over same WebSocket |

### Model Cache

Models are downloaded on first use and cached at `~/.insightface/models/` (default InsightFace path).

On RunPod, models are cached at `/workspace/models/insightface/` if the path exists (Network Volume), otherwise falls back to `~/.insightface/models/`.

### Development Tips

- Use `--log-level debug` for verbose output
- Run `python pipeline.py --help` to see all flags
- Test batch mode with example files: `python pipeline.py -s .github/examples/source.jpg -t .github/examples/target.mp4 -o /tmp/test.mp4`

---

**Questions?** Check `ARCHITECTURE.md` for system design details.
