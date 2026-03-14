# Phantom — RunPod Cloud Deployment Guide

Step-by-step instructions for deploying Phantom to RunPod.io for GPU-accelerated remote face-swapping.

---

## GPU Tier Recommendations

| GPU | VRAM | Use Case | Est. Cost/hr |
|-----|------|----------|-------------|
| RTX 4090 | 24 GB | Stream mode, fast inference | ~$0.74 |
| A100 80GB | 80 GB | Batch processing, large videos | ~$1.99 |
| A40 | 48 GB | Good balance: batch + stream | ~$0.76 |
| RTX 3090 | 24 GB | Budget stream mode | ~$0.44 |

**Recommended**: RTX 4090 for real-time stream mode (best latency/cost ratio).

---

## Step 1: Create a Pod

1. Go to [RunPod.io](https://runpod.io) → **Deploy**
2. Select **GPU Pod**
3. Choose GPU: **RTX 4090** (or A40 for batch)
4. Set **Container Image**: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
5. Set **Container Disk**: 50 GB (for system + code)
6. **Expose port 9000** — add to "Expose HTTP Ports" field
7. (Optional) Set **Volume Mount** for model caching (see Step 2)

---

## Step 2: Network Volume (Model Caching)

To avoid re-downloading models on every pod start:

1. Go to **Storage** → **Network Volumes** → **Create**
2. Name: `phantom-models`, Size: 20 GB, Region: same as your pod
3. When creating the pod, attach this volume at `/workspace/models`

Phantom automatically checks `/workspace/models/insightface/` before downloading.

---

## Step 3: Start the Pod

Once the pod is running:

```bash
# SSH into the pod
ssh root@<pod-ip>

# Or use RunPod's web terminal
```

---

## Step 4: Install Phantom

```bash
cd /workspace

# Clone the repository
git clone https://github.com/yourusername/phantom.git
cd phantom

# Run startup script (installs FFmpeg, checks CUDA, warms up models)
bash runpod/startup.sh

# Install Python dependencies (GPU)
pip install -r requirements-pipeline-gpu.txt
```

---

## Step 5: Configure Environment

```bash
cp .env.example .env
# Edit as needed — EXECUTION_PROVIDER=cuda is the key setting
```

---

## Step 6: Start the Pipeline

```bash
# Stream mode (realtime)
python pipeline.py --stream --execution-provider cuda

# Or batch mode via WebSocket commands
python pipeline.py --execution-provider cuda
```

The server listens at `ws://0.0.0.0:9000/ws`.

---

## Step 7: Connect the Desktop GUI

RunPod exposes port 9000 via a proxy URL. Find it in your pod's **Connect** tab:

```
wss://<pod-id>-9000.proxy.runpod.net/ws
```

Set `PHANTOM_API_URL` on the desktop machine:

```bash
PHANTOM_API_URL=wss://abc123-9000.proxy.runpod.net/ws python desktop.py
```

Or export permanently:

```bash
export PHANTOM_API_URL=wss://abc123-9000.proxy.runpod.net/ws
python desktop.py
```

---

## WebSocket Protocol

All communication over a single WebSocket connection:

### Commands (Desktop → Pipeline)

```json
{"action": "set_source", "path": "/workspace/source.jpg"}
{"action": "start_stream"}
{"action": "stop"}
{"action": "health"}
```

### Events (Pipeline → Desktop)

```json
{"type": "event", "event": "STATUS_CHANGED", "message": "..."}
{"type": "event", "event": "PIPELINE_STARTED"}
{"type": "event", "event": "PIPELINE_STOPPED"}
```

### Frames (Pipeline → Desktop)

Binary WebSocket messages: raw JPEG bytes at quality 85.

---

## Troubleshooting

### Port 9000 not reachable

1. In RunPod pod settings, confirm port 9000 is in "Expose HTTP Ports"
2. Check the pipeline is running: `netstat -tlnp | grep 9000`
3. Look for `WebSocket server listening on ws://0.0.0.0:9000/ws` in logs

### CUDA out of memory

```bash
# Check GPU usage
nvidia-smi

# Reduce quality preset (fewer model passes)
python pipeline.py --quality fast --execution-provider cuda

# Use CPU fallback
python pipeline.py --execution-provider cpu
```

### Models not found

```bash
# Pre-download models
python -c "
from pipeline.config import CONFIG
from pipeline.services.face_detection import FaceDetector
d = FaceDetector(CONFIG)
d._get_analyser()  # triggers download
print('Done')
"
```

### Connection drop / disconnection

The desktop client uses exponential backoff reconnection (1s → 2s → 4s → ... up to 30s).
Check `PHANTOM_API_URL` is correct. Use `wss://` for RunPod proxy (TLS required).

### First frame spike (1–3s delay)

Expected: models load on first frame. Subsequent frames will be fast.
To pre-warm models, run the startup script which performs a warmup pass.

---

## Cost Optimization

- Use **Spot Instances** for up to 70% savings (acceptable if occasional interruptions are OK)
- Stop pod when not in use — models persist on Network Volume
- For batch-only workloads, use cheaper A40 pods
- Monitor GPU usage: if consistently <50%, try a smaller GPU tier

---

## Security Notes

- RunPod proxy URLs include authentication; do not share your pod URL
- The WebSocket server has no built-in authentication — rely on RunPod's proxy security
- For production, place behind an authenticated reverse proxy (nginx + Basic Auth)

---

## Production Checklist

- [ ] Pod created with correct GPU and disk size
- [ ] Network Volume attached at `/workspace/models`
- [ ] Port 9000 exposed in pod settings
- [ ] `runpod/startup.sh` executed successfully
- [ ] CUDA detected: `nvidia-smi` shows GPU
- [ ] Pipeline starts: `WebSocket server listening` in logs
- [ ] Desktop connects via `wss://` RunPod proxy URL
- [ ] Frames appear in desktop preview
- [ ] Health check passes: `{"action": "health"}` returns `{"status": "healthy", ...}`
