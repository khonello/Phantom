# Phantom — RunPod Cloud Deployment Guide

Step-by-step instructions for deploying Phantom to RunPod.io for GPU-accelerated remote face-swapping. This guide reflects the exact steps taken during the first successful deployment, including issues encountered and how they were resolved.

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

1. Go to [RunPod.io](https://runpod.io) → **Deploy** → **GPU Pod**
2. Choose your GPU tier (RTX 4090 recommended)
3. Select the **PyTorch template** — this pre-installs PyTorch, CUDA, and JupyterLab
4. Click **Customize Deployment** to open **Pod Template Overrides**

### Pod Template Overrides

Configure the following before deploying:

| Setting | Value | Notes |
|---------|-------|-------|
| Container Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Pre-filled by template |
| Container Disk | 20 GB | Temporary, erased on pod stop |
| Volume Disk | 20 GB | Persistent, survives restarts |
| Volume Mount Path | `/workspace` | Where models cache |
| Expose HTTP Ports | `8888` | JupyterLab |
| Expose TCP Ports | `9000` | Phantom WebSocket API |

> **Important**: Port 9000 must go in **Expose TCP Ports**, not HTTP Ports. This is what your local desktop connects to.

> **Environment Variables**: Leave blank — Phantom auto-detects CUDA and reads config from code. No manual env vars needed.

5. Click **Set Overrides** then deploy the pod.

---

## Step 2: Set Up SSH Access

Once the pod is running, the **Connect** tab will show an SSH setup prompt. Do this — it gives you a reliable terminal that doesn't break when your browser tab closes.

**On your local machine:**

```bash
# Generate an SSH key if you don't have one
ssh-keygen -t ed25519 -C "you@email.com"

# Copy your public key
cat ~/.ssh/id_ed25519.pub
```

Paste the public key into the SSH public key field on the RunPod Connect tab and click **Save**.

RunPod will then show you a connection command like:

```bash
# Format: ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
# Example:
ssh sipo66pbzzdcir-64411f5f@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Use that exact command — it routes through RunPod's SSH proxy and is more reliable than the direct IP.

> **Note**: The Connect tab also shows a **Direct TCP** address. Example: `213.192.2.110:40152 → :9000`. This is the direct address for port 9000 — use it for `PHANTOM_API_URL` on your local machine (see Step 6).

---

## Step 3: Connect to the Pod

Wait for JupyterLab to show **Ready** (green dot) on the Connect tab, then SSH in:

```bash
# Format: ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
# Example:
ssh sipo66pbzzdcir-64411f5f@ssh.runpod.io -i ~/.ssh/id_ed25519
```

You will land in `/` (root). Navigate to the workspace:

```bash
cd /workspace
```

---

## Step 4: Fix DNS (if needed)

> **Known issue**: Some RunPod pods start with broken DNS, blocking `git clone` and `pip install`.

Test internet connectivity first:

```bash
curl -s https://github.com || echo "no internet"
```

If you see `no internet`, fix DNS:

```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
curl -s https://github.com || echo "still no internet"
```

Retry the curl after fixing. If it still fails, terminate the pod and create a new one — occasionally a pod container starts with a broken network stack that doesn't recover.

---

## Step 5: Install Phantom

```bash
cd /workspace

# Clone the repository (use token for private repos — see note below)
git clone https://github.com/khonello/Phantom.git
cd Phantom

# Run startup script (installs FFmpeg, checks CUDA, sets up model dirs)
bash runpod/startup.sh

# Install Python dependencies
pip install -r requirements-pipeline-gpu.txt
```

> **Private repository**: If the repo is private, authenticate with a GitHub Personal Access Token:
> ```bash
> git clone https://<your_token>@github.com/khonello/Phantom.git
> ```
> Generate a token at GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → repo scope.

### Known dependency issues

**numpy conflict (Python 3.11)**

If you see:
```
ERROR: Cannot install numpy==1.23.5 ... onnxruntime-gpu requires numpy>=1.24.2
```
This means the requirements file had a pinned numpy too old for Python 3.11. Pull the latest version of the repo and retry — this has been fixed.

**torch downgrade conflict**

If you see:
```
torchaudio X.X requires torch==X.X, but you have torch Y.Y which is incompatible
```
The pod's PyTorch image ships with a newer torch pre-installed. The GPU requirements file no longer pins torch — it uses whatever is already on the image. Pull the latest and retry.

If you encounter either issue before pulling the fix, the install may still complete with warnings. Check with:
```bash
python -c "import torch; import cv2; import insightface; print('OK')"
```

---

## Step 6: Start the Pipeline

**Always use `--execution-provider cuda` on RunPod** — you are paying for a GPU, use it:

```bash
python pipeline.py --stream --execution-provider cuda
```

### `--execution-provider` explained

| Command | Provider | Inference runs on | Expected latency |
|---------|----------|-------------------|-----------------|
| `python pipeline.py --stream` | CPU | All CPU cores | High — 500ms+ per frame, not suitable for real-time |
| `python pipeline.py --stream --execution-provider cuda` | GPU (CUDA) | RTX 3090 / 4090 | Low — 30–80ms per frame, real-time capable |

**What to expect without `--cuda`**: The pipeline starts and works but face detection and swapping run on CPU. On a video feed you will see significant lag — frames process slowly and the desktop preview will stutter or freeze. The GPU sits idle despite you paying for it.

**What to expect with `--cuda`**: Face detection and the ONNX swap model both run on the GPU. Frame processing is fast enough for real-time display. You will see `Applied providers: ['CUDAExecutionProvider']` in the logs confirming GPU is active.

### Expected startup output (with CUDA)

```
[CORE] INFO: GPU available: NVIDIA GeForce RTX 3090
[API_SERVER] INFO: WebSocket API server started on port 9000
[CORE] INFO: Starting in stream mode
[PIPELINE] INFO: Stream pipeline started
[FACE_DETECTOR] INFO: Using RunPod model cache: /workspace/models/insightface
[API_SERVER] INFO: WebSocket server listening on ws://0.0.0.0:9000/ws
Applied providers: ['CUDAExecutionProvider'], with options: {...}
```

> **First run only**: InsightFace will download `buffalo_l.zip` (~275MB) on first start. This is a one-time download — it caches to `/workspace/models/insightface/` on the Network Volume and will not re-download on subsequent pod starts.

> Use `--stream` for real-time mode. Without it, the pipeline runs in batch mode and exits after processing a file.

---

## Step 7: Connect the Desktop GUI

On your **local machine**, set `PHANTOM_API_URL` in your `.env` file to the pod's direct TCP address shown on the Connect tab:

```
PHANTOM_API_URL=ws://213.192.2.110:40152/ws
```

> Use `ws://` with the direct TCP address (IP:port format), not `wss://`. The RunPod proxy URL (`wss://<pod-id>-9000.proxy.runpod.net/ws`) also works but the direct TCP connection is lower latency.

Then run:

```bash
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

### Pod has no internet after start

Fix DNS:
```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
```
If this doesn't work, terminate and recreate the pod.

### Port 9000 not reachable

1. Confirm port 9000 is in **Expose TCP Ports** (not HTTP Ports) in pod settings
2. Check the pipeline is running: `netstat -tlnp | grep 9000`
3. Look for `WebSocket server listening on ws://0.0.0.0:9000/ws` in logs

### CUDA out of memory

```bash
# Check GPU usage
nvidia-smi

# Use CPU fallback
python pipeline.py --stream --execution-provider cpu
```

### Models not found

Models auto-download on first run to `/workspace/models/insightface/`. If the Network Volume is attached at `/workspace`, they persist across pod restarts. If you see repeated downloads every start, check the volume is mounted correctly.

### Connection drop / disconnection

The desktop client uses exponential backoff reconnection. Check `PHANTOM_API_URL` is correct and the pipeline is still running on the pod.

### First frame spike (1–3s delay)

Expected on first run — models load on first frame. Subsequent frames will be fast. The startup script performs a warmup pass to reduce this.

---

## Cost Optimization

- Use **Spot Instances** for up to 70% savings (pod may be interrupted occasionally)
- **Stop the pod** when not in use — models persist on the Network Volume
- For batch-only workloads, cheaper A40 pods are sufficient
- Monitor GPU usage: `nvidia-smi` — if consistently below 50%, a smaller GPU tier works

---

## Security Notes

- RunPod proxy URLs include pod-specific routing; do not share your pod URL publicly
- The WebSocket server has no built-in authentication — rely on RunPod's network security
- For production, place behind an authenticated reverse proxy

---

## Production Checklist

- [ ] Pod created with correct GPU and disk size
- [ ] Port 9000 in **Expose TCP Ports** in pod settings
- [ ] Network Volume attached at `/workspace` (20GB+)
- [ ] SSH key saved and connection tested
- [ ] DNS working: `curl -s https://github.com` succeeds
- [ ] Repo cloned to `/workspace/Phantom`
- [ ] `runpod/startup.sh` executed successfully
- [ ] `pip install -r requirements-pipeline-gpu.txt` completed
- [ ] CUDA detected: `nvidia-smi` shows GPU
- [ ] Pipeline starts: `WebSocket server listening` in logs
- [ ] `PHANTOM_API_URL` set on local machine
- [ ] Desktop connects and frames appear in preview
- [ ] Health check passes: `{"action": "health"}` returns `{"status": "healthy", ...}`
