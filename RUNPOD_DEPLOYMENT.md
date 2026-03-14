# Phantom — RunPod Cloud Deployment Guide

Step-by-step instructions for deploying Phantom to RunPod.io for GPU-accelerated remote face-swapping. This guide reflects the exact steps taken during the first successful deployment, including issues encountered and how they were resolved.

---

## Table of Contents

1. [GPU Tier Recommendations](#gpu-tier-recommendations)
2. [First-Time Setup](#first-time-setup)
   - [Step 1: Create a Pod](#step-1-create-a-pod)
   - [Step 2: Set Up SSH Access](#step-2-set-up-ssh-access)
   - [Step 3: Connect to the Pod](#step-3-connect-to-the-pod)
   - [Step 4: Fix DNS (if needed)](#step-4-fix-dns-if-needed)
   - [Step 5: Install Phantom](#step-5-install-phantom)
   - [Step 6: Start the Pipeline](#step-6-start-the-pipeline)
   - [Step 7: Connect the Desktop GUI](#step-7-connect-the-desktop-gui)
3. [Reconnecting to a New Pod](#reconnecting-to-a-new-pod)
4. [WebSocket Protocol](#websocket-protocol)
5. [Troubleshooting](#troubleshooting)
6. [Cost Optimization](#cost-optimization)
7. [Security Notes](#security-notes)
8. [Production Checklist](#production-checklist)

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

## First-Time Setup

### Step 1: Create a Pod

1. Go to [RunPod.io](https://runpod.io) → **Deploy** → **GPU Pod**
2. Choose your GPU tier (RTX 4090 recommended)
3. Select the **PyTorch template** — this pre-installs PyTorch, CUDA, and JupyterLab
4. Click **Customize Deployment** to open **Pod Template Overrides**

#### Pod Template Overrides

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

### Step 2: Set Up SSH Access

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

> **Note**: The Connect tab also shows a **Direct TCP Ports** section at the bottom. It looks like: `213.192.2.110:40152 → :9000`. The left side (`213.192.2.110:40152`) is the public address RunPod assigned to your pod's port 9000. You will use this for `PHANTOM_API_URL` (see Step 7).

---

### Step 3: Connect to the Pod

Wait for JupyterLab to show **Ready** (green dot) on the Connect tab, then SSH in:

```bash
ssh sipo66pbzzdcir-64411f5f@ssh.runpod.io -i ~/.ssh/id_ed25519
```

You will land in `/` (root). Navigate to the workspace:

```bash
cd /workspace
```

---

### Step 4: Fix DNS (if needed)

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

### Step 5: Install Phantom

```bash
cd /workspace

# Clone the repository (use token for private repos — see note below)
git clone https://github.com/khonello/Phantom.git
cd Phantom

# Run the RunPod startup script — located at runpod/startup.sh inside the repo.
# Run it from the repo root (do NOT cd into runpod/ first).
# It installs FFmpeg, checks CUDA, and creates the model cache directory.
#
# Note: local/setup.sh is a separate local-dev script (creates a venv on
# Linux/macOS). Do NOT use it on RunPod.
bash runpod/startup.sh

# Install Python dependencies
pip install -r requirements-pipeline-gpu.txt
```

> **Private repository**: If the repo is private, authenticate with a GitHub Personal Access Token:
> ```bash
> git clone https://<your_token>@github.com/khonello/Phantom.git
> ```
> Generate a token at GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → repo scope.

#### Known dependency issues

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

### Step 6: Start the Pipeline

**Always use tmux** — if you run the pipeline directly in the SSH terminal and your connection drops (WiFi loss, laptop sleep, etc.), the process dies and you lose the session. tmux keeps it running independently.

```bash
# Start a named tmux session
tmux new -s phantom

# Inside tmux, start the pipeline
python pipeline.py --stream --execution-provider cuda

# Detach from tmux (pipeline keeps running): Ctrl+B, then D
# Reattach later: tmux attach -t phantom
```

#### If the process is stuck / won't respond to Ctrl+C

This happens when an SSH session disconnects mid-run (e.g. WiFi drops). Reconnect and kill it:

```bash
ssh sipo66pbzzdcir-64411f5f@ssh.runpod.io -i ~/.ssh/id_ed25519

pkill -f pipeline.py

# Confirm it's gone
ps aux | grep pipeline.py

# Restart cleanly
tmux new -s phantom
python pipeline.py --stream --execution-provider cuda
```

**Always use `--execution-provider cuda` on RunPod** — you are paying for a GPU, use it.

#### `--execution-provider` explained

| Command | Provider | Inference runs on | Expected latency |
|---------|----------|-------------------|-----------------|
| `python pipeline.py --stream` | CPU | All CPU cores | High — 500ms+ per frame, not suitable for real-time |
| `python pipeline.py --stream --execution-provider cuda` | GPU (CUDA) | RTX 3090 / 4090 | Low — 30–80ms per frame, real-time capable |

**What to expect without `--cuda`**: The pipeline starts and works but face detection and swapping run on CPU. On a video feed you will see significant lag — frames process slowly and the desktop preview will stutter or freeze. The GPU sits idle despite you paying for it.

**What to expect with `--cuda`**: Face detection and the ONNX swap model both run on the GPU. Frame processing is fast enough for real-time display. You will see `Applied providers: ['CUDAExecutionProvider']` in the logs confirming GPU is active.

#### Expected startup output (with CUDA)

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

---

### Step 7: Connect the Desktop GUI

#### Finding your connection URL

1. Go to your pod on RunPod → click **Connect**
2. Scroll to the **Direct TCP Ports** section at the bottom
3. You will see a line like: `213.192.2.110:40152 → :9000`
   - `213.192.2.110` — the pod's public IP
   - `40152` — the public port RunPod assigned to your internal port 9000
   - This mapping changes every time you restart the pod — always check here for the current value

#### Set up your local `.env`

On your **local machine**, in the root of the Phantom project:

```bash
cp .env.example .env
```

Open `.env` and set:

```
# Format: ws://<public-ip>:<public-port>/ws
# Example (replace with your actual values from the Connect tab):
PHANTOM_API_URL=ws://213.192.2.110:40152/ws
```

> Use `ws://` with the direct TCP address. The RunPod proxy URL (`wss://<pod-id>-9000.proxy.runpod.net/ws`) also works but the direct TCP connection is lower latency.

#### Start the desktop

```bash
python desktop.py
```

---

## Reconnecting to a New Pod

When you **delete a pod and create a new one**, two things change: the pod's SSH host key and its public IP/port. Both need to be refreshed on your local machine.

### 1 — Remove the old host key

Every pod has a unique SSH host key. When you connect to a new pod at the same address, SSH will refuse with **"WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED"** and refuse to connect. Clear the old entry first:

```bash
# If you used the RunPod SSH proxy URL:
ssh-keygen -R "ssh.runpod.io"

# If you used a direct IP (replace with the old pod's actual IP):
ssh-keygen -R "[213.192.2.110]:40152"
```

Or open `~/.ssh/known_hosts` and manually delete the line that contains the old pod's address.

> **Your private key stays the same** — do not delete `~/.ssh/id_ed25519`. Only the known_hosts entry changes.

On your first SSH connection to the new pod, you will be prompted:

```
The authenticity of host 'ssh.runpod.io' can't be established.
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` — this adds the new pod's host key to `known_hosts`.

### 2 — Get the new connection details

The new pod will have a different SSH command and a different Direct TCP address. From the RunPod **Connect** tab:

- Copy the new SSH command (new pod ID / hash)
- Scroll to **Direct TCP Ports** and note the new public IP and port for `PHANTOM_API_URL`

### 3 — Update your local `.env`

Open `.env` and update `PHANTOM_API_URL` with the new pod's address:

```
PHANTOM_API_URL=ws://<new-ip>:<new-port>/ws
```

### 4 — Re-install Phantom on the new pod

The **container disk is wiped** when a pod is deleted. The **Network Volume** (`/workspace`) persists — your model cache survives. On the new pod:

```bash
cd /workspace
git clone https://github.com/khonello/Phantom.git   # or git pull if already cloned
cd Phantom

# runpod/startup.sh is inside the repo — run from the repo root
bash runpod/startup.sh

pip install -r requirements-pipeline-gpu.txt
```

Models in `/workspace/models/insightface/` are already there from the previous session — no re-download needed.

### Quick reconnect checklist

- [ ] Removed old host key from `~/.ssh/known_hosts`
- [ ] Accepted new host key on first SSH connection
- [ ] Updated `PHANTOM_API_URL` in local `.env` with new pod's Direct TCP address
- [ ] Re-cloned / pulled repo on new pod (container disk was wiped)
- [ ] `runpod/startup.sh` and `pip install` completed
- [ ] Pipeline started in tmux with `--execution-provider cuda`

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

### SSH refuses connection — "REMOTE HOST IDENTIFICATION HAS CHANGED"

You deleted the old pod and the new one has a different host key. Remove the stale entry:
```bash
ssh-keygen -R "ssh.runpod.io"
# or for direct IP:
ssh-keygen -R "[<old-ip>]:<old-port>"
```
Then reconnect and accept the new host key.

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

### Desktop shows "disconnected — reconnecting..."

- Check `PHANTOM_API_URL` matches the current pod's Direct TCP address (changes on every new pod)
- Confirm the pipeline is still running on the pod (`tmux attach -t phantom`)
- The desktop client will automatically reconnect once the URL is correct and the server is up

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

### First-time setup
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

### Reconnecting after pod deletion
- [ ] Old host key removed from `~/.ssh/known_hosts`
- [ ] New host key accepted on first SSH connection
- [ ] `PHANTOM_API_URL` updated in local `.env`
- [ ] Repo re-cloned / pulled on new pod
- [ ] `runpod/startup.sh` and `pip install` re-run
- [ ] Pipeline running in tmux with `--execution-provider cuda`
