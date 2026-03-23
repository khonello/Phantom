# Phantom — RunPod Deployment Guide

Phantom runs its GPU pipeline on a RunPod cloud pod and connects to it from your local desktop via WebSocket. The `runpod/orchestrator.py` script manages the full pod lifecycle from your local machine.

---

## How We Got Here

### Stage 1 — Manual (before orchestrator)

Everything was done by hand through the RunPod dashboard and terminal:

```
1. RunPod dashboard → pick a GPU → deploy pod
2. Copy the SSH command from the dashboard
3. ssh root@<pod-address> -i ~/.ssh/id_ed25519
4. git clone https://github.com/khonello/Phantom.git /workspace/Phantom
5. cd /workspace/Phantom && bash runpod/startup.sh
6. /workspace/venv/bin/python pipeline.py --execution-provider cuda
7. Manually copy the pod's public IP:port into .env as PHANTOM_API_URL
8. python desktop.py
```

To stop: dashboard → stop pod. To resume: dashboard → start pod → repeat steps 2-7 (minus git clone if already on volume). To update code: ssh in → `cd /workspace/Phantom && git pull`.

**Pain points:** every session meant opening the dashboard, copying addresses, running SSH commands manually, remembering what to skip on resume vs first deploy.

---

### Stage 2 — SSH orchestrator (current, for development)

`RUNPOD_DEPLOY_MODE=ssh`

The orchestrator automates everything from stage 1 via the RunPod API and paramiko:

```
orchestrator.py start
  ├─ resume or deploy pod (API)
  ├─ wait for SSH port
  ├─ SSH: git clone (first time only)
  ├─ SSH: startup.sh (installs ffmpeg, creates venv on first run)
  ├─ SSH: start pipeline in tmux
  ├─ wait for port 9000
  └─ update .env PHANTOM_API_URL

orchestrator.py stop
  └─ stop pod (API)
```

Same result as stage 1, but one command instead of 7 manual steps. Code updates are still `git pull` on the pod (ssh in, or the orchestrator can be extended to do it).

**Use for:** active development. You can ssh into the pod, attach to the tmux session, tail logs, `git pull` to test changes instantly without rebuilding anything.

---

### Stage 3 — Docker orchestrator (future, for production)

`RUNPOD_DEPLOY_MODE=docker`

All dependencies, ffmpeg, and application code are baked into a Docker image. The pod boots and the pipeline is already running — no SSH, no setup:

```
orchestrator.py start
  ├─ resume or deploy pod (API, using custom Docker image)
  ├─ wait for port 9000 (pipeline auto-starts via Docker CMD)
  └─ update .env PHANTOM_API_URL

orchestrator.py stop
  └─ stop pod (API)
```

**Use for:** stable releases. Faster boot, reproducible environment, no SSH key management. Code changes require `docker build && docker push` then redeploying the pod.

---

## One-Time Setup

This walks through everything from a fresh RunPod account to a working orchestrator. Each step produces a value that goes into `.env` — by the end you'll have all of them.

### 1 — Create a RunPod account and get your API key

Sign up at [runpod.io](https://www.runpod.io). Add billing (prepaid credit or card).

Dashboard → **Settings** → **API Keys** → **Create API Key**. Copy it — this is your `RUNPOD_API_KEY`.

### 2 — Generate an SSH key pair (ssh mode only)

If you don't already have one:

```bash
ssh-keygen -t ed25519 -C "you@email.com"
```

This creates two files:
- `~/.ssh/id_ed25519` — your **private** key (stays on your machine, goes into `RUNPOD_SSH_KEY_PATH`)
- `~/.ssh/id_ed25519.pub` — your **public** key (uploaded to RunPod)

If you already have a key pair, skip this step and use your existing path.

### 3 — Register your public key on RunPod

Dashboard → **Settings** → **SSH Public Keys** → **Add SSH Key**.

```bash
# Copy your public key to clipboard
cat ~/.ssh/id_ed25519.pub
```

Paste the output into RunPod and save. This is what lets the orchestrator (and you manually) SSH into pods as root. Not needed for docker mode, but useful to have registered for debugging.

### 4 — Create a network volume

Dashboard → **Storage** → **Network Volumes** → **Create**:

| Setting | Value |
|---------|-------|
| Name | `phantom-workspace` |
| Size | 20 GB |
| Datacenter | Choose one and note it (e.g. `EU-RO-1`) |

After creation, copy the **volume ID** from the storage list — this is your `RUNPOD_NETWORK_VOLUME_ID`. The datacenter you chose is your `RUNPOD_DATACENTER_ID`.

> The volume datacenter determines where your pods must run. All pods must be deployed in the same datacenter as the volume.

### 5 — Verify with a manual first connection (recommended)

Before trusting the orchestrator, confirm everything works by hand. This is exactly what stage 1 looked like:

```bash
# Deploy a pod from the dashboard:
#   Dashboard → GPU Cloud → Deploy
#   Pick any GPU in your volume's datacenter
#   Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04
#   Attach your network volume
#   Enable SSH
#   Deploy

# Once the pod is running, the dashboard shows a Connect button.
# It gives you an SSH command like:
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# If this connects and you get a root shell, your SSH key is working.
# Check the volume is mounted:
ls /workspace/
# Should be empty or have files from a previous session.

# Check GPU:
nvidia-smi

# Clone and test manually (optional — the orchestrator does this for you):
git clone https://github.com/khonello/Phantom.git /workspace/Phantom
cd /workspace/Phantom
bash runpod/startup.sh
/workspace/venv/bin/python pipeline.py --execution-provider cuda

# When done, stop the pod from the dashboard.
# Note the pod ID from the URL or dashboard — this is your RUNPOD_POD_ID
# (or leave it blank in .env and let the orchestrator create a fresh one).
```

**What you've confirmed:** SSH key works, volume mounts correctly, GPU is visible, repo clones, startup.sh runs, pipeline starts. If any step fails here, fix it before moving to the orchestrator.

### 6 — Configure .env

```bash
cp .env.example .env
```

Fill in everything you collected above:

```env
RUNPOD_API_KEY=rp_xxxxxxxxxx       # from step 1
RUNPOD_POD_ID=                      # leave blank or paste from step 5
RUNPOD_DEPLOY_MODE=ssh              # ssh (development) or docker (production)

RUNPOD_DATACENTER_ID=EU-RO-1       # from step 4
RUNPOD_GPU_TYPES=NVIDIA GeForce RTX 3090,NVIDIA RTX A4000,NVIDIA GeForce RTX 4090
RUNPOD_NETWORK_VOLUME_ID=vol_xxx   # from step 4

# ssh mode: generic base image
RUNPOD_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04
# docker mode: your custom image
# RUNPOD_IMAGE=<your-registry>/phantom-pipeline:latest

# ssh mode only
RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519           # from step 2
RUNPOD_REPO_URL=https://github.com/khonello/Phantom.git
```

### 7 — Install orchestrator dependencies

```bash
pip install -r requirements-orchestrator.txt
```

Setup is complete. Run `python runpod/orchestrator.py start` to test the automated flow.

---

## Daily Workflow

### Starting a session

```bash
python runpod/orchestrator.py start
python desktop.py
```

`start` handles everything — pod creation/resume, setup (ssh mode) or waiting (docker mode), and updating `.env`.

### Ending a session

```bash
python runpod/orchestrator.py stop
```

Pod pauses. `/workspace` is preserved. No GPU billing while stopped.

### Other commands

```bash
python runpod/orchestrator.py status     # GPU, cost/hr, uptime, current URL
python runpod/orchestrator.py terminate  # delete pod (network volume survives)
```

---

## What `start` Does

### SSH mode

```
1.  Get a running pod       resume existing, or deploy new
                             (GPU priority list, filtered to datacenter)
                             RUNPOD_POD_ID saved to .env if new

2.  Wait for ports           poll until SSH (22) and WS (9000) are assigned

3.  Wait for SSH             poll TCP until port 22 accepts connections

4.  SSH setup                git clone (first time only)
                             startup.sh: installs ffmpeg, creates venv (first time)
                             kills leftover tmux, starts pipeline in new tmux session

5.  Wait for pipeline        poll TCP at port 9000

6.  Update .env              PHANTOM_API_URL=ws://<ip>:<port>/ws
```

On resume: git clone and venv install are skipped (already on volume). FFmpeg is reinstalled (system package, lost on container stop). Pipeline always starts fresh in tmux.

### Docker mode

```
1.  Get a running pod       resume existing, or deploy new

2.  Wait for port 9000       poll until assigned by RunPod

3.  Wait for pipeline        poll TCP (pipeline auto-started by Docker CMD)

4.  Update .env              PHANTOM_API_URL=ws://<ip>:<port>/ws
```

No SSH step. The image has everything. Models are on the `/workspace` volume.

---

## Building the Docker Image (docker mode only)

The `Dockerfile` bakes in ffmpeg, Python dependencies, and application code. Models live on `/workspace` (not in the image).

```bash
docker build -t <your-registry>/phantom-pipeline:latest .
docker push <your-registry>/phantom-pipeline:latest
```

Set `RUNPOD_IMAGE` in `.env` to the pushed tag and `RUNPOD_DEPLOY_MODE=docker`.

Rebuild when: pipeline code changes, dependencies change, or system packages change. Not needed for runtime config changes (those go through `.env` and the WebSocket API).

---

## Switching Between Modes

Change one line in `.env`:

```env
# Development — iterate fast, ssh in to debug
RUNPOD_DEPLOY_MODE=ssh
RUNPOD_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04

# Production — fast boot, reproducible
RUNPOD_DEPLOY_MODE=docker
RUNPOD_IMAGE=<your-registry>/phantom-pipeline:latest
```

Then `terminate` the current pod and `start` fresh (the mode affects how the pod is created). The network volume carries over either way.

---

## GPU Tier Recommendations

| GPU | VRAM | Use Case | Est. Cost/hr |
|-----|------|----------|-------------|
| RTX 3090 | 24 GB | Budget stream mode | ~$0.44 |
| RTX A4000 | 16 GB | Good balance | ~$0.76 |
| RTX 4090 | 24 GB | Best latency | ~$0.74 |
| A100 80GB | 80 GB | Large batch jobs | ~$1.99 |

---

## Troubleshooting

### `start` fails — pod not found

`RUNPOD_POD_ID` refers to a deleted pod. `start` automatically deploys a new one. No action needed.

### `start` fails — no GPU capacity

Wait and retry, add more GPU types, or change datacenter (requires new network volume).

### `start` fails — SSH timeout (ssh mode)

Pod is up but SSH isn't reachable. Usually transient — retry `start`.

### Pipeline not starting (ssh mode)

SSH into the pod and check the tmux session:

```bash
ssh root@<pod-address> -i ~/.ssh/id_ed25519
tmux attach -t phantom
```

### Pipeline not starting (docker mode)

Check pod logs on the RunPod dashboard, or SSH in if SSH is enabled on the pod.

### Desktop shows "disconnected — reconnecting..."

`PHANTOM_API_URL` is stale. Run `start` to update it.

### CUDA not used

Check pipeline logs for `Applied providers: ['CUDAExecutionProvider']`. If not, check `nvidia-smi` on the pod.

---

## Security Notes

- RunPod proxy URLs are pod-specific; do not share them publicly
- The WebSocket server has no authentication — rely on RunPod's network isolation
- SSH keys registered on RunPod grant root access to the pod container
- For production, place the WebSocket behind an authenticated reverse proxy
