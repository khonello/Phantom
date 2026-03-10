# Phantom — Production Deployment

## Philosophy

The pipeline only needs to run during active sessions. The goal is:
- Zero idle cost (or near zero)
- Under 60 seconds from "I want to stream" to pipeline ready
- No manual dependency installation after first setup

---

## Recommended Providers

### Tier 1 — RunPod (Primary)

**Best for:** Regular use, reliability matters, semi-production sessions.

| GPU | VRAM | Price | Verdict |
|---|---|---|---|
| RTX 3090 | 24GB | ~$0.44/hr | **Recommended** |
| RTX 4090 | 24GB | ~$0.74/hr | Overkill unless running `production` preset at scale |

**Why RunPod:**
- Consistent uptime — proper cloud infrastructure, not spare capacity
- Pod templates — one-time setup, instant boot every session
- Stopped pods cost only storage (~$0.07/GB/month) — negligible idle cost
- Clean port exposure — TCP 9000, 9001, 5000 all work out of the box
- Web terminal + SSH — easy to manage

---

### Tier 2 — Vast.ai (Budget)

**Best for:** Occasional use, maximum cost savings, don't mind slight instability.

| GPU | VRAM | Price | Verdict |
|---|---|---|---|
| RTX 3080 | 10GB | ~$0.20–0.35/hr | **Recommended** |
| RTX 3090 | 24GB | ~$0.30–0.50/hr | If 3080 VRAM feels tight |

**Why Vast.ai:**
- Cheapest real GPU rental available
- Destroy instance after session → $0 idle cost
- Larger GPU pool — more availability
- Templates supported

**Caveats:**
- Host reliability varies — occasional bad actors
- Re-verify TCP port availability before committing to a host
- Slightly more setup friction than RunPod

---

## Cost Estimates

**RTX 3080 — Vast.ai (~$0.28/hr avg)**

| Usage | Est. Monthly |
|---|---|
| 1 hr/day | ~$9 |
| 2 hrs/day | ~$17 |
| 3 hrs/day | ~$26 |
| 4 hrs/day | ~$34 |

**RTX 3090 — Vast.ai (~$0.40/hr avg)**

| Usage | Est. Monthly |
|---|---|
| 1 hr/day | ~$12 |
| 2 hrs/day | ~$24 |
| 3 hrs/day | ~$36 |
| 4 hrs/day | ~$48 |

**RTX 3090 — RunPod (~$0.44/hr avg)**

| Usage | Est. Monthly |
|---|---|
| 1 hr/day | ~$13 |
| 2 hrs/day | ~$27 |
| 3 hrs/day | ~$40 |
| 4 hrs/day | ~$53 |

**Sweet spot:** RTX 3090 on RunPod at 2 hrs/day — reliable, ~$27/month.

---

## One-Time Setup

Do this once. Save as a template. Never repeat.

### 1. Create the instance

**RunPod:**
- Go to RunPod → Secure Cloud → Deploy
- Select **RTX 3090**
- Base image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- Expose TCP ports: `9000, 9001, 5000`
- Volume: 20GB (for models)

**Vast.ai:**
```bash
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 inet_up>200 reliability>0.95 tcp_ports=9000'
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
  --disk 20 \
  --ports 9000/tcp 9001/tcp 5000/tcp
```

---

### 2. Install dependencies

SSH into the instance:

```bash
ssh root@<instance-ip> -p <port>
```

Run the setup:

```bash
# System deps
apt-get update && apt-get install -y ffmpeg git libgl1

# Clone repo
git clone https://github.com/yourrepo/phantom.git /workspace/phantom
cd /workspace/phantom

# Python deps
pip install -r requirements-pipeline-gpu.txt

# Download face swap model
mkdir -p pipeline/models
# Place inswapper_128.onnx in pipeline/models/
# Place GFPGANv1.4.pth in pipeline/models/ (optional, for enhancement)
```

---

### 3. Create startup script

```bash
cat > /workspace/start.sh << 'EOF'
#!/bin/bash
cd /workspace/phantom
python pipeline.py --execution-provider cuda
EOF

chmod +x /workspace/start.sh
```

---

### 4. Save as template

**RunPod:** Pod menu → Save as Template → name it `phantom-pipeline`

**Vast.ai:**
```bash
vastai create template \
  --name "phantom-pipeline" \
  --image <your-image-id> \
  --onstart "/workspace/start.sh"
```

From this point forward — boot template, pipeline is live in ~30 seconds.

---

## Per-Session Workflow

### Start a session

**RunPod:**
1. Go to Templates → `phantom-pipeline` → Deploy
2. Wait ~30 seconds for boot + auto-start
3. Copy the public IP from the pod dashboard
4. On your desktop:
```bash
python desktop.py --host <pod-ip> --port 9000
```

**Vast.ai:**
```bash
# Find your saved template
vastai search offers 'gpu_name=RTX_3090 reliability>0.95'

# Launch with startup script
vastai create instance <OFFER_ID> --template phantom-pipeline

# Get instance IP
vastai show instance <INSTANCE_ID>

# Connect desktop
python desktop.py --host <instance-ip> --port 9000
```

---

### End a session

**RunPod:** Stop pod (keeps storage, ~$0.07/GB/month idle) or Terminate (zero cost, redeploy next time from template).

**Vast.ai:**
```bash
vastai destroy instance <INSTANCE_ID>
```
Zero ongoing cost. Next session creates a fresh instance from template.

---

## Port Reference

| Port | Protocol | Direction | Must be open |
|---|---|---|---|
| 9000 | TCP | Desktop → Pipeline | Yes — HTTP control |
| 9001 | TCP | Pipeline → Desktop | Yes — WebSocket frames |
| 5000 | TCP | Desktop → Pipeline | Yes — webcam feed |

Confirm all three are open before starting a session. On RunPod check the pod's port mapping page. On Vast.ai verify with:

```bash
vastai show instance <ID> | grep ports
```

---

## Quality Preset by GPU

| GPU | VRAM | Recommended Preset | GFPGAN |
|---|---|---|---|
| RTX 3080 | 10GB | `optimal` | interval 10+ |
| RTX 3090 | 24GB | `optimal` or `production` | interval 5 |
| RTX 4090 | 24GB | `production` | interval 1 |

Start with `optimal`. Switch to `production` if you have headroom and want maximum quality.

---

## Troubleshooting

**Desktop shows "cannot reach server"**
- Confirm pipeline booted (check instance logs)
- Confirm ports 9000 and 9001 are exposed and mapped
- Check the IP/port you passed to `--host` and `--port`

**OUTPUT panel stays blank after START**
- Port 5000 may not be open — pipeline not receiving webcam feed
- Check instance firewall rules
- Verify `tcp://0.0.0.0:5000?listen` in pipeline logs

**Frames arrive but quality is poor**
- Upgrade quality preset: `optimal` → `production`
- Check GPU utilisation on instance — if maxed, drop to `fast`

**High latency between action and output**
- Expected: ~200–500ms over internet
- On LAN or same-city cloud: ~50–100ms
- Reduce with `fast` preset (lower model complexity)

**Instance keeps failing (Vast.ai)**
- Switch to a host with `reliability > 0.97`
- Or move to RunPod for the session
