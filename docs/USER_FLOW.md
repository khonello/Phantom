# Phantom — User Flow

## Overview

Phantom runs as two separate programs that talk to each other over the network.

| Program | Command | Role |
|---|---|---|
| Pipeline | `python pipeline.py` | Headless AI worker — does the face swap |
| Desktop | `python desktop.py` | GUI — controls the pipeline, displays output |

They can run on the same machine or different machines. The pipeline does all the heavy lifting; the desktop just controls it and shows results.

---

## Starting Up

### 1. Start the pipeline first

```bash
python pipeline.py
# with CUDA:
python pipeline.py --execution-provider cuda
# on a different port:
python pipeline.py --port 9000
```

The pipeline starts two servers and waits for commands:
- HTTP control server on **port 9000**
- WebSocket frame server on **port 9001**

### 2. Start the desktop

```bash
python desktop.py
# connecting to a remote pipeline:
python desktop.py --host 192.168.1.10 --port 9000
```

On startup the desktop:
- Opens your webcam and shows it in the **SOURCE panel** (always on)
- Begins polling the pipeline every 2 seconds (connection dot in header)

---

## Step-by-Step User Flow

### Step 1 — Select a face source

Click **Select Source Images** in the sidebar.

| Selection | What happens |
|---|---|
| Single image | Sent directly to pipeline as the swap source |
| Multiple images | Pipeline averages the face embeddings across all images — better identity consistency. Status shows "creating embedding…" then "embedding ready" |

> The face source can be changed at any time, even mid-stream.

---

### Step 2 — Configure settings (optional)

| Setting | Description |
|---|---|
| **Webcam Index** | Which camera to use. `0` is default. Change if you have multiple cameras. |
| **Quality** | `fast` — low latency, less accurate. `optimal` — balanced (default). `production` — highest quality, more GPU load. |
| **Stream URL** | RTMP endpoint to push the output to (e.g. `rtmp://live.twitch.tv/live/YOUR_KEY`). Only needed if you want to stream externally. |

---

### Step 3 — Press START

What happens:
1. Desktop tells pipeline to begin (HTTP command)
2. Desktop starts piping your webcam to the pipeline over **UDP port 5000** (via ffmpeg)
3. Pipeline reads the webcam feed, swaps faces frame by frame, pushes results back over **WebSocket port 9001**
4. Desktop receives swapped frames and displays them in the **OUTPUT panel** (right side)
5. SOURCE panel continues showing your raw webcam — unchanged

> The pipeline uses your selected face image (or embedding) as the source and your webcam as the target.

---

### Step 4 — Press STREAM (optional)

Requires a URL in the **Stream URL** field.

What happens:
1. Desktop opens an ffmpeg process
2. Each incoming face-swapped frame is piped to ffmpeg → encoded as H.264 → pushed to your RTMP endpoint
3. Border of OUTPUT panel turns green, label shows **LIVE**

> Streaming is handled entirely by the desktop. The pipeline is unaware of the RTMP destination.

To stop streaming without stopping the pipeline, press **STREAMING** again to toggle off. The OUTPUT panel continues showing the live feed.

---

### Step 5 — Press STOP

What happens:
1. RTMP stream is stopped (if active)
2. WebSocket receiver disconnects
3. UDP webcam broadcast stops
4. Pipeline is told to stop its face-swap loop
5. SOURCE panel continues showing your webcam (it never stopped)

---

## Ports Reference

| Port | Protocol | Direction | Purpose |
|---|---|---|---|
| 9000 | HTTP | Desktop → Pipeline | Commands (start, stop, set source, etc.) and status polling |
| 9001 | WebSocket | Pipeline → Desktop | Continuous face-swapped JPEG frame push |
| 5000 | TCP | Desktop → Pipeline | Raw webcam feed (H.264 MPEG-TS via ffmpeg) |

---

## Panel Reference

| Panel | Always visible | Content |
|---|---|---|
| SOURCE (left) | Yes | Raw webcam feed — never affected by pipeline state |
| OUTPUT (right) | After START | Face-swapped frames from pipeline. Green border when streaming to RTMP. |

---

## Status Messages

| Message | Meaning |
|---|---|
| `idle` | Desktop started, no action taken yet |
| `connecting...` | Trying to reach pipeline |
| `face set: filename.jpg` | Single source image accepted |
| `creating embedding from N images...` | Pipeline is averaging N face images |
| `embedding ready` | Embedding complete, ready to start |
| `no face detected in selected images` | Pipeline couldn't find a face in the selected images |
| `pipeline connected · processing` | Pipeline running, frames flowing |
| `streaming...` | RTMP stream active |
| `no signal from server` | WebSocket connected but no frames received for 3+ seconds |
| `cannot reach server — ...` | HTTP request to pipeline failed |
| `stopped` | Pipeline stopped cleanly |

---

## Connection Indicator (Header)

| State | Colour | Meaning |
|---|---|---|
| Pulsing green dot | Green | Pipeline reachable and responding |
| Red dot | Red | Pipeline unreachable or not running |

The label next to the dot shows `host:port` of the pipeline being polled.

---

## Notes

- The pipeline can run on a separate GPU machine; point the desktop at it with `--host`
- Face source can be swapped mid-stream — the pipeline reloads it automatically
- Quality preset takes effect on the next `start_stream` command
- If the WebSocket drops mid-stream, the receiver reconnects automatically every second
- The RTMP ffmpeg process is separate from the UDP broadcast ffmpeg — each can fail independently without affecting the other
