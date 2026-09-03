# Running the pipeline on your own GPU

For a machine with an NVIDIA card — a gaming desktop, an Alienware, a
workstation — instead of a rented pod.

The short version:

```bash
git clone <repo> && cd Phantom
python -m venv .venv && .venv\Scripts\activate     # Linux: source .venv/bin/activate
python tools/setup_local_gpu.py
```

That script installs and then verifies. If it ends with *"GPU pipeline is
ready"*, you are done. Everything below explains what it did and what to do
when it does not.

---

## Why `pip install -r requirements-pipeline-gpu.txt` is not enough

That file was written for the rented instance, whose base image already ships PyTorch
and CUDA. Three things it cannot express, each of which ends in a
**working-looking install that silently runs on CPU** — which for this pipeline
is seconds per frame rather than a live call.

### 1. It does not install PyTorch on Windows or Linux

Look at the top of the file:

```
torch; sys_platform == 'darwin'
```

macOS only. On Windows and Linux nothing installs torch until `gfpgan` pulls
one in as a dependency — and **the default PyPI wheel on Windows is the CPU
build**. A requirements file cannot name PyTorch's own package index, so this
has to happen separately:

```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2. insightface drags in the CPU onnxruntime

`insightface` depends on `onnxruntime` — the CPU wheel — and both packages
install into the same `onnxruntime/` directory. Whichever pip resolves last
wins, so the GPU build can be silently overwritten by the CPU one, and
`CUDAExecutionProvider` disappears.

The fix is after the fact, which is why it lives in a script rather than a
requirements file:

```bash
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.15.0"
```

### 3. cuDNN 9 has to be findable

`onnxruntime-gpu` loads cuDNN at runtime. `nvidia-cudnn-cu12` installs it
inside site-packages, where the loader does not look by default:

| | Location | Library |
|---|---|---|
| Linux | `nvidia/cudnn/lib` | `libcudnn.so.9` |
| Windows | `nvidia/cudnn/bin` | `cudnn64_9.dll` |

`--no-deps` matters, or it will pull its own torch over the one you just
installed:

```bash
pip install --no-deps "nvidia-cudnn-cu12>=9.0"
```

Then put that directory on the loader path:

```bash
# Windows
set PATH=<...>\site-packages\nvidia\cudnn\bin;%PATH%

# Linux
export LD_LIBRARY_PATH=<...>/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

`tools/setup_local_gpu.py --check` prints the exact path for your machine.

---

## Versions, and why they are pinned

| | Version | Why |
|---|---|---|
| Python | **3.10** | What everything here is built and tested against |
| torch | **2.2.0+cu121** | What runs in production |
| numpy | **< 2** | Not a free choice — see below |
| onnxruntime-gpu | **>= 1.15** | Resolves to 1.23.x, which needs cuDNN 9 |

**numpy 2 will break it.** torch 2.2.0 is compiled against numpy 1.x, and numpy
2 breaks the bridge between them:

```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

This is not hypothetical — it happened on the pod mid-session, from a stray
`pip install` that upgraded numpy as a side effect. `requirements-pipeline-gpu.txt`
pins `numpy>=1.24.2,<2.0` for exactly this reason. If something later drags
numpy 2 in, `tools/setup_local_gpu.py --check` names it.

Newer torch and CUDA very likely work. They are simply not what has been run,
and the pinned set is known-good.

---

## Verifying it actually uses the GPU

**This is the step to not skip.** ONNX Runtime does not raise when a provider
fails to initialise — it silently falls back to CPU. Every model that decides
how the output looks is ONNX (the swapper, the restorer, XSeg), so that
fallback is the difference between a call and a slideshow.

```bash
python tools/setup_local_gpu.py --check
```

```
== Verifying ==
  [ok] NVIDIA driver            NVIDIA GeForce RTX 4090, 560.94
  [ok] torch                    2.2.0+cu121 (CUDA 12.1)
  [ok] numpy                    1.26.4
  [ok] onnxruntime providers    TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
  [ok] ffmpeg                   ffmpeg version 7.1
  [ok] cuDNN                    ...\site-packages\nvidia\cudnn\bin
```

It exits non-zero when the GPU path is not usable, so it works in a script.

The pipeline checks this too, at startup: `pipeline/services/execution.py::verify`
raises `ExecutionProviderError` rather than running on CPU. That is deliberate
and should not be downgraded to a warning — a silent CPU fallback produces
unusable output while appearing to work.

Once running, `tools/stats.py` reports the same from the outside.

---

## Also required

- **FFmpeg** on `PATH` — RENDER mode decodes and encodes with it.
- **Model weights** download on first use into `pipeline/models/` (~1.6 GB:
  inswapper or hyperswap, the restorer, XSeg, buffalo_l). No action needed,
  but the first run is slow and needs the network.

## Then

```bash
python pipeline.py --execution-provider cuda
```

And point the desktop at it. If the desktop is the **same machine**, that is
`localhost` and the network round trip disappears — and the playout delay
follows it down on its own, since D is measured from the link over the first
few seconds rather than assumed. Watch for the
`[SYNC] playout delay calibrated to Nms` line; `PHANTOM_PLAYOUT_DELAY_MS`
overrides it if you want a specific number.

You still need OBS and a virtual audio cable on that machine, because you are
still the operator. See [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md).

---

## What this buys you

Everything measured on a rented RTX 4090 applies: ~28ms per frame at 640x360,
holding a 50ms deadline with room. The difference is the **network** — a local
pipeline removes the ~350ms round trip that dominates the felt latency on a
remote pod, which is the single largest improvement available to this system
and the one no amount of GPU work can achieve.

A local 4090 is not faster than a rented 4090. It is closer, and that is the
part that has been hurting.

---

## Honest status

`tools/setup_local_gpu.py` mirrors `vast/startup.sh`, which is proven on a
pod, but **the local install path has not been run against a real NVIDIA
machine**. The verification half is safe to run anywhere and is what tells you
whether it worked; `--dry-run` prints every command without executing it if you
would rather drive it by hand.
