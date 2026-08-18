# Phantom — GPU Pipeline Docker Image
#
# Bakes in all dependencies so the pod boots straight into the pipeline.
# Models live on the /workspace network volume (not in the image).
#
# Build:
#   docker build -t <registry>/phantom-pipeline:latest .
#
# Push:
#   docker push <registry>/phantom-pipeline:latest

# Pinned to the same tag RUNPOD_IMAGE uses, so development and production are
# the same machine. Two traps here, both already paid for once:
#   - `runtime` is not a published tag for runpod/pytorch (TROUBLESHOOTING
#     section 5). `devel` is the only option, and is also the smallest.
#   - drifting this away from RUNPOD_IMAGE means production runs a different
#     Python, torch and CUDA than anything ever tested over SSH.
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ffmpeg tmux \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — cached layer, only rebuilds when requirements change
WORKDIR /app
COPY requirements-pipeline-gpu.txt .
# Just this helper, ahead of the full COPY, so the cuDNN layer below stays
# cacheable across code changes.
COPY runpod/cudnn_path.py /tmp/cudnn_path.py
RUN pip install --no-cache-dir -r requirements-pipeline-gpu.txt

# cuDNN 9 for ONNX Runtime. The SSH path installs this in startup.sh step 6b;
# without the equivalent here onnxruntime-gpu finds no libcudnn.so.9 and falls
# back to CPU — silently, and for the swapper, CodeFormer and XSeg alike, which
# is every model that matters. Seconds per frame rather than a live call.
#
# --no-deps for the same reason startup.sh uses it: take the .so files without
# letting the resolver touch the image's torch/CUDA build.
#
# Registered with ldconfig rather than exported as LD_LIBRARY_PATH, because CMD
# does not run a login shell and an /etc/profile.d script would never be
# sourced. It also avoids hardcoding a site-packages path that moves with the
# base image's Python version.
#
# The CDLL check fails the build rather than shipping an image that quietly
# runs on CPU — the whole point is that this cannot go wrong unnoticed.
RUN pip install --no-cache-dir --no-deps 'nvidia-cudnn-cu12>=9.0' \
    && python /tmp/cudnn_path.py > /etc/ld.so.conf.d/cudnn.conf \
    && ldconfig \
    && python -c "import ctypes; ctypes.CDLL('libcudnn.so.9')" \
    && echo "cuDNN 9 resolved."

# Application code
COPY . .

EXPOSE 9000

# Network volume mounts at /workspace — models persist there across pods, which
# is why weights are not baked into the image. That makes them the one thing
# still fetched at run time, so the entrypoint pre-warms them before the
# pipeline reports ready: otherwise the first frame of a paid session pays for a
# 384 MB download. The SSH path does the same in startup.sh, and
# tests/test_wiring.py asserts both paths still do.
RUN chmod +x /app/runpod/entrypoint.sh
ENTRYPOINT ["/app/runpod/entrypoint.sh"]
