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

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ffmpeg tmux \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — cached layer, only rebuilds when requirements change
WORKDIR /app
COPY requirements-pipeline-gpu.txt .
RUN pip install --no-cache-dir -r requirements-pipeline-gpu.txt

# Application code
COPY . .

EXPOSE 9000

# Network volume mounts at /workspace — models persist there across pods.
# InsightFace auto-downloads on first run if /workspace/models/insightface/ is empty.
CMD ["python", "pipeline.py", "--execution-provider", "cuda"]
