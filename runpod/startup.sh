#!/usr/bin/env bash
# Phantom — RunPod Pod Startup Script
# Run once after pod creation to prepare the environment.
#
# Usage:
#   bash runpod/startup.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS_DIR="${WORKSPACE}/models"
PHANTOM_DIR="${WORKSPACE}/phantom"

echo "=== Phantom RunPod Startup ==="
echo "Workspace: ${WORKSPACE}"
echo "Models dir: ${MODELS_DIR}"

# ── 1. Install FFmpeg ──────────────────────────────────────────────────────────
echo ""
echo "--- Installing FFmpeg ---"
if command -v ffmpeg &>/dev/null; then
    echo "FFmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
else
    apt-get update -qq && apt-get install -y -qq ffmpeg
    echo "FFmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
fi

# ── 2. Check CUDA ──────────────────────────────────────────────────────────────
echo ""
echo "--- CUDA Check ---"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo "GPU detected: ${GPU_NAME}"
    echo "Driver version: ${CUDA_VERSION}"
else
    echo "WARNING: nvidia-smi not found. No GPU acceleration available."
fi

# ── 3. Create /workspace/models if not present ─────────────────────────────────
echo ""
echo "--- Model Cache Setup ---"
if [ -d "${MODELS_DIR}" ]; then
    echo "Model cache directory exists: ${MODELS_DIR}"
    echo "Contents:"
    ls -lh "${MODELS_DIR}/" 2>/dev/null || echo "  (empty)"
else
    echo "Creating model cache directory: ${MODELS_DIR}"
    mkdir -p "${MODELS_DIR}/insightface"
    echo "Directory created."
fi

# ── 4. Python dependency check ─────────────────────────────────────────────────
echo ""
echo "--- Python Environment ---"
python3 --version || echo "WARNING: python3 not found"
pip3 --version || echo "WARNING: pip3 not found"

# ── 5. Optional model pre-warming ─────────────────────────────────────────────
echo ""
echo "--- Model Pre-Warm (optional) ---"
if [ -f "${PHANTOM_DIR}/pipeline/__init__.py" ]; then
    echo "Phantom found at ${PHANTOM_DIR}. Running model warmup..."
    cd "${PHANTOM_DIR}"
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from pipeline.config import CONFIG
    from pipeline.services.face_detection import FaceDetector
    print('Loading InsightFace model...')
    det = FaceDetector(CONFIG)
    det._get_analyser()
    print('InsightFace model ready.')
except Exception as e:
    print(f'Model warmup skipped: {e}')
" 2>&1 || echo "Warmup failed (models will load on first request)."
else
    echo "Phantom not found at ${PHANTOM_DIR}. Skipping warmup."
    echo "Run: cd ${WORKSPACE} && git clone <repo> phantom && cd phantom && pip install -r requirements-pipeline-gpu.txt"
fi

# ── 6. Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Startup Complete ==="
echo ""
echo "To start the pipeline:"
echo "  cd ${PHANTOM_DIR}"
echo "  python pipeline.py --stream --execution-provider cuda"
echo ""
echo "Connect desktop GUI:"
echo "  PHANTOM_API_URL=wss://<pod-id>-9000.proxy.runpod.net/ws python desktop.py"
echo ""
