#!/usr/bin/env bash
# Phantom — RunPod Pod Startup Script
#
# Run once after the very first pod creation to prepare the environment.
# On pod resume or new pod deployment (same network volume), most steps
# are skipped because the venv and models already live on /workspace.
#
# Dependency sync: on every run, compares requirements-pipeline-gpu.txt
# against a snapshot stored on the volume. If requirements changed since
# the last install, pip install runs again to pick up new/removed packages.
#
# Usage (from repo root):
#   bash runpod/startup.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS_DIR="${WORKSPACE}/models"
VENV_DIR="${WORKSPACE}/venv"
# Derive repo root from the script's own location (runpod/startup.sh → repo root)
PHANTOM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
REQUIREMENTS="${PHANTOM_DIR}/requirements-pipeline-gpu.txt"
REQUIREMENTS_SNAPSHOT="${VENV_DIR}/.requirements-snapshot"

# RunPod exposes pod-level environment variables (everything orchestrator.py
# forwards) through this file. An SSH session usually inherits them via .bashrc,
# but "usually" is not good enough for settings that decide what a paid session
# measures — and _shell_run wraps each command in a subshell, so sourcing from
# the orchestrator side would not survive into this script anyway.
if [ -f /etc/rp_environment ]; then
    # shellcheck disable=SC1091
    . /etc/rp_environment
fi

echo "=== Phantom RunPod Startup ==="
echo "Workspace:  ${WORKSPACE}"
echo "Venv:       ${VENV_DIR}"
echo "Models dir: ${MODELS_DIR}"
echo "Phantom:    ${PHANTOM_DIR}"

# ── Phase timing ──────────────────────────────────────────────────────────────
# Cold start is the only number that still matters for this product, and nobody
# has one. Timing it with a stopwatch tells you the total; this tells you which
# part to attack, which is the difference between "cold start is slow" and
# "pip is 80% of cold start, so bake an image".
#
# Emitted as `PHASE <name> <seconds>` so the orchestrator can parse it out of
# the SSH transcript without the two having to agree on anything else.
_PHASE_CUR=""
_PHASE_T0=0
_PHASE_LINES=""
_RUN_T0=$(date +%s)

_phase() {
    local now
    now=$(date +%s)
    if [ -n "${_PHASE_CUR}" ]; then
        _PHASE_LINES="${_PHASE_LINES}PHASE ${_PHASE_CUR} $((now - _PHASE_T0))"$'
'
    fi
    _PHASE_CUR="$1"
    _PHASE_T0="${now}"
}

# Whether this volume already had the expensive artefacts on it. The whole point
# of measuring cold start is comparing these two cases, so the run has to say
# which one it was rather than leaving it to be remembered.
VOLUME_STATE="empty"
if [ -d "${VENV_DIR}" ] && [ -d "${MODELS_DIR}" ]; then
    VOLUME_STATE="warm"
fi

# ── 1. Install system packages (re-installs on each new container) ────────────
echo ""
echo "--- System Packages ---"
_phase "apt-get"
NEED_INSTALL=false
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg
else
    echo "Already installed: ffmpeg"
fi

# ── 2. Pull latest code ───────────────────────────────────────────────────────
echo ""
echo "--- Code Sync ---"
_phase "git-pull"
if [ -d "${PHANTOM_DIR}/.git" ]; then
    echo "Pulling latest changes..."
    git -C "${PHANTOM_DIR}" pull --ff-only 2>&1 || echo "WARNING: git pull failed — using existing code."
else
    echo "Not a git repo — skipping pull."
fi

# ── 3. Check CUDA ──────────────────────────────────────────────────────────────
echo ""
echo "--- CUDA ---"
_phase "cuda-check"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo "GPU:    ${GPU_NAME}"
    echo "Driver: ${DRIVER}"
else
    echo "WARNING: nvidia-smi not found. No GPU acceleration available."
fi

# ── 4. Create model cache directory ───────────────────────────────────────────
echo ""
echo "--- Model Cache ---"
_phase "model-cache"
if [ -d "${MODELS_DIR}" ]; then
    echo "Exists: ${MODELS_DIR}"
    ls -lh "${MODELS_DIR}/" 2>/dev/null || echo "  (empty)"
else
    mkdir -p "${MODELS_DIR}/insightface"
    echo "Created: ${MODELS_DIR}"
fi

# ── 5. Create or reuse /workspace/venv ────────────────────────────────────────
# The venv lives on the network volume so it survives pod restarts and
# new pod deployments. Packages are installed on first run, and re-synced
# whenever requirements-pipeline-gpu.txt changes.
echo ""
echo "--- Python Venv ---"
_phase "venv"
if [ -d "${VENV_DIR}" ]; then
    # Ensure venv has system-site-packages enabled (needed for image's PyTorch/CUDA)
    CFG="${VENV_DIR}/pyvenv.cfg"
    if grep -q "include-system-site-packages = false" "${CFG}" 2>/dev/null; then
        echo "Upgrading venv to use system-site-packages (for CUDA-compatible PyTorch)..."
        sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "${CFG}"
    fi
    echo "Venv already exists at ${VENV_DIR}."
    echo "Python: $(${PYTHON} --version 2>&1)"
else
    echo "Creating venv at ${VENV_DIR}..."
    # --system-site-packages inherits the image's pre-installed PyTorch/torchvision
    # which are compiled for the correct CUDA version on this host.
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "Created. Python: $(${PYTHON} --version 2>&1)"

    ${PIP} install --upgrade pip --quiet
fi

# ── 6. Sync dependencies ─────────────────────────────────────────────────────
# Compare current requirements against the snapshot from last install.
# If they differ (or no snapshot exists), run pip install to sync.
echo ""
echo "--- Dependencies ---"
_phase "pip-install"
if [ -f "${REQUIREMENTS}" ]; then
    if [ -f "${REQUIREMENTS_SNAPSHOT}" ] && diff -q "${REQUIREMENTS}" "${REQUIREMENTS_SNAPSHOT}" &>/dev/null; then
        echo "Requirements unchanged — skipping pip install."
    else
        if [ -f "${REQUIREMENTS_SNAPSHOT}" ]; then
            echo "Requirements changed since last install — syncing..."
        else
            echo "First install — installing all dependencies..."
        fi
        ${PIP} install -r "${REQUIREMENTS}"
        cp "${REQUIREMENTS}" "${REQUIREMENTS_SNAPSHOT}"
        echo "Dependencies synced."
    fi
else
    echo "WARNING: requirements-pipeline-gpu.txt not found at ${PHANTOM_DIR}."
    echo "Run manually: ${PIP} install -r requirements-pipeline-gpu.txt"
fi

# ── 6b. cuDNN 9 for ONNX Runtime ─────────────────────────────────────────────
# onnxruntime-gpu requires libcudnn.so.9 which most RunPod base images don't
# ship. Install nvidia-cudnn-cu12 with --no-deps to get just the .so files
# without letting pip's dependency resolver upgrade torch or other packages.
echo ""
echo "--- cuDNN 9 ---"
_phase "cudnn"
CUDNN_OK=$(${PYTHON} -c "
import ctypes
try: ctypes.CDLL('libcudnn.so.9'); print('yes')
except: print('no')
" 2>/dev/null || echo "no")

if [ "${CUDNN_OK}" = "yes" ]; then
    echo "libcudnn.so.9 already available."
else
    echo "Installing nvidia-cudnn-cu12 (--no-deps to avoid torch upgrade)..."
    ${PIP} install --no-deps 'nvidia-cudnn-cu12>=9.0'
    echo "Installed."
fi

# Ensure the cuDNN .so is on LD_LIBRARY_PATH for onnxruntime.
# Export for this session (inherited by nohup pipeline) and persist
# to /etc/profile.d/ for manual SSH sessions.
# The same helper the Docker build uses, so the two cannot diverge again.
# Errors are shown rather than swallowed: the previous `2>/dev/null` turned
# a real TypeError into an empty result and a misleading warning.
CUDNN_LIB_DIR=$(${PYTHON} "${PHANTOM_DIR}/runpod/cudnn_path.py" || echo "")
if [ -n "${CUDNN_LIB_DIR}" ] && [ -d "${CUDNN_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    echo "export LD_LIBRARY_PATH=\"${CUDNN_LIB_DIR}:\${LD_LIBRARY_PATH:-}\"" \
        > /etc/profile.d/cudnn.sh
    echo "LD_LIBRARY_PATH: ${CUDNN_LIB_DIR}"
else
    echo "WARNING: cuDNN lib dir not found."
fi

# Verify rather than assume. Every step above can succeed and still leave the
# library unloadable, and a silent CPU fallback is the most expensive failure on
# this path: the swapper, CodeFormer and XSeg are all ONNX, so it is seconds per
# frame instead of a live call, with nothing raised to notice.
#
# Fatal on purpose. Failing here is obvious and costs a redeploy; booting anyway
# spends a paid GPU hour producing output nobody can use, which is the entire
# reason for renting the pod in the first place.
CUDNN_FINAL=$(${PYTHON} -c "
import ctypes
try: ctypes.CDLL('libcudnn.so.9'); print('yes')
except Exception: print('no')
" 2>/dev/null || echo "no")

if [ "${CUDNN_FINAL}" = "yes" ]; then
    echo "Verified: libcudnn.so.9 loads."
else
    echo "ERROR: libcudnn.so.9 cannot be loaded after installation."
    echo "       onnxruntime-gpu would silently fall back to CPU."
    echo "       See runpod/TROUBLESHOOTING.md section 5b."
    exit 1
fi

# ── 7. GFPGAN model download ──────────────────────────────────────────────────
echo ""
echo "--- GFPGAN Model ---"
_phase "gfpgan-download"
GFPGAN_PATH="${MODELS_DIR}/GFPGANv1.4.pth"
GFPGAN_URL="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
if [ -f "${GFPGAN_PATH}" ]; then
    echo "Already downloaded: ${GFPGAN_PATH} ($(du -h "${GFPGAN_PATH}" | cut -f1))"
else
    echo "Downloading GFPGANv1.4.pth..."
    wget -q --show-progress -O "${GFPGAN_PATH}" "${GFPGAN_URL}"
    echo "Downloaded: ${GFPGAN_PATH} ($(du -h "${GFPGAN_PATH}" | cut -f1))"
fi

# ── 8. Model pre-warm ─────────────────────────────────────────────────────────
echo ""
echo "--- Model Pre-Warm ---"
_phase "model-load"
if [ -f "${PHANTOM_DIR}/pipeline/__init__.py" ]; then
    echo "Loading models into cache..."
    cd "${PHANTOM_DIR}"
    # A separate script rather than an inline heredoc: this needs to import the
    # pipeline, apply the configured model profile and load four models, and
    # that is past the point where quoting it inside bash stays readable.
    #
    # Non-fatal by design. A pre-warm failure means a slow first frame, not a
    # broken pod, and the pipeline downloads on demand anyway. The cuDNN check
    # above is the one that genuinely must stop the deploy.
    if ! ${PYTHON} "${PHANTOM_DIR}/runpod/prewarm.py" 2>&1; then
        echo "WARNING: pre-warm incomplete — models will load on first request."
    fi
else
    echo "Phantom not found at ${PHANTOM_DIR} — skipping warmup."
fi

# ── 9. Summary ─────────────────────────────────────────────────────────────────
echo ""
_phase ""

echo "=== Startup Complete ==="
echo ""
echo "--- Phase Timings (volume: ${VOLUME_STATE}) ---"
printf "%b" "${_PHASE_LINES}"
echo "PHASE total $(( $(date +%s) - _RUN_T0 ))"
echo "VOLUME ${VOLUME_STATE}"
echo ""
echo "To start the pipeline (always use the workspace venv):"
echo "  cd ${PHANTOM_DIR}"
echo "  ${PYTHON} pipeline.py --execution-provider cuda"
echo ""
echo "Or in background (survives SSH disconnects):"
echo "  nohup ${PYTHON} pipeline.py --execution-provider cuda > /workspace/phantom-pipeline.log 2>&1 &"
echo "  tail -f /workspace/phantom-pipeline.log"
echo ""
