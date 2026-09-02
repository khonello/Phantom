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
# This pull updates the repository that contains *this script*, while bash is
# part-way through executing it. Bash reads a script incrementally, by byte
# offset, so a file that changes underneath it is resumed at the old offset in
# the new content: blocks get skipped, repeated, or spliced together. A boot
# that pulled a fix then behaved as though it had not is this, and it is
# indistinguishable from the fix being wrong.
#
# So: if the pull moved HEAD, hand over to the new copy with exec and start
# again from the top. PHANTOM_STARTUP_REEXEC makes that at most once, so a pull
# that somehow keeps moving cannot loop.
if [ -d "${PHANTOM_DIR}/.git" ]; then
    echo "Pulling latest changes..."
    _GIT_BEFORE=$(git -C "${PHANTOM_DIR}" rev-parse HEAD 2>/dev/null || echo none)

    # Never let git ask a question. There is no terminal to answer on, so a
    # remote that wants credentials — an expired token in the clone URL is the
    # usual way — does not fail, it *blocks*, forever, at exactly this line.
    # From outside that is indistinguishable from a slow boot.
    export GIT_TERMINAL_PROMPT=0
    export GIT_ASKPASS=/bin/true
    export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

    # And bound it, because a half-open connection to the remote hangs a fetch
    # rather than erroring. Two minutes is far more than this pull ever needs.
    # Retried, because the failure this actually hits is transient. GitHub
    # refuses *anonymous* traffic from shared datacenter addresses under load
    # and answers 401, so a pod can pull fine for days and then not, with
    # nothing changed at either end — it cleared on its own once, and returned.
    # Three attempts over ~20s costs nothing on the happy path and rides
    # straight through the version of this that resolves by itself.
    _GIT_STATUS=1
    for _GIT_TRY in 1 2 3; do
        set +e
        _GIT_OUTPUT=$(timeout 120 git -C "${PHANTOM_DIR}" pull --ff-only 2>&1)
        _GIT_STATUS=$?
        set -e
        echo "${_GIT_OUTPUT}"
        [ ${_GIT_STATUS} -eq 0 ] && break
        if [ ${_GIT_TRY} -lt 3 ]; then
            echo "  pull attempt ${_GIT_TRY} failed (exit ${_GIT_STATUS}) — retrying..."
            sleep $((_GIT_TRY * 5))
        fi
    done

    if [ ${_GIT_STATUS} -ne 0 ]; then
        # This used to be one swallowed WARNING, and that is why a stale pod is
        # hard to recognise: the boot continues, the pipeline starts, everything
        # looks healthy, and it is running code from before the fix that is
        # being tested. The operator sees changes "not arriving" and suspects
        # the change rather than the deploy.
        echo ""
        echo "=============================================================="
        if [ ${_GIT_STATUS} -eq 124 ]; then
            echo "ERROR: git pull timed out after 120s."
            echo "The remote accepted the connection and then stopped talking,"
            echo "or it wanted credentials this script refuses to supply."
        else
            echo "ERROR: git pull failed (exit ${_GIT_STATUS})."
        fi
        echo ""
        echo "Local HEAD:  $(git -C "${PHANTOM_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "Branch:      $(git -C "${PHANTOM_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        echo "Remote:      $(git -C "${PHANTOM_DIR}" remote get-url origin 2>/dev/null | sed 's#//[^@]*@#//***@#' || echo unknown)"
        echo ""
        # Only *tracked* modifications block a fast-forward. Untracked files do
        # not, and reporting them as the cause sends the reader after the wrong
        # thing — which it did, pointing at a stray directory while the real
        # answer was sitting in the 401 above.
        _GIT_MODIFIED=$(git -C "${PHANTOM_DIR}" status --porcelain --untracked-files=no 2>/dev/null | head -20)
        if [ -n "${_GIT_MODIFIED}" ]; then
            echo "Tracked files are modified, and --ff-only will not overwrite"
            echo "them. The checkout is on the network volume, so this outlives"
            echo "the pod and a fresh GPU changes nothing:"
            echo ""
            echo "${_GIT_MODIFIED}"
            echo ""
            echo "To discard them and take the remote's version:"
            echo "  python runpod/orchestrator.py run \\"
            echo "    \"git -C ${PHANTOM_DIR} reset --hard\""
        fi

        case "${_GIT_OUTPUT}" in
            *401*|*"could not read Username"*|*"Authentication failed"*)
                echo "This is an authentication failure, and the remote above is"
                echo "the thing to look at rather than the working tree."
                echo ""
                echo "If that repository is public, GitHub is refusing this pod"
                echo "*anonymously* — shared datacenter addresses get rate"
                echo "limited, which is why it can work for days and then not."
                echo "An authenticated pull is not subject to that limit, so the"
                echo "fix is the same either way: give the URL a token."
                echo ""
                echo "  python runpod/orchestrator.py run \\"
                echo "    \"git -C ${PHANTOM_DIR} remote set-url origin \\"
                echo "     https://<token>@github.com/<owner>/<repo>.git\""
                echo ""
                echo "Set RUNPOD_REPO_URL in .env to the same, so a future pod"
                echo "clones with it rather than rediscovering this."
                ;;
        esac
        echo ""
        echo "Booting anyway would run code that is not the code you pushed."
        echo "Set PHANTOM_ALLOW_STALE=1 to continue regardless."
        echo "=============================================================="

        if [ -z "${PHANTOM_ALLOW_STALE:-}" ]; then
            exit 1
        fi
        echo "PHANTOM_ALLOW_STALE set — continuing with the existing checkout."
    fi

    _GIT_AFTER=$(git -C "${PHANTOM_DIR}" rev-parse HEAD 2>/dev/null || echo none)

    if [ "${_GIT_BEFORE}" != "${_GIT_AFTER}" ]; then
        if [ -n "${PHANTOM_STARTUP_REEXEC:-}" ]; then
            echo "Code changed again after re-exec — continuing with what is loaded."
        else
            echo "Code changed (${_GIT_BEFORE:0:7} -> ${_GIT_AFTER:0:7}); re-running startup.sh from the new copy."
            export PHANTOM_STARTUP_REEXEC=1
            exec bash "${BASH_SOURCE[0]}" "$@"
        fi
    fi
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

# ── 6a. One onnxruntime, and it has to be the GPU one ───────────────────
# insightface depends on `onnxruntime` — the CPU wheel — and both packages
# install into the same `onnxruntime/` directory. Whichever pip writes last
# wins, and it writes the CPU one last, quietly replacing the GPU binaries
# installed moments earlier in the same command.
#
# Nothing about that looks like a failure. The only symptom is
# get_available_providers() returning ['AzureExecutionProvider',
# 'CPUExecutionProvider'], which core.py turns into the argparse choices for
# --execution-provider. So the pipeline rejects its own launch flag:
#
#   pipeline.py: error: argument --execution-provider: invalid choice: 'cuda'
#
# Outside the requirements-hash guard on purpose: a warm volume skips the pip
# sync, and this has to be true on every boot, not just the ones that install.
echo ""
echo "--- ONNX Runtime ---"
_phase "onnxruntime"

_ort_providers() {
    ${PYTHON} -c "
import onnxruntime
print(','.join(onnxruntime.get_available_providers()))
" 2>/dev/null || echo ""
}

# Remove the CPU wheel first: while it is installed it owns the shared files,
# so reinstalling the GPU build under it achieves nothing.
if ${PIP} list --format=freeze 2>/dev/null | grep -q '^onnxruntime=='; then
    echo "CPU onnxruntime present (pulled in by insightface) - removing."
    ${PIP} uninstall -y onnxruntime
fi

# Then repair on the evidence, not on whether that removal just happened.
# Uninstalling the CPU wheel deletes the files it overwrote, which are the GPU
# build's own — so a venv can reach this point with no CPU wheel installed and
# a gutted GPU one, and gating the reinstall on "did we just remove something"
# skips the repair exactly when it is needed. The provider list is the thing
# that has to be true; test that instead.
ORT_PROVIDERS=$(_ort_providers)
case "${ORT_PROVIDERS}" in
    *CUDAExecutionProvider*)
        echo "CUDAExecutionProvider already available."
        ;;
    *)
        echo "No CUDAExecutionProvider (have: ${ORT_PROVIDERS:-<none>}) - reinstalling GPU build."
        ${PIP} install --force-reinstall --no-deps 'onnxruntime-gpu>=1.15.0'
        ORT_PROVIDERS=$(_ort_providers)
        ;;
esac

echo "Providers: ${ORT_PROVIDERS:-<none>}"
case "${ORT_PROVIDERS}" in
    *CUDAExecutionProvider*)
        echo "Verified: CUDAExecutionProvider is available."
        ;;
    *)
        echo "ERROR: onnxruntime offers no CUDAExecutionProvider."
        echo "       --execution-provider cuda would be rejected as an invalid"
        echo "       choice and the pipeline would never start."
        echo "       Installed onnxruntime packages:"
        ${PIP} list --format=freeze 2>/dev/null | grep -i onnxruntime || echo "       (none)"
        exit 1
        ;;
esac

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

# ── 6c. TensorRT for ONNX Runtime (opt-in) ───────────────────────────────────
# onnxruntime-gpu ships the TensorRT execution provider but dlopens libnvinfer
# at runtime, and nothing in the base image provides it. Without this the
# provider never registers and TRT=true does nothing at all.
#
# Gated on TRT because the wheels are ~2GB and most sessions do not want them.
# The flag is forwarded from .env by the orchestrator, so asking for TensorRT in
# .env is what installs it.
#
# NOT fatal, deliberately — and this is the one place that differs from 6b
# above. Missing cuDNN means every ONNX model silently runs on CPU, which is a
# paid GPU hour producing nothing usable, so that exits 1. Missing TensorRT
# means the models run on CUDA instead: still on the GPU, still holding a live
# call, merely not as fast as intended. Halting the session over it would cost
# the operator more than the fallback does. See docs/COMPILATION.md.
if [ "${TRT:-}" = "true" ] || [ "${TRT:-}" = "1" ]; then
    echo ""
    echo "--- TensorRT ---"
    _phase "tensorrt"

    TRT_OK=$(${PYTHON} -c "
import onnxruntime
print('yes' if 'TensorrtExecutionProvider' in onnxruntime.get_available_providers() else 'no')
" 2>/dev/null || echo "no")

    if [ "${TRT_OK}" = "yes" ]; then
        echo "TensorrtExecutionProvider already available."
    else
        echo "Installing tensorrt (~2GB, this is why it is opt-in)..."
        ${PIP} install --no-deps tensorrt tensorrt-cu12 tensorrt_libs 2>/dev/null             || ${PIP} install tensorrt             || echo "WARNING: tensorrt install failed."

        TRT_LIB_DIR=$(${PYTHON} "${PHANTOM_DIR}/runpod/tensorrt_path.py" || echo "")
        if [ -n "${TRT_LIB_DIR}" ]; then
            export LD_LIBRARY_PATH="${TRT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
            echo "export LD_LIBRARY_PATH=\"${TRT_LIB_DIR}:\${LD_LIBRARY_PATH:-}\""                 > /etc/profile.d/tensorrt.sh
            echo "LD_LIBRARY_PATH: ${TRT_LIB_DIR}"
        else
            echo "WARNING: TensorRT lib dir not found."
        fi
    fi

    # Ask onnxruntime, not the filesystem. A library that loads proves nothing
    # about whether the provider registered against this onnxruntime build —
    # the TensorRT major version has to match, and a mismatch shows up here
    # rather than as an error anywhere else.
    TRT_FINAL=$(${PYTHON} -c "
import onnxruntime
print('yes' if 'TensorrtExecutionProvider' in onnxruntime.get_available_providers() else 'no')
" 2>/dev/null || echo "no")

    if [ "${TRT_FINAL}" = "yes" ]; then
        echo "Verified: TensorrtExecutionProvider is registered."
    else
        echo "WARNING: TensorRT was requested but the provider is not registered."
        echo "         Models will run on CUDA instead — correct, just not as fast."
        echo "         Usually a version mismatch between tensorrt and onnxruntime-gpu."
        echo "         Continuing: a CUDA session is worth more than no session."
    fi
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
