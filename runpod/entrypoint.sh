#!/usr/bin/env bash
# Phantom — Docker container entrypoint.
#
# Docker mode skips startup.sh entirely: the image already has the system
# packages, the venv and cuDNN baked in, and the build fails if cuDNN cannot
# load. What it does *not* have is the model weights — those live on the
# /workspace network volume by design, so they are the one thing still fetched
# at run time.
#
# Which means without this, the first frame of a paid session pays for the
# download: 384 MB for hyperswap, plus the detector, restorer and occluder. That
# cost belongs here, before the pipeline announces itself as ready, so it lands
# on container start rather than on a customer.
#
# This is the Docker counterpart to startup.sh steps 7 and 8. Keep the two in
# step — tests/test_wiring.py asserts both paths pre-warm.

set -uo pipefail

PHANTOM_DIR="${PHANTOM_DIR:-/app}"
cd "${PHANTOM_DIR}"

echo "=== Phantom container starting ==="
echo "Swap model: ${SWAPPER_MODEL:-inswapper_128 (default)}"

# RunPod exposes pod-level environment variables through this file for shells
# that do not inherit them. Sourcing it is harmless when it is absent.
if [ -f /etc/rp_environment ]; then
    # shellcheck disable=SC1091
    . /etc/rp_environment
fi

echo ""
echo "--- Model Pre-Warm ---"
# Deliberately `;` rather than `&&`: a pre-warm failure means a slow first
# frame, not a broken pod, and the pipeline downloads on demand anyway. The
# checks that genuinely must stop things are the cuDNN test baked into the image
# build and the execution-provider check inside the pipeline itself.
if ! python "${PHANTOM_DIR}/runpod/prewarm.py"; then
    echo "WARNING: pre-warm incomplete — models will load on first request."
fi

echo ""
echo "--- Pipeline ---"
# exec so the pipeline becomes PID 1 and receives SIGTERM directly. Without it
# the shell holds PID 1, signals never reach Python, and RunPod stopping the pod
# turns into a hard kill — which loses the guard telemetry and latency report
# that are written when the stream shuts down cleanly.
exec python "${PHANTOM_DIR}/pipeline.py" --execution-provider cuda "$@"
