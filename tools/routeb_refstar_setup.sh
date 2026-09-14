#!/usr/bin/env bash
# Route B measurement — set up RefSTAR (AAAI 2026) beside the pipeline, on a pod.
#
# RESEMBLANCE.md §4 Route B. This is the MEASUREMENT step, not the integration:
# a reference-guided restorer, run on stills of our swapped output with the
# source's own photographs as references, timed, and read against GPEN on
# `tools/restore_probe.py`. If it wins, it becomes a registry entry.
#
# Deliberately its own venv: the pipeline's is the production runtime and
# RefSTAR pins diffusers 0.23 / transformers 4.37.2, which nothing here should
# be made to live with. torch is pinned to the version the pipeline venv already
# has so pip's wheel cache serves it rather than the network.
#
# Known trap, patched below: basicsr imports
# `torchvision.transforms.functional_tensor`, removed in torchvision 0.17. The
# import is a one-line sed away from working and there is no reason to hold
# torch back for it.
set -euo pipefail

ROOT=/workspace
REPO=$ROOT/RefSTAR
VENV=$ROOT/venv-refstar
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
# The Google Drive folder the README names for the pretrained weights.
GDRIVE_FOLDER=1-PW53efDRvFpxFhnHyl20lEKVW1jAF59

echo "== $(date -u +%H:%M:%S) clone"
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/yinzhicun/RefSTAR.git "$REPO"
fi

echo "== $(date -u +%H:%M:%S) venv"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
fi
"$PIP" install -q --upgrade pip wheel setuptools

echo "== $(date -u +%H:%M:%S) torch (pinned to the pipeline venv's, cu121)"
"$PIP" install -q torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

echo "== $(date -u +%H:%M:%S) requirements"
"$PIP" install -q -r "$REPO/requirements.txt"
# torch 2.2 was built against numpy 1.x; the requirements' `numpy>=1.23` pulls
# numpy 2 and torch then reports "Numpy is not available" from inside basicsr.
# And opencv-python 5 requires numpy 2, so OpenCV is held at the last numpy-1
# release with it. Both pins are about torch 2.2's ABI, not about RefSTAR.
"$PIP" install -q "numpy<2" "opencv-python==4.10.0.84"
# diffusers 0.23 imports `cached_download`, gone from huggingface_hub 0.26.
"$PIP" install -q "huggingface_hub<0.26"
# peft >= 0.11 imports `EncoderDecoderCache`, which transformers 4.37 lacks.
"$PIP" install -q "peft==0.10.0"
# face-alignment 1.4 renamed LandmarksType._2D to TWO_D; the repo uses the old name.
"$PIP" install -q "face-alignment==1.3.5"
"$PIP" install -q gdown

echo "== $(date -u +%H:%M:%S) patch basicsr for torchvision >= 0.17"
# Located without importing basicsr, which cannot be imported until this very
# patch has been applied.
BASICSR_DEG=$("$PY" - <<'EOF'
import importlib.util, os
spec = importlib.util.find_spec('basicsr')
print(os.path.join(os.path.dirname(spec.origin), 'data', 'degradations.py'))
EOF
)
if grep -q "functional_tensor" "$BASICSR_DEG"; then
  sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$BASICSR_DEG"
  echo "   patched $BASICSR_DEG"
fi

echo "== $(date -u +%H:%M:%S) patch RefSTAR for face-alignment >= 1.3.5 (LandmarksType._2D -> TWO_D)"
grep -rl "LandmarksType._2D" "$REPO/test" | xargs -r sed -i 's/LandmarksType\._2D/LandmarksType.TWO_D/g'

echo "== $(date -u +%H:%M:%S) weights (Google Drive folder -> test/pretrained_models)"
mkdir -p "$REPO/test/pretrained_models"
cd "$REPO/test/pretrained_models"
"$PY" -m gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_FOLDER" -O . || \
  echo "   WARN: gdown folder download failed or was partial; see listing below"
echo "-- pretrained_models tree:"
find . -maxdepth 3 | head -60
du -sh . 2>/dev/null | tail -1

echo "== $(date -u +%H:%M:%S) smoke import"
cd "$REPO/test"
"$PY" - <<'EOF'
import torch, diffusers, transformers, basicsr, insightface
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)
EOF

echo "== $(date -u +%H:%M:%S) done"
df -h /workspace | tail -1
