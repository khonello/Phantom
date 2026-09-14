#!/usr/bin/env bash
# Route B measurement â€” run RefSTAR on one swapped still, timed.
#
#   routeb_refstar_run.sh <swapped-frame.png> <reference-photo.jpg> <out-dir>
#
# The swapped frame is the pipeline's output with restoration OFF (RefSTAR is a
# candidate to REPLACE the restorer, so it gets the unrestored composite); the
# reference is one source photograph. RefSTAR matches reference to input by
# filename stem, so both are copied under one name.
set -euo pipefail

LQ=${1:?swapped frame}
REF=${2:?reference photograph}
OUT=${3:?output dir}
PY=/workspace/venv-refstar/bin/python
WORK=/workspace/routeb/work

rm -rf "$WORK"
mkdir -p "$WORK/lq" "$WORK/ref" "$OUT"
cp "$LQ" "$WORK/lq/swap.png"
cp "$REF" "$WORK/ref/swap.jpg"

cd /workspace/RefSTAR/test
echo "== imports"
"$PY" - <<'EOF'
import torch, diffusers, transformers, basicsr, insightface
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)
EOF

echo "== refstar (full-frame mode: detect, restore at 512, paste back)"
START=$(date +%s.%N)
"$PY" test.py --testsets "$WORK/lq" --testsets_ref "$WORK/ref" \
  --output_path "$OUT" --detect_faces True 2>&1 | grep -v -i warning | tail -30
END=$(date +%s.%N)
echo "== wall time (includes model load): $(echo "$END - $START" | bc) s"
ls -la "$OUT"
df -h /workspace | tail -1
