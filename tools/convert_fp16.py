#!/usr/bin/env python3
"""
Convert an ONNX model to half precision, beside the original.

fp16 is the largest single inference win available to this pipeline and the
only one that can change what the output looks like, so this writes a *copy* —
`codeformer.onnx` stays put and `codeformer-fp16.onnx` appears next to it. The
pipeline picks the copy up when `--fp16` is set and falls back to the original,
loudly, when it is absent. Reverting is a config change, not a 384 MB download.

**`keep_io_types=True` is deliberate and load-bearing.** The converter can
either expose fp16 inputs and outputs or wrap the graph in casts so callers
still hand it float32. We take the casts: `_CodeFormerBackend.restore` builds a
float32 NCHW blob and `FaceCompositor` expects float32 back, and a conversion
that silently changed those contracts would be a second edit in a second file
for every model converted. The casts cost a fraction of what the conversion
saves.

**The block list is not optional.** Reductions and normalisations accumulate in
the accumulator's precision, and in fp16 a `ReduceMean` over a 512x512 feature
map can saturate. Those ops stay in fp32; everything around them converts. A
model that produces NaN is not a faster model.

Usage:
    python tools/convert_fp16.py pipeline/models/codeformer.onnx
    python tools/convert_fp16.py /workspace/models/*.onnx
    python tools/convert_fp16.py --check pipeline/models/codeformer-fp16.onnx
"""

import argparse
import os
import sys
from typing import List, Optional

# Ops kept at fp32. Reductions and normalisations accumulate across a whole
# feature map, which is exactly where fp16's narrow exponent runs out; the
# range ops are here because a clamp bound that saturates changes behaviour
# rather than precision.
_BLOCK_LIST = [
    'ReduceMean',
    'ReduceSum',
    'ReduceL2',
    'InstanceNormalization',
    'LayerNormalization',
    'GroupNormalization',
    'Softmax',
    'Exp',
    'Pow',
    'Sqrt',
    'Range',
    'Min',
    'Max',
]

_FP16_SUFFIX = '-fp16.onnx'


def _output_path(model_path: str) -> str:
    """The `-fp16.onnx` path beside `model_path`."""
    base, _ = os.path.splitext(model_path)
    return base + _FP16_SUFFIX


def convert(model_path: str, force: bool = False) -> Optional[str]:
    """
    Convert one model to fp16.

    Args:
        model_path: Path to the fp32 .onnx file
        force: Overwrite an existing converted copy

    Returns:
        The path written, or None if nothing was written
    """
    if not os.path.isfile(model_path):
        print(f'  not found: {model_path}')
        return None

    if model_path.endswith(_FP16_SUFFIX):
        print(f'  already fp16, skipping: {os.path.basename(model_path)}')
        return None

    destination = _output_path(model_path)
    if os.path.isfile(destination) and not force:
        print(f'  exists, skipping: {os.path.basename(destination)} (use --force)')
        return None

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print(
            '  onnx and onnxconverter-common are required:\n'
            '    pip install onnx onnxconverter-common',
        )
        return None

    print(f'  loading {os.path.basename(model_path)}...')
    model = onnx.load(model_path)

    print('  converting...')
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=_BLOCK_LIST,
        disable_shape_infer=False,
    )

    onnx.save(converted, destination)

    before = os.path.getsize(model_path) / (1024 * 1024)
    after = os.path.getsize(destination) / (1024 * 1024)
    print(
        f'  wrote {os.path.basename(destination)} '
        f'({before:.0f} MB -> {after:.0f} MB)',
    )
    return destination


def check(model_path: str) -> bool:
    """
    Load a converted model and run one random input through it.

    A conversion that saves without error can still produce NaN on the first
    real call. This is the cheapest way to find that out here rather than in a
    paid session — it does not prove the output *looks* right, which is what
    `--debug-frames` and `tools/compare_frames.py` are for.

    Args:
        model_path: Path to the converted .onnx file

    Returns:
        True if it loaded and produced finite numbers
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print('  onnxruntime is required for --check')
        return False

    try:
        session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'],
        )
    except Exception as e:
        print(f'  FAILED to load: {type(e).__name__}: {e}')
        return False

    inputs = {}
    for meta in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        dtype = np.float16 if 'float16' in meta.type else np.float32
        if 'double' in meta.type:
            dtype = np.float64
        inputs[meta.name] = (np.random.rand(*shape) * 0.5).astype(dtype)

    try:
        outputs = session.run(None, inputs)
    except Exception as e:
        print(f'  FAILED to run: {type(e).__name__}: {e}')
        return False

    for index, output in enumerate(outputs):
        if not np.all(np.isfinite(np.asarray(output, dtype=np.float32))):
            print(f'  FAILED: output {index} contains NaN or Inf')
            return False

    print(f'  ok: {os.path.basename(model_path)} ran and produced finite output')
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert ONNX models to fp16, beside the original.',
    )
    parser.add_argument('models', nargs='+', help='.onnx files to convert')
    parser.add_argument(
        '--force', action='store_true',
        help='overwrite an existing -fp16.onnx',
    )
    parser.add_argument(
        '--check', action='store_true',
        help='load and smoke-test the given models instead of converting',
    )
    args = parser.parse_args(argv)

    failures = 0
    for model_path in args.models:
        print(model_path)
        if args.check:
            if not check(model_path):
                failures += 1
            continue

        written = convert(model_path, force=args.force)
        if written and not check(written):
            failures += 1

    if failures:
        print(f'\n{failures} model(s) failed. Do not ship these.')
        return 1

    print('\nDone. Enable with --fp16 (or FP16=1 in .env), then A/B on footage:')
    print('  python pipeline.py --stream --debug-frames fp32/')
    print('  python pipeline.py --stream --fp16 --debug-frames fp16/')
    print('  python tools/compare_frames.py fp16/ --against fp32/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
