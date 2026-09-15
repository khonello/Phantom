#!/usr/bin/env python3
"""
Verify the XSeg ConvTranspose rewrite where the pipeline runs.

    python tools/xseg_cuda_check.py [--model /workspace/models/dfl_xseg.onnx]

Rewrites the graph to `<name>-cuda.onnx`, then answers the three questions
that decide whether the pipeline may load it:

    exact       max |original - rewritten| on a real input, both graphs run
                CPU-only so the comparison is arithmetic, not provider
    placed      how many nodes the CUDA session puts on the CPU — must be 0
    time        raw session.run, warm, original vs rewritten, on CUDA
"""

import argparse
import os
import sys
import tempfile
import time
from typing import Callable, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FaceSwapConfig                        # noqa: E402  (cuDNN path setup)
from pipeline.core import decode_execution_providers              # noqa: E402
from pipeline.services import graph_fixes                         # noqa: E402

import onnxruntime as ort                                         # noqa: E402


def _timed(fn: Callable[[], object], runs: int = 40) -> Tuple[float, float]:
    for _ in range(3):
        fn()
    samples: List[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 95))


def _placed_on_cpu(path: str) -> int:
    options = ort.SessionOptions()
    options.log_severity_level = 0
    options.log_verbosity_level = 1
    with tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False) as capture:
        saved = os.dup(2)
        os.dup2(capture.fileno(), 2)
        try:
            ort.InferenceSession(path, options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        capture.flush()
        capture.seek(0)
        log = capture.read()
    count = 0
    for line in log.splitlines():
        if '[CPUExecutionProvider]' in line and 'placed on' in line:
            try:
                count += int(line.rsplit(':', 1)[-1])
            except ValueError:
                pass
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', default='/workspace/models/dfl_xseg.onnx')
    parser.add_argument('--image', default='target/face-3.jpeg')
    args = parser.parse_args()

    # The pipeline's own load order, so CUDA initialises the way it does for
    # a real session: the detector comes up first and brings cuDNN into the
    # process with it. A bare `import onnxruntime` here left every node on
    # the CPU and timed nothing useful.
    config = FaceSwapConfig()
    config.execution_providers = decode_execution_providers(['cuda'])
    from pipeline.services.face_detection import FaceDetector      # noqa: E402
    FaceDetector(config).detect_one(np.zeros((64, 64, 3), np.uint8))

    rewritten_path = graph_fixes.cuda_sibling(args.model)
    if os.path.isfile(rewritten_path):
        os.remove(rewritten_path)
    count = graph_fixes.symmetrise_convtranspose(args.model, rewritten_path)
    print('rewritten nodes:', count)
    if not count:
        return 1

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit('unreadable image')
    crop = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    blob = np.expand_dims(crop, 0)

    original_cpu = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    rewritten_cpu = ort.InferenceSession(rewritten_path, providers=['CPUExecutionProvider'])
    name = original_cpu.get_inputs()[0].name
    a = original_cpu.run(None, {name: blob})[0]
    b = rewritten_cpu.run(None, {name: blob})[0]
    print('output shapes:', a.shape, b.shape)
    print('exact: max |diff| = {:.3e}   mean |diff| = {:.3e}'.format(
        float(np.abs(a - b).max()), float(np.abs(a - b).mean())))

    print('nodes on CPU — original: {}   rewritten: {}'.format(
        _placed_on_cpu(args.model), _placed_on_cpu(rewritten_path)))

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    original_cuda = ort.InferenceSession(args.model, providers=providers)
    rewritten_cuda = ort.InferenceSession(rewritten_path, providers=providers)
    p50, p95 = _timed(lambda: original_cuda.run(None, {name: blob}))
    print('time on CUDA — original:  p50 {:6.2f}ms  p95 {:6.2f}ms'.format(p50, p95))
    p50, p95 = _timed(lambda: rewritten_cuda.run(None, {name: blob}))
    print('time on CUDA — rewritten: p50 {:6.2f}ms  p95 {:6.2f}ms'.format(p50, p95))
    c = rewritten_cuda.run(None, {name: blob})[0]
    print('CUDA rewritten vs CPU original: max |diff| = {:.3e}'.format(float(np.abs(a - c).max())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
