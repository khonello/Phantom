#!/usr/bin/env python3
"""
Where the `mask` stage's time goes, on the machine that runs it.

    python tools/mask_profile.py [--execution-provider cuda]

The stage measured 17-26ms on the pod while the CPU work around XSeg profiles
at ~2ms on a laptop, so the time is in the inference or in how it is run.
This prints, for the XSeg session the masker actually builds:

    provider      what the session reports it is running on — the truth,
                  rather than the log line at load
    threads       how many CPU cores ORT is spreading its CPU-side work over,
                  and how many the host has
    raw run       session.run on the real input, warm, p50/p95
    bound run     the IOBinding path, if the masker got one
    build         FaceMasker.build end to end, as the compositor calls it

Read `raw run` first. Under ~3ms and the stage's cost is elsewhere; 15ms+ on
a modern card is a session that is not really on the GPU, or one whose
CPU-side work is starved by everything else on a host with few cores.
"""

import argparse
import os
import sys
import time
from typing import Callable, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FaceSwapConfig                        # noqa: E402
from pipeline.core import (                                        # noqa: E402
    decode_execution_providers,
    suggest_default_execution_providers,
)
from pipeline.processing.geometry import (                        # noqa: E402
    alignment_template,
    estimate_similarity,
)
from pipeline.services import swapper_models                      # noqa: E402
from pipeline.services.face_detection import FaceDetector         # noqa: E402
from pipeline.services.masking import FaceMasker                  # noqa: E402


def _timed(fn: Callable[[], object], runs: int = 60) -> Tuple[float, float]:
    """p50 and p95 milliseconds over `runs`, after three warm calls."""
    for _ in range(3):
        fn()
    samples: List[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 95))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--execution-provider', nargs='+',
                        default=suggest_default_execution_providers())
    parser.add_argument('--image', default='target/face-3.jpeg',
                        help='a frame with a face in it')
    args = parser.parse_args()

    config = FaceSwapConfig()
    config.execution_providers = decode_execution_providers(args.execution_provider)

    detector = FaceDetector(config)
    masker = FaceMasker(config, detector)

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit('unreadable image: {}'.format(args.image))
    frame = cv2.resize(frame, (640, 360)) if frame.shape[1] > 640 else frame
    detection = detector.detect_one(frame)
    if detection is None:
        raise SystemExit('no face in {}'.format(args.image))
    face = detection.face

    size = 256
    template = alignment_template(swapper_models.resolve(config.swapper_model).template)
    matrix = estimate_similarity(np.asarray(face.kps, dtype=np.float64), template * size)
    if matrix is None:
        raise SystemExit('could not fit the alignment')
    aligned = cv2.warpAffine(frame, matrix, (size, size))

    print('host cores: {}'.format(os.cpu_count()))

    # Force the occluder to load, the way the first frame does.
    session = masker._get_session()
    if session is None:
        print('XSeg session: NOT AVAILABLE (the mask stage is hull-only here)')
        return 1

    try:
        providers = session.get_providers()
    except Exception:
        providers = ['?']
    print('XSeg providers (in use):', providers)
    try:
        options = session.get_session_options()
        print('ORT intra-op threads: {}  inter-op: {}'.format(
            options.intra_op_num_threads, options.inter_op_num_threads))
    except Exception:
        pass
    print('bound runner:', 'yes' if masker._runner is not None else 'no (plain session.run)')
    print('input: {} {}x{}'.format(
        'NCHW' if masker._input_nchw else 'NHWC', masker._input_size, masker._input_size))

    blob = cv2.resize(aligned, (masker._input_size, masker._input_size)).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)
    if masker._input_nchw:
        blob = blob.transpose(0, 3, 1, 2)
    inputs = {masker._input_name: blob}

    print()
    p50, p95 = _timed(lambda: session.run(None, inputs))
    print('  raw session.run       p50 {:6.2f}ms  p95 {:6.2f}ms'.format(p50, p95))
    if masker._runner is not None:
        runner = masker._runner
        p50, p95 = _timed(lambda: runner.run(inputs))
        print('  bound runner.run      p50 {:6.2f}ms  p95 {:6.2f}ms'.format(p50, p95))
    p50, p95 = _timed(lambda: masker._occlusion_mask(aligned))
    print('  _occlusion_mask       p50 {:6.2f}ms  p95 {:6.2f}ms   (pre + run + post)'.format(p50, p95))
    p50, p95 = _timed(lambda: masker.build(
        face, matrix, aligned, frame.shape[:2], swapped=aligned, template=template))
    print('  FaceMasker.build      p50 {:6.2f}ms  p95 {:6.2f}ms   (the whole stage bar the warps)'.format(p50, p95))

    config.occluder = False
    p50, p95 = _timed(lambda: masker.build(
        face, matrix, aligned, frame.shape[:2], swapped=aligned, template=template))
    print('  build, occluder OFF   p50 {:6.2f}ms  p95 {:6.2f}ms   (hull, valid, erode, feather only)'.format(p50, p95))
    return 0


if __name__ == '__main__':
    sys.exit(main())
