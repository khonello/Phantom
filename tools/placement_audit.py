#!/usr/bin/env python3
"""
Which nodes of each ONNX model does the CUDA provider hand to the CPU?

    python tools/placement_audit.py [--models /workspace/models]

For every `.onnx` under the directory, build a CUDA session the way the
pipeline does and read ORT's own placement log: how many nodes landed on the
CPU provider, and which op types. A session that reports CUDA can still run
part of its graph on the CPU with a device round trip on either side —
that was XSeg's decoder for the life of the project, 18ms of every frame —
and neither `execution.verify` nor the provider list can see it. This can.

Zero CPU nodes is the answer wanted for every live model. Anything else
names the op type to go and look at.
"""

import argparse
import os
import re
import sys
import tempfile
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FaceSwapConfig                        # noqa: E402
from pipeline.core import decode_execution_providers              # noqa: E402


def _placements(path: str) -> Tuple[int, int, Counter]:
    """(cpu nodes, cuda nodes, Counter of CPU op types) for one model."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.log_severity_level = 0
    options.log_verbosity_level = 1
    with tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False) as capture:
        saved = os.dup(2)
        os.dup2(capture.fileno(), 2)
        try:
            ort.InferenceSession(path, options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception as e:
            os.dup2(saved, 2)
            os.close(saved)
            print('   could not load: {}'.format(str(e)[:120]))
            return -1, -1, Counter()
        finally:
            try:
                os.dup2(saved, 2)
                os.close(saved)
            except OSError:
                pass
        capture.flush()
        capture.seek(0)
        lines = capture.read().splitlines()

    cpu = cuda = 0
    ops: Counter = Counter()
    for index, line in enumerate(lines):
        m = re.search(r'Node\(s\) placed on \[(\w+)\]\. Number of nodes: (\d+)', line)
        if not m:
            continue
        count = int(m.group(2))
        if m.group(1) == 'CPUExecutionProvider':
            cpu += count
            for follow in lines[index + 1:index + 1 + count]:
                text = follow.split('] ', 1)[-1].strip()
                op = text.split(' ', 1)[0]
                if op and not text.startswith('Node'):
                    ops[op] += 1
        elif m.group(1) == 'CUDAExecutionProvider':
            cuda += count
    return cpu, cuda, ops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--models', default='/workspace/models')
    args = parser.parse_args()

    # Bring CUDA up the way the pipeline does — through the detector — so the
    # audit measures the real thing rather than a CPU-only fallback.
    config = FaceSwapConfig()
    config.execution_providers = decode_execution_providers(['cuda'])
    from pipeline.services.face_detection import FaceDetector      # noqa: E402
    FaceDetector(config).detect_one(np.zeros((64, 64, 3), np.uint8))

    files: List[str] = []
    for root, _, names in os.walk(args.models):
        for name in sorted(names):
            if name.endswith('.onnx'):
                files.append(os.path.join(root, name))
    home = os.path.expanduser('~/.insightface/models')
    if os.path.isdir(home):
        for root, _, names in os.walk(home):
            for name in sorted(names):
                if name.endswith('.onnx'):
                    files.append(os.path.join(root, name))

    print('{:<48} {:>6} {:>6}  {}'.format('model', 'cpu', 'cuda', 'cpu op types'))
    print('-' * 90)
    worst: Dict[str, int] = {}
    for path in files:
        label = os.path.relpath(path, os.path.dirname(args.models))[:48]
        cpu, cuda, ops = _placements(path)
        if cpu < 0:
            continue
        flag = '' if cpu == 0 else '   <-- look here'
        print('{:<48} {:>6} {:>6}  {}{}'.format(
            label, cpu, cuda, ', '.join('{} x{}'.format(k, v) for k, v in ops.most_common()), flag))
        if cpu:
            worst[label] = cpu
    print()
    if worst:
        print('models with CPU-placed nodes:', ', '.join(sorted(worst)))
        return 1
    print('every model is entirely on CUDA')
    return 0


if __name__ == '__main__':
    sys.exit(main())
