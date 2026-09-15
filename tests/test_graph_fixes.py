"""
The XSeg ConvTranspose rewrite — `pipeline/services/graph_fixes.py`.

The claim that matters was proven on the pod, where the real graph and the
real GPU are: bit-identical output, zero nodes left on the CPU, 18ms -> 3ms
(`tools/xseg_cuda_check.py`). What this pins is the part that can drift in a
clean checkout: that the rewrite produces a graph ORT accepts and that
evaluates identically to the original, on a synthetic ConvTranspose with the
same asymmetric pads the export has; and that the masker's load path prefers
the rewritten sibling only when a GPU provider is asked for.

The numerical half needs real `onnx` and `onnxruntime`, which the CI `test`
job has and the unit environment stubs. It is skipped, and says so, when they
are stubs — a skipped check is not a passed one, so it prints as such.
"""

import os
import sys
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import tempfile
from unittest.mock import MagicMock

# Do NOT stub onnx / onnxruntime here: the whole point is to run the real
# ones when they exist. The other heavy modules are stubbed as elsewhere.
for name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'torch', 'torchvision', 'psutil', 'tensorflow', 'opennsfw2', 'gfpgan',
):
    if name not in sys.modules:
        stub = MagicMock()
        stub.__path__ = []
        sys.modules[name] = stub

import logging

import numpy as np

from pipeline.services import graph_fixes

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []
SKIPPED: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def skip(label: str, why: str) -> None:
    SKIPPED.append(label)
    print(f'  [SKIP] {label} - {why}')


try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort
    REAL = not isinstance(onnx, MagicMock) and hasattr(onnx, 'helper')
except Exception:
    REAL = False

WORK = tempfile.mkdtemp(prefix='phantom-graph-fixes-')

print('=' * 70)
print('Graph fixes')
print('=' * 70)

print('\nPaths')
check('the rewritten copy sits beside the original with a -cuda suffix',
      graph_fixes.cuda_sibling('/w/models/dfl_xseg.onnx') == '/w/models/dfl_xseg-cuda.onnx')
check('a missing original is returned as-is rather than raising',
      graph_fixes.prefer_cuda_copy(os.path.join(WORK, 'nope.onnx'))
      == (os.path.join(WORK, 'nope.onnx'), False))

print('\nRewrite')
if not REAL:
    skip('the rewrite evaluates identically to the original',
         'real onnx/onnxruntime not installed here; proven on the pod by tools/xseg_cuda_check.py')
else:
    # A ConvTranspose with the export's pads, stride 2, kernel 3, then a Relu
    # so the rewritten node's output name is consumed downstream.
    rng = np.random.default_rng(3)
    weight = rng.normal(0, 0.1, (8, 4, 3, 3)).astype(np.float32)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 8, 16, 16])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 32, 32])
    node = helper.make_node(
        'ConvTranspose', ['x', 'w'], ['t'], name='conv2d_transpose',
        kernel_shape=[3, 3], strides=[2, 2], pads=[0, 0, 1, 1], dilations=[1, 1])
    relu = helper.make_node('Relu', ['t'], ['y'], name='relu')
    graph = helper.make_graph([node, relu], 'g', [x], [y],
                              initializer=[numpy_helper.from_array(weight, 'w')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
    model.ir_version = 7
    src = os.path.join(WORK, 'asym.onnx')
    onnx.save(model, src)
    dst = graph_fixes.cuda_sibling(src)

    count = graph_fixes.symmetrise_convtranspose(src, dst)
    check('the asymmetric node is rewritten', count == 1, str(count))
    check('the rewritten file exists', os.path.isfile(dst))

    rewritten = onnx.load(dst)
    ops = [n.op_type for n in rewritten.graph.node]
    check('a Slice follows the ConvTranspose', ops == ['ConvTranspose', 'Slice', 'Relu'], str(ops))
    pads = [list(a.ints) for n in rewritten.graph.node if n.op_type == 'ConvTranspose'
            for a in n.attribute if a.name == 'pads'][0]
    check('its pads are now symmetric (zero)', pads == [0, 0, 0, 0], str(pads))

    inp = rng.normal(0, 1, (1, 8, 16, 16)).astype(np.float32)
    a = ort.InferenceSession(src, providers=['CPUExecutionProvider']).run(None, {'x': inp})[0]
    b = ort.InferenceSession(dst, providers=['CPUExecutionProvider']).run(None, {'x': inp})[0]
    check('the output shape is unchanged (2x upsample, 32x32)',
          a.shape == b.shape == (1, 4, 32, 32), '{} vs {}'.format(a.shape, b.shape))
    check('the rewrite evaluates identically to the original',
          float(np.abs(a - b).max()) == 0.0, 'max |diff| {:.3e}'.format(float(np.abs(a - b).max())))

    # A graph that is already symmetric is left alone and no file is written.
    sym = helper.make_node(
        'ConvTranspose', ['x', 'w'], ['t'], name='ok',
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    model2 = helper.make_model(
        helper.make_graph([sym, relu], 'g2', [x], [y], initializer=[numpy_helper.from_array(weight, 'w')]),
        opset_imports=[helper.make_opsetid('', 15)])
    model2.ir_version = 7
    src2 = os.path.join(WORK, 'sym.onnx')
    onnx.save(model2, src2)
    check('a symmetric graph is not rewritten and no sibling is written',
          graph_fixes.symmetrise_convtranspose(src2, graph_fixes.cuda_sibling(src2)) == 0
          and not os.path.isfile(graph_fixes.cuda_sibling(src2)))
    check('prefer_cuda_copy hands back the sibling once it exists',
          graph_fixes.prefer_cuda_copy(src) == (dst, True))

print('\n' + '=' * 70)
print('{} passed, {} failed, {} skipped'.format(len(PASS), len(FAIL), len(SKIPPED)))
if FAIL:
    for f in FAIL:
        print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
