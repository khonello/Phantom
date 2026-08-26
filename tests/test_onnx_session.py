"""
Exercise the shared ONNX session factory.

Four speed levers hook one moment — session construction — and they interact:
fp16 changes which file loads, TensorRT changes which provider sees the graph
first, CUDA graphs are only legal on static shapes, and the engine cache is only
correct if its key covers everything an engine is invalid across.

None of that can be tested against a real GPU here (the suite runs without
onnxruntime at all), so what these check is the decision-making: which weights,
which providers, in what order, keyed how. The inference itself is covered by a
session on a pod, not by this.

The case this exists for is the bug it was written alongside: a `SessionOptions`
built, configured, and never passed to the model. Nothing owned session
construction, so nothing noticed.
"""

import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from unittest.mock import MagicMock


class StubModule(MagicMock):
    __path__: list = []


for name in ('insightface', 'insightface.app', 'insightface.app.common',
             'insightface.model_zoo', 'insightface.utils',
             'insightface.utils.face_align', 'onnxruntime', 'torch',
             'torchvision', 'psutil', 'tensorflow', 'opennsfw2', 'gfpgan', 'onnx'):
    sys.modules.setdefault(name, StubModule())

import logging
import shutil
import tempfile

from pipeline.config import FaceSwapConfig
from pipeline.services import onnx_session

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label: str, condition: bool, detail: str = '') -> None:
    """Record one assertion, printing it as it runs."""
    if condition:
        PASS.append(label)
        print('  ok   {}'.format(label))
    else:
        FAIL.append(label)
        print('  FAIL {}{}'.format(label, ' — ' + detail if detail else ''))


def cuda_config(**overrides: object) -> FaceSwapConfig:
    """A config asking for CUDA, with the speed levers off unless overridden."""
    config = FaceSwapConfig()
    config.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


TEMP = tempfile.mkdtemp(prefix='phantom-onnx-')

# ── fp16 weight resolution ────────────────────────────────────────────
print('\nfp16 weight resolution')

model = _os.path.join(TEMP, 'codeformer.onnx')
open(model, 'wb').write(b'fp32 weights')

config = cuda_config()
check('fp32 by default', onnx_session.resolve_weights(config, model) == model)

config = cuda_config(fp16=True)
check('fp16 requested but absent falls back to fp32',
      onnx_session.resolve_weights(config, model) == model)

half = _os.path.join(TEMP, 'codeformer-fp16.onnx')
open(half, 'wb').write(b'fp16 weights')
check('fp16 copy is preferred once it exists',
      onnx_session.resolve_weights(config, model) == half)

config = cuda_config(fp16=False)
check('fp16 copy is ignored when not asked for',
      onnx_session.resolve_weights(config, model) == model)

# ── Which GPUs are worth an engine build ──────────────────────────────
print('\nTensorRT GPU eligibility')

config = cuda_config(trt_gpus='')
check('an empty list means never', onnx_session.trt_enabled_for_gpu(config) is False)

_real_identity = onnx_session.gpu_identity


def fake_identity(name: str):
    """Pin `gpu_identity` to a known device for the checks below."""
    def identity() -> str:
        return name
    return identity


onnx_session.gpu_identity = fake_identity('NVIDIA-GeForce-RTX-4090')
config = cuda_config(trt_gpus='RTX 4090,H100')
check('a listed GPU is eligible', onnx_session.trt_enabled_for_gpu(config) is True)

config = cuda_config(trt_gpus='H100,L40S')
check('an unlisted GPU is not', onnx_session.trt_enabled_for_gpu(config) is False)

onnx_session.gpu_identity = fake_identity('Tesla-V100-SXM2-16GB')
config = cuda_config(trt_gpus='RTX 4090,H100')
check('a slow GPU is not worth the build time',
      onnx_session.trt_enabled_for_gpu(config) is False)

# ── Provider assembly ─────────────────────────────────────────────────
print('\nProvider assembly')

onnx_session.gpu_identity = fake_identity('NVIDIA-GeForce-RTX-4090')


def provider_names(providers: list) -> list:
    """Strip options, leaving just the ordered provider names."""
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def options_for(providers: list, name: str) -> dict:
    """The options dict attached to `name`, or empty."""
    for entry in providers:
        if isinstance(entry, tuple) and entry[0] == name:
            return entry[1]
    return {}


config = cuda_config()
built = onnx_session.build_providers(config, model, static_shapes=True)
check('defaults leave the requested list alone',
      provider_names(built) == ['CUDAExecutionProvider', 'CPUExecutionProvider'],
      str(provider_names(built)))
check('no CUDA graph unless asked for',
      'enable_cuda_graph' not in options_for(built, 'CUDAExecutionProvider'))

config = cuda_config(cuda_graphs=True)
built = onnx_session.build_providers(config, model, static_shapes=True)
check('CUDA graph is enabled on a static-shape model',
      options_for(built, 'CUDAExecutionProvider').get('enable_cuda_graph') == 1)

built = onnx_session.build_providers(config, model, static_shapes=False)
check('CUDA graph is refused on a dynamic-shape model',
      'enable_cuda_graph' not in options_for(built, 'CUDAExecutionProvider'),
      'a captured graph records fixed buffer addresses')

config = cuda_config(trt=True, trt_gpus='RTX 4090')
built = onnx_session.build_providers(config, model, static_shapes=True)
check('TensorRT goes first, or it is never offered the graph',
      provider_names(built)[0] == 'TensorrtExecutionProvider',
      str(provider_names(built)))
check('the engine cache is enabled',
      options_for(built, 'TensorrtExecutionProvider').get('trt_engine_cache_enable') is True)
check('the timing cache is enabled too',
      options_for(built, 'TensorrtExecutionProvider').get('trt_timing_cache_enable') is True)

config = cuda_config(trt=True, trt_gpus='H100')
built = onnx_session.build_providers(config, model, static_shapes=True)
check('TensorRT is skipped on an ineligible GPU',
      'TensorrtExecutionProvider' not in provider_names(built),
      str(provider_names(built)))

config = cuda_config(trt=True, trt_gpus='RTX 4090', fp16=True)
built = onnx_session.build_providers(config, model, static_shapes=True)
check('fp16 reaches the TensorRT options',
      options_for(built, 'TensorrtExecutionProvider').get('trt_fp16_enable') is True)

# ── Engine cache keying ───────────────────────────────────────────────
print('\nEngine cache keying')

config = cuda_config()
fp32_dir = onnx_session.engine_cache_dir(config, model)
config = cuda_config(fp16=True)
fp16_dir = onnx_session.engine_cache_dir(config, model)
check('precision changes the cache directory', fp32_dir != fp16_dir)

onnx_session.gpu_identity = fake_identity('NVIDIA-H100-PCIe')
other_gpu_dir = onnx_session.engine_cache_dir(cuda_config(), model)
check('a different GPU gets a different cache directory',
      other_gpu_dir != fp32_dir,
      'an engine built for one architecture is invalid on another')

onnx_session.gpu_identity = fake_identity('NVIDIA-GeForce-RTX-4090')
second = _os.path.join(TEMP, 'inswapper_128.onnx')
open(second, 'wb').write(b'different weights entirely')
check('different weights get a different cache directory',
      onnx_session.engine_cache_dir(cuda_config(), second) != fp32_dir)

check('the same model on the same GPU is stable across calls',
      onnx_session.engine_cache_dir(cuda_config(), model) == fp32_dir,
      'otherwise every session rebuilds and the cache never warms')

# ── IOBinding falls back rather than failing ──────────────────────────
print('\nBoundRunner')


class _Meta:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class _Session:
    """Minimal stand-in exposing only what BoundRunner touches."""

    def __init__(self, shape: list, binding_works: bool = True) -> None:
        self._shape = shape
        self._binding_works = binding_works
        self.plain_runs = 0
        self.bound_runs = 0

    def get_outputs(self) -> list:
        return [_Meta('out', self._shape)]

    def run(self, _outputs: object, _inputs: object) -> list:
        self.plain_runs += 1
        return ['plain']

    def io_binding(self) -> object:
        if not self._binding_works:
            raise RuntimeError('no binding on this build')
        return MagicMock()

    def run_with_iobinding(self, _binding: object) -> None:
        self.bound_runs += 1


session = _Session([1, 3, 8, 8])
runner = onnx_session.BoundRunner(session, 'test')
runner.run({'in': None})
check('a static output shape is bound', session.bound_runs == 1 and session.plain_runs == 0)

runner.run({'in': None})
check('the binding is reused rather than rebuilt', session.bound_runs == 2)

session = _Session([1, 3, 'height', 'width'])
runner = onnx_session.BoundRunner(session, 'test')
runner.run({'in': None})
check('a symbolic output shape falls back to a plain run',
      session.plain_runs == 1 and session.bound_runs == 0,
      'preallocation needs a shape known ahead of time')

session = _Session([1, 3, 8, 8], binding_works=False)
runner = onnx_session.BoundRunner(session, 'test')
runner.run({'in': None})
runner.run({'in': None})
check('a build without IOBinding degrades quietly and stays degraded',
      session.plain_runs == 2,
      'this is a performance path; a warning per frame would cost more than it saves')

onnx_session.gpu_identity = _real_identity
shutil.rmtree(TEMP, ignore_errors=True)

print('\n' + '=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
for f in FAIL:
    print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """
    Surface the checks above to pytest as one assertion.

    The bodies run at import: these are scripts first, so they stay runnable
    directly (`python tests/test_x.py`) when a failure needs poking at, and the
    per-check output is the diagnostic.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
