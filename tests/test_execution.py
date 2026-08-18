"""
Exercise execution provider verification.

The case this exists for: onnxruntime silently running every ONNX model on CPU
when CUDA was asked for. That is not a degraded product, it is a paid GPU hour
producing seconds-per-frame output, so it must fail rather than continue.
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

from pipeline.config import FaceSwapConfig
from pipeline.services import execution

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print(f'  [{"PASS" if condition else "FAIL"}] {label}' + (f' - {detail}' if detail else ''))


def session(providers):
    """A stub holding a session that reports `providers`."""
    holder = MagicMock()
    holder._session.get_providers.return_value = list(providers)
    del holder.get_providers          # force the nested lookup path
    return holder


def with_available(providers):
    execution.available_providers = lambda: list(providers)


CUDA, CPU = 'CUDAExecutionProvider', 'CPUExecutionProvider'

print('=' * 70)
print('Execution provider verification')
print('=' * 70)

# -- Session discovery --------------------------------------------------
print('\nSession discovery')
direct = MagicMock()
direct.get_providers.return_value = [CUDA, CPU]
check('finds a session that is itself the session',
      execution.session_providers(direct) == [CUDA, CPU])
check('finds a session held privately by a service',
      execution.session_providers(session([CUDA])) == [CUDA])

analysis = MagicMock()
del analysis.get_providers
del analysis._session
del analysis.session
analysis.models = {'detection': session([CUDA, CPU])}
check('finds a session inside a FaceAnalysis model dict',
      execution.session_providers(analysis) == [CUDA, CPU])

check('an unloaded service reports None, not CPU',
      execution.session_providers(None) is None,
      'not loaded must not be confused with fell back')

# -- The failure this exists for ----------------------------------------
print('\nCUDA requested but unavailable')
config = FaceSwapConfig()
config.execution_providers = [CUDA, CPU]
with_available([CPU])

raised = False
try:
    execution.verify(config, [('swap', session([CPU]))])
except execution.ExecutionProviderError as exc:
    raised = True
    message = str(exc)
check('raises rather than continuing on CPU', raised)
check('the error names the likely cause', raised and 'libcudnn.so.9' in message,
      message[:80] if raised else '')
check('the error says what it costs', raised and 'seconds per frame' in message)

summary = execution.verify(config, [('swap', session([CPU]))], strict=False)
check('strict=False still reports the failure', summary['ok'] is False)

# -- Available but not actually used ------------------------------------
print('\nCUDA available but a model fell back anyway')
with_available([CUDA, CPU])
raised = False
try:
    execution.verify(config, [
        ('swap', session([CUDA, CPU])),
        ('occluder', session([CPU])),
    ])
except execution.ExecutionProviderError as exc:
    raised = True
    message = str(exc)
check('a per-model fallback is caught, not just a missing provider', raised)
check('the error names which model fell back',
      raised and 'occluder' in message and 'swap' not in message,
      message[:90] if raised else '')

# -- The healthy case ---------------------------------------------------
print('\nEverything on CUDA')
summary = execution.verify(config, [
    ('swap', session([CUDA, CPU])),
    ('occluder', session([CUDA, CPU])),
])
check('passes when every model is on CUDA', summary['ok'] is True)
check('the summary records what was actually used',
      summary['actual']['swap'] == [CUDA, CPU], str(summary['actual']))

# -- Explicit CPU is a supported choice ---------------------------------
print('\nCPU requested explicitly')
cpu_config = FaceSwapConfig()
cpu_config.execution_providers = [CPU]
with_available([CPU])
summary = execution.verify(cpu_config, [('swap', session([CPU]))])
check('asking for CPU does not raise', summary['ok'] is True,
      '--execution-provider cpu is the supported escape hatch')

# -- Models not loaded yet ----------------------------------------------
print('\nModels not loaded')
with_available([CUDA, CPU])
summary = execution.verify(config, [('swap', None), ('occluder', None)])
check('unloaded models do not trigger a false failure', summary['ok'] is True)

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
    per-check output is the diagnostic. This function is what makes the same
    file a pytest test without duplicating any of it.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
