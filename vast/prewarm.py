"""
Pre-warm every model the pipeline needs, so none of them download or
initialise on the first frame of a paid session.

Run by vast/startup.sh. Each model is independent: one failing must not
stop the others, because a missing optional model degrades the pipeline
while a missing required one is worth knowing about before a customer waits.

Previously this warmed only the detector and called `Enhancer()` with no
arguments, which raises TypeError and was swallowed by a bare except — so
restoration has never actually been pre-warmed, and CodeFormer downloaded
on the first frame instead. The swapper was never warmed at all, which
matters more now: hyperswap is a 384 MB download.
"""

import os
import sys
from typing import Callable, List

sys.path.insert(0, '.')

from pipeline.config import CONFIG  # noqa: E402

# Apply the configured model's realism profile before anything loads, so the
# warm-up exercises the same configuration the session will run.
CONFIG.apply_model_profile()

# And the same execution provider. This script set none, so it inherited the
# config default of CPU and built four CPU sessions — warming the file cache
# while proving nothing about whether CUDA works, and printing
# "CPUExecutionProvider" in the deploy log for every model, which reads exactly
# like the silent fallback this project exits non-zero to prevent.
#
# EXECUTION_PROVIDER matches what startup.sh passes the pipeline itself, so the
# two cannot disagree about what is being warmed.
_PROVIDER = os.environ.get('EXECUTION_PROVIDER', 'cuda').strip().lower()
if _PROVIDER and _PROVIDER != 'cpu':
    CONFIG.set('execution_providers', [
        '{}ExecutionProvider'.format(_PROVIDER.upper() if _PROVIDER in ('cuda', 'rocm')
                                     else _PROVIDER.capitalize()),
        'CPUExecutionProvider',
    ])

FAILED: List[str] = []


def warm(label: str, fn: Callable[[], None], required: bool = True) -> None:
    """Load one model, reporting rather than raising."""
    try:
        fn()
        print('  ok       {}'.format(label))
    except Exception as exc:
        mark = 'REQUIRED' if required else 'optional'
        print('  {:<8} {} - {}: {}'.format(mark, label, type(exc).__name__, exc))
        if required:
            FAILED.append(label)


def _detector() -> None:
    from pipeline.services.face_detection import FaceDetector
    FaceDetector(CONFIG)._get_analyser()


def _swapper() -> None:
    from pipeline.services.face_swapping import FaceSwapper

    swapper = FaceSwapper(CONFIG)
    model = swapper.model()
    if model.kind == 'inswapper':
        if not swapper.pre_check():
            raise RuntimeError('inswapper weights unavailable')
        swapper._get_swapper()
    else:
        # Downloads on first call. 384 MB for hyperswap, which is exactly the
        # kind of wait that should not land on a customer.
        if swapper._get_session(model) is None:
            raise RuntimeError('{} weights unavailable'.format(model.name))
    print('           model: {} ({}px native)'.format(model.name, model.size))


def _enhancer() -> None:
    from pipeline.services.enhancement import Enhancer

    enhancer = Enhancer(CONFIG)
    enhancer.load()
    if not enhancer.available:
        raise RuntimeError('restoration backend unavailable')


def _masker() -> None:
    from pipeline.services.masking import FaceMasker
    if FaceMasker(CONFIG)._get_session() is None:
        raise RuntimeError('occluder unavailable')


print('Pre-warming models (swapper: {})'.format(CONFIG.swapper_model))
warm('detection', _detector)
warm('swap', _swapper)
# Both degrade gracefully at runtime, so neither is fatal here — but a slow
# first frame is still worth avoiding.
warm('restoration', _enhancer, required=False)
warm('occluder', _masker, required=False)

if FAILED:
    print('Pre-warm incomplete: {}'.format(', '.join(FAILED)))
    sys.exit(1)

print('All required models ready.')
