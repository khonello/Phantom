"""
Exercise the swap-model registry and the hyperswap inference path.

Two classes of bug this guards against, both of which fail *silently* rather
than raising:

- **The embedding-name trap.** facefusion calls the normalised 512-d vector
  `embedding_norm`; InsightFace uses that name for a *scalar* magnitude and
  calls the vector `normed_embedding`. Feeding the scalar produces garbage
  output, not an error.
- **Profile drift.** The realism knobs are tuned per model. Switching model
  without switching profile makes a better model look worse, which invites
  exactly the wrong conclusion.
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


for _name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(_name, StubModule())

import logging

import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.services import swapper_models
from pipeline.services.face_swapping import FaceSwapper, _ARCFACE_TEMPLATE

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}'.format('PASS' if condition else 'FAIL', label)
          + (' - {}'.format(detail) if detail else ''))


print('=' * 70)
print('Swap model registry and hyperswap path')
print('=' * 70)

# ── Registry ───────────────────────────────────────────────────────────
print('\nRegistry')
check('the incumbent is the default',
      swapper_models.DEFAULT_SWAPPER_MODEL == 'inswapper_128')
check('all three hyperswap variants are registered',
      all('hyperswap_1{}_256'.format(v) in swapper_models.SWAPPER_MODELS
          for v in 'abc'))

inswapper = swapper_models.resolve('inswapper_128')
hyperswap = swapper_models.resolve('hyperswap_1a_256')

check('an unknown name falls back rather than raising',
      swapper_models.resolve('does_not_exist').name == 'inswapper_128',
      'a typo in .env must not take down a paid session')

check('both use the same alignment template',
      inswapper.template == hyperswap.template == 'arcface_128',
      'this is why the compositor, masker and guards need no change')
check('both take an embedding source',
      inswapper.kind == 'inswapper' and hyperswap.kind == 'hyperswap',
      'image-source families are deliberately absent')

check('image-source models are not registered',
      not any(m.kind in ('uniface', 'blendswap')
              for m in swapper_models.SWAPPER_MODELS.values()),
      'they would break multi-photo embedding averaging')

check('hyperswap is 256 native, inswapper 128',
      hyperswap.size == 256 and inswapper.size == 128)
check('the model URL is pinned to the tag that serves it',
      'models-3.3.0' in hyperswap.url,
      'models-3.0.0 and 3.4.0 both 404 for these files')
check('the incumbent has no URL (resolved locally)', inswapper.url == '')

# ── Alignment template ─────────────────────────────────────────────────
print('\nAlignment template')
# InsightFace builds a 128px arcface crop as arcface_dst + [8, 0].
arcface_dst = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float64)
expected = (arcface_dst + np.array([8.0, 0.0])) / 128.0
check('template matches InsightFace arcface_128 exactly',
      np.allclose(_ARCFACE_TEMPLATE, expected, atol=1e-6),
      'max delta {:.2e}'.format(float(np.abs(_ARCFACE_TEMPLATE - expected).max())))
check('template is normalised, so one constant serves 128 and 256',
      _ARCFACE_TEMPLATE.max() < 1.0 and _ARCFACE_TEMPLATE.min() > 0.0)

# ── Look profiles ──────────────────────────────────────────────────────
print('\nLook profiles')
check('a 256 model asks for less restoration',
      hyperswap.enhance_strength < inswapper.enhance_strength,
      '{} vs {} - less invented detail to blend in'.format(
          hyperswap.enhance_strength, inswapper.enhance_strength))
check('a 256 model stays closer to its own output',
      hyperswap.enhancer_weight > inswapper.enhancer_weight,
      '{} vs {} - fidelity, 1 = closest to input'.format(
          hyperswap.enhancer_weight, inswapper.enhancer_weight))
check('the compositing floor rises to the model native size',
      hyperswap.aligned_min == 256 and inswapper.aligned_min == 128,
      'compositing below native would discard model output')

config = FaceSwapConfig()
check('default config starts on the incumbent',
      config.swapper_model == 'inswapper_128')

config.apply_model_profile('hyperswap_1a_256')
check('applying a profile moves the knobs',
      config.enhance_strength == 0.5 and config.enhancer_weight == 0.8
      and config.aligned_min == 256,
      'strength={} weight={} floor={}'.format(
          config.enhance_strength, config.enhancer_weight, config.aligned_min))
check('applying a profile records the model',
      config.swapper_model == 'hyperswap_1a_256')

config.apply_model_profile('inswapper_128')
check('switching back restores the incumbent profile',
      config.enhance_strength == 0.7 and config.aligned_min == 128)

# ── Presets no longer own appearance ───────────────────────────────────
print('\nPreset / profile separation')
from pipeline.api.schema import PRESETS  # noqa: E402

owns_look = [name for name, preset in PRESETS.items()
             if 'enhance_strength' in preset or 'enhancer_weight' in preset]
check('no preset sets appearance knobs any more', not owns_look,
      'the model owns those; the preset owns compute')
check('presets still own the compute knobs',
      all('aligned_size' in p and 'det_size' in p for p in PRESETS.values()))

fresh = FaceSwapConfig()
fresh.apply_preset('production')
fresh.apply_model_profile('hyperswap_1a_256')
check('profile applies after preset without the two fighting',
      fresh.aligned_size == 320 and fresh.aligned_min == 256,
      'ceiling {} from preset, floor {} from model'.format(
          fresh.aligned_size, fresh.aligned_min))

# ── Compositing floor is honoured ──────────────────────────────────────
print('\nCompositing floor')
from pipeline.processing.compositor import FaceCompositor  # noqa: E402

compositor = FaceCompositor(fresh, MagicMock(), MagicMock())
# A tiny face: scale large => small extent => would previously pick 128.
small_face = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]], dtype=np.float32)
chosen = compositor._aligned_size(small_face, 256)
check('a distant face is not composited below the model native size',
      chosen >= 256, 'chose {} with floor {}'.format(chosen, fresh.aligned_min))

legacy = FaceSwapConfig()
legacy.apply_preset('optimal')
legacy.apply_model_profile('inswapper_128')
legacy_compositor = FaceCompositor(legacy, MagicMock(), MagicMock())
check('a 128 model may still drop to 128 for a distant face',
      legacy_compositor._aligned_size(small_face, 128) == 128,
      'the floor is per-model, not a blanket raise')

# ── The embedding-name trap ────────────────────────────────────────────
print('\nHyperswap inference')


def fake_session(size=256):
    """A session echoing a normalised-looking output of the right shape."""
    session = MagicMock()
    session.get_inputs.return_value = [
        MagicMock(name='s'), MagicMock(name='t'),
    ]
    session.get_inputs.return_value[0].name = 'source'
    session.get_inputs.return_value[1].name = 'target'
    captured = {}

    def run(_outputs, feeds):
        captured.update(feeds)
        return [np.full((1, 3, size, size), 0.25, dtype=np.float32)]

    session.run.side_effect = run
    session._captured = captured
    return session


cfg = FaceSwapConfig()
cfg.apply_model_profile('hyperswap_1a_256')
swapper = FaceSwapper(cfg)
session = fake_session()
swapper._session = session
swapper._session_model = 'hyperswap_1a_256'
swapper._source_input = 'source'
swapper._target_input = 'target'

rng = np.random.default_rng(5)
vector = rng.normal(size=512).astype(np.float32)
vector /= np.linalg.norm(vector)

source = MagicMock()
source.normed_embedding = vector
source.embedding_norm = 27.4        # InsightFace: a SCALAR. Must not be used.

target = MagicMock()
target.kps = np.array([
    [40.0, 50.0], [80.0, 50.0], [60.0, 70.0], [45.0, 90.0], [75.0, 90.0],
], dtype=np.float32)

frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
result = swapper._swap_session(hyperswap, source, target, frame)

check('the aligned swap returns a crop and an affine', result is not None)
if result is not None:
    crop, matrix = result
    check('crop is at the model native size',
          crop.shape[:2] == (256, 256), str(crop.shape))
    check('crop is uint8 image data', crop.dtype == np.uint8)
    check('affine is a 2x3 frame-to-crop matrix', matrix.shape == (2, 3))

    fed = session._captured['source']
    check('the NORMALISED VECTOR is fed, not the scalar norm',
          fed.shape == (1, 512),
          'shape {} - feeding embedding_norm would be a scalar'.format(fed.shape))
    check('the fed vector is the source embedding unchanged',
          np.allclose(fed.ravel(), vector, atol=1e-6),
          'hyperswap takes it directly, with no emap projection')

    blob = session._captured['target']
    check('target blob is NCHW float32', blob.shape == (1, 3, 256, 256)
          and blob.dtype == np.float32, str(blob.shape))
    check('target blob is normalised by the model mean/std',
          -1.05 <= float(blob.min()) and float(blob.max()) <= 1.05,
          'range [{:.2f}, {:.2f}] for mean 0.5 std 0.5'.format(
              float(blob.min()), float(blob.max())))

    # Output 0.25 with mean .5 / std .5 -> 0.25*0.5+0.5 = 0.625 -> ~159
    check('output is denormalised back through mean/std',
          abs(int(crop.mean()) - 159) <= 2,
          'mean {} - expected ~159'.format(int(crop.mean())))

no_embedding = MagicMock()
no_embedding.normed_embedding = None
check('a source without an embedding is refused, not guessed at',
      swapper._swap_session(hyperswap, no_embedding, target, frame) is None)

bad_kps = MagicMock()
bad_kps.kps = np.array([[1.0, 2.0]], dtype=np.float32)
check('a face without five keypoints is refused',
      swapper._swap_session(hyperswap, source, bad_kps, frame) is None)

# ── Routing ────────────────────────────────────────────────────────────
print('\nRouting')
check('the configured model reaches the swapper',
      FaceSwapper(cfg).model().name == 'hyperswap_1a_256')

incumbent_cfg = FaceSwapConfig()
check('the incumbent still routes to InsightFace',
      FaceSwapper(incumbent_cfg).model().kind == 'inswapper')

swapper.clear()
check('clear() drops both model caches',
      swapper._session is None and swapper._session_model == '')


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
