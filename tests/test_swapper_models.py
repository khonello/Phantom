"""
Exercise the swap-model registry and the alphaface inference path.

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
from pipeline.processing import geometry
from pipeline.services import swapper_models
from pipeline.services.face_swapping import FaceSwapper, _ARCFACE_TEMPLATE

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}'.format('PASS' if condition else 'FAIL', label)
          + (' - {}'.format(detail) if detail else ''))


print('=' * 70)
print('Swap model registry and the session inference path')
print('=' * 70)

# ── Registry ───────────────────────────────────────────────────────────
print('\nRegistry')
check('the incumbent is the default',
      swapper_models.DEFAULT_SWAPPER_MODEL == 'inswapper_128')
check('the model that targeted the opposite objective is gone',
      not any('hyperswap' in n for n in swapper_models.names()),
      'tuned to respect the target, slower than the incumbent, and worst '
      'by eye — three 384 MB entries nobody should reach for')

inswapper = swapper_models.resolve('inswapper_128')
alphaface = swapper_models.resolve('alphaface_256')

check('an unknown name falls back rather than raising',
      swapper_models.resolve('does_not_exist').name == 'inswapper_128',
      'a typo in .env must not take down a paid session')

check('both use the same alignment template',
      inswapper.template == alphaface.template == 'arcface_128',
      'this is why the compositor, masker and guards need no change')
check('both take an embedding source',
      inswapper.source_kind == alphaface.source_kind == 'embedding')

# This used to assert the opposite — that image-source models were kept OUT
# because they "would break multi-photo embedding averaging". That was the
# pipeline's architecture choosing which models were permitted to exist, which
# is backwards: the source contract is a fact about the weights.
check('image-source models are registered rather than excluded',
      {m.kind for m in swapper_models.SWAPPER_MODELS.values()}
      >= {'uniface', 'blendswap'},
      'averaging is inapplicable to them, not broken by them')

check('alphaface is 256 native, inswapper 128',
      alphaface.size == 256 and inswapper.size == 128)
check('the model URL is pinned to the tag that serves it',
      'models-3.9.0' in alphaface.url,
      'the tag differs per model; an asset under one 404s under another')
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
      alphaface.enhance_strength < inswapper.enhance_strength,
      '{} vs {} - less invented detail to blend in'.format(
          alphaface.enhance_strength, inswapper.enhance_strength))
check('a 256 model stays closer to its own output',
      alphaface.enhancer_weight > inswapper.enhancer_weight,
      '{} vs {} - fidelity, 1 = closest to input'.format(
          alphaface.enhancer_weight, inswapper.enhancer_weight))
check('the compositing floor rises to the model native size',
      alphaface.aligned_min == 256 and inswapper.aligned_min == 128,
      'compositing below native would discard model output')

config = FaceSwapConfig()
check('default config starts on the incumbent',
      config.swapper_model == 'inswapper_128')

config.apply_model_profile('alphaface_256')
check('applying a profile moves the knobs',
      config.enhance_strength == 0.5 and config.enhancer_weight == 0.8
      and config.aligned_min == 256,
      'strength={} weight={} floor={}'.format(
          config.enhance_strength, config.enhancer_weight, config.aligned_min))
check('applying a profile records the model',
      config.swapper_model == 'alphaface_256')

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
fresh.apply_model_profile('alphaface_256')
# Read from the schema rather than written out. This asserted `== 320`, which
# was production's aligned ceiling until the preset was retuned for uplink
# rather than compute - so a preset change broke a test about model profiles,
# which is the wrong thing to be coupled to. What matters is the SOURCE of each
# number: the ceiling comes from the preset, the floor from the model.
check('profile applies after preset without the two fighting',
      fresh.aligned_size == PRESETS['production']['aligned_size']
      and fresh.aligned_min == 256,
      'ceiling {} from preset, floor {} from model'.format(
          fresh.aligned_size, fresh.aligned_min))

# The two really are independent, which equal numbers would not demonstrate.
other = FaceSwapConfig()
other.apply_preset('fast')
other.apply_model_profile('inswapper_128')
check('a different preset and model move them independently',
      other.aligned_size == PRESETS['fast']['aligned_size']
      and other.aligned_min == 128,
      'ceiling {} from preset, floor {} from model'.format(
          other.aligned_size, other.aligned_min))

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
cfg.apply_model_profile('alphaface_256')
swapper = FaceSwapper(cfg)
session = fake_session()
swapper._session = session
swapper._session_model = 'alphaface_256'
swapper._source_input = 'source'
swapper._target_input = 'target'

rng = np.random.default_rng(5)
vector = rng.normal(size=512).astype(np.float32)
vector /= np.linalg.norm(vector)

source = MagicMock()
source.normed_embedding = vector
# alphaface is conditioned on the RAW vector, so the magnitude is part of
# the signal and this stand-in has to carry a realistic one.
source.embedding = vector * 21.0
source.embedding_norm = 27.4        # InsightFace: a SCALAR. Must not be used.

target = MagicMock()
target.kps = np.array([
    [40.0, 50.0], [80.0, 50.0], [60.0, 70.0], [45.0, 90.0], [75.0, 90.0],
], dtype=np.float32)

frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
result = swapper._swap_session(alphaface, source, target, frame)

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
    check('the RAW vector is fed, at its own magnitude',
          np.allclose(fed.ravel(), vector * 21.0, atol=1e-4),
          'alphaface is conditioned on the unnormalised embedding; a unit '
          'vector here is a quieter identity and no error')

    blob = session._captured['target']
    check('target blob is NCHW float32', blob.shape == (1, 3, 256, 256)
          and blob.dtype == np.float32, str(blob.shape))
    check('target blob is in [0, 1] for mean 0 / deviation 1',
          -0.05 <= float(blob.min()) and float(blob.max()) <= 1.05,
          'range [{:.2f}, {:.2f}]'.format(
              float(blob.min()), float(blob.max())))

    # alphaface emits [0, 1] directly and is NOT denormalised, so a 0.25
    # output is 0.25*255 = ~64. Applying the inverse would give ~159 here,
    # and on a model with real mean/deviation it would tint the crop —
    # which is the whole reason `denormalize_output` is a registry field.
    check('output is NOT denormalised, as this export requires',
          abs(int(crop.mean()) - 64) <= 2,
          'mean {} - expected ~64; ~159 would mean the inverse ran'.format(
              int(crop.mean())))

no_embedding = MagicMock()
no_embedding.normed_embedding = None
check('a source without an embedding is refused, not guessed at',
      swapper._swap_session(alphaface, no_embedding, target, frame) is None)

bad_kps = MagicMock()
bad_kps.kps = np.array([[1.0, 2.0]], dtype=np.float32)
check('a face without five keypoints is refused',
      swapper._swap_session(alphaface, source, bad_kps, frame) is None)

# ── Routing ────────────────────────────────────────────────────────────
print('\nRouting')
check('the configured model reaches the swapper',
      FaceSwapper(cfg).model().name == 'alphaface_256')

incumbent_cfg = FaceSwapConfig()
check('the incumbent still routes to InsightFace',
      FaceSwapper(incumbent_cfg).model().kind == 'inswapper')

swapper.clear()
check('clear() drops both model caches',
      swapper._session is None and swapper._session_model == '')

# ── Per-family conventions, pinned against facefusion's reference ──────
#
# Every entry below is a fact about an export, and every one of them fails
# *silently* when wrong: a raw vector fed as a unit vector gives a blander
# identity, an un-needed denormalisation gives a tinted crop. Neither raises,
# so neither would be noticed while judging a model on footage — which is
# exactly when a wrong convention would be blamed on the model.
print('\nPer-family source and output conventions')

# (name, kind, template, size, source_form, normalise_source, denormalize,
#  has_converter)
_REFERENCE = (
    ('inswapper_128', 'inswapper', 'arcface_128', 128, 'normed', True, True, False),
    ('hififace_unofficial_256', 'hififace', 'mtcnn_512', 256, 'raw', True, True, True),
    ('alphaface_256', 'alphaface', 'arcface_128', 256, 'raw', False, False, False),
    ('ghost_1_256', 'ghost', 'arcface_112_v1', 256, 'raw', False, True, True),
    ('simswap_256', 'simswap', 'arcface_112_v1', 256, 'raw', True, False, True),
    ('simswap_unofficial_512', 'simswap', 'arcface_112_v1', 512, 'raw', True, False, True),
)

for (_name, _kind, _template, _size, _form, _norm, _denorm,
     _converted) in _REFERENCE:
    _m = swapper_models.resolve(_name)
    check('{} matches facefusion on every convention'.format(_name),
          (_m.name == _name and _m.kind == _kind and _m.template == _template
           and _m.size == _size and _m.source_form == _form
           and _m.normalise_source is _norm
           and _m.denormalize_output is _denorm
           and bool(_m.converter_filename) is _converted),
          'kind={} tmpl={} size={} form={} norm={} denorm={} conv={}'.format(
              _m.kind, _m.template, _m.size, _m.source_form,
              _m.normalise_source, _m.denormalize_output,
              _m.converter_filename or '-'))

# ── The source contract is per model, not per architecture ────────────
#
# `blendswap` and `uniface` used to be excluded on the grounds that an image
# source would break multi-photo averaging, .npy sources and the outlier
# guard. That was the pipeline deciding which models were allowed to exist.
# The guards run at upload and are untouched; averaging is inapplicable, not
# broken; and an .npy set is told once rather than swapping badly.
print()
print('The source contract is a fact about the weights')

for _name, _src_kind, _src_template, _src_size in (
    ('blendswap_256', 'image', 'arcface_112_v2', 112),
    ('uniface_256', 'image', 'ffhq_512', 256),
):
    _m = swapper_models.resolve(_name)
    check('{} is conditioned on a picture, in its own framing'.format(_name),
          (_m.source_kind == _src_kind
           and _m.source_template == _src_template
           and _m.source_size == _src_size),
          'src={} tmpl={} size={}'.format(
              _m.source_kind, _m.source_template, _m.source_size))

check('an image model wants a DIFFERENT space for source and target',
      swapper_models.resolve('blendswap_256').source_template
      != swapper_models.resolve('blendswap_256').template,
      'assuming one from the other degrades quietly rather than failing')

_uniface = swapper_models.resolve('uniface_256')
check('uniface happens to share one space between its two inputs',
      (_uniface.source_template == _uniface.template
       and _uniface.source_size == _uniface.size),
      'which is exactly why blendswap must be carried separately rather '
      'than generalised from this one')

check('every image model names a template that resolves',
      all(geometry.alignment_template(
          swapper_models.resolve(n).source_template) is not None
          for n in swapper_models.names()
          if swapper_models.resolve(n).source_kind == 'image'))

check('every embedding model declares no source framing',
      all(not swapper_models.resolve(n).source_template
          for n in swapper_models.names()
          if swapper_models.resolve(n).source_kind == 'embedding'),
      'an unused framing would be a claim nothing checks')

check('the incumbent is still an embedding model',
      swapper_models.resolve('inswapper_128').source_kind == 'embedding')

check('an unknown name still falls back rather than raising',
      swapper_models.resolve('not_a_model').name == 'inswapper_128')
check('every registered template resolves to a real one',
      all(geometry.alignment_template(swapper_models.resolve(n).template)
          is not None for n in swapper_models.names()))
check('simswap 256 and 512 disagree about input normalisation, as they do '
      'upstream',
      swapper_models.resolve('simswap_256').mean != (0.0, 0.0, 0.0)
      and swapper_models.resolve('simswap_unofficial_512').mean == (0.0, 0.0, 0.0),
      'assuming one convention for both would tint the 256')
check('the 512 model raises the compositing floor to its native size',
      swapper_models.resolve('simswap_unofficial_512').aligned_min == 512,
      'and the floor beats the preset ceiling, deliberately')
check('only the one ghost variant is registered',
      [n for n in swapper_models.names() if n.startswith('ghost')]
      == ['ghost_1_256'],
      'kept for its licence, not for identity; 2 and 3 were 1.5 GB of '
      'catalogue for a model nothing here rates')
check('and it does not collide with the STUDIO backend named ghost_2',
      'ghost_2' not in swapper_models.names())
check('every model declares a download size, so a truncation is visible',
      all(swapper_models.resolve(n).size_bytes > 0
          for n in swapper_models.names()
          if swapper_models.resolve(n).url))


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
