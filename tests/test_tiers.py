"""
The three tiers, and the rule that a model cannot cross them by accident.

The distinction used to live in three implicit places — a speed comment in one
registry, a refusal in the stream loop, and a paragraph in CLAUDE.md — and an
implicit rule is one that drifts. What is pinned here is that the rule is now
*derived* and *enforced*, in both directions:

- **Derived.** No registry writes its tier down. Each declares two facts —
  `live_capable` and `needs_training` — and the tier follows. A model cannot be
  filed under the wrong constraint by editing a label.
- **Enforced.** Every registry answers `require_live` the same way, and the
  stream consults it before any model is warmed. There is no default that lets
  an undeclared model through.

The combination worth naming is TRAINED: **fast enough for a call, and unusable
until someone has trained it**. Collapsing the tiers into a speed ladder would
file it beside a diffusion model it has nothing in common with, and would hide
the only question that matters about it — not "how fast" but "trained on whom".
"""

import sys
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
import os
import shutil
import tempfile

from pipeline.config import FaceSwapConfig
from pipeline.services import (
    downloads, identity_models, studio_swappers, swapper_models, tiers,
)

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}'.format('PASS' if condition else 'FAIL', label)
          + (' - {}'.format(detail) if detail else ''))


print('=' * 70)
print('Tiers: live, studio, trained')
print('=' * 70)

# ── The taxonomy ───────────────────────────────────────────────────────
print()
print('The taxonomy is derived from two facts, not written down')

check('there are exactly three tiers',
      set(tiers.TIERS) == {'live', 'studio', 'trained'})

check('live capable and untrained is LIVE',
      tiers.classify(True, False) == tiers.LIVE)
check('not live capable is STUDIO',
      tiers.classify(False, False) == tiers.STUDIO)
check('needing training is TRAINED even though it is fast',
      tiers.classify(True, True) == tiers.TRAINED,
      'this is the pair a speed ladder would lose')

_raised = False
try:
    tiers.classify(False, True)
except ValueError:
    _raised = True
check('slow AND trained has no tier, and raises rather than inventing one',
      _raised,
      'it would be the worst of both; nothing here is that')

check('every tier describes its constraint, not its technology',
      all(tiers.describe(t).summary and tiers.describe(t).label
          for t in tiers.TIERS))
check('exactly one tier needs training',
      sum(tiers.describe(t).needs_training for t in tiers.TIERS) == 1)
check('exactly one tier is not live capable',
      sum(not tiers.describe(t).live_capable for t in tiers.TIERS) == 1)

# ── Every registry declares, and agrees ────────────────────────────────
print()
print('All three registries answer the same question about themselves')

check('every swap model is LIVE',
      all(swapper_models.resolve(n).tier == tiers.LIVE
          for n in swapper_models.names()),
      '{} models'.format(len(swapper_models.names())))

check('every studio backend is STUDIO',
      all(studio_swappers.resolve(n).tier == tiers.STUDIO
          for n in studio_swappers.names()))

check('a trained model declares TRAINED',
      identity_models.IdentityModel(name='x', path='x.dfm').tier
      == tiers.TRAINED)

check('a trained model is live capable, unlike a studio one',
      (identity_models.IdentityModel(name='x', path='x.dfm').live_capable
       and not studio_swappers.resolve('reface').live_capable),
      'the tiers are not a speed ranking')

check('is_live_safe now answers from the declared fact',
      all(studio_swappers.resolve(n).is_live_safe()
          is studio_swappers.resolve(n).live_capable
          for n in studio_swappers.names()),
      'a backend cannot be cleared for a call by editing that method')

# ── The gate refuses, and says which fix applies ───────────────────────
print()
print('require_live refuses, and the wording distinguishes the two failures')

check('a live model is cleared',
      tiers.require_live(tiers.LIVE, 'inswapper_128', 50.0) is None)

_studio_reason = tiers.require_live(tiers.STUDIO, 'reface', 50.0)
check('a studio model is refused', bool(_studio_reason))
check('and the refusal names the deadline it cannot hold',
      '50ms' in (_studio_reason or ''), _studio_reason or '')
check('and points at the jobs it IS for',
      'RENDER' in (_studio_reason or ''),
      'a refusal has to be actionable')

check('an undeclared tier is refused rather than allowed',
      bool(tiers.require_live('nonsense', 'mystery', 50.0)),
      'there must be no default that lets an unknown model onto a call')

check('a trained model is cleared for live once its artifact exists',
      tiers.require_live(tiers.TRAINED, 'someone', 50.0) is None)

_missing = tiers.require_trained_artifact(tiers.TRAINED, 'someone', False)
check('a trained model with no artifact is refused', bool(_missing))
check('and the refusal is about TRAINING, not about speed',
      'trained per person' in (_missing or '')
      and 'deadline' not in (_missing or ''),
      'the right tool that does not exist yet is a different fix')
check('a present artifact clears it',
      tiers.require_trained_artifact(tiers.TRAINED, 'someone', True) is None)
check('the artifact rule does not apply to untrained tiers',
      tiers.require_trained_artifact(tiers.LIVE, 'inswapper_128', False)
      is None)

# ── The trained registry is scanned, not declared ──────────────────────
print()
print('Trained models are a directory listing, not a registry')

config = FaceSwapConfig()
check('the trained tier is off by default',
      config.identity_model == '')
check('and goes all the way to the trained identity when on',
      config.identity_morph == 1.0)

_work = tempfile.mkdtemp(prefix='phantom-tiers-')
_previous = os.environ.get('IDENTITY_MODEL_DIR')
try:
    os.environ['IDENTITY_MODEL_DIR'] = _work
    check('an empty directory offers nothing',
          identity_models.names() == ())

    with open(os.path.join(_work, 'jane_doe_320.dfm'), 'wb') as fh:
        fh.write(b'stub')
    with open(os.path.join(_work, 'notes.txt'), 'w') as fh:
        fh.write('ignore me')

    check('a .dfm is found and named by its stem',
          identity_models.names() == ('jane_doe_320',),
          str(identity_models.names()))
    check('anything else in the directory is ignored',
          'notes' not in identity_models.names())
    check('it is reported available',
          identity_models.available('jane_doe_320'))
    check('and an unknown name is not',
          not identity_models.available('someone_else'))

    _raised = False
    try:
        identity_models.resolve('someone_else')
    except KeyError:
        _raised = True
    check('resolving an unknown trained model RAISES',
          _raised,
          'unlike a swap model, a different name here is a different person')

    check('the model aligns to the wider dfl_whole_face crop',
          identity_models.resolve('jane_doe_320').template == 'dfl_whole_face',
          'jaw and forehead, which is where the silhouette advantage is')

    # Scanned every call: these arrive by being copied in, with no restart
    # between training one and wanting it.
    with open(os.path.join(_work, 'second_256.dfm'), 'wb') as fh:
        fh.write(b'stub')
    check('a model added after the first scan is seen without a restart',
          set(identity_models.names()) == {'jane_doe_320', 'second_256'})
finally:
    if _previous is None:
        os.environ.pop('IDENTITY_MODEL_DIR', None)
    else:
        os.environ['IDENTITY_MODEL_DIR'] = _previous
    shutil.rmtree(_work, ignore_errors=True)

# ── Download policy ────────────────────────────────────────────────────
print()
print('Downloads are a stated policy rather than an unstated default')

_previous = os.environ.get('MODEL_DOWNLOADS')
try:
    for value in (None, '', 'selected'):
        if value is None:
            os.environ.pop('MODEL_DOWNLOADS', None)
        else:
            os.environ['MODEL_DOWNLOADS'] = value
        check('policy {!r} allows what the configuration selected'.format(value),
              downloads.allowed('alphaface_256')[0])
        check('policy {!r} refuses a speculative fetch'.format(value),
              not downloads.allowed('alphaface_256', selected=False)[0],
              'a pod fetches what it runs, not the catalogue')

    os.environ['MODEL_DOWNLOADS'] = 'none'
    _ok, _why = downloads.allowed('alphaface_256')
    check('none refuses even a selected model', not _ok)
    check('and the refusal names the variable that would allow it',
          'MODEL_DOWNLOADS' in _why, _why)

    os.environ['MODEL_DOWNLOADS'] = 'all'
    check('all allows a speculative fetch, for pre-seeding',
          downloads.allowed('anything_at_all', selected=False)[0])

    os.environ['MODEL_DOWNLOADS'] = 'alphaface_256, gpen_bfr_256'
    check('a list allows what it names',
          downloads.allowed('alphaface_256')[0])
    check('an extension is ignored when matching',
          downloads.allowed('alphaface_256.onnx')[0],
          'the same weight is named both ways across the codebase')
    check('and refuses what it does not name',
          not downloads.allowed('blendswap_256')[0],
          'the 1.6 GB one')
finally:
    if _previous is None:
        os.environ.pop('MODEL_DOWNLOADS', None)
    else:
        os.environ['MODEL_DOWNLOADS'] = _previous

check('every fetch site consults the policy',
      all('downloads.refuse_reason' in open(
          _os.path.join(_REPO_ROOT, *path), encoding='utf-8').read()
          for path in (
              ('pipeline', 'services', 'face_swapping.py'),
              ('pipeline', 'services', 'enhancement.py'),
              ('pipeline', 'services', 'masking.py'),
          )),
      'a site that skipped it would be an unstated exception')


print()
print('=' * 70)
print('{} passed, {} failed'.format(len(PASS), len(FAIL)))
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
