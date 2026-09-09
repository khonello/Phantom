"""
Exercise the studio swap backends — the ones that cannot run on a call.

Nothing here runs a model: these are tens of gigabytes of weights behind three
separate checkouts, and none of it exists on a CI box. What *is* testable is
everything that decides whether the right process is started with the right
arguments, and everything that decides what happens when it is not — which is
where the damage lives:

- **A backend on the live path would emit nothing.** The fastest is ~0.6s per
  image against a 50ms deadline, so a stream would present a frozen frame while
  the connection, the virtual camera and the badges all read healthy. The gate
  is the only thing between a set variable and that call.
- **A failure must leave no output file.** `guards.py` establishes that a photo
  which cannot be swapped writes nothing, because a copy of the input wearing
  the output's name is indistinguishable from success. A subprocess that
  crashes, times out or writes nowhere has to obey the same rule.
- **A silent fallback would be the wrong model.** An unavailable or
  media-mismatched backend must refuse rather than quietly handing the job back
  to the ONNX path, which would produce a result from a model nobody chose.
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
import os
import shutil
import tempfile

from pipeline.config import FaceSwapConfig
from pipeline.services import studio_swappers
from pipeline.processing import studio

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}'.format('PASS' if condition else 'FAIL', label)
          + (' - {}'.format(detail) if detail else ''))


def env(**values):
    """Set environment variables, returning the previous values."""
    previous = {k: os.environ.get(k) for k in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def restore(previous):
    """Put back what `env` replaced."""
    env(**previous)


print('=' * 70)
print('Studio swap backends')
print('=' * 70)

# ── Registry ───────────────────────────────────────────────────────────
print('\nRegistry')

check('all three backends are registered',
      set(studio_swappers.names()) == {'reface', 'ghost_2', 'dreamid_v'},
      ', '.join(studio_swappers.names()))

try:
    studio_swappers.resolve('not_a_backend')
    _raised = False
except KeyError:
    _raised = True
check('an unknown name raises rather than falling back',
      _raised,
      'there is no incumbent to fall back to: a different backend is a '
      'different architecture, licence and media type')

check('no backend claims to be live safe',
      not any(studio_swappers.resolve(n).is_live_safe()
              for n in studio_swappers.names()),
      'this is what the stream gate reads')

check('exactly one backend is video-native',
      [studio_swappers.resolve(n).media
       for n in studio_swappers.names()].count('video') == 1)

check('the head-swapping backends are the image ones',
      all(studio_swappers.resolve(n).swaps_head
          for n in ('reface', 'ghost_2'))
      and not studio_swappers.resolve('dreamid_v').swaps_head,
      'head swap is the only property that raises the identity ceiling '
      'rather than moving along it')

check('every backend names its own interpreter variable',
      len({studio_swappers.resolve(n).python_env
           for n in studio_swappers.names()}) == 3,
      'one environment cannot satisfy all three')

# ── Selection ──────────────────────────────────────────────────────────
print('\nSelection')

config = FaceSwapConfig()
check('the default is off, so nothing changes for a live session',
      config.studio_swapper == '')
check('nothing is resolved when it is off',
      studio.resolve_backend(config) is None)

config.studio_swapper = 'reface'
backend = studio.resolve_backend(config)
check('a set name resolves to its backend',
      isinstance(backend, studio.RefaceSwapper))

config.studio_swapper = 'ghost_2'
check('ghost_2 resolves to its own class',
      isinstance(studio.resolve_backend(config), studio.Ghost2Swapper))

config.studio_swapper = 'dreamid_v'
check('dreamid_v resolves to its own class',
      isinstance(studio.resolve_backend(config), studio.DreamIdVSwapper))

config.studio_swapper = 'typo_here'
check('a typo degrades to the ordinary path rather than failing the job',
      studio.resolve_backend(config) is None,
      'reported, not raised — the compositing path still works')

# ── Availability reports the missing thing by name ─────────────────────
print('\nAvailability')

config.studio_swapper = 'reface'
_previous = env(REFACE_REPO=None, REFACE_PYTHON=None)
backend = studio.resolve_backend(config)
ready, reason = backend.available()
check('an unconfigured backend is unavailable',
      not ready)
check('and the reason names the variable that would fix it',
      'REFACE_REPO' in reason,
      reason)

_work = tempfile.mkdtemp(prefix='phantom-test-')
try:
    env(REFACE_REPO=_work)
    ready, reason = backend.available()
    check('a checkout without the entrypoint is refused',
          not ready and 'one_inference.py' in reason,
          reason)

    os.makedirs(os.path.join(_work, 'scripts'), exist_ok=True)
    with open(os.path.join(_work, 'scripts', 'one_inference.py'), 'w') as fh:
        fh.write('')
    ready, reason = backend.available()
    check('a checkout with no interpreter set is refused',
          not ready and 'REFACE_PYTHON' in reason,
          reason)

    check('the interpreter is never inherited from this process',
          backend.interpreter() == '',
          'ours is the environment that cannot satisfy these requirements')

    # ── Nothing produced means nothing written ─────────────────────────
    print('\nA failure writes no file')

    destination = os.path.join(_work, 'result.png')
    check('an unavailable backend reports failure',
          not backend.swap(
              os.path.join(_work, 'src.jpg'),
              os.path.join(_work, 'tgt.jpg'),
              destination))
    check('and leaves no output behind',
          not os.path.exists(destination),
          'an unswapped copy named like a result is worse than no file')

    check('_deliver refuses to invent a file from nothing',
          not studio.StudioSwapper._deliver(None, destination)
          and not studio.StudioSwapper._deliver(
              os.path.join(_work, 'absent.png'), destination))
    check('and still leaves no output behind',
          not os.path.exists(destination))

    # ── Media kind is refused, not attempted ───────────────────────────
    print('\nMedia kind')

    config.studio_swapper = 'dreamid_v'
    env(DREAMIDV_REPO=_work, DREAMIDV_PYTHON=sys.executable)
    video_backend = studio.resolve_backend(config)
    check('the video backend refuses a still',
          not video_backend._media_matches('photo.jpg'))
    check('and accepts a clip',
          video_backend._media_matches('clip.mp4'))

    config.studio_swapper = 'reface'
    image_backend = studio.resolve_backend(config)
    check('an image backend refuses a clip',
          not image_backend._media_matches('clip.mp4'))
    check('and accepts a still',
          image_backend._media_matches('photo.png'))

    # ── The command lines match each project's own documentation ───────
    print('\nCommand lines')

    env(REFACE_PYTHON=sys.executable)
    # Real files: REFace stages them into one-image folders, so `_argv` copies.
    _src = os.path.join(_work, 'src.jpg')
    _tgt = os.path.join(_work, 'tgt.jpg')
    for _p in (_src, _tgt):
        with open(_p, 'wb') as fh:
            fh.write(b'stub')

    argv = image_backend._argv(_src, _tgt, 'out.png', _work)
    check('REFace is driven by folders, as its own Demo.sh does',
          '--src_folder' in argv and '--target_folder' in argv
          and '--outdir' in argv and '--Base_dir' in argv)
    check('REFace carries its config and checkpoint',
          '--config' in argv and '--ckpt' in argv)
    check('REFace defaults to the published sampling settings',
          argv[argv.index('--ddim_steps') + 1] == '50'
          and argv[argv.index('--scale') + 1] == '3.5')
    check('the staged source and target are real files for the child',
          os.path.isfile(os.path.join(_work, 'source', 'src.jpg'))
          and os.path.isfile(os.path.join(_work, 'target', 'tgt.jpg')),
          'copied rather than symlinked: the child may run as another user')

    env(REFACE_DDIM_STEPS='5')
    argv = image_backend._argv(_src, _tgt, 'out.png', _work)
    check('the step count is the documented lever and it is reachable',
          argv[argv.index('--ddim_steps') + 1] == '5',
          '5 steps against 50 is ~4.7s against under a second')
    env(REFACE_DDIM_STEPS=None)

    config.studio_swapper = 'ghost_2'
    env(GHOST2_REPO=_work, GHOST2_PYTHON=sys.executable, GHOST2_USE_KANDI=None)
    ghost = studio.resolve_backend(config)
    argv = ghost._argv(_src, _tgt, 'out.png', _work)
    check('GHOST takes source, target and destination directly',
          '--source' in argv and '--target' in argv and '--save_path' in argv)
    check('GHOST writes into scratch, not straight to the destination',
          argv[argv.index('--save_path') + 1].startswith(_work),
          'a failed run must not leave a partial file where a finished one goes')
    check('GHOST loads both checkpoints',
          '--ckpt_a' in argv and '--ckpt_b' in argv)
    check('the Kandinsky post-blend is off unless asked for',
          '--use_kandi' not in argv,
          'a second diffusion model on an already slow path')

    env(GHOST2_USE_KANDI='true')
    check('and is reachable when asked for',
          '--use_kandi' in ghost._argv(_src, _tgt, 'o.png', _work))
    env(GHOST2_USE_KANDI=None)

    config.studio_swapper = 'dreamid_v'
    env(DREAMIDV_PYTHON=sys.executable, DREAMIDV_WAN_DIR=_work,
        DREAMIDV_CKPT=os.path.join(_work, 'dreamidv_faster.pth'))
    dream = studio.resolve_backend(config)
    argv = dream._argv(_src, os.path.join(_work, 'clip.mp4'),
                       'out.mp4', _work)
    check('DreamID-V takes a reference image and a reference video',
          '--ref_image' in argv and '--ref_video' in argv
          and '--save_file' in argv)
    check('DreamID-V carries both checkpoint trees',
          '--ckpt_dir' in argv and '--dreamidv_ckpt' in argv,
          'its own weights plus the Wan 2.1 VAE and text encoder')
    check('DreamID-V defaults to the README sampling settings',
          argv[argv.index('--sample_steps') + 1] == '16'
          and argv[argv.index('--size') + 1] == '832*480')

    ready, reason = dream.available()
    check('a checkout missing the entrypoint is refused first',
          not ready and 'generate_dreamidv_faster.py' in reason,
          reason)

    with open(os.path.join(_work, 'generate_dreamidv_faster.py'), 'w') as fh:
        fh.write('')
    ready, reason = dream.available()
    check('and then the missing checkpoint is named specifically',
          not ready and 'DREAMIDV_CKPT' in reason,
          reason)

    env(DREAMIDV_WAN_DIR=None)
    ready, reason = dream.available()
    check('the Wan backbone is named separately from its own weights',
          not ready and 'DREAMIDV_WAN_DIR' in reason,
          'it supplies the VAE and text encoder, and is a second download')
    env(DREAMIDV_WAN_DIR=_work)

    # ── Every argv starts with the backend's own interpreter ───────────
    check('every backend runs its own interpreter, never ours implicitly',
          argv[0] == sys.executable and argv[1] == dream.model.entrypoint,
          'ours is set here only because the test set it')
finally:
    restore(_previous)
    env(REFACE_REPO=None, REFACE_PYTHON=None, GHOST2_REPO=None,
        GHOST2_PYTHON=None, DREAMIDV_REPO=None, DREAMIDV_PYTHON=None,
        DREAMIDV_WAN_DIR=None, DREAMIDV_CKPT=None)
    shutil.rmtree(_work, ignore_errors=True)

# ── The live path refuses them ─────────────────────────────────────────
print('\nThe live gate')

with open(_os.path.join(_REPO_ROOT, 'pipeline', 'processing',
                        'pipeline.py'), encoding='utf-8') as fh:
    _pipeline_src = fh.read()

_stream_impl = _pipeline_src.split('def _run_stream_impl')[1].split('def ')[0]
check('the stream clears every model for live BEFORE warming them',
      '_clear_for_live()' in _stream_impl
      and _stream_impl.index('_clear_for_live()')
      < _stream_impl.index('_warm_up_models'),
      'a pod must not bill for a session that cannot work')

_gate = _pipeline_src.split('def _clear_for_live')[1].split('\n    def ')[0]
check('the gate reads the tier system rather than a per-registry constant',
      'tiers.require_live' in _gate,
      'a model must not be cleared by editing a speed comment')
check('all three tiers are consulted: offline, trained, and the swap model',
      ('resolve_backend' in _gate
       and 'identity_model' in _gate
       and 'swapper_models.resolve' in _gate),
      'a model could otherwise reach a call via the registry nobody checked')
check('a trained model is refused for a MISSING ARTIFACT, not for being slow',
      'require_trained_artifact' in _gate,
      'it is the right tool that does not exist yet — a different fix')
check('every refusal is an error, not a warning',
      _gate.count('emit_error') >= 3
      and 'emit_warning' not in _gate,
      'the desktop reads a failed start from whether an error arrived')
check('the gate returns False rather than raising',
      'return False' in _gate and 'raise' not in _gate,
      'a refused session must end cleanly, not as a crash')

_image_batch = _pipeline_src.split('def _process_image_batch')[1].split(
    '\n    def ')[0]
check('the image path delegates before it reads the file',
      'resolve_backend' in _image_batch
      and _image_batch.index('resolve_backend')
      < _image_batch.index('cv2.imread'),
      'the backend does its own decoding, detection and alignment')
check('a refused offline job skips rather than falling through',
      'PhotoResult.skipped' in _image_batch.split('resolve_backend')[1][:600],
      'falling through would produce a result from a model nobody chose')

_video_batch = _pipeline_src.split('def _process_video_batch')[1].split(
    '\n    def ')[0]
check('the video path delegates before extracting frames',
      'resolve_backend' in _video_batch
      and _video_batch.index('resolve_backend')
      < _video_batch.index('extract_frames'))


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
