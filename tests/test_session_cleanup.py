"""
Erasing a session: the operator's face, and everything made from it.

The pipeline runs on a rented pod that is handed to somebody else afterwards,
and the network volume survives a stop - so `/workspace/tmp/phantom/uploads`
outlives the customer who uploaded into it unless something removes it. What is
being proved here:

1. **One delete covers everything transient.** Sources, uploaded target photos
   and videos, a template job's output directory and the `_swapped` files
   written beside each target all live under one tree, deliberately, so the
   erase cannot miss a category that a later feature adds.
2. **Deleting the files is not enough.** The averaged embedding sits on the
   swapping processor and the per-image faces sit in the database cache;
   neither is on disk, and an `rmtree` does not touch either.
3. **A dropped socket is not a departure.** `PipelineClient` reconnects
   indefinitely by design, so the sweep runs on a grace period and a reconnect
   inside it calls it off. Wiping on the disconnect itself would delete the
   operator's face mid-call every time the link hiccuped.
"""

import os
import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import tempfile
import threading
import time
from unittest.mock import MagicMock


class StubModule(MagicMock):
    """MagicMock that also satisfies `from x.y import z` for nested paths."""

    __path__: list = []


for name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(name, StubModule())

import logging

WORK = tempfile.mkdtemp(prefix='phantom-cleanup-test-')
# Read by `get_temp_root`, which is what `_upload_dir` is built on. Set before
# the pipeline modules are imported so nothing has resolved a path yet.
os.environ['PHANTOM_TEMP_DIR'] = WORK

from pipeline.api import handlers                          # noqa: E402
from pipeline.api.server import WebSocketAPIServer         # noqa: E402
from pipeline.config import FaceSwapConfig                 # noqa: E402
from pipeline.events import EventBus                       # noqa: E402
from pipeline.processing.pipeline import ProcessingPipeline  # noqa: E402

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print('  [{}] {}'.format(mark, label) + (' - {}'.format(detail) if detail else ''))


def seed_uploads() -> dict:
    """One file of every shape the upload tree is known to hold."""
    uploads = handlers._upload_dir()
    paths = {
        'the source photo': os.path.join(uploads, 'face.jpg'),
        'an uploaded target photo': os.path.join(uploads, 'target.png'),
        'a photo swap output': os.path.join(uploads, 'target_swapped.png'),
    }

    template_job = os.path.join(uploads, 'template_abc')
    os.makedirs(template_job, exist_ok=True)
    paths['a template job output'] = os.path.join(template_job, 'scene_swapped.png')

    video_job = os.path.join(uploads, 'video_xyz')
    os.makedirs(video_job, exist_ok=True)
    paths['an uploaded target video'] = os.path.join(video_job, 'clip.mp4')

    for path in paths.values():
        with open(path, 'wb') as handle:
            handle.write(b'not really an image')
    return paths


class SweepOnly:
    """
    The server's sweep machinery with the socket layer left out.

    Borrows the real methods rather than reimplementing them, so this cannot
    drift into testing a copy of the logic instead of the logic.
    """

    _arm_session_sweep = WebSocketAPIServer._arm_session_sweep
    _cancel_session_sweep = WebSocketAPIServer._cancel_session_sweep

    def __init__(self, grace: float) -> None:
        self.config = FaceSwapConfig()
        self.pipeline = None
        self._clients: set = set()
        self._clients_lock = threading.Lock()
        self._session_grace = grace
        self._session_sweep = None
        self._sweep_lock = threading.Lock()
        self.swept = threading.Event()

    def _sweep_session(self) -> None:
        """`WebSocketAPIServer._sweep_session`, with the erase itself flagged."""
        with self._clients_lock:
            if self._clients:
                return
        with self._sweep_lock:
            self._session_sweep = None
        self.swept.set()


print('=' * 70)
print('Session cleanup')
print('=' * 70)

# ── Everything transient goes ─────────────────────────────────────────
print('\nThe upload tree')

config = FaceSwapConfig()
pipeline = ProcessingPipeline(config, EventBus())
written = seed_uploads()

config.source_path = written['the source photo']
config.source_paths = [written['the source photo']]
config.target_paths = [written['an uploaded target photo']]
config.target_face_points = [(0.5, 0.5)]
config.output_dir = os.path.dirname(written['a template job output'])

# Stand in for what a live session leaves in memory.
pipeline._swapping_proc = MagicMock()
pipeline._swapping_proc.source_face = object()
pipeline._database = MagicMock()
pipeline._photo_results = ['a result']

reply = handlers.handle_cleanup_session(config, pipeline)

check('the erase reports success', reply.success, str(reply.error))
for label, path in written.items():
    check('{} is gone'.format(label), not os.path.exists(path), path)

check('the source is unset',
      not config.source_path and not config.source_paths)
check('targets are unset',
      not config.target_paths and not config.target_path)
check('a face the operator picked is forgotten',
      not config.target_face_points,
      'it names a face in a photo that no longer exists')
check('the template job output directory is unset', config.output_dir is None)

# ── What a delete on disk cannot reach ────────────────────────────────
print('\nIn memory')

check('the averaged embedding is dropped',
      pipeline._swapping_proc.source_face is None,
      'an rmtree does not unload what was built from the files')
check('the per-image face cache is cleared',
      pipeline._database.clear.called)
check('photo results are dropped',
      pipeline._photo_results == [],
      'they name output paths that were just deleted')

# ── A blip is not a departure ─────────────────────────────────────────
print('\nThe grace period')

server = SweepOnly(grace=0.15)
server._arm_session_sweep()
check('nothing is erased immediately', not server.swept.is_set(),
      'a dropped socket is usually a live session, not a finished one')

server._clients.add(object())
server._cancel_session_sweep()
time.sleep(0.35)
check('a reconnect inside the grace period cancels the erase',
      not server.swept.is_set())

server._clients.clear()
server._arm_session_sweep()
check('the erase runs once the grace period passes with no client',
      server.swept.wait(timeout=2.0))

racing = SweepOnly(grace=0.05)
racing._arm_session_sweep()
racing._clients.add(object())
time.sleep(0.35)
check('a client present when the timer fires is not erased under',
      not racing.swept.is_set(),
      'the client set is re-checked in the sweep, not trusted from arming')

disabled = SweepOnly(grace=0.0)
disabled._arm_session_sweep()
check('a grace period of 0 disables the sweep',
      disabled._session_sweep is None,
      'for a pipeline that several clients come and go from')

print('=' * 70)
print('{} passed, {} failed'.format(len(PASS), len(FAIL)))
if FAIL:
    for failure in FAIL:
        print('  FAILED:', failure)
print('=' * 70)


def test_everything_passed() -> None:
    """
    Surface the checks above to pytest as one assertion.

    Same shape as the rest of the suite: the body runs at import so the file
    stays runnable directly when a failure needs poking at, and this function
    is what makes it a pytest test without duplicating any of it.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
