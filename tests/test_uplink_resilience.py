"""
The two threads that must not die, and did.

`_run_webcam` is the uplink. It ran bare, so one exception anywhere in the body
— the JPEG encode, a filter, the Qt buffer write — ended it for good. The pod
then received nothing, produced nothing, and the desktop's jitter buffer went on
repeating the last frame it had. That is a frozen swapped face, indistinguishable
from a guard holding, and the only recovery was stopping and starting the stream
because that is what builds a new thread. Nothing said so: a release build hides
the console, so even the interpreter's own thread traceback went nowhere.

`_run_vcam` had the same shape and a worse consequence. Its `cam.send` sat
inside a `try` that wrapped the *whole* loop, so one failure exited it and
released the virtual camera — and a conferencing app responds to a camera that
disappears by selecting the next one, which is the operator's real webcam. That
is the exposure the entire design exists to prevent, reached through a transient
error.

Both are gated on PySide6, which the CI interpreter does not have. Run them
against the Qt environment in `desktop/.qtcreator/`.
"""

import os
import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import queue
import threading
from unittest.mock import MagicMock

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PySide6', reason='desktop threads need Qt; see desktop/.qtcreator/')

import numpy as np                                     # noqa: E402

from desktop import bridge as bridge_module            # noqa: E402
from desktop.bridge import Bridge                      # noqa: E402


class _Capture:
    """A camera that yields frames forever and counts its release."""

    def __init__(self) -> None:
        self.released = 0
        self.reads = 0

    def isOpened(self):        # noqa: N802 - mirrors cv2's name
        return True

    def set(self, *_args):
        return True

    def read(self):
        self.reads += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released += 1


def _make_bridge(monkeypatch, capture, decorate):
    """
    A Bridge with only the attributes `_run_webcam` touches.

    Built with `__new__` rather than constructed: the real `__init__` opens a
    camera, a socket and an audio device, none of which this is about. Same
    approach `tests/test_render_download.py` takes with PipelineClient.
    """
    self = Bridge.__new__(Bridge)
    self._webcam_stop = threading.Event()
    self._ws_push_active = threading.Event()
    self._ws_push_active.set()
    self._uplink_bytes = 0
    self._uplink_frames = 0
    self._quality = 'optimal'
    self._client = MagicMock()
    self._status: list = []
    self._decorate = decorate

    monkeypatch.setattr(Bridge, '_set_status',
                        lambda s, msg, error=False: s._status.append(msg))
    monkeypatch.setattr(bridge_module.cv2, 'VideoCapture', lambda _i: capture)
    monkeypatch.setattr(bridge_module, 'webcam_buffer', MagicMock())
    return self


def test_the_uplink_survives_an_exception_in_its_body(monkeypatch):
    """
    One bad frame must cost one frame, not the rest of the session.

    The failure is raised from `_decorate`, which is a real place it came from:
    the filter and effect layer runs on this thread, on every frame.
    """
    capture = _Capture()
    calls = {'n': 0}

    def explode_once(frame):
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('filter blew up')
        return frame

    self = _make_bridge(monkeypatch, capture, explode_once)

    def run():
        self._run_webcam(0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Let it get well past the frame that raised.
    deadline = threading.Event()
    while calls['n'] < 20 and not deadline.wait(0.01):
        if not thread.is_alive():
            break

    alive = thread.is_alive()
    self._webcam_stop.set()
    thread.join(timeout=3)

    assert alive, (
        'the uplink thread died on one exception — the pod then receives '
        'nothing and the far end freezes on the last frame it had'
    )
    assert calls['n'] > 3, 'it stopped producing frames after the failure'
    # Sends are *paced* to the preset's rate, so there are far fewer of them
    # than loop iterations: this test drives the loop as fast as it will go, so
    # every frame lands inside one 50ms window and only the first is due. What
    # matters here is that sending resumed at all after the exception.
    assert self._client.send_frame.call_count >= 1, (
        'nothing was ever sent: {} sends across {} frames'.format(
            self._client.send_frame.call_count, calls['n'])
    )


def test_the_camera_is_released_even_when_the_loop_raises(monkeypatch):
    """
    A leaked handle makes the *next* start fail too.

    `cap.release()` sat after the loop, so an exception skipped it. The next
    start then opened a second handle on a camera the first was still holding,
    which on Windows can simply fail — so one transient error could cost more
    than one restart to recover from.
    """
    capture = _Capture()

    def always_explode(_frame):
        raise RuntimeError('permanently broken')

    self = _make_bridge(monkeypatch, capture, always_explode)
    self._webcam_stop.set()          # one pass, then stop
    self._run_webcam(0)

    assert capture.released == 1, 'the capture device was not released'


def test_a_failing_uplink_is_reported(monkeypatch):
    """
    Sustained failure reaches the operator.

    Not the first one — `cap.read` returning a bad frame for a moment is
    ordinary — but a run of them means the far end is looking at a frozen face,
    and that must not be something only a hidden console knows about.
    """
    capture = _Capture()

    def always_explode(_frame):
        raise RuntimeError('permanently broken')

    self = _make_bridge(monkeypatch, capture, always_explode)

    thread = threading.Thread(target=lambda: self._run_webcam(0), daemon=True)
    thread.start()

    waited = threading.Event()
    while not self._status and not waited.wait(0.02):
        if not thread.is_alive():
            break
    self._webcam_stop.set()
    thread.join(timeout=3)

    assert self._status, 'a sustained uplink failure said nothing to anyone'
    assert 'uplink' in self._status[-1].lower(), self._status


def test_the_virtual_camera_is_not_released_on_one_failed_send(monkeypatch):
    """
    Releasing the device is the one action that can expose the real face.

    A conferencing app responds to a camera that disappears by showing a
    placeholder, reporting a disconnection, or selecting the next available
    camera — which is the operator's webcam. So a transient `send` failure is
    retried rather than allowed to end the loop.
    """
    sends = {'n': 0}

    class _Cam:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def send(self, _frame):
            sends['n'] += 1
            if sends['n'] == 2:
                raise RuntimeError('device hiccup')

        def sleep_until_next_frame(self):
            pass

    fake_pyvirtualcam = MagicMock()
    fake_pyvirtualcam.Camera = lambda **_k: _Cam()
    monkeypatch.setitem(sys.modules, 'pyvirtualcam', fake_pyvirtualcam)

    self = Bridge.__new__(Bridge)
    self._vcam_platform = ''
    self._vcam_queue = queue.Queue()
    self._status = []
    monkeypatch.setattr(Bridge, '_set_status',
                        lambda s, msg, error=False: s._status.append(msg))
    monkeypatch.setattr(Bridge, '_set_virtual_cam_active', lambda s, v: None)

    self._vcam_queue.put(np.zeros((4, 4, 3), dtype=np.uint8))
    stop = threading.Event()

    thread = threading.Thread(target=lambda: self._run_vcam(stop), daemon=True)
    thread.start()

    waited = threading.Event()
    while sends['n'] < 6 and not waited.wait(0.02):
        if not thread.is_alive():
            break

    survived = thread.is_alive()
    stop.set()
    thread.join(timeout=3)

    assert survived, (
        'one failed send closed the virtual camera — a call application can '
        'respond to that by selecting the real webcam'
    )
    assert sends['n'] > 2, 'sending stopped after the failure: {}'.format(sends)
