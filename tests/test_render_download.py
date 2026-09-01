"""
Getting a finished render back off the pod.

A render writes on the pipeline's filesystem. On a pod that is another machine,
so the file has to be read back or it does not exist as far as the operator is
concerned — and if the pod is later terminated, it stops existing at all. That
is what happened: the download ran on the WebSocket receive thread, could not
be answered, and reported success anyway.
"""

import threading

import pytest

from desktop.controller import PipelineClient


class _FakeWS:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)


@pytest.fixture
def client():
    c = PipelineClient.__new__(PipelineClient)
    c._ws = _FakeWS()
    c._ws_lock = threading.Lock()
    c._pending_lock = threading.Lock()
    c._response_events = {}
    c._response_data = {}
    c._recv_thread = None
    return c


# ── A failure must not read as a success ───────────────────────────────


def test_timeout_reply_is_marked_unsuccessful(client):
    """
    Callers read `reply.get('success', True)` — defaulting to True, because a
    handler that answers without the field has succeeded. So an error dict
    lacking the field was read as a success carrying no data, which is how a
    failed download reported "finished" and wrote nothing.
    """
    reply = client._send('get_output_info', _timeout=0.01)
    assert reply.get('success', True) is False
    assert 'error' in reply


def test_send_failure_is_marked_unsuccessful(client):
    class _Broken:
        def send(self, payload):
            raise OSError('socket gone')

    client._ws = _Broken()
    reply = client._send('get_output_info', _timeout=0.01)
    assert reply.get('success', True) is False


def test_not_connected_is_unsuccessful(client):
    client._ws = None
    reply = client._send('get_output_info')
    assert reply.get('success', True) is False


# ── The deadlock itself ────────────────────────────────────────────────


def test_a_request_from_the_receive_thread_is_refused(client):
    """
    The reply is delivered by the receive loop, so a request issued from that
    thread can never be answered — it can only burn its timeout. Refused
    immediately and loudly instead, because the failure is a programming error
    and silence cost a whole render.
    """
    client._recv_thread = threading.current_thread()

    reply = client._send('get_output_info', _timeout=30.0)

    assert reply.get('success', True) is False
    assert 'receive thread' in reply['error']
    # And it must not have burned the timeout waiting.
    assert client._response_events == {}


def test_a_request_from_any_other_thread_proceeds(client):
    """The guard must not fire for ordinary calls."""
    client._recv_thread = threading.Thread(target=lambda: None)

    reply = client._send('get_output_info', _timeout=0.01)

    assert 'receive thread' not in reply.get('error', '')


# ── The output path is derived per render ──────────────────────────────


def test_output_path_is_never_reused_across_renders():
    """
    After a download `_output_path` holds the *local* copy. Carrying it into a
    second render would name a Windows path to a pod, where it is not a path at
    all, and the render fails at the last step after all the work.
    """
    import io
    import os

    # Read as text rather than importing: the bridge needs PySide6, which the
    # test environment deliberately does not install.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = io.open(os.path.join(root, 'desktop', 'bridge.py'),
                  encoding='utf-8').read()

    guard = 'if not self._output_path:' + chr(10) + '            import os'
    assert guard not in src, (
        'the output path must be derived from the target every render, '
        'not kept from the previous one'
    )
    assert 'pipeline_target = self._target_remote_path or self._target_path' in src
