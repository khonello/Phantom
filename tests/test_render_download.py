"""
Getting a finished render back off the pod.

A render writes on the pipeline's filesystem. On a pod that is another machine,
so the file has to be read back or it does not exist as far as the operator is
concerned — and if the pod is later terminated, it stops existing at all. That
is what happened: the download ran on the WebSocket receive thread, could not
be answered, and reported success anyway.
"""

import json
import threading
import time
from collections import OrderedDict

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
    c._pending = OrderedDict()
    c._request_seq = 0
    c._recv_thread = None
    # A reply with no waiter falls through to the event path.
    c.on_event = None
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
    assert not client._pending


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


# ── A reply must answer the request that asked for it ──────────────────


def test_a_late_reply_cannot_answer_the_next_request(client):
    """
    Waiters used to be keyed by action name, so a reply the caller had already
    given up on satisfied the *next* request of the same name.

    That is what an operator saw as the source upload misbehaving: the first
    upload timed out client-side while the server kept working, they uploaded
    again, and the first attempt's review came back and unblocked the retry —
    reporting a verdict on a request nobody was waiting for, while the retry's
    own reply was dropped for having no waiter left.
    """
    # First request times out; the server has not answered yet.
    first = client._send('upload_source', _timeout=0.01)
    assert first.get('success', True) is False

    first_id = json.loads(client._ws.sent[0])['request_id']

    # The operator retries.
    done = {}

    def _retry():
        done['reply'] = client._send('upload_source', _timeout=2.0)

    thread = threading.Thread(target=_retry)
    thread.start()
    while len(client._ws.sent) < 2:
        time.sleep(0.01)
    second_id = json.loads(client._ws.sent[1])['request_id']
    assert second_id != first_id

    # Now the first attempt's reply finally lands.
    client._dispatch_message({
        'type': 'response',
        'action': 'upload_source',
        'request_id': first_id,
        'success': True,
        'data': {'count': 1},
    })
    time.sleep(0.05)
    assert 'reply' not in done, 'a stale reply answered the retry'

    # The retry's own reply is what unblocks it.
    client._dispatch_message({
        'type': 'response',
        'action': 'upload_source',
        'request_id': second_id,
        'success': True,
        'data': {'count': 3},
    })
    thread.join(timeout=3.0)
    assert done['reply']['data'] == {'count': 3}


def test_a_reply_without_an_id_still_matches_by_action(client):
    """A pipeline old enough not to echo the id must still be answerable."""
    done = {}

    def _ask():
        done['reply'] = client._send('get_output_info', _timeout=2.0)

    thread = threading.Thread(target=_ask)
    thread.start()
    while not client._pending:
        time.sleep(0.01)

    client._dispatch_message({
        'type': 'response',
        'action': 'get_output_info',
        'success': True,
        'data': {'size': 7},
    })
    thread.join(timeout=3.0)
    assert done['reply']['data'] == {'size': 7}
