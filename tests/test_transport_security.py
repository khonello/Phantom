"""
The two things the Vast migration must not get wrong.

RunPod terminated TLS at its proxy and handed out a hostname nobody could
guess. Vast gives a random external port on a shared public IP and terminates
nothing, so the same code that was merely *unauthenticated* on RunPod would be
unauthenticated **and readable** here — with the operator's face in it.

`docs/ACCEPTED_RISKS.md` excused the missing authentication partly because the
proxy URL was "pod-specific and unguessable in practice". An IP and a port is
neither, so the move has to close that rather than inherit it.

Two properties are pinned here, and the first is the one worth the file:

  1. **An unauthenticated client never joins the broadcast set.** Frames go to
     every socket in `_clients`, so a connection admitted first and checked
     second would receive swapped video in the window between the two. The
     ordering is the security property; a token check that ran late would pass
     every test that only asked "was it rejected".

  2. **A missing certificate is fatal, not a downgrade.** Falling back to
     cleartext is the silent-CPU-fallback mistake somewhere worse: the session
     works, looks identical, and is readable by anyone on the path.
"""

import json
import logging
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.api.server import WebSocketAPIServer  # noqa: E402

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'ok  ' if condition else 'FAIL'
    print('{} {}{}'.format(mark, label, ('  — ' + detail) if detail else ''))


class FakeSocket:
    """One client connection, enough of it for `_authenticate` to run."""

    def __init__(self, first_frame=None, raise_on_recv: bool = False) -> None:
        self._first = first_frame
        self._raise = raise_on_recv
        self.closed_with = None
        self.sent: list = []

    def recv(self, timeout: float = 0):
        if self._raise:
            raise TimeoutError('no frame')
        return self._first

    def send(self, payload) -> None:
        self.sent.append(payload)

    def close(self, code: int = 1000, reason: str = '') -> None:
        self.closed_with = (code, reason)


class AuthOnly:
    """
    `WebSocketAPIServer`'s authentication, without the rest of the server.

    The real methods are bound onto a minimal stand-in rather than copied, for
    the reason the session-cleanup test gives: a copy drifts, and then the test
    passes against logic the product does not run.
    """

    _authenticate = WebSocketAPIServer._authenticate
    _reject = WebSocketAPIServer._reject
    _build_ssl_context = WebSocketAPIServer._build_ssl_context

    def __init__(self) -> None:
        self._clients: set = set()
        self._clients_lock = threading.Lock()
        self.dispatched: list = []

    def _handle_text_message(self, websocket, message) -> None:
        self.dispatched.append(message)


def _admit(server: AuthOnly, sock: FakeSocket) -> bool:
    """
    The handler's admission sequence, in the order the handler runs it.

    Kept in one place so the ordering property is what the tests below
    actually exercise, rather than each of them re-deciding it.
    """
    if not server._authenticate(sock, 'test-client'):
        return False
    with server._clients_lock:
        server._clients.add(sock)
    return True


# ── 1. No token configured: local development is unchanged ───────────────────

os.environ.pop('PHANTOM_API_TOKEN', None)

server = AuthOnly()
sock = FakeSocket()
check('no token configured admits a client',
      _admit(server, sock) and sock in server._clients,
      'a local pipeline has no network to protect')
check('no token configured consumes no frame',
      server.dispatched == [],
      'the first frame is an ordinary command when auth is off')


# ── 2. Correct token ─────────────────────────────────────────────────────────

os.environ['PHANTOM_API_TOKEN'] = 'a' * 64

server = AuthOnly()
good = FakeSocket(json.dumps({'action': 'health', 'token': 'a' * 64}))
check('correct token admits a client',
      _admit(server, good) and good in server._clients)
check('the opening frame is answered, not swallowed',
      len(server.dispatched) == 1,
      'otherwise every client would have to send health twice')


# ── 3. Wrong, missing and malformed tokens ───────────────────────────────────

cases = {
    'wrong token': json.dumps({'action': 'health', 'token': 'b' * 64}),
    'no token field': json.dumps({'action': 'health'}),
    'empty token': json.dumps({'action': 'health', 'token': ''}),
    'not json': 'health please',
    'json but not an object': json.dumps(['health']),
}

for label, frame in cases.items():
    server = AuthOnly()
    bad = FakeSocket(frame)
    admitted = _admit(server, bad)
    check('{} is refused'.format(label), not admitted)
    # The property that matters. A client can be refused and still have been
    # added, and that is the bug this file exists to prevent.
    check('{} never joins the broadcast set'.format(label),
          bad not in server._clients and not server._clients,
          'frames go to every socket in _clients')
    check('{} is closed with 1008'.format(label),
          bad.closed_with is not None and bad.closed_with[0] == 1008)
    check('{} is told nothing useful'.format(label),
          bad.sent == [],
          'no reply that would distinguish wrong from missing')

server = AuthOnly()
silent = FakeSocket(raise_on_recv=True)
check('a client that sends nothing is refused',
      not _admit(server, silent) and not server._clients,
      'a held-open socket must not occupy a slot forever')


# ── 4. Binary first frame ────────────────────────────────────────────────────
# A JPEG arriving before authentication must not be treated as a credential,
# and must not reach `_handle_binary_frame` either.

server = AuthOnly()
binary = FakeSocket(b'\xff\xd8\xff\xe0 not a token')
check('a binary opening frame is refused',
      not _admit(server, binary) and not server._clients)


# ── 5. TLS: configured, absent, and broken ───────────────────────────────────

os.environ.pop('PHANTOM_TLS_CERT', None)
os.environ.pop('PHANTOM_TLS_KEY', None)
check('no certificate configured leaves the server on ws://',
      AuthOnly()._build_ssl_context() is None,
      'local runs need no certificate')

os.environ['PHANTOM_TLS_CERT'] = '/nonexistent/phantom.pem'
os.environ['PHANTOM_TLS_KEY'] = '/nonexistent/phantom.key'
raised = False
try:
    AuthOnly()._build_ssl_context()
except SystemExit:
    raised = True
except Exception:
    pass
check('an unreadable certificate stops the server',
      raised,
      'falling back to cleartext would work and be readable')

os.environ.pop('PHANTOM_TLS_CERT', None)
os.environ.pop('PHANTOM_TLS_KEY', None)
os.environ.pop('PHANTOM_API_TOKEN', None)


# ── 6. The desktop pin ───────────────────────────────────────────────────────

from desktop.controller import _check_pin, _pinned_ssl_context  # noqa: E402


class FakeTLS:
    def __init__(self, der: bytes) -> None:
        self.socket = self

    der = b''

    def getpeercert(self, binary_form: bool = False) -> bytes:
        return self.der


import hashlib  # noqa: E402

cert_bytes = b'a pretend DER certificate'
digest = hashlib.sha256(cert_bytes).hexdigest()

os.environ.pop('PHANTOM_TLS_FINGERPRINT', None)
check('no fingerprint set leaves ordinary verification in place',
      _pinned_ssl_context() is None,
      'so a real certificate later needs no flag')

os.environ['PHANTOM_TLS_FINGERPRINT'] = digest
matching = FakeTLS(cert_bytes)
matching.der = cert_bytes
ok = True
try:
    _check_pin(matching)
except Exception:
    ok = False
check('a matching certificate passes the pin', ok)

mismatched = FakeTLS(cert_bytes)
mismatched.der = b'a different certificate'
rejected = False
try:
    _check_pin(mismatched)
except Exception:
    rejected = True
check('a different certificate fails the pin',
      rejected,
      'a swapped certificate on a video call is not a reason to carry on')


class NoTLS:
    socket = object()


refused = False
try:
    _check_pin(NoTLS())
except Exception:
    refused = True
check('a pinned client refuses a plaintext connection',
      refused,
      'wss:// was expected; ws:// must not silently satisfy a pin')

os.environ.pop('PHANTOM_TLS_FINGERPRINT', None)


print('=' * 70)
print('{} passed, {} failed'.format(len(PASS), len(FAIL)))
if FAIL:
    for failure in FAIL:
        print('  FAILED:', failure)
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
