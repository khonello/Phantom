"""
The TLS-and-token path, run for real against a live server.

`test_transport_security.py` pins the pieces in isolation. This one starts an
actual `WebSocketAPIServer` over TLS on a real socket and connects to it with
the same clients the product uses, because every piece passing separately is
exactly the state a deployment can be in while still not working.

It is also the only place the **openssl commands from `vast/startup.sh`** are
executed. Those run on a rented machine minutes into a paid session; the
failure they would produce — a certificate Python cannot load, or a fingerprint
that never matches — is silent until the desktop refuses to connect. Generating
the cert here with the identical command is what makes that a test rather than
a hope.

Skipped when openssl is unavailable. CI has it; a developer machine might not,
and a skipped integration test is better than a fake one.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

openssl = shutil.which('openssl')
pytestmark = pytest.mark.skipif(openssl is None, reason='openssl not available')

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:  # pragma: no cover
    ws_connect = None


TOKEN = 'f' * 64


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _make_cert(directory):
    """
    Generate a certificate exactly as `vast/startup.sh` does.

    Kept byte-identical to the script on purpose: if the flags there stop
    producing something `ssl.load_cert_chain` accepts, this is where it shows
    up, rather than on a rented GPU.
    """
    cert = os.path.join(directory, 'phantom-tls.pem')
    key = os.path.join(directory, 'phantom-tls.key')
    subprocess.run(
        [openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
         '-keyout', key, '-out', cert,
         '-days', '3650', '-subj', '/CN=phantom-pipeline'],
        check=True, capture_output=True,
    )

    # And the fingerprint the same way: DER, not PEM. Hashing the PEM produces
    # a value that never matches what getpeercert(binary_form=True) returns,
    # which would make the pin always fail — and a pin that always fails gets
    # disabled, which is worse than never having had one.
    der = subprocess.run([openssl, 'x509', '-in', cert, '-outform', 'DER'],
                         check=True, capture_output=True).stdout
    import hashlib
    return cert, key, hashlib.sha256(der).hexdigest()


@pytest.fixture(scope='module')
def live_server(tmp_path_factory):
    """A real WebSocketAPIServer, over TLS, with a token, on a real port."""
    if ws_connect is None:
        pytest.skip('websockets not installed')

    work = str(tmp_path_factory.mktemp('tls'))
    cert, key, fingerprint = _make_cert(work)

    os.environ['PHANTOM_TLS_CERT'] = cert
    os.environ['PHANTOM_TLS_KEY'] = key
    os.environ['PHANTOM_API_TOKEN'] = TOKEN
    # Never let a test start the billing timer.
    os.environ.pop('VAST_MAX_UPTIME', None)

    from pipeline.api.server import WebSocketAPIServer

    port = _free_port()
    server = WebSocketAPIServer(port=port)
    server.start()

    # The server binds on a background thread; wait for the socket rather than
    # sleeping a guessed amount.
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                break
        except OSError:
            time.sleep(0.1)
    else:  # pragma: no cover
        server.stop()
        pytest.fail('server never bound port {}'.format(port))

    yield {'port': port, 'fingerprint': fingerprint, 'cert': cert}

    server.stop()
    for name in ('PHANTOM_TLS_CERT', 'PHANTOM_TLS_KEY', 'PHANTOM_API_TOKEN'):
        os.environ.pop(name, None)


def _pinned_ssl(fingerprint, port):
    """Build the client context the way tools/pipeline_link.py does."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    os.environ['PHANTOM_TLS_FINGERPRINT'] = fingerprint
    import importlib
    import pipeline_link
    importlib.reload(pipeline_link)
    _url, kwargs = pipeline_link.build('127.0.0.1', port)
    return kwargs.get('ssl')


# ── The happy path ───────────────────────────────────────────────────────────

def test_a_pinned_client_with_the_token_gets_a_health_reply(live_server):
    """
    The whole chain in one assertion: openssl's certificate, served by the real
    server, pinned by the real client helper, authenticated with the real
    token, answered by the real handler.
    """
    ctx = _pinned_ssl(live_server['fingerprint'], live_server['port'])
    url = 'wss://127.0.0.1:{}/ws'.format(live_server['port'])

    with ws_connect(url, ssl=ctx, open_timeout=15, close_timeout=5) as ws:
        ws.send(json.dumps({'action': 'health', 'token': TOKEN}))
        for _ in range(20):
            reply = json.loads(ws.recv(timeout=15))
            if reply.get('status') == 'healthy':
                return
    pytest.fail('no healthy reply')


def test_the_openssl_fingerprint_matches_what_the_server_presents(live_server):
    """
    The DER-vs-PEM trap, proven against a live handshake rather than argued.
    """
    import hashlib
    import ssl
    probe = ssl.create_default_context()
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    with socket.create_connection(('127.0.0.1', live_server['port']), timeout=10) as raw:
        with probe.wrap_socket(raw, server_hostname='127.0.0.1') as tls:
            der = tls.getpeercert(True)
    assert hashlib.sha256(der).hexdigest() == live_server['fingerprint']


# ── The refusals ─────────────────────────────────────────────────────────────

def test_a_wrong_token_is_refused(live_server):
    ctx = _pinned_ssl(live_server['fingerprint'], live_server['port'])
    url = 'wss://127.0.0.1:{}/ws'.format(live_server['port'])

    with pytest.raises(Exception):
        with ws_connect(url, ssl=ctx, open_timeout=15, close_timeout=5) as ws:
            ws.send(json.dumps({'action': 'health', 'token': 'a' * 64}))
            # The server closes with 1008; recv raises rather than returning.
            for _ in range(5):
                ws.recv(timeout=10)


def test_no_token_at_all_is_refused(live_server):
    ctx = _pinned_ssl(live_server['fingerprint'], live_server['port'])
    url = 'wss://127.0.0.1:{}/ws'.format(live_server['port'])

    with pytest.raises(Exception):
        with ws_connect(url, ssl=ctx, open_timeout=15, close_timeout=5) as ws:
            ws.send(json.dumps({'action': 'health'}))
            for _ in range(5):
                ws.recv(timeout=10)


def test_a_wrong_pin_is_refused_before_anything_is_sent(live_server):
    """
    The client must fail while building its context, not after connecting.

    That is why the pin is installed as a trust anchor rather than compared
    after the handshake: with only this certificate trusted, a different one
    cannot complete a handshake at all, so no credential can reach it even by
    mistake. Here the mismatch is caught one step earlier still — while
    fetching the certificate to build the anchor from.
    """
    with pytest.raises(RuntimeError, match='fingerprint mismatch'):
        _pinned_ssl('0' * 64, live_server['port'])


def test_plain_ws_cannot_reach_a_tls_server(live_server):
    """A cleartext client must fail loudly, not silently downgrade."""
    url = 'ws://127.0.0.1:{}/ws'.format(live_server['port'])
    with pytest.raises(Exception):
        with ws_connect(url, open_timeout=10, close_timeout=5) as ws:
            ws.send(json.dumps({'action': 'health', 'token': TOKEN}))
            ws.recv(timeout=10)
