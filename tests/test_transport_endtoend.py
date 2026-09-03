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

import io
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


# ── The startup.sh block that produces all of this ───────────────────────────

def _tls_block():
    """The TLS section of vast/startup.sh, as a runnable fragment."""
    script = io.open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'vast', 'startup.sh'), encoding='utf-8').read()
    start = script.index('# \u2500\u2500 8b. TLS certificate and API token')
    end = script.index('# \u2500\u2500 9. Summary')
    # `_phase` belongs to the surrounding script's timing harness.
    return 'set -euo pipefail\nWORKSPACE="$1"\n' + \
        script[start:end].replace('_phase "tls"', ':')


@pytest.mark.skipif(shutil.which('bash') is None, reason='bash not available')
def test_the_startup_tls_block_is_idempotent(tmp_path):
    """
    Running it twice must reuse the certificate, not regenerate it.

    This is the property that makes `resume` work at all. The certificate's
    fingerprint is pinned in the operator's `.env`; regenerating it on every
    boot would break the pin after every stop/start, and the desktop would
    refuse the connection — which reads as an attack rather than as a restart.

    Executed rather than argued, because "the `if [ -f ... ]` looks right" is
    exactly what one would have said about the version that silently failed:
    an `openssl` error was going to /dev/null, so the files were never created
    and every run took the generate branch.
    """
    block = tmp_path / 'block.sh'
    # newline='' so LF survives. write_text() would translate to CRLF on
    # Windows, and bash then rejects the CR at the end of the first line as
    # part of the option name -- the same way a CRLF startup.sh would fail
    # on the instance.
    with io.open(str(block), 'w', encoding='utf-8', newline='') as fh:
        fh.write(_tls_block())

    env = dict(os.environ)
    # MSYS rewrites the POSIX -subj argument into a Windows path on Git Bash.
    # A local-shell artefact, not a script bug — the pod is Linux.
    env['MSYS_NO_PATHCONV'] = '1'

    def run():
        # Relative paths, run from inside tmp_path: MSYS_NO_PATHCONV stops
        # Git Bash converting the -subj argument, but it stops it converting
        # the script path too, and a Windows path reaches bash mangled.
        done = subprocess.run(['bash', 'block.sh', '.'], cwd=str(tmp_path),
                              capture_output=True, text=True, env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        return done.stdout

    first = run()
    second = run()

    def field(out, name):
        for line in out.splitlines():
            if line.startswith(name + ' '):
                return line.split()[-1]
        return None

    assert 'Generating a self-signed certificate' in first
    assert 'Reusing existing certificate' in second, \
        'a second boot regenerated the certificate, which breaks the pin'

    fp1, fp2 = field(first, 'CERT_FINGERPRINT'), field(second, 'CERT_FINGERPRINT')
    tk1, tk2 = field(first, 'API_TOKEN'), field(second, 'API_TOKEN')

    assert fp1 and len(fp1) == 64 and int(fp1, 16) >= 0, 'fingerprint is not 64 hex chars'
    assert tk1 and len(tk1) == 64 and int(tk1, 16) >= 0, 'token is not 64 hex chars'
    assert fp1 == fp2, 'the fingerprint changed between boots'
    assert tk1 == tk2, 'the token changed between boots'


@pytest.mark.skipif(shutil.which('bash') is None, reason='bash not available')
def test_the_block_reports_a_fingerprint_the_server_actually_presents(tmp_path):
    """
    Close the loop: the value startup.sh prints is the value a TLS server built
    from those same files hands to a client.

    Both halves were verified separately — openssl produces a DER hash, and the
    server serves the cert — but the thing that matters is that they are the
    same number, because the orchestrator writes one and the desktop checks the
    other.
    """
    block = tmp_path / 'block.sh'
    # newline='' so LF survives. write_text() would translate to CRLF on
    # Windows, and bash then rejects the CR at the end of the first line as
    # part of the option name -- the same way a CRLF startup.sh would fail
    # on the instance.
    with io.open(str(block), 'w', encoding='utf-8', newline='') as fh:
        fh.write(_tls_block())
    env = dict(os.environ)
    env['MSYS_NO_PATHCONV'] = '1'
    out = subprocess.run(['bash', 'block.sh', '.'], cwd=str(tmp_path),
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr

    reported = None
    for line in out.stdout.splitlines():
        if line.startswith('CERT_FINGERPRINT '):
            reported = line.split()[-1]

    import hashlib
    import ssl
    cert = str(tmp_path / 'phantom-tls.pem')
    key = str(tmp_path / 'phantom-tls.key')

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert, key)

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    import threading as _threading

    def serve():
        try:
            conn, _ = listener.accept()
            try:
                server_ctx.wrap_socket(conn, server_side=True).close()
            except OSError:
                conn.close()
        except OSError:
            pass

    _threading.Thread(target=serve, daemon=True).start()

    probe = ssl.create_default_context()
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    with socket.create_connection(('127.0.0.1', port), timeout=10) as raw:
        with probe.wrap_socket(raw, server_hostname='127.0.0.1') as tls:
            der = tls.getpeercert(True)
    listener.close()

    assert hashlib.sha256(der).hexdigest() == reported
