"""
One place that knows how to open a WebSocket to a running pipeline.

`sweep_levers.py`, `stats.py` and `realism.py` each built their own
`ws://host:port/ws` string. That was fine while RunPod terminated TLS at its
proxy and the API had no authentication: any of them could reach any pod with a
host and a port.

Neither half is true on Vast. The instance serves `wss://` with a self-signed
certificate, and the server drops a client that does not present the shared
token in its first frame — so all three tools would fail, in three slightly
different ways, against a pipeline that is working perfectly.

Three call sites is also exactly how a codebase acquires three subtly different
answers to the same question, which is the argument `onnx_session.py` already
makes about session construction. So this is the only place that builds the
connection.

Everything is read from the environment, which `vast/orchestrator.py start`
writes into `.env` — so a tool run from the repo root against a live instance
needs no extra flags:

    PHANTOM_TLS_FINGERPRINT   pin the instance's certificate
    PHANTOM_API_TOKEN         the shared secret for the first frame

Both unset means a plain local pipeline, which is what these tools do against
`localhost` and what CI does.
"""

import hashlib
import json
import os
import ssl
import sys
from typing import Any, Dict, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env() -> None:
    """
    Read .env if python-dotenv is available, without requiring it.

    These tools are run from the repo root against an instance the
    orchestrator just configured, so the values are already on disk. Failing
    to find dotenv is not worth an error: an explicit environment still works,
    which is what CI uses.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(_REPO_ROOT, '.env'))


def _fingerprint() -> str:
    return os.environ.get('PHANTOM_TLS_FINGERPRINT', '').strip().lower()


def token() -> str:
    """The shared secret, or '' when the pipeline is running open."""
    return os.environ.get('PHANTOM_API_TOKEN', '').strip()


def build(host: str, port: int, tls: Optional[bool] = None) -> Tuple[str, Dict[str, Any]]:
    """
    URL and connect kwargs for `websockets.connect`.

    `tls` forces the scheme; None decides from whether a fingerprint is set,
    which is the case that matters — the orchestrator writes the fingerprint
    exactly when it has put the pipeline behind TLS, so the two cannot
    disagree.

    Verification is disabled and replaced by the pin. That is not a weakening:
    the certificate is self-signed on an IP that moves with the host, so there
    is no name to check and no CA that could vouch for it, and pinning one
    specific key is a stronger statement than trusting whoever a CA signed for.
    """
    _load_env()
    secure = _fingerprint() != '' if tls is None else tls
    scheme = 'wss' if secure else 'ws'

    # Port 443 means the address is already a hostname serving standard TLS,
    # so appending the port would produce a URL nothing answers.
    if secure and port == 443:
        url = 'wss://{}/ws'.format(host)
    else:
        url = '{}://{}:{}/ws'.format(scheme, host, port)

    kwargs: Dict[str, Any] = {'max_size': None}
    if secure:
        kwargs['ssl'] = _pinned_context(host, port)
    return url, kwargs


def _pinned_context(host: str, port: int) -> ssl.SSLContext:
    """
    A context that will complete a handshake with one certificate and no other.

    Built by fetching the certificate, checking its fingerprint, and then
    installing that exact certificate as the only trust anchor. The handshake
    then does the enforcing.

    The obvious alternative — connect with verification off and compare the
    peer certificate afterwards — was tried first and rejected twice over.
    Reaching the peer certificate through `websockets` means reading a
    different private attribute on each of its client implementations, so the
    check silently becomes a no-op on a version bump; and a check that runs
    after the handshake has already exchanged data is a check in the wrong
    place. Loading the certificate as its own CA is version-independent and
    happens before any bytes are trusted.

    `check_hostname` stays off because the certificate is self-signed for a
    name that is not the IP it is served from — there is nothing to match. The
    key itself is the identity, which is what the fingerprint asserts.
    """
    expected = _fingerprint()
    context = ssl.create_default_context()
    if not expected:
        return context

    probe = ssl.create_default_context()
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    import socket as _socket
    with _socket.create_connection((host, port), timeout=15) as raw:
        with probe.wrap_socket(raw, server_hostname=host) as tls:
            der = tls.getpeercert(True)

    if not der:
        raise RuntimeError('{}:{} presented no certificate'.format(host, port))
    actual = hashlib.sha256(der).hexdigest()
    if actual.lower() != expected:
        raise RuntimeError(
            'certificate fingerprint mismatch at {}:{} — expected {}..., got '
            '{}...\nEither the instance was rebuilt (re-run '
            '`vast/orchestrator.py start`) or something is in the middle.'.format(
                host, port, expected[:16], actual[:16])
        )

    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
    return context


def opening_frame() -> Optional[str]:
    """
    The first frame every client must send when the API is protected.

    `health` rather than a bare credential, because the server dispatches this
    frame as an ordinary command once the token checks out — so the handshake
    doubles as the readiness probe instead of costing an extra round trip.
    """
    secret = token()
    if not secret:
        return None
    return json.dumps({'action': 'health', 'token': secret})


def describe(url: str) -> str:
    """A one-line summary for a tool to print before it connects."""
    bits = []
    if _fingerprint():
        bits.append('pinned cert')
    if token():
        bits.append('token')
    suffix = ' ({})'.format(', '.join(bits)) if bits else ' (open)'
    return url + suffix


def warn_if_unprotected(url: str) -> None:
    """
    Say so when a remote address is being reached without TLS.

    Not fatal — a pipeline on a trusted LAN is a legitimate thing to measure —
    but silence here is how the cleartext mistake gets made, so a remote
    address with no pin is called out on stderr.
    """
    if url.startswith('wss://'):
        return
    host = url.split('://', 1)[-1].split(':', 1)[0].split('/', 1)[0]
    if host in ('localhost', '127.0.0.1', '::1'):
        return
    print(
        'WARNING: {} is not local and not encrypted. Frames and commands '
        'travel in the clear.'.format(host),
        file=sys.stderr,
    )
