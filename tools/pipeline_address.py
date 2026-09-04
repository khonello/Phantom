"""
Where the running pipeline is, without being told twice.

`orchestrator.py start` already resolves the address and writes it into `.env`
as `PHANTOM_API_URL` — it has to, because the desktop reads it from there. Every
measurement tool then asked for `--host` and `--port` again anyway, so the
operator copied a value out of one command's output and into the next, and the
port changes on every stop/resume, so a stale copy is the normal case rather
than the careless one.

The address is derived here instead, and `--host` becomes an override for the
case it exists for: pointing a tool at something other than the instance the
orchestrator most recently started.

Deliberately one module rather than three copies of the parsing. The same
argument `onnx_session.py` makes about session construction: three call sites is
how a codebase acquires three subtly different answers to one question — and the
question here is "which pipeline", where two answers is already a bug.
"""

import os
import re
from typing import Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_REPO_ROOT, '.env')

_URL_RE = re.compile(
    r'^wss?://'
    r'(?P<host>[^/:]+)'
    r'(?::(?P<port>\d+))?'
)


def _read_env_url() -> Optional[str]:
    """
    `PHANTOM_API_URL` from the process environment, else from `.env`.

    The file is parsed rather than loaded through dotenv so this works without
    the dependency — these tools are run from a checkout, and failing to find
    an optional package is a worse error than reading two lines of text.
    """
    from_process = os.environ.get('PHANTOM_API_URL', '').strip()
    if from_process:
        return from_process

    if not os.path.isfile(_ENV_PATH):
        return None
    with open(_ENV_PATH, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('PHANTOM_API_URL='):
                value = line.split('=', 1)[1].strip().strip('"').strip("'")
                return value or None
    return None


def resolve(host: Optional[str], port: Optional[int]) -> Tuple[str, int]:
    """
    The address to connect to, and where it came from.

    An explicit `--host` always wins; `--port` defaults to the one in the URL,
    then to 9000. Raises with an actionable message rather than returning
    something plausible, because a measurement pointed at the wrong pipeline is
    worse than one that did not run.
    """
    url = _read_env_url()
    env_host: Optional[str] = None
    env_port: Optional[int] = None

    if url:
        match = _URL_RE.match(url)
        if match:
            env_host = match.group('host')
            raw_port = match.group('port')
            # A proxy hostname carries no port: it is standard TLS on 443.
            env_port = int(raw_port) if raw_port else (
                443 if url.startswith('wss://') else 9000)

    final_host = host or env_host
    final_port = port or env_port or 9000

    if not final_host:
        raise SystemExit(
            'No pipeline address. Either pass --host, or start an instance so\n'
            '  PHANTOM_API_URL is written to .env:\n'
            '    python runpod/orchestrator.py start\n'
            '  (looked in the environment and {})'.format(_ENV_PATH))

    return final_host, final_port


def describe(host: str, port: int, explicit: bool) -> str:
    """One line saying which pipeline is about to be talked to, and why."""
    source = 'from --host' if explicit else 'from PHANTOM_API_URL in .env'
    return '{}:{}  ({})'.format(host, port, source)
