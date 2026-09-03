#!/usr/bin/env python3
"""
Change realism settings on a running pipeline, from the command line.

`set_realism` existed only as a method on `desktop/controller.py` and as
something `tools/sweep_levers.py` sent from inside its own loop. There was no
way for a person to use it: the desktop deliberately exposes no UI for these
knobs, and the CLI flags are read once at process start. So the documented way
to "A/B live" required writing a WebSocket client first.

    python tools/realism.py --host 1.2.3.4 --port 19278 \\
        swapper_model=hyperswap_1a_256 enhance_strength=0.7

    python tools/realism.py --host 1.2.3.4 --port 19278 enhance=false
    python tools/realism.py --host 1.2.3.4 --port 19278 --show

Values are parsed by shape: `true`/`false` become booleans, bare numbers become
int or float, everything else stays a string. The server validates and clamps,
and reports anything it refused rather than silently ignoring it — so a typo
comes back named instead of quietly doing nothing.

Take the host and port from what `orchestrator.py push` prints; they change on
every stop/resume. On Git Bash, prefix with `MSYS_NO_PATHCONV=1` if any value
you pass starts with a slash.
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional


def _parse_value(raw: str) -> Any:
    """
    Turn a command-line string into the type the server expects.

    Shape decides, because a settings command that needed types declared would
    be worse to use than the thing it replaced.

    Args:
        raw: The text after the `=`

    Returns:
        bool, int, float or str
    """
    low = raw.strip().lower()
    if low in ('true', 'yes', 'on'):
        return True
    if low in ('false', 'no', 'off'):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_settings(pairs: List[str]) -> Dict[str, Any]:
    """
    Parse `key=value` arguments.

    Args:
        pairs: Raw `key=value` strings

    Returns:
        Mapping of field name to typed value

    Raises:
        SystemExit: on an argument with no `=`, since guessing what was meant
                    is how the wrong setting gets changed
    """
    values: Dict[str, Any] = {}
    for pair in pairs:
        if '=' not in pair:
            print("ERROR: expected key=value, got '{}'".format(pair))
            raise SystemExit(2)
        key, _, raw = pair.partition('=')
        values[key.strip()] = _parse_value(raw)
    return values


async def _run(args: argparse.Namespace) -> int:
    # Imported here, not at module scope: the formatting and parsing
    # helpers above are worth testing on a machine that has no
    # WebSocket client installed, and CI is one of them.
    import websockets
    from pipeline_link import build, describe, opening_frame, warn_if_unprotected

    url, kwargs = build(args.host, args.port)
    warn_if_unprotected(url)
    print('connecting to {}'.format(describe(url)))

    async with websockets.connect(url, open_timeout=20, **kwargs) as ws:
        hello = opening_frame()
        if hello:
            await ws.send(hello)

        async def collect(action: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
            """Read until the reply to `action` arrives, ignoring frames."""
            loop = asyncio.get_event_loop()
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if message.get('type') == action or message.get('action') == action:
                    return dict(message)
            return None

        if args.show:
            await ws.send(json.dumps({'action': 'get_stats'}))
            reply = await collect('get_stats')
            print(json.dumps(reply, indent=2) if reply else 'no reply')
            return 0

        values = _parse_settings(args.settings)
        if not values:
            print('nothing to set — pass key=value pairs, or --show')
            return 2

        print('sending: {}'.format(', '.join(
            '{}={!r}'.format(k, v) for k, v in values.items())))
        await ws.send(json.dumps({'action': 'set_realism', 'values': values}))

        reply = await collect('set_realism')
        if reply is None:
            print('no acknowledgement within 10s — is the pipeline running?')
            return 1

        data = reply.get('data') or {}
        applied = data.get('applied') or {}
        rejected = data.get('rejected') or {}

        for key, value in applied.items():
            print('  applied  {} = {}'.format(key, value))
        for key, reason in rejected.items():
            print('  REJECTED {}: {}'.format(key, reason))

        # A rejection is the whole point of reporting: the previous way to send
        # these dropped the reply on the floor, so a misspelled field looked
        # exactly like a successful change.
        return 1 if rejected else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Change realism settings on a running pipeline.',
        epilog='Example: realism.py --host 1.2.3.4 --port 19278 enhance=false',
    )
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--show', action='store_true',
                        help='print the pipeline status instead of changing anything')
    parser.add_argument('settings', nargs='*', help='key=value pairs')
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print('ERROR: could not reach {}:{} — {}'.format(args.host, args.port, exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
