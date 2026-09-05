#!/usr/bin/env python3
"""
Ask a running pipeline what it is actually running.

    python tools/stats.py --host 1.2.3.4 --port 19278
    python tools/stats.py --host 1.2.3.4 --port 19278 --json

Answers the question that previously needed the pipeline log and knowing what
to grep for: which GPU, which swap model, which restoration model, whether
restoration is even on, whether the models are on CUDA or quietly on CPU, and
how much of the paid hour is left before the pod stops itself.

**It reports resolved values, not requested ones.** Both model registries fall
back on an unknown name rather than failing — a typo in `.env` degrades to the
default instead of taking down a pod that is already being paid for — so what
the config asked for and what loaded can differ. That difference is usually the
thing being looked for.

Exit status is 0 when the pipeline answered and nothing looks wrong, 1 when it
could not be reached, and 2 when it answered but a check failed — an
accelerator was requested and is not available, which is the silent CPU
fallback that costs a GPU hour and produces nothing usable.
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional


def _line(label: str, value: Any) -> str:
    return '  {:<22} {}'.format(label, value)


def _render(data: Dict[str, Any]) -> List[str]:
    """Format the report for a terminal, in the order someone reads it."""
    out: List[str] = []
    server = data.get('server') or {}
    swapper = data.get('swapper') or {}
    enhancer = data.get('enhancer') or {}
    realism = data.get('realism') or {}
    capture = data.get('capture') or {}
    levers = data.get('levers') or {}

    out.append('')
    out.append('PIPELINE')
    out.append(_line('ready', data.get('ready')))
    out.append(_line('streaming', data.get('pipeline_running')))
    out.append(_line('source loaded', data.get('source_loaded')))

    uptime = server.get('uptime_seconds')
    if uptime is not None:
        out.append(_line('uptime', '{:.0f}m {:.0f}s'.format(uptime // 60, uptime % 60)))
    remaining = server.get('auto_stop_remaining_seconds')
    if remaining is not None:
        out.append(_line('auto-stop in', '{:.0f} min'.format(remaining / 60)))
    elif server.get('auto_stop_minutes') == 0:
        out.append(_line('auto-stop', 'disabled'))
    if 'clients' in server:
        out.append(_line('clients', server['clients']))

    out.append('')
    out.append('HARDWARE')
    out.append(_line('gpu', data.get('gpu')))
    out.append(_line('providers requested', ', '.join(data.get('execution_providers') or [])))
    out.append(_line('providers available', ', '.join(data.get('available_providers') or [])))

    out.append('')
    out.append('MODELS')
    out.append(_line('swap', '{}  ({}px native)'.format(
        swapper.get('model'), swapper.get('native_size'))))
    if enhancer.get('enabled'):
        out.append(_line('restoration', '{}  ({}px crop)'.format(
            enhancer.get('model'), enhancer.get('crop'))))
    else:
        out.append(_line('restoration', 'OFF  (would be {})'.format(enhancer.get('model'))))

    out.append('')
    out.append('LOOK')
    out.append(_line('enhance_strength', realism.get('enhance_strength')))
    weight = realism.get('enhancer_weight')
    if realism.get('enhancer_weight_active'):
        out.append(_line('enhancer_weight', weight))
    else:
        # Saying so matters: it is CodeFormer's fidelity input and inert on
        # every other model, so the number looks configured while doing nothing.
        out.append(_line('enhancer_weight', '{}  (inert - model has no weight input)'.format(weight)))
    out.append(_line('aligned floor/ceiling', '{} / {}'.format(
        realism.get('aligned_floor'), realism.get('aligned_ceiling'))))
    out.append(_line('temporal_alpha', realism.get('temporal_alpha')))
    out.append(_line('colour / grain / xseg', '{} / {} / {}'.format(
        realism.get('color_correction'), realism.get('grain'), realism.get('occluder'))))

    out.append('')
    out.append('CAPTURE')
    out.append(_line('preset', capture.get('quality')))
    out.append(_line('resolution', '{}x{} @ {}fps'.format(
        capture.get('width'), capture.get('height'), capture.get('fps'))))
    out.append(_line('jpeg / det_size', '{} / {}'.format(
        capture.get('jpeg_quality'), capture.get('det_size'))))

    active = [name for name, on in levers.items() if on]
    out.append('')
    out.append(_line('speed levers', ', '.join(active) if active else 'none'))
    out.append('')
    return out


def _warnings(data: Dict[str, Any]) -> List[str]:
    """Checks worth failing on, rather than leaving for someone to notice."""
    problems: List[str] = []

    requested = [p for p in (data.get('execution_providers') or [])
                 if p != 'CPUExecutionProvider']
    available = data.get('available_providers') or []
    for provider in requested:
        if provider not in available:
            problems.append(
                '{} was requested but onnxruntime does not offer it — '
                'this pipeline is on CPU.'.format(provider))

    if not data.get('source_loaded'):
        problems.append('No source face loaded — a stream will refuse to start.')

    return problems


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
        # The token frame doubles as the readiness probe; the server dispatches
        # it as an ordinary command once the credential checks out.
        hello = opening_frame()
        if hello:
            await ws.send(hello)
        await ws.send(json.dumps({'action': 'get_stats'}))

        loop = asyncio.get_event_loop()
        deadline = loop.time() + 15.0
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
            if message.get('type') != 'get_stats':
                continue

            data = message.get('data') or {}
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True))
            else:
                print('\n'.join(_render(data)))

            problems = _warnings(data)
            for problem in problems:
                print('  WARNING: {}'.format(problem))
            return 2 if problems else 0

        print('No reply within 15s. Is this an older pipeline without get_stats?')
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Report what a running pipeline is configured with.')
    parser.add_argument('--host', help='override PHANTOM_API_URL from .env')
    parser.add_argument('--port', type=int,
                        help='override the port in PHANTOM_API_URL')
    parser.add_argument('--json', action='store_true', help='raw JSON instead of a report')
    args = parser.parse_args(argv)
    # The address comes from PHANTOM_API_URL in .env, which `start` wrote,
    # unless --host overrides it. The port changes on every stop/resume, so a
    # value copied by hand is stale more often than not.
    from pipeline_address import resolve, describe
    explicit = args.host is not None
    args.host, args.port = resolve(args.host, args.port)
    print('pipeline: {}'.format(describe(args.host, args.port, explicit)))

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print('ERROR: could not reach {}:{} - {}'.format(args.host, args.port, exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
