#!/usr/bin/env python3
"""
Measure each inference speed lever against the same clip, in one pod session.

The levers are decided at ONNX session construction, so changing one used to
mean restarting the pipeline — and on a rented pod that is a cold start per
measurement, which would be most of a paid hour spent waiting rather than
measuring. `set_realism` now accepts them and the pipeline rebuilds its
sessions in place, so a whole sweep fits in one session.

**Feed it a file, not a camera.** A pod has no webcam, and more importantly a
camera gives every configuration different frames — so a latency difference
could be the lever or could be that someone moved. Point the pipeline at a clip
on the network volume and every run sees identical input:

    python pipeline.py --stream --input-url /workspace/clip.mp4

The pipeline does not need restarting for that. `input_url` is read when a
stream starts rather than when the process does, so `--input-url` sends
`set_input_url` and a pipeline already running under nohup picks it up.

    python tools/sweep_levers.py --host <ip> --port <port> \
        --input-url /tmp/phantom_uploads/<id>/clip.mp4 \
        --source /workspace/Phantom/.github/examples/source.jpg \
        --seconds 60 --out sweep.json

Use the host and port `orchestrator.py push` prints. Port 9000 is exposed
as tcp, so the pod has a public IP and a mapped port rather than a
`proxy.runpod.net` hostname, and `--tls` does not apply to it.

Each configuration starts the stream, runs for `--seconds`, stops it, and
captures the latency report the pipeline prints on stop. Order matters and is
not alphabetical: baseline first so everything else has something to be
compared against, and `cuda_graphs` before `cuda_streams` because they conflict
by design — graphs win where both are set, so measuring them together would
tell you nothing about streams.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Configurations to measure, in order. Each is (label, {lever: value}); every
# lever not named is left off. Baseline is first because a number with nothing
# to compare it against is not a measurement.
_SWEEP: List[Any] = [
    ('baseline', {}),
    ('async_encode', {'async_encode': True}),
    ('cuda_graphs', {'cuda_graphs': True}),
    ('cuda_streams', {'cuda_streams': True}),
    ('fp16', {'fp16': True}),
    ('fp16+cuda_graphs', {'fp16': True, 'cuda_graphs': True}),
    ('trt+fp16', {'trt': True, 'fp16': True}),
]

_ALL_LEVERS = ('fp16', 'cuda_graphs', 'cuda_streams', 'trt', 'async_encode')

# The pipeline prints the latency budget as a STATUS_CHANGED with this scope
# when a stream stops.
_PERF_SCOPE = 'PERF'


def _parse_report(text: str) -> Dict[str, Any]:
    """
    Pull the per-stage numbers out of the pipeline's formatted latency report.

    Parsed rather than requested because the report is already emitted on stop
    and already carries exactly what is needed; adding a command to return it
    as JSON would be a second representation to keep in step.

    Args:
        text: The report as the pipeline formatted it

    Returns:
        A mapping of stage name to its p50/p95/p99, plus the verdict
    """
    stages: Dict[str, Any] = {}
    for line in text.splitlines():
        match = re.match(
            r'\s+(\S+)\s+p50=\s*([\d.]+)ms\s+p95=\s*([\d.]+)ms\s+p99=\s*([\d.]+)ms',
            line,
        )
        if match:
            stages[match.group(1)] = {
                'p50': float(match.group(2)),
                'p95': float(match.group(3)),
                'p99': float(match.group(4)),
            }

    verdict = 'HOLDS' if '[HOLDS]' in text else ('MISSES' if '[MISSES]' in text else '?')
    over = re.search(r'([\d.]+)% over deadline', text)

    return {
        'verdict': verdict,
        'over_deadline_pct': float(over.group(1)) if over else None,
        'stages': stages,
    }


async def _run(args: argparse.Namespace) -> int:
    try:
        import websockets
    except ImportError:
        print('websockets is required: pip install websockets', file=sys.stderr)
        return 1

    scheme = 'wss' if args.tls else 'ws'
    url = '{}://{}:{}/ws'.format(scheme, args.host, args.port)
    if args.tls and args.port == 443:
        url = 'wss://{}/ws'.format(args.host)

    results: Dict[str, Any] = {'url': url, 'seconds': args.seconds, 'runs': {}}

    print('Connecting to {}'.format(url))
    async with websockets.connect(url, max_size=None) as socket:

        async def send(action: str, **payload: Any) -> Dict[str, Any]:
            """
            Send a command and wait for its acknowledgement.

            The first version fired and forgot, which made every failure look
            identical: a refused `set_source`, a stream that never started and
            a stream that ran perfectly all produced the same silent "no report
            captured". A measurement tool that cannot say why it found nothing
            is not much of a measurement tool.

            Args:
                action: Command name
                **payload: Command fields

            Returns:
                The server response, or an empty dict if none arrived
            """
            await socket.send(json.dumps({'action': action, **payload}))

            deadline = time.time() + 15.0
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        socket.recv(), timeout=max(0.1, deadline - time.time()),
                    )
                except asyncio.TimeoutError:
                    break
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue

                if args.verbose:
                    print('    < {}'.format(str(message)[:200]))

                # An error from anywhere is worth surfacing, not only one
                # carrying this command's name.
                if message.get('event') == 'ERROR' or message.get('level') == 'error':
                    print('    ERROR: {}'.format(
                        message.get('message') or message.get('error')))

                if message.get('action') == action or message.get('type') == action:
                    if not message.get('success', True):
                        print('    REFUSED {}: {}'.format(
                            action, message.get('error', 'no reason given')))
                    return dict(message)

            print('    no acknowledgement for {} within 15s'.format(action))
            return {}

        async def collect(seconds: float) -> str:
            """Read messages for `seconds`, returning any PERF status text."""
            report = ''
            deadline = time.time() + seconds
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        socket.recv(), timeout=max(0.1, deadline - time.time()),
                    )
                except asyncio.TimeoutError:
                    break
                if isinstance(raw, bytes):
                    continue  # a video frame
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if args.verbose:
                    print('    < {}'.format(str(message)[:200]))
                if message.get('event') == 'ERROR' or message.get('level') == 'error':
                    print('    ERROR: {}'.format(
                        message.get('message') or message.get('error')))
                if message.get('scope') == _PERF_SCOPE:
                    report += str(message.get('message', '')) + '\n'
            return report

        # Point the pipeline at the clip before anything else. This is what
        # makes a restart unnecessary: `input_url` is read when a stream
        # starts, not when the process does, so a pipeline already running
        # under nohup on a pod can be redirected at a file without a shell.
        if args.input_url:
            await send('set_input_url', url=args.input_url)
            await collect(1.0)

        if args.source:
            await send('set_source', path=args.source)
            await collect(3.0)

        for label, levers in _SWEEP:
            settings = {name: False for name in _ALL_LEVERS}
            settings.update(levers)

            print('\n=== {} ==='.format(label))
            print('  {}'.format(
                ', '.join(k for k, v in settings.items() if v) or 'everything off'))

            await send('set_realism', values=settings)
            # Sessions rebuild lazily on the next frame, so give the rebuild
            # somewhere to land before the clock starts. TensorRT may build an
            # engine here, which is minutes on a first run for that GPU.
            await collect(args.settle)

            await send('start_stream')
            await collect(args.seconds)
            await send('stop_stream')

            report = await collect(args.settle)
            parsed = _parse_report(report)
            results['runs'][label] = {'levers': settings, **parsed, 'raw': report}

            stages = parsed['stages']
            if stages:
                total = stages.get('total', {})
                restore = stages.get('restore', {})
                print('  total p95 {:.1f}ms   restore p95 {:.1f}ms   {}'.format(
                    total.get('p95', float('nan')),
                    restore.get('p95', float('nan')),
                    parsed['verdict'],
                ))
            else:
                print('  no report captured. The pipeline reports only when it '
                      'processed at least one frame, so the stream produced '
                      'none. Re-run with --verbose to see why.')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
        print('\nWrote {}'.format(args.out))

    _summarise(results)
    return 0


def _summarise(results: Dict[str, Any]) -> None:
    """Print the comparison the sweep exists to produce."""
    runs = results.get('runs', {})
    baseline = runs.get('baseline', {}).get('stages', {}).get('total', {}).get('p95')

    print('\n' + '=' * 62)
    print('{:<20} {:>10} {:>10} {:>10}'.format('config', 'total p95', 'vs base', 'verdict'))
    print('-' * 62)
    for label, run in runs.items():
        total = run.get('stages', {}).get('total', {}).get('p95')
        if total is None:
            print('{:<20} {:>10}'.format(label, 'no data'))
            continue
        delta = ''
        if baseline:
            delta = '{:+.1f}%'.format((total - baseline) / baseline * 100.0)
        print('{:<20} {:>9.1f}ms {:>10} {:>10}'.format(
            label, total, delta, run.get('verdict', '?')))
    print('=' * 62)
    print(
        'Read `restore` against `swap+composite` in the JSON before anything '
        'else.\nIf restoration is not the dominant term, the reasoning behind '
        'fp16 and\nTensorRT is wrong for this workload — see docs/COMPILATION.md.'
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Measure each speed lever against the same clip.',
    )
    parser.add_argument('--host', required=True, help='pipeline host')
    parser.add_argument('--port', type=int, default=9000, help='pipeline port')
    parser.add_argument('--tls', action='store_true', help='use wss (RunPod proxy)')
    parser.add_argument('--source', help='source face path on the pipeline filesystem')
    parser.add_argument('--input-url', dest='input_url',
                        help='clip path on the pipeline filesystem, as printed by '
                             'orchestrator.py push. Sent as set_input_url, so a '
                             'running pipeline does not need restarting.')
    parser.add_argument('--seconds', type=float, default=60.0,
                        help='how long to stream per configuration')
    parser.add_argument('--settle', type=float, default=8.0,
                        help='pause after a lever change, for session rebuild')
    parser.add_argument('--out', default='sweep.json', help='write results here')
    parser.add_argument('--verbose', action='store_true',
                        help='print every message the pipeline sends')
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
