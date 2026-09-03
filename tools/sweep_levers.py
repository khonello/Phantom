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

    # The measurement that bounds every other one. Restoration is ~68% of the
    # frame on a 4090, so this is the floor: whatever is left is what no amount
    # of work on restoration can remove. Take it before optimising anything.
    # It has still never been taken — the two sessions that tried both ran
    # against a pod whose `set_realism` did not yet accept `enhance`.
    ('no_restore', {'enhance': False}),

    # The config-only route to the same saving, and the one a product would
    # actually ship: skip restoration for a face below this size, needing no
    # model change. Set past the face in the clip so it skips throughout, which
    # makes this `no_restore` reached by the shippable lever — the two should
    # agree, and a gap between them is a bug in the threshold.
    ('restore_skip_small', {'restore_min_face': 200}),

    # The working resolution below restoration — the one knob that already
    # follows face size.
    ('aligned_128', {'aligned_size': 128}),

    # hyperswap is 256px native against inswapper's 128, and its profile asks
    # for *less* restoration (enhance_strength 0.5 vs 0.7) because the swap it
    # produces needs less. A bigger swap that buys a cheaper restore may be a
    # net win; it may also just be a bigger swap. Untested either way.
    ('hyperswap', {'swapper_model': 'hyperswap_1a_256'}),
    ('hyperswap+no_restore', {'swapper_model': 'hyperswap_1a_256', 'enhance': False}),
]

# Deliberately not swept, each for a reason established by measurement rather
# than by argument. Kept here because a lever that was tried and abandoned is
# worth more written down than silently missing — the next session should not
# have to rediscover any of it.
#
#   restore_256 / restore_128   `codeformer.onnx` declares input
#       {'restore_size': 256}   [1, 3, 512, 512] — static and square. The size
#                               is fixed in the graph, so `Enhancer.crop_size`
#                               warns once and holds at 512 and these rows read
#                               identical to baseline. Restoring smaller needs a
#                               re-export, not a config change. Check a model's
#                               declared shape before sweeping a shape lever.
#
#   fp16                        No valid fp16 weights exist. The conversion
#       {'fp16': True}          runs and halves the file (359 -> 180 MB), then
#                               fails its own load check on a Cast node whose
#                               output stayed float16 against a float input.
#                               Without weights the flag falls back silently,
#                               which is what made this row read flat twice.
#
#   trt+fp16                    Registering TensorRT broke cuDNN loading and
#       {'trt': True, ...}      dropped the swapper and restorer to CPU; the
#                               execution-provider check correctly halted the
#                               stream. It is last in any list for that reason.
#
#   cuda_graphs / cuda_streams  Measured flat on an L4 (144.4 / 146.1ms against
#                               a 146ms baseline). A model that spends its time
#                               inside one large graph is not waiting on kernel
#                               launch overhead.
#
#   async_encode                Measured flat. The encode is not the cost.

# Reset between configurations, so one run cannot inherit another's state.
# Every key any entry sets must appear here with its default.
_BASE: Dict[str, Any] = {
    'fp16': False,
    'cuda_graphs': False,
    'cuda_streams': False,
    'trt': False,
    'async_encode': False,
    'enhance': True,
    'aligned_size': 256,
    'restore_size': 512,
    'restore_min_face': 0,
    'swapper_model': 'inswapper_128',
    # No entry sets this, and it is here anyway: it is an appearance lever with
    # a per-frame cost, so a sweep started after someone A/B'd it live would
    # measure it in every configuration without saying so.
    'texture_strength': 0.0,
    'mask_feather': 0.04,
    'mask_erode': 0.03,
    'diffuse_strength': 0.0,
}

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
    # The latency report is emitted around PIPELINE_STOPPED, so it can arrive
    # while waiting for that event rather than in the collect() after it.
    nonlocal_report: List[str] = []

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

        async def wait_stopped(timeout: float = 20.0) -> None:
            """
            Wait for PIPELINE_STOPPED after a stop.

            `stop` acknowledges immediately — the reply means "stop requested",
            not "stopped". Starting the next configuration before the previous
            one has finished stopping made `start_stream` answer
            `{'rejoined': True}`, joining a pipeline already on its way down,
            which then stopped and produced no frames at all. Baseline and
            async_encode both reported "no data" for exactly that reason.
            """
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        socket.recv(), timeout=max(0.1, deadline - time.time()),
                    )
                except asyncio.TimeoutError:
                    return
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if args.verbose:
                    print('    < {}'.format(str(message)[:200]))
                if message.get('scope') == _PERF_SCOPE:
                    nonlocal_report.append(str(message.get('message', '')))
                if message.get('event') == 'PIPELINE_STOPPED':
                    return

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
            reply = await send('set_source', path=args.source)
            await collect(3.0)
            # Every configuration needs this one thing, so a refusal here is
            # not a bad run, it is a bad sweep. Without this the whole list
            # still executes — each stream refused for "Source path not set",
            # each producing "no data" — and the minutes come off a paid pod
            # before anyone reads the first line of output.
            if reply and not reply.get('success', True):
                print('')
                print('ABORTED: the pipeline refused the source, so no '
                      'configuration can run.')
                print('  {}'.format(reply.get('error', 'no reason given')))
                print('  The path is resolved on the *pipeline\'s* filesystem, '
                      'not this machine\'s.')
                print('  On Git Bash for Windows, note that an argument '
                      'beginning with / is')
                print('  rewritten into a Windows path unless '
                      'MSYS_NO_PATHCONV=1 is set.')
                return 2

        # Warm-up run, discarded. The first configuration otherwise pays model
        # load and first-inference cost inside its own measurement window, and
        # since baseline runs first that is exactly the number everything else
        # is compared against. It showed up as baseline reporting 490ms against
        # 147ms for the next configuration — a difference that was almost
        # entirely warm-up, and would have been read as a 3x win.
        print('')
        print('warm-up (discarded)')
        await send('start_stream')
        await collect(args.warmup)
        await send('stop')
        await wait_stopped()

        for label, levers in _SWEEP:
            settings = dict(_BASE)
            settings.update(levers)

            print('\n=== {} ==='.format(label))
            changed = {k: v for k, v in settings.items() if _BASE.get(k) != v}
            print('  {}'.format(
                ', '.join('{}={}'.format(k, v) for k, v in sorted(changed.items()))
                or 'defaults, everything off'))

            await send('set_realism', values=settings)
            # Sessions rebuild lazily on the next frame, so give the rebuild
            # somewhere to land before the clock starts. TensorRT may build an
            # engine here, which is minutes on a first run for that GPU.
            await collect(args.settle)

            await send('start_stream')
            await collect(args.seconds)
            # 'stop', not 'stop_stream'. CMD_STOP_STREAM is a dead constant
            # in the schema — never in COMMANDS, never dispatched — so the
            # wiring test does not catch it and the server answers "Unknown
            # command". The desktop's PipelineClient.stop_stream() is an alias
            # that sends 'stop' for the same reason.
            #
            # This mattered more than a wrong name usually does: the report is
            # only emitted when a stream stops, so a refused stop meant every
            # configuration measured nothing.
            await send('stop')
            nonlocal_report.clear()
            await wait_stopped()

            joined = chr(10).join(nonlocal_report)
            report = joined + await collect(2.0)
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

    # A sweep where nothing reported is a failed sweep, not an empty one.
    # Returning 0 here meant a run that measured nothing at all looked exactly
    # like a run that measured everything, which is how a wasted pod session
    # gets noticed by reading rather than by exit code.
    measured = sum(1 for run in results['runs'].values() if run.get('stages'))
    if not measured:
        print('\nFAILED: no configuration produced a report.')
        return 1
    if measured < len(results['runs']):
        print('\nPARTIAL: {} of {} configurations reported.'.format(
            measured, len(results['runs'])))
        return 1
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
    parser.add_argument('--warmup', type=float, default=20.0,
                        help='discarded first run, so model load does not land '
                             'inside the baseline measurement')
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
