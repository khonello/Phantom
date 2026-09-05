"""
What the link between this machine and a running pipeline actually delivers.

The question this answers is not "how fast is the GPU". `tools/stats.py` says
what is loaded and `LatencyBudget` says what the pipeline spends per stage.
Neither can see the part that turned out to dominate: the operator's own
upstream bandwidth.

The measurement that matters is a **comparison between presets in one
session**, not an absolute number. Absolutes move with whatever the connection
is doing that afternoon, and a single bad day is indistinguishable from a
standing problem. Two presets measured minutes apart on the same link are
internally controlled: if halving the bitrate collapses the latency, the link
is the constraint, and that conclusion holds whether the day was good or bad.

That is the test docs/PENDING_WORK.md asked for and CLAUDE.md predicted the
answer to. First run, 2026-09-05, against a Denmark RTX 4090:

    optimal   3.96 Mbps   61% delivered   p50 1222ms
    fast      1.58 Mbps   94% delivered   p50  366ms

856ms for a preset change worth ~10ms of compute. Not a compute result.

Two things make the numbers trustworthy:

  - **One clock.** The pipeline echoes the 8-byte capture timestamp back on the
    swapped frame, so a sample is `now - my_own_stamp`. No clock sync needed.
  - **Delivery rate is reported, not just latency.** A saturated uplink shows up
    first as frames that never come back, and a p50 computed over the survivors
    looks healthier than the experience is.

Run it against a live instance from the repo root:

    python tools/measure_link.py                    # fast and optimal
    python tools/measure_link.py --presets all      # adds production
    python tools/measure_link.py --out link.json    # keep it for comparison

Results are stamped and appended, so runs on different days can be compared -
which is the whole point.
"""

import argparse
import asyncio
import json
import os
import statistics
import struct
import sys
import time
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

from pipeline.api.schema import PRESETS  # noqa: E402

# Where the source face lives on the *pipeline's* filesystem. The repo is
# cloned there by the orchestrator, so its own example ships with it.
DEFAULT_POD_SOURCE = '/workspace/Phantom/.github/examples/source.jpg'
DEFAULT_LOCAL_FRAME = os.path.join(_REPO_ROOT, '.github', 'examples', 'source.jpg')

# Discarded before each measurement. The first stream after a pipeline start pays
# model load - tens of seconds - and those frames come back carrying round
# trips of exactly that size.
WARMUP_SECONDS = 12.0
_TS = '<q'


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _encode_frame(path: str, width: int, height: int, quality: int) -> bytes:
    """
    One JPEG at a preset's capture settings.

    It has to contain a face. While a stream runs the live path emits a frame
    only if it was swapped, so a frame with no face produces no reply at all
    and the measurement silently reports zero delivery - see
    tests/test_live_exposure.py for why that rule exists.
    """
    import cv2
    image = cv2.imread(path)
    if image is None:
        raise SystemExit('cannot read {} - need a face image to send'.format(path))
    resized = cv2.resize(image, (width, height))
    ok, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise SystemExit('JPEG encode failed')
    return bytes(buffer.tobytes())


async def _drain(ws: Any, quiet_for: float = 2.0, limit: float = 30.0) -> None:
    """
    Read until the socket has been silent for `quiet_for` seconds.

    Draining until the *first* gap is not enough, and the failure is not
    theoretical: a saturated uplink leaves seconds of frames in flight, so the
    first read after `stop_stream` times out while a queue is still arriving.
    Those frames then land inside the next preset's window and are counted as
    its own - which reported `fast` at 24% delivered and `optimal` at 89%, the
    exact reverse of the truth.
    """
    deadline = asyncio.get_event_loop().time() + limit
    while asyncio.get_event_loop().time() < deadline:
        try:
            await asyncio.wait_for(ws.recv(), timeout=quiet_for)
        except asyncio.TimeoutError:
            return
        except Exception:
            return


async def _network_rtt(ws: Any, probes: int) -> Dict[str, float]:
    """
    Round trip with no frame in it, which is the network on its own.

    Matched on the echoed `request_id` when there is one, and otherwise on the
    action. `health` is short-circuited in the server *before* dispatch_command,
    so it is the one reply that never carries a request_id back - matching on it
    alone waits out the timeout against a pipeline answering perfectly.

    The queue is drained first for the opposite reason: the server broadcasts
    status events on the same socket, and matching loosely against a backlog
    reports one of those instead, which reads as a 1ms network.
    """
    await _drain(ws)
    samples: List[float] = []
    for index in range(probes):
        marker = 'measure-link-{}'.format(index)
        started = time.perf_counter()
        await ws.send(json.dumps({'action': 'health', 'request_id': marker}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            if isinstance(raw, bytes):
                continue
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if message.get('request_id') == marker:
                break
            if message.get('action') == 'health':
                break
        samples.append((time.perf_counter() - started) * 1000.0)
        await asyncio.sleep(0.2)

    return {
        'p50': statistics.median(samples),
        'p95': _percentile(samples, 0.95),
        'min': min(samples),
        'max': max(samples),
        'samples': float(len(samples)),
    }


async def _measure_preset(
    ws: Any,
    name: str,
    seconds: float,
    frame_path: str,
) -> Dict[str, Any]:
    """
    Stream one preset at its own capture settings and time the round trip.

    Sender and receiver are separate tasks. Interleaving them in one loop
    starves the sender - the first version of this managed 71 frames in 30
    seconds instead of 600, and reported a latency for a rate it never reached.
    """
    preset = PRESETS[name]
    width = int(preset['capture_width'])
    height = int(preset['capture_height'])
    fps = float(preset['capture_fps'])
    quality = int(preset['jpeg_quality'])

    jpeg = _encode_frame(frame_path, width, height, quality)
    mbps = len(jpeg) * 8 * fps / 1e6

    print('')
    print('{}: {}x{} q{} @ {:.0f}fps - {:.1f} KB/frame, {:.2f} Mbps up'.format(
        name, width, height, quality, fps, len(jpeg) / 1024.0, mbps))

    await ws.send(json.dumps({'action': 'set_quality', 'preset': name}))
    await asyncio.sleep(1.0)
    await ws.send(json.dumps({'action': 'start_stream'}))
    print('  warm-up {:.0f}s (model load lands here, not in the sample)'.format(
        WARMUP_SECONDS))
    await asyncio.sleep(WARMUP_SECONDS)
    await _drain(ws)

    samples: List[float] = []
    counters = {'sent': 0, 'received': 0}
    finished = asyncio.Event()
    stop_at = time.time() + seconds
    # Belt to the drain's braces. A frame already on the wire when this window
    # opens carries a stamp from before it, and counting it would credit this
    # preset with the previous one's work.
    run_start = time.perf_counter_ns()

    async def sender() -> None:
        interval = 1.0 / fps
        due = time.perf_counter()
        while time.time() < stop_at:
            await ws.send(struct.pack(_TS, time.perf_counter_ns()) + jpeg)
            counters['sent'] += 1
            due += interval
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        finished.set()

    async def receiver() -> None:
        while not finished.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, bytes) or len(raw) <= 8:
                continue
            stamp = struct.unpack(_TS, raw[:8])[0]
            if stamp >= run_start:
                samples.append((time.perf_counter_ns() - stamp) / 1e6)
                counters['received'] += 1

    print('  measuring {:.0f}s...'.format(seconds))
    await asyncio.gather(sender(), receiver())
    await ws.send(json.dumps({'action': 'stop_stream'}))
    await asyncio.sleep(2.0)
    await _drain(ws)

    sent = counters['sent']
    received = counters['received']
    delivered = 100.0 * received / sent if sent else 0.0

    result: Dict[str, Any] = {
        'preset': name,
        'width': width,
        'height': height,
        'fps': fps,
        'jpeg_quality': quality,
        'frame_bytes': len(jpeg),
        'uplink_mbps': round(mbps, 2),
        'sent': sent,
        'received': received,
        'delivered_pct': round(delivered, 1),
    }
    if samples:
        result.update({
            'p50_ms': round(statistics.median(samples), 1),
            'p95_ms': round(_percentile(samples, 0.95), 1),
            'min_ms': round(min(samples), 1),
            'max_ms': round(max(samples), 1),
        })
        print('  {}/{} returned ({:.0f}%)  p50 {:.0f}ms  p95 {:.0f}ms  min {:.0f}ms'.format(
            received, sent, delivered, result['p50_ms'], result['p95_ms'],
            result['min_ms']))
    else:
        print('  NOTHING RETURNED. A frame leaves the pipeline only if it was')
        print('  swapped, so check the source loaded and the test frame has a face.')
    return result


def _verdict(network: Dict[str, float], runs: List[Dict[str, Any]]) -> List[str]:
    """
    Say what the comparison means, since the numbers alone do not.

    The interesting quantity is not any single latency. It is what happens to
    latency and delivery when the bitrate changes, because that separates a
    saturated link from a slow pipeline - and it does so without needing a good
    day to compare against.
    """
    lines = ['', 'READING']
    usable = [r for r in runs if r.get('received')]
    if not usable:
        lines.append('  Nothing came back. Not a link result - see the note above.')
        return lines

    floor = min(r['min_ms'] for r in usable)
    lines.append(
        '  Network alone is {:.0f}ms p50. The best frame round trip is {:.0f}ms,'
        .format(network['p50'], floor))
    lines.append(
        '  so the pipeline and both JPEG hops cost about {:.0f}ms. That part is'
        .format(max(0.0, floor - network['min'])))
    lines.append('  not the network, and a nearer datacenter will not move it.')

    starved = [r for r in usable if r['delivered_pct'] < 90.0]
    for run in starved:
        lines.append('')
        lines.append(
            '  {} lost {:.0f}% of frames at {:.2f} Mbps up. Frames that never arrive'
            .format(run['preset'], 100.0 - run['delivered_pct'], run['uplink_mbps']))
        lines.append(
            '  are the first symptom of a saturated uplink, and they flatter the p50')
        lines.append('  by leaving it computed over the survivors.')

    if len(usable) >= 2:
        cheap = min(usable, key=lambda r: r['uplink_mbps'])
        dear = max(usable, key=lambda r: r['uplink_mbps'])
        if cheap is not dear:
            saved = dear['p50_ms'] - cheap['p50_ms']
            gained = cheap['delivered_pct'] - dear['delivered_pct']
            lines.append('')
            lines.append(
                '  {} -> {} cuts the uplink {:.2f} -> {:.2f} Mbps: p50 {:+.0f}ms, '
                'delivery {:+.0f}pp.'.format(
                    dear['preset'], cheap['preset'], dear['uplink_mbps'],
                    cheap['uplink_mbps'], -saved, gained))

            # Delivery is weighed with latency, not after it. Smoothness is
            # frames arriving consistently, not a median - and a preset can
            # hold its p50 while dropping one frame in six, which is what an
            # operator sees. The first version of this keyed on p50 alone,
            # reported "bitrate barely moved it" for a pair that differed by
            # 7 points of delivery, and was corrected by someone watching the
            # picture rather than the numbers.
            if saved > 100.0 or gained >= 5.0:
                reasons = []
                if saved > 100.0:
                    reasons.append('{:.0f}ms off the p50'.format(saved))
                if gained >= 5.0:
                    reasons.append('{:.0f}pp more frames delivered'.format(gained))
                lines.append('  {} for a change worth ~10ms of compute.'.format(
                    ' and '.join(reasons)))
                lines.append(
                    '  The constraint is bandwidth, and no GPU or datacenter '
                    'changes it.')
            elif saved < 30.0 and abs(gained) < 5.0:
                lines.append(
                    '  Neither latency nor delivery moved: bandwidth is not the '
                    'constraint here.')

    worst_spread = max(r['p95_ms'] - r['p50_ms'] for r in usable)
    if worst_spread > 300.0:
        lines.append('')
        lines.append(
            '  p95 sits {:.0f}ms above p50 at worst. That spread is jitter, and'
            .format(worst_spread))
        lines.append(
            '  jitter is what the playout buffer turns into a fixed delay - so it')
        lines.append('  costs more than the number looks.')
    return lines


def _write(path: str, payload: Dict[str, Any]) -> None:
    """Append this run, so days can be compared rather than remembered."""
    history: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                existing = json.load(handle)
            if isinstance(existing, list):
                history = existing
            elif isinstance(existing, dict):
                history = [existing]
        except (ValueError, OSError):
            history = []
    history.append(payload)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
    print('')
    print('appended to {} ({} runs recorded)'.format(path, len(history)))


async def _run(args: argparse.Namespace) -> int:
    import websockets
    from pipeline_address import resolve, describe as describe_address
    from pipeline_link import build, describe, opening_frame, warn_if_unprotected

    host, port = resolve(args.host, args.port)
    print(describe_address(host, port, args.host is not None))
    url, kwargs = build(host, port)
    warn_if_unprotected(url)
    print('connecting to {}'.format(describe(url)))

    async with websockets.connect(url, open_timeout=20, **kwargs) as ws:
        hello = opening_frame()
        if hello:
            await ws.send(hello)

        print('')
        print('network round trip, {} probes, no frame...'.format(args.probes))
        network = await _network_rtt(ws, args.probes)
        print('  p50 {:.0f}ms  p95 {:.0f}ms  min {:.0f}ms  max {:.0f}ms'.format(
            network['p50'], network['p95'], network['min'], network['max']))

        await ws.send(json.dumps({'action': 'set_source', 'path': args.source}))
        await asyncio.sleep(3.0)

        runs: List[Dict[str, Any]] = []
        for name in args.presets:
            runs.append(await _measure_preset(ws, name, args.seconds, args.frame))

    print('')
    print('{:<11} {:>9} {:>11} {:>9} {:>9} {:>9}'.format(
        'preset', 'Mbps up', 'delivered', 'p50', 'p95', 'min'))
    for run in runs:
        print('{:<11} {:>9.2f} {:>10.0f}% {:>8.0f}ms {:>8.0f}ms {:>8.0f}ms'.format(
            run['preset'], run['uplink_mbps'], run['delivered_pct'],
            run.get('p50_ms', 0), run.get('p95_ms', 0), run.get('min_ms', 0)))

    for line in _verdict(network, runs):
        print(line)

    if args.out:
        _write(args.out, {
            'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'pipeline': '{}:{}'.format(host, port),
            'network_rtt_ms': {k: round(v, 1) for k, v in network.items()},
            'presets': runs,
        })
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Measure what the link to a running pipeline delivers.')
    parser.add_argument('--host', help='override PHANTOM_API_URL from .env')
    parser.add_argument('--port', type=int, help='override the port')
    parser.add_argument('--presets', default='fast,optimal',
                        help='comma-separated, or "all". Measured in this order.')
    parser.add_argument('--seconds', type=float, default=40.0,
                        help='sampling window per preset, after warm-up')
    parser.add_argument('--probes', type=int, default=15,
                        help='network round-trip probes')
    parser.add_argument('--source', default=DEFAULT_POD_SOURCE,
                        help='source face, on the PIPELINE filesystem')
    parser.add_argument('--frame', default=DEFAULT_LOCAL_FRAME,
                        help='local image sent as the target; it needs a face')
    parser.add_argument('--out', help='append the run to this JSON file')
    parsed = parser.parse_args(argv)

    if str(parsed.presets).strip().lower() == 'all':
        names = list(PRESETS)
    else:
        names = [p.strip() for p in str(parsed.presets).split(',') if p.strip()]
    unknown = [p for p in names if p not in PRESETS]
    if unknown:
        print('unknown preset(s): {}. Known: {}'.format(
            ', '.join(unknown), ', '.join(PRESETS)))
        return 1
    parsed.presets = names

    try:
        return asyncio.run(_run(parsed))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
