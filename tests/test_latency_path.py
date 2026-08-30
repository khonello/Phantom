"""
The latency the client adds on top of the network.

The pipeline reports its own work — ~27ms on a 4090 — while an operator on a
remote pod reports "sluggish". Everything between those two numbers lives here:
the playout buffer, the inbound queue, and the uplink. These cover the parts
this codebase controls, and the reporting that makes the parts it does not
control visible.
"""

import queue

import pytest

from desktop.audio import JitterBuffer, RTTTracker

_MS = 1_000_000


def _tracker(rtts_ms, count=40):
    """A tracker fed `count` samples cycling through `rtts_ms`."""
    tracker = RTTTracker()
    for i in range(count):
        rtt = rtts_ms[i % len(rtts_ms)] * _MS
        tracker.record(1_000_000_000, 1_000_000_000 + rtt)
    return tracker


# ── The delay a fresh session starts with ──────────────────────────────


def test_initial_delay_is_not_half_a_second():
    """
    Every session pays this before a single RTT sample exists, and it used to
    be 400ms — long enough that a nearby pod felt heavily delayed for the first
    seconds, which is exactly when an impression forms.
    """
    assert RTTTracker.INITIAL_DELAY_NS <= 150 * _MS


def test_floor_is_about_one_frame_interval():
    """
    The floor is added even on a perfect link. Below one frame interval the
    buffer cannot absorb ordinary arrival quantisation; far above it, it is
    just latency. 20fps makes one interval 50ms.
    """
    assert 30 * _MS <= RTTTracker.FLOOR_NS <= 60 * _MS


def test_a_fresh_tracker_reports_the_initial_delay():
    assert RTTTracker().target_delay_ns == RTTTracker.INITIAL_DELAY_NS


# ── Adaptation ─────────────────────────────────────────────────────────


def test_a_slow_link_raises_the_target():
    tracker = _tracker([300, 320, 310, 330])
    assert tracker.target_delay_ns > RTTTracker.INITIAL_DELAY_NS


def test_a_fast_link_lowers_the_target_toward_the_floor():
    tracker = _tracker([20, 22, 21, 19], count=200)
    assert tracker.target_delay_ns < RTTTracker.INITIAL_DELAY_NS
    assert tracker.target_delay_ns >= RTTTracker.FLOOR_NS


def test_it_rises_faster_than_it_falls():
    """
    Asymmetric on purpose. A buffer that grows late glitches visibly; one that
    shrinks early underruns. A single alpha has to be slow in one direction,
    and the old symmetric 0.2 was slow in both.
    """
    assert RTTTracker.SMOOTHING_ALPHA_UP > RTTTracker.SMOOTHING_ALPHA_DOWN

    rising = RTTTracker()
    falling = RTTTracker()
    for _ in range(RTTTracker.UPDATE_INTERVAL * 2):
        rising.record(0 + 1, 1 + 400 * _MS)     # far above the initial delay
        falling.record(0 + 1, 1 + 10 * _MS)     # far below it

    rose = rising.target_delay_ns - RTTTracker.INITIAL_DELAY_NS
    fell = RTTTracker.INITIAL_DELAY_NS - falling.target_delay_ns
    assert rose > fell


def test_clamped_within_bounds():
    assert _tracker([10_000], count=200).target_delay_ns <= RTTTracker.CEILING_NS
    assert _tracker([1], count=200).target_delay_ns >= RTTTracker.FLOOR_NS


def test_a_negative_sample_is_ignored():
    """A clock anomaly must not poison the window."""
    tracker = RTTTracker()
    tracker.record(1_000, 500)
    assert tracker.target_delay_ns == RTTTracker.INITIAL_DELAY_NS


# ── What the readout reports ───────────────────────────────────────────


def test_stats_report_percentiles_not_just_the_mean():
    """
    A mean hides a tail, and a tail is what a playout buffer exists for. The
    readout has to carry p95 or "it feels fine but occasionally jumps" cannot
    be told apart from "it is uniformly slow".
    """
    buf = JitterBuffer()
    for rtt in (100, 100, 100, 100, 900):
        buf.push(1_000_000_000, b'x')
        buf._rtt.record(1_000_000_000, 1_000_000_000 + rtt * _MS)

    stats = buf.sync_stats()
    for key in ('rtt_p50_ms', 'rtt_p95_ms', 'target_delay_ms',
                'buffer_depth', 'rtt_samples', 'rtt_stddev_ms'):
        assert key in stats, key
    assert stats['rtt_p95_ms'] > stats['rtt_p50_ms']


def test_stats_are_safe_with_no_samples():
    stats = JitterBuffer().sync_stats()
    assert stats['rtt_samples'] == 0
    assert stats['rtt_p50_ms'] == 0.0
    assert stats['rtt_p95_ms'] == 0.0


# ── The inbound queue keeps the newest frame ───────────────────────────


def test_a_full_inbound_queue_evicts_the_oldest():
    """
    The pod's handler drops the *oldest* frame under pressure, not the
    arriving one. Dropping the arrival is backwards for a live call: it keeps a
    backlog of stale frames and discards the only current one, so the face lags
    by the whole queue depth and stays there.

    Mirrors `WebSocketAPIServer._on_binary`; kept as a behavioural statement
    because the property matters more than the call site.
    """
    fq: 'queue.Queue[tuple]' = queue.Queue(maxsize=2)

    def offer(item):
        try:
            fq.put_nowait(item)
        except queue.Full:
            try:
                fq.get_nowait()
            except queue.Empty:
                pass
            try:
                fq.put_nowait(item)
            except queue.Full:
                pass

    for seq in range(5):
        offer((seq, b''))

    held = [fq.get_nowait()[0] for _ in range(fq.qsize())]
    assert held == [3, 4], 'the queue must hold the newest frames, not the first two'


def test_inbound_queue_depth_is_small():
    """
    Anything waiting in this queue is a frame the operator has already moved
    past. Ten deep was half a second at 20fps — felt as lag, not seen as
    stutter.
    """
    import inspect

    from pipeline.api import server

    src = inspect.getsource(server)
    assert 'queue.Queue(maxsize=2)' in src
    assert 'queue.Queue(maxsize=10)' not in src


# ── Uplink accounting ──────────────────────────────────────────────────


@pytest.mark.parametrize('kb_per_frame, fps, expected_mbps', [
    (30, 20, 4.8),   # 640x360 q70 at optimal — plausibly saturates a home uplink
    (12, 15, 1.44),  # 480x270 q60, the fast preset
])
def test_uplink_arithmetic(kb_per_frame, fps, expected_mbps):
    """
    The number that decides whether the link or the GPU is the problem. Sending
    a JPEG per captured frame upstream is the direction most likely to
    saturate, and saturation shows as latency rather than as stutter.
    """
    mbps = (kb_per_frame * 1000 * fps * 8) / 1_000_000
    assert mbps == pytest.approx(expected_mbps, rel=0.01)
