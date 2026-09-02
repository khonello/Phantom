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

from desktop.audio import (
    DEFAULT_PLAYOUT_DELAY_NS, JitterBuffer, RTTTracker,
)

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


# ── Fixed playout: one delay, held for both streams ────────────────────


def test_the_delay_is_fixed_by_default():
    """
    Adaptive is right for video alone and wrong once audio shares the clock:
    every adjustment becomes a skip or a silence. Jitter is more damaging than
    delay — people adapt to a constant 550ms and never to one that moves.

    550ms is now the *provisional* value, replaced once by a measured one — but
    only after a full RTT window, which six samples is not.
    """
    buf = JitterBuffer()
    assert buf.target_delay_ns == DEFAULT_PLAYOUT_DELAY_NS

    for rtt in (20, 25, 22, 900, 30, 28):
        buf._rtt.record(1_000_000_000, 1_000_000_000 + rtt * _MS)

    assert buf.target_delay_ns == DEFAULT_PLAYOUT_DELAY_NS, (
        'the delay must not follow the network once it is fixed'
    )


def test_rtt_is_still_measured_while_fixed():
    """The telemetry says whether the fixed value is still right — keep it."""
    buf = JitterBuffer()
    for _ in range(20):
        buf._rtt.record(1_000_000_000, 1_000_000_000 + 300 * _MS)
    assert buf.sync_stats()['rtt_p50_ms'] == pytest.approx(300, rel=0.05)


def test_adaptive_is_still_reachable():
    buf = JitterBuffer(fixed_delay_ns=0)
    assert buf.target_delay_ns == RTTTracker.INITIAL_DELAY_NS


# ── The slot rule ──────────────────────────────────────────────────────


def _buf_with(frames, now, delay_ms=100):
    buf = JitterBuffer(fixed_delay_ns=delay_ms * _MS, calibrate=False)
    for ts, payload in frames:
        buf.push(ts, payload)
    return buf


def test_an_on_time_frame_is_shown(monkeypatch):
    buf = _buf_with([(0, b'a')], now=0)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 200 * _MS)
    assert buf.next_for_slot() == (0, b'a')


def test_an_empty_slot_repeats_the_last_shown_frame(monkeypatch):
    """
    The slot fires on time regardless. Waiting would slip the schedule, and
    audio is locked to the same clock — a stall is a gap in speech.
    """
    buf = _buf_with([(0, b'a')], now=0)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 200 * _MS)

    assert buf.next_for_slot() == (0, b'a')
    assert buf.next_for_slot() == (0, b'a')     # nothing new arrived
    assert buf.next_for_slot() == (0, b'a')
    assert buf.sync_stats()['repeats'] == 2


def test_nothing_to_show_before_the_first_frame(monkeypatch):
    """No frame has ever arrived, so there is nothing to repeat."""
    buf = JitterBuffer()
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)
    assert buf.next_for_slot() is None
    assert buf.sync_stats()['repeats'] == 0


def test_a_late_frame_does_not_displace_a_newer_one(monkeypatch):
    """
    A frame that missed its slot is stale. Showing it would shift everything
    one slot later and the pattern would never recover — the next frame to
    show is the next one that is on time.
    """
    buf = _buf_with([(0, b'old'), (100 * _MS, b'new')], now=0)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 400 * _MS)

    assert buf.next_for_slot() == (100 * _MS, b'new')


def test_clear_forgets_the_held_frame(monkeypatch):
    """It belongs to the session that ended; repeating it would be a stale face."""
    buf = _buf_with([(0, b'a')], now=0)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 200 * _MS)
    buf.next_for_slot()
    buf.clear()
    assert buf.next_for_slot() is None


# ── Escalation happens on evidence, not per frame ──────────────────────


def test_sustained_repeats_raise_the_delay(monkeypatch):
    """
    One repeat is invisible; a sustained rate is a frozen face while audio
    continues, which reads as a broken swap rather than a slow link.
    """
    buf = _buf_with([(0, b'a')], now=0)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 200 * _MS)
    before = buf.target_delay_ns

    for _ in range(400):
        buf.next_for_slot()

    assert buf.target_delay_ns > before
    assert buf.sync_stats()['escalations'] >= 1


def test_a_healthy_stream_never_escalates(monkeypatch):
    """A step is visible, so ordinary operation must not cause one."""
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    buf = JitterBuffer(fixed_delay_ns=100 * _MS)
    for i in range(400):
        buf.push(i * 30 * _MS, b'f')
        clock['now'] = i * 30 * _MS + 150 * _MS
        buf.next_for_slot()

    assert buf.sync_stats()['escalations'] == 0


# ── One delay, and it is the one both streams read ─────────────────────


def test_release_uses_the_delay_in_force_not_the_rtt_estimate(monkeypatch):
    """
    The defect this file exists to prevent recurring.

    `pop_eligible` read `self._rtt.target_delay_ns` while audio positioned
    itself against `self.target_delay_ns`. Video was therefore released on
    arrival while audio waited the fixed delay, and on a call the sound trailed
    the lips by the difference — several hundred milliseconds on a link whose
    RTT is nowhere near the fixed value.

    Nothing caught it because every existing assertion advanced the clock past
    *both* numbers. This one drives them apart on purpose.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    buf = JitterBuffer(fixed_delay_ns=600 * _MS, calibrate=False)
    for _ in range(200):
        buf._rtt.record(1_000_000_000, 1_000_000_000 + 20 * _MS)
    assert buf._rtt.target_delay_ns < 100 * _MS, 'estimate not driven apart'

    buf.push(1, b'frame')

    clock['now'] = 300 * _MS
    assert buf.next_for_slot() is None, (
        'released against the RTT estimate instead of the delay in force'
    )

    clock['now'] = 650 * _MS
    assert buf.next_for_slot() == (1, b'frame')


def test_stats_report_the_delay_actually_in_force():
    """
    The readout named the adaptive estimate, so `[SYNC] delay=` and the badge
    never showed the number either stream was held to — which is most of why
    the split above survived so long. Both numbers are reported now, because
    the interesting state is when they disagree.
    """
    buf = JitterBuffer(fixed_delay_ns=600 * _MS, calibrate=False)
    for _ in range(200):
        buf._rtt.record(1_000_000_000, 1_000_000_000 + 20 * _MS)

    stats = buf.sync_stats()
    assert stats['target_delay_ms'] == 600.0
    assert stats['rtt_target_ms'] < 600.0


def test_escalation_moves_the_delay_video_reads(monkeypatch):
    """
    Escalation raised `_fixed_delay_ns`, which only audio read — so a link bad
    enough to trigger it pushed audio further behind the picture while doing
    nothing about the repeats it was raised to steady.
    """
    clock = {'now': 200 * _MS}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    buf = _buf_with([(1, b'a')], now=0)
    assert buf.next_for_slot() == (1, b'a')
    for _ in range(400):
        buf.next_for_slot()

    assert buf.sync_stats()['escalations'] >= 1
    assert buf.target_delay_ns > 100 * _MS

    # A frame younger than the raised delay must now be held back too.
    buf.push(clock['now'] - 150 * _MS, b'b')
    assert buf.next_for_slot() == (1, b'a'), 'video ignored the raised delay'


# ── Calibrate once, then freeze ────────────────────────────────────────


def _calibrated(monkeypatch, rtt_ns=200 * _MS, frames=120):
    """A buffer fed `frames` frames at a steady round trip.

    Enough to clear the warm-up discard and then fill a whole window.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])
    buf = JitterBuffer()
    # The clock starts past the round trip, so no frame carries a capture
    # stamp at or below zero — which `RTTTracker.record` discards as a clock
    # anomaly and `pop_eligible` treats as a legacy frame.
    for i in range(1, frames + 1):
        clock['now'] = rtt_ns + i * 50 * _MS
        buf.push(clock['now'] - rtt_ns, b'f')
    return buf, clock


def test_calibration_commits_once_and_then_holds(monkeypatch):
    """
    D cannot be smaller than what video costs — a frame cannot be shown before
    it arrives. What it must not be is *larger*, which is what a constant tuned
    against one datacenter guarantees everywhere else: 550ms against a 200ms
    link holds audio a third of a second longer than the picture needs.

    p95 (200) + the spread floor (50) + margin (80) = 330, quantised to 325.
    """
    buf, clock = _calibrated(monkeypatch)

    assert buf.calibrated
    assert buf.target_delay_ns == 325 * _MS
    assert buf.target_delay_ns < DEFAULT_PLAYOUT_DELAY_NS

    for i in range(121, 1121):
        clock['now'] = 200 * _MS + i * 50 * _MS
        buf.push(clock['now'] - 200 * _MS, b'f')

    assert buf.target_delay_ns == 325 * _MS, 'D moved after it was committed'


def test_calibration_moves_the_epoch_once(monkeypatch):
    """Audio's cursor will not follow a changing delay; the epoch tells it to."""
    buf, clock = _calibrated(monkeypatch)
    epoch = buf.delay_epoch
    assert epoch > 0

    for i in range(121, 460):
        clock['now'] = 200 * _MS + i * 50 * _MS
        buf.push(clock['now'] - 200 * _MS, b'f')

    assert buf.delay_epoch == epoch, 'one commit, one reposition'


def test_a_new_session_is_measured_again(monkeypatch):
    """
    `clear()` runs on reconnect, and a new pod is a new link — inheriting the
    last one's answer is how a delay measured in Romania ends up in force
    somewhere else.
    """
    buf, _clock = _calibrated(monkeypatch)
    epoch = buf.delay_epoch

    buf.clear()

    assert not buf.calibrated
    assert buf.target_delay_ns == DEFAULT_PLAYOUT_DELAY_NS
    assert buf.delay_epoch > epoch, 'audio was not told the number changed'


def test_a_pinned_delay_is_never_calibrated(monkeypatch):
    """
    A measurement run pins it: two sessions cannot be compared if the delay
    chose itself differently in each.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])
    buf = JitterBuffer(fixed_delay_ns=400 * _MS, calibrate=False)
    for i in range(1, 201):
        clock['now'] = 200 * _MS + i * 50 * _MS
        buf.push(clock['now'] - 200 * _MS, b'f')

    assert not buf.calibrated
    assert buf.target_delay_ns == 400 * _MS
    assert buf.delay_epoch == 0


# ── What calibration must refuse to believe ────────────────────────────


def test_warm_up_round_trips_do_not_set_the_delay(monkeypatch):
    """
    The first stream after a pipeline start pays model warm-up — tens of
    seconds — and those frames come back carrying round trips of exactly that
    size. Calibrating on them put D straight onto the ceiling, and a delay the
    buffer cannot hold is a picture that stops moving. It happened on the first
    real run.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])
    buf = JitterBuffer()

    for i in range(1, 31):                      # warm-up: 20s round trips
        clock['now'] = 30_000 * _MS + i * _MS
        buf.push(clock['now'] - 20_000 * _MS, b'f')
    for i in range(1, 200):                     # then a steady 150ms link
        clock['now'] = 60_000 * _MS + i * 50 * _MS
        buf.push(clock['now'] - 150 * _MS, b'f')

    assert buf.calibrated
    assert buf.target_delay_ns == 275 * _MS, (
        'calibrated on the model load rather than the link'
    )


def test_an_unsettled_link_keeps_the_provisional_delay(monkeypatch):
    """
    A tail far above the middle describes a regime change, not a link. Better
    to hold the conservative provisional and let escalation raise it on real
    starvation than to freeze a number onto an artefact.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])
    buf = JitterBuffer()

    for i in range(1, 1000):
        clock['now'] = 100_000 * _MS + i * 50 * _MS
        rtt = 1_500 * _MS if i % 5 == 0 else 100 * _MS
        buf.push(clock['now'] - rtt, b'f')

    assert buf.calibrated, 'it must give up rather than retry forever'
    assert buf.target_delay_ns == DEFAULT_PLAYOUT_DELAY_NS
    assert buf.delay_epoch == 0


# ── A quiet slot is not a starving one ─────────────────────────────────


def test_a_stream_slower_than_the_display_tick_never_escalates(monkeypatch):
    """
    The runaway this pins.

    The display ticks at 30/s while `optimal` streams at 20fps, so a third of
    all slots have no new frame due — on a perfect link, forever. Escalation
    counted those as evidence the link needed headroom, which was harmless only
    for as long as video ignored the number it raised. Pointing video at it sent
    D to the 2s ceiling and froze the picture.

    Frames are waiting in the buffer at every one of those slots. That is D
    being generous, and raising it is precisely backwards.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    start = 1_000 * _MS
    tick, frame, rtt = 33 * _MS, 50 * _MS, 100 * _MS
    buf = JitterBuffer(fixed_delay_ns=200 * _MS, calibrate=False)

    sent = 0
    for i in range(900):                         # ~30s, three whole windows
        clock['now'] = start + i * tick
        while start + sent * frame + rtt <= clock['now']:
            buf.push(start + sent * frame, b'f')
            sent += 1
        buf.next_for_slot()

    stats = buf.sync_stats()
    assert stats['repeats'] > 0, 'the mismatch this is about did not happen'
    assert stats['escalations'] == 0, 'a healthy 20fps stream escalated'
    assert buf.target_delay_ns == 200 * _MS


def test_a_dead_stream_does_not_escalate(monkeypatch):
    """
    Nothing arrived, so there is nothing a bigger buffer would have held.
    Stepping D for a dead pipeline only delays the picture that comes back.
    """
    clock = {'now': 200 * _MS}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    buf = _buf_with([(1, b'a')], now=0)
    buf.next_for_slot()
    buf._pushes = 0                              # the push above is spent
    for _ in range(900):
        buf.next_for_slot()

    assert buf.sync_stats()['escalations'] == 0


# ── A delay the buffer cannot hold ─────────────────────────────────────


def test_a_full_buffer_shows_the_oldest_rather_than_freezing(monkeypatch):
    """
    Once the buffer is full the next push evicts the frame at the front before
    it reaches its deadline — and the one after that. That is a picture which
    never moves again. Early by a fraction of D beats frozen, and it is counted
    so the cause is not read as a network fault.
    """
    clock = {'now': 10_000 * _MS}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    buf = JitterBuffer(fixed_delay_ns=2_000 * _MS, calibrate=False)
    for i in range(JitterBuffer.MAX_FRAMES + 20):
        buf.push(clock['now'] - 500 * _MS + i, b'f')

    assert buf.next_for_slot() is not None, 'a full buffer froze the picture'
    assert buf.sync_stats()['forced'] > 0


def test_the_buffer_outlasts_the_escalation_ceiling():
    """
    D holds `(D - rtt) * fps` frames in flight, so a buffer smaller than that
    evicts frames before they are ever shown. 60 was exactly 2s at 30fps — the
    escalation ceiling, with no headroom at all.
    """
    fps = 30
    ceiling_s = 2.0
    assert JitterBuffer.MAX_FRAMES > ceiling_s * fps
