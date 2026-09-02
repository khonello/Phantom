"""
Audio playout: continuity.

Video can drop a frame and nobody notices. A 23ms hole in speech is a click,
and a repeated 23ms is a stutter — so the two cannot share a read strategy.
Playback used to re-derive its position every block from `now - target_delay`,
against a target the *video* RTT tracker moves continuously. Real telemetry
from a session showed that target swinging 380 → 500 → 420 → 490ms every two
seconds; each swing discarded audio or inserted silence.

These pin the properties that make it continuous instead.
"""

import numpy as np
import pytest

from desktop.audio import AudioPlayback, AudioRingBuffer, JitterBuffer

_RATE = 44100
_CHUNK = 1024
_CHUNK_NS = int(_CHUNK / _RATE * 1_000_000_000)


class _FakeJitter:
    """Stands in for JitterBuffer, exposing only what playback reads."""

    def __init__(self, delay_ns, epoch=0):
        self.target_delay_ns = delay_ns
        # Bumped when D is decided rather than estimated. Playback follows this
        # and never the delay itself — see the tests below.
        self.delay_epoch = epoch


def _pcm(value):
    return np.full((_CHUNK, 1), float(value), dtype=np.float32)


def _fill(ring, count, start_ts=0):
    """Contiguous chunks, each tagged with a distinct sample value."""
    for i in range(count):
        ring.append(start_ts + i * _CHUNK_NS, _pcm(i + 1))


@pytest.fixture
def rig():
    ring = AudioRingBuffer(max_chunks=200, sample_rate=_RATE)
    jitter = _FakeJitter(200_000_000)
    playback = AudioPlayback(ring, jitter, sample_rate=_RATE,
                             channels=1, block_size=_CHUNK)
    # Attached and already aware of this buffer's delay generation, so the
    # tests below exercise the seek, trim and resync paths rather than the
    # one-off reposition a fresh epoch triggers.
    playback._delay_epoch = jitter.delay_epoch
    return playback, ring, jitter


def _pull(playback, frames=_CHUNK):
    out = np.zeros((frames, 1), dtype=np.float32)
    playback._output_callback(out, frames, None, None)
    return out


# ── Continuity ─────────────────────────────────────────────────────────


def test_consecutive_blocks_are_contiguous(rig, monkeypatch):
    """
    Every sample captured, in order, with nothing dropped or repeated. This is
    the whole property — the old callback could not offer it, because each
    block re-derived where to read from.
    """
    playback, ring, _ = rig
    _fill(ring, 8)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    values = []
    for _ in range(6):
        block = _pull(playback)
        values.append(float(block[0][0]))

    assert values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_moving_target_delay_does_not_break_the_stream(rig, monkeypatch):
    """
    The measured session swung the target 380 → 500 → 420 → 490ms. Playback
    must not follow that: the delay estimate exists for video, and audio has
    already been positioned.
    """
    playback, ring, jitter = rig
    _fill(ring, 12)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    values = []
    for delay_ms in (380, 500, 420, 490, 400, 460):
        jitter.target_delay_ns = delay_ms * 1_000_000
        values.append(float(_pull(playback)[0][0]))

    assert values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (
        'playback followed the video delay estimate and lost continuity'
    )


def test_partial_chunks_carry_over(rig, monkeypatch):
    """A block smaller than a chunk must resume mid-chunk, not restart it."""
    playback, ring, _ = rig
    _fill(ring, 4)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    first = _pull(playback, frames=_CHUNK // 2)
    second = _pull(playback, frames=_CHUNK // 2)
    third = _pull(playback, frames=_CHUNK // 2)

    assert float(first[0][0]) == 1.0
    assert float(second[0][0]) == 1.0   # same chunk, second half
    assert float(third[0][0]) == 2.0    # next chunk


# ── Seeking into a chunk, not to its edge ──────────────────────────────


def test_seek_lands_inside_the_straddling_chunk(rig, monkeypatch):
    """
    A chunk that straddles the playback point used to be played from its first
    sample, so the position quantised to a ~23ms boundary and jumped whenever
    the boundary drifted past. That is the missing-pieces symptom.
    """
    playback, ring, jitter = rig
    _fill(ring, 10)
    jitter.target_delay_ns = 0
    # Half-way into chunk index 3.
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 3 * _CHUNK_NS + _CHUNK_NS // 2)

    playback._needs_seek = True
    block = _pull(playback)

    assert float(block[0][0]) == 4.0, 'should resume inside chunk 4'
    assert playback._underruns == 0
    # The second half of chunk 4 is ~512 samples, so chunk 5 follows within
    # the same block — proving it did not restart at a chunk edge.
    assert float(block[-1][0]) == 5.0


# ── Underrun ───────────────────────────────────────────────────────────


def test_empty_ring_is_silence_and_is_counted(rig, monkeypatch):
    playback, ring, _ = rig
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    block = _pull(playback)
    assert not block.any()
    assert playback._underruns == 1


def test_one_gap_stays_one_gap(rig, monkeypatch):
    """
    After silence, the next block continues from where audio resumes rather
    than from a recomputed clock position — so an underrun costs exactly the
    audio that was missing.
    """
    playback, ring, _ = rig
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    _pull(playback)                      # underrun
    _fill(ring, 3)
    values = [float(_pull(playback)[0][0]) for _ in range(3)]

    assert values == [1.0, 2.0, 3.0]
    assert playback._underruns == 1


# ── Latency is corrected by depth, rarely ──────────────────────────────


def test_a_deep_backlog_is_trimmed(rig, monkeypatch):
    """
    Capture outrunning playback must not grow the delay without bound. The
    correction is a trim, taken outside the fill so it is rare rather than
    per-block.
    """
    playback, ring, jitter = rig
    jitter.target_delay_ns = 100_000_000
    # ~580ms queued against a 100ms target: past the 250ms trim tolerance, but
    # well short of the resync threshold, so this exercises the trim and not
    # the "cursor is meaningless, start again" path.
    _fill(ring, 25)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 25 * _CHUNK_NS)

    playback._needs_seek = False
    before = playback._buffered_ns()
    _pull(playback)

    assert playback._trims == 1
    assert playback._resyncs == 0
    assert playback._buffered_ns() < before


def test_a_stalled_cursor_resyncs_rather_than_trims(rig, monkeypatch):
    """
    Past a point the cursor means nothing — the device stalled, or capture
    stopped — and continuing would play something seconds old.
    """
    playback, ring, jitter = rig
    jitter.target_delay_ns = 100_000_000
    _fill(ring, 200)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 200 * _CHUNK_NS)

    playback._needs_seek = False
    _pull(playback)

    assert playback._resyncs == 1


def test_a_normal_depth_is_not_trimmed(rig, monkeypatch):
    """A trim is audible, so ordinary jitter must not cause one."""
    playback, ring, jitter = rig
    jitter.target_delay_ns = 200_000_000
    _fill(ring, 10)                       # ~230ms, around the target
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    playback._needs_seek = False
    for _ in range(4):
        _pull(playback)

    assert playback._trims == 0


def test_stats_report_the_counters():
    ring = AudioRingBuffer(max_chunks=10, sample_rate=_RATE)
    playback = AudioPlayback(ring, _FakeJitter(0), sample_rate=_RATE,
                             channels=1, block_size=_CHUNK)
    stats = playback.stats()
    for key in ('buffered_ms', 'underruns', 'trims', 'resyncs'):
        assert key in stats


# ── Where the delayed audio goes ───────────────────────────────────────


def _fake_devices(monkeypatch, devices):
    """Stand in for sounddevice.query_devices()."""
    import types

    fake = types.SimpleNamespace(
        query_devices=lambda index=None: (devices if index is None
                                          else devices[index]),
    )
    monkeypatch.setitem(__import__('sys').modules, 'sounddevice', fake)


def test_finds_a_virtual_cable(monkeypatch):
    """
    The counterpart of the virtual camera. Without it the delayed audio goes to
    the operator's speakers while the call still receives their real microphone
    undelayed — so the delay makes the desync worse, not better.
    """
    from desktop import audio

    _fake_devices(monkeypatch, [
        {'name': 'Speakers (Realtek)', 'max_output_channels': 2},
        {'name': 'CABLE Input (VB-Audio Virtual Cable)', 'max_output_channels': 2},
    ])
    assert audio.find_virtual_output() == 1


def test_prefers_a_real_cable_over_a_loose_name_match(monkeypatch):
    """`virtual` is the loosest hint and must lose to an actual cable."""
    from desktop import audio

    _fake_devices(monkeypatch, [
        {'name': 'Some Virtual Thing', 'max_output_channels': 2},
        {'name': 'CABLE Input (VB-Audio)', 'max_output_channels': 2},
    ])
    assert audio.find_virtual_output() == 1


def test_prefers_the_lowest_latency_instance(monkeypatch):
    """
    Windows lists the same device once per host API and enumeration puts MME
    first. On a real machine the same VB-Audio cable offered 90ms on MME,
    120ms on DirectSound and 2ms on WASAPI — so taking the first match added
    88ms to a path already carrying a 350ms round trip, and it would have read
    as more of the latency everything else was being blamed for.
    """
    from desktop import audio

    _fake_devices(monkeypatch, [
        {'name': 'CABLE Input (VB-Audio Virtual C', 'max_output_channels': 2,
         'default_low_output_latency': 0.090},                       # MME
        {'name': 'CABLE Input (VB-Audio Virtual Cable)', 'max_output_channels': 2,
         'default_low_output_latency': 0.120},                       # DirectSound
        {'name': 'CABLE Input (VB-Audio Virtual Cable)', 'max_output_channels': 2,
         'default_low_output_latency': 0.002},                       # WASAPI
    ])
    assert audio.find_virtual_output() == 2


def test_a_missing_latency_field_does_not_break_selection(monkeypatch):
    """Not every backend reports one; absence must not exclude the device."""
    from desktop import audio

    _fake_devices(monkeypatch, [
        {'name': 'CABLE Input', 'max_output_channels': 2},
    ])
    assert audio.find_virtual_output() == 0


def test_ignores_input_only_devices(monkeypatch):
    """A virtual *microphone* is not somewhere to write audio."""
    from desktop import audio

    _fake_devices(monkeypatch, [
        {'name': 'CABLE Output (VB-Audio)', 'max_output_channels': 0},
    ])
    assert audio.find_virtual_output() is None


def test_none_when_nothing_is_installed(monkeypatch):
    from desktop import audio

    _fake_devices(monkeypatch, [
        {'name': 'Speakers', 'max_output_channels': 2},
        {'name': 'Headphones', 'max_output_channels': 2},
    ])
    assert audio.find_virtual_output() is None


def test_discovery_survives_no_sounddevice(monkeypatch):
    """Audio is optional; missing it must not take the application down."""
    import sys

    from desktop import audio

    monkeypatch.setitem(sys.modules, 'sounddevice', None)
    assert audio.find_virtual_output() is None


def test_stats_name_the_output(rig):
    playback, _ring, _jitter = rig
    stats = playback.stats()
    assert 'device' in stats and 'virtual' in stats
    assert stats['virtual'] is False


# ── The delay changes exactly once, and says so ────────────────────────


def test_a_delay_epoch_change_repositions_the_cursor(rig, monkeypatch):
    """
    Calibration commits a delay measured from the link, and escalation raises
    it. The cursor is continuous precisely so it does not chase an *estimate* —
    but a committed decision is not an estimate, and audio has to land on the
    new number rather than sit at the old one for the rest of the session.
    """
    playback, ring, jitter = rig
    _fill(ring, 20)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    assert float(_pull(playback)[0][0]) == 1.0
    assert float(_pull(playback)[0][0]) == 2.0

    jitter.target_delay_ns = 0
    jitter.delay_epoch += 1
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 10 * _CHUNK_NS)

    assert float(_pull(playback)[0][0]) == 11.0, 'did not follow the new delay'
    assert float(_pull(playback)[0][0]) == 12.0, 'not continuous afterwards'


def test_the_delay_alone_never_moves_the_cursor(rig, monkeypatch):
    """
    The converse, and the reason the epoch exists at all. A delay that moved on
    its own would be followed per block, which is the skip-and-silence the
    fixed delay was introduced to remove.
    """
    playback, ring, jitter = rig
    _fill(ring, 12)
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns', lambda: 0)

    values = []
    for delay_ms in (380, 500, 420, 490, 400, 460):
        jitter.target_delay_ns = delay_ms * 1_000_000   # epoch unchanged
        values.append(float(_pull(playback)[0][0]))

    assert values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_output_device_latency_is_taken_off_the_read_point(rig, monkeypatch):
    """
    A block written here is audible once the device has played what it already
    holds, so the real delay was D *plus* the device — and the same VB-Audio
    cable reports 2ms on WASAPI and 90ms on MME. `find_virtual_output` picks
    the low-latency instance; this accounts for whatever is left.
    """
    playback, ring, jitter = rig
    _fill(ring, 40)
    jitter.target_delay_ns = 10 * _CHUNK_NS
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: 20 * _CHUNK_NS)

    playback._output_latency_ns = 5 * _CHUNK_NS
    playback._needs_seek = True

    # 20 - 10 + 5 lands on chunk index 15, whose value is 16.
    assert float(_pull(playback)[0][0]) == 16.0


# ── The two streams present the same moment ────────────────────────────


def test_video_and_audio_present_the_same_capture_instant(monkeypatch):
    """
    The test whose absence let the split ship.

    Both streams claim to present `capture + D`. Nothing ever compared what
    they *actually* presented, so video reading the adaptive estimate while
    audio read the fixed delay survived a test suite, a log line and a badge —
    and reached a live call as speech arriving after the lips had stopped.

    Video is allowed to be up to one frame interval older, because frames are
    discrete and the newest eligible one may have been captured an interval
    before the deadline. Anything beyond that is the fault this pins.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    delay = 20 * _CHUNK_NS
    jitter = JitterBuffer(fixed_delay_ns=delay, calibrate=False)
    ring = AudioRingBuffer(max_chunks=200, sample_rate=_RATE)
    playback = AudioPlayback(ring, jitter, sample_rate=_RATE, channels=1,
                             block_size=_CHUNK)

    # The same capture instants on both streams.
    for i in range(60):
        ts = i * _CHUNK_NS + 1
        ring.append(ts, _pcm(i + 1))
        jitter.push(ts, b'v')

    clock['now'] = 40 * _CHUNK_NS
    jitter.next_for_slot()
    _pull(playback)

    video_ms = jitter.sync_stats()['video_age_ms']
    audio_ms = playback.stats()['audio_age_ms']
    interval_ms = _CHUNK_NS / 1_000_000

    assert video_ms > 0 and audio_ms > 0
    assert abs(video_ms - audio_ms) <= interval_ms * 1.5, (
        'video +{}ms against audio +{}ms'.format(video_ms, audio_ms)
    )


def test_the_skew_is_visible_when_the_delays_disagree(monkeypatch):
    """
    And it has to *fail* when they do — a readout that always says 'aligned' is
    the state this was in before.
    """
    clock = {'now': 0}
    monkeypatch.setattr('desktop.audio.time.perf_counter_ns',
                        lambda: clock['now'])

    jitter = JitterBuffer(fixed_delay_ns=20 * _CHUNK_NS, calibrate=False)
    ring = AudioRingBuffer(max_chunks=200, sample_rate=_RATE)
    playback = AudioPlayback(ring, jitter, sample_rate=_RATE, channels=1,
                             block_size=_CHUNK)

    for i in range(60):
        ts = i * _CHUNK_NS + 1
        ring.append(ts, _pcm(i + 1))
        jitter.push(ts, b'v')

    clock['now'] = 40 * _CHUNK_NS
    jitter.next_for_slot()

    # Audio alone held twice as long — the shape of the original defect.
    jitter._fixed_delay_ns = 40 * _CHUNK_NS
    playback._needs_seek = True
    _pull(playback)

    skew = (jitter.sync_stats()['video_age_ms']
            - playback.stats()['audio_age_ms'])
    assert skew < -10 * _CHUNK_NS / 1_000_000, (
        'a several-chunk split reported as aligned'
    )
