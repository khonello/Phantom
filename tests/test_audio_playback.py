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

from desktop.audio import AudioPlayback, AudioRingBuffer

_RATE = 44100
_CHUNK = 1024
_CHUNK_NS = int(_CHUNK / _RATE * 1_000_000_000)


class _FakeJitter:
    """Stands in for JitterBuffer, exposing only the delay playback reads."""

    def __init__(self, delay_ns):
        self.target_delay_ns = delay_ns


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
