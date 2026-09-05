"""
Local audio capture and A/V synchronized playback.

Audio is captured from the local microphone with precise timestamps aligned to
the same monotonic clock (time.perf_counter_ns) used for video frame capture.
Audio never leaves the local machine — it is buffered here and played back in
sync with processed video frames returned from the remote GPU.

Components:
- AudioRingBuffer: thread-safe deque of timestamped PCM chunks
- AudioCapture: sounddevice.InputStream wrapper, stores (capture_ts, pcm) chunks
- RTTTracker: sliding-window RTT estimator, computes adaptive playout delay
- JitterBuffer: FIFO video frame buffer with timed release based on RTT
- AudioPlayback: sounddevice.OutputStream that reads audio at the playout offset

Requires: sounddevice (pip install sounddevice)
"""

import collections
import sys
import time
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from desktop.voice import VoiceTransformer

# PCM chunk: (capture_ts_ns, pcm_data as float32 numpy array)
AudioChunk = Tuple[int, npt.NDArray[Any]]

# Default audio parameters.
#
# 48 kHz rather than 44.1: this is a fallback for when a device will not say
# what it wants, and virtual cables overwhelmingly want 48. `resolve_sample_rate`
# asks the device first, and should almost always answer instead of this.
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEFAULT_BLOCK_SIZE = 1024  # ~21ms at 48000 Hz
DEFAULT_BUFFER_SECONDS = 10  # ring buffer capacity

# Fixed playout delay, in nanoseconds. Every frame and every audio sample is
# presented exactly this long after it was captured, whatever the network did
# in between.
#
# **Fixed, not adaptive, because audio shares the clock.** An adaptive delay is
# right for video alone — it chases the network and the viewer sees nothing.
# The moment audio is played against the same number, every adjustment becomes
# a discontinuity: move the read point forward and samples are skipped, move it
# back and silence is inserted. A measured session had the adaptive target
# swinging 380 -> 500 -> 420 -> 490ms every couple of seconds, and that is what
# an operator hears as speech breaking up. Jitter is far more damaging than
# delay: people adapt to a constant 550ms, nobody adapts to a delay that moves.
#
# 550ms is chosen from measurement, not taste: RTT p50 ~350ms, p95 ~450ms, with
# occasional 700ms outliers. It covers p95 with margin, and it is barely above
# what the adaptive buffer was already averaging — so it costs almost nothing
# in latency and removes the variance entirely.
#
# Set 0 to restore the adaptive behaviour.
DEFAULT_PLAYOUT_DELAY_NS = 550_000_000

# D is *measured*, then frozen — not assumed.
#
# 550ms above was picked from one session against one datacenter, and there was
# no way to change it short of editing this line. On a link with a 350ms round
# trip that holds audio 200ms longer than video actually costs, which is heard
# as the sound trailing the lips. Calibration keeps the property that matters —
# D never moves once speech is running — while letting the number come from the
# link rather than from a constant: measure for the first `_CALIBRATE_SAMPLES`
# frames, commit once, and hold.
#
# The floor is not `RTTTracker.FLOOR_NS`. D also has to absorb the 33ms display
# tick and the virtual camera's own queue, neither of which is RTT, so a
# calibrated delay below ~100ms would freeze the picture to save latency
# nothing can perceive.
# **The warm-up frames describe the model load, not the link.** The first
# stream after a pipeline start pays tens of seconds of model warm-up, and those
# frames come back carrying round trips of exactly that size. Calibrating on
# them puts D straight onto the ceiling and freezes the picture — which is what
# it did the first time this ran. They are discarded, and the window that
# follows has to look settled before it is believed.
_CALIBRATE_SKIP = 30                 # frames discarded before measuring
_CALIBRATE_SAMPLES = 60              # a full RTT window, ~3s at 20fps
_CALIBRATE_MARGIN_NS = 80_000_000    # headroom above p95, so slots stay fed
_CALIBRATE_SPREAD_FLOOR_NS = 50_000_000   # one frame interval at 20fps
_CALIBRATE_FLOOR_NS = 100_000_000
# A window whose tail is more than this much worse than its middle is not a
# link, it is a regime change — a warm-up ending, a reconnect, a stall. And a
# round trip beyond `_CALIBRATE_MAX_NS` is not something to calibrate *to*;
# escalation exists for links genuinely that bad, and it steps on evidence.
_CALIBRATE_SPREAD_RATIO = 2.0
_CALIBRATE_MAX_NS = 1_000_000_000
_CALIBRATE_ATTEMPTS = 5              # then keep the provisional and say so
_DELAY_QUANTUM_NS = 25_000_000       # a delay reported to 1ms invites tuning it

# Raising the delay is a visible step, so it happens on evidence and rarely:
# only when slots have been **starving** — frames arriving already past their
# deadline, which is the one thing more headroom fixes.
#
# It used to count *repeats*, and that was structurally always true. The
# display ticks at 30/s while the stream runs at the preset's rate — 20fps at
# `optimal`, 15 at `fast` — so a third to a half of all slots have no new frame
# due, on a perfect link, forever. The condition never stopped being met.
#
# That was invisible while `_fixed_delay_ns` was a number only audio read.
# Pointing video at it turned a permanently-true condition into a runaway: D
# climbed 100ms every ten seconds to the 2s ceiling, and the picture froze.
#
# A slot with frames waiting in the buffer is not starving — it is D being
# generous, and raising it is precisely backwards.
_ESCALATE_AFTER_SLOTS = 300        # ~10s at 30fps
_ESCALATE_STARVED_RATIO = 0.20
_ESCALATE_STEP_NS = 100_000_000
_ESCALATE_CEILING_NS = 2_000_000_000


# Name fragments of virtual audio devices, lowercased. A virtual *output* is
# the counterpart of the virtual camera: the conferencing app selects it as a
# microphone, and what this application writes to it is what the call hears.
#
# **None of these can be created from Python.** pyvirtualcam ships a video
# device; there is no equivalent for audio, because a virtual microphone is a
# kernel driver. So this finds one the operator has installed and says clearly
# when there is none — the alternative is writing to the default output, which
# means the operator hears their own voice half a second late and the call
# hears their real voice with no delay at all.
_VIRTUAL_OUTPUT_HINTS = (
    'cable input',       # VB-Audio Virtual Cable (Windows) — the common one
    'vb-audio',
    'voicemeeter',       # VoiceMeeter's virtual inputs
    'blackhole',         # macOS
    'soundflower',       # macOS, older
    'pulse',             # Linux, when a null sink is exposed through PulseAudio
    'virtual',           # generic, last because it is the loosest match
)


def find_virtual_output() -> Optional[int]:
    """
    Index of an installed virtual audio output, or None.

    Matched by name against `_VIRTUAL_OUTPUT_HINTS`, in that order, so a real
    virtual cable is preferred over anything that merely has "virtual" in its
    name. Only devices with output channels are considered.

    **Among devices with the same name, the lowest-latency one wins.** Windows
    exposes each device once per host API, and the difference is not small: one
    machine offered the same VB-Audio cable at 90ms on MME, 120ms on
    DirectSound and **2ms on WASAPI**. Enumeration order puts MME first, so
    taking the first match added 88ms to a path already carrying a 350ms round
    trip — and it would never have looked like a bug, just like more of the
    latency everything else was being blamed for.

    Sorting on the driver's own reported latency rather than matching API names
    keeps this working on macOS and Linux, where the same idea has different
    labels.

    Returns:
        sounddevice device index, or None when nothing suitable is installed
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception:
        return None

    for hint in _VIRTUAL_OUTPUT_HINTS:
        matches = []
        for index, device in enumerate(devices):
            try:
                if int(device.get('max_output_channels', 0)) <= 0:
                    continue
                if hint not in str(device.get('name', '')).lower():
                    continue
                latency = float(device.get('default_low_output_latency', 0.0) or 0.0)
                matches.append((latency, index))
            except Exception:
                continue
        if matches:
            matches.sort()
            return matches[0][1]
    return None


def device_sample_rate(index: Optional[int]) -> Optional[int]:
    """
    The rate a device says it wants, or None if it will not say.

    Args:
        index: sounddevice device index, or None for the system default

    Returns:
        Integer sample rate, or None when it cannot be read
    """
    try:
        import sounddevice as sd
        rate = sd.query_devices(index).get('default_samplerate')
        return int(rate) if rate else None
    except Exception:
        return None


def resolve_sample_rate(
    output_device: Optional[int],
    input_device: Optional[int] = None,
) -> int:
    """
    The rate to run the whole audio path at, taken from the output device.

    **The output device decides, because it is the one with no alternative.**
    It is chosen automatically and for a reason — `find_virtual_output` picks
    the lowest-latency instance of the cable, which on Windows is the WASAPI
    one — and WASAPI in shared mode will not resample: ask it for a rate its
    endpoint is not configured for and the stream simply refuses to open. The
    microphone is the system's, usually happy at either rate, and is the side
    that can bend.

    That asymmetry is what this exists for. The rate used to be a constant, and
    the constant was 44.1 kHz while VB-CABLE's endpoint was configured at 48 —
    so playback could not start at all. Changing the constant to 48000 fixes
    that machine and breaks the next one whose cable is set to 44.1, because
    the endpoint's rate is a dropdown in the Windows sound control panel and
    nothing about it is ours to assume. Asking is the fix; the constant is only
    the fallback for a device that reports nothing.

    Worth recording why this surfaced when it did: MME and DirectSound resample
    silently, so before the low-latency selection landed, the mismatch was
    being papered over by the same slow path that cost 88ms. The faster device
    did not introduce this — it stopped hiding it.

    Args:
        output_device: Where audio is played; the rate is taken from here
        input_device: Checked against that rate, and warned about if it
                      disagrees. None means the system default input

    Returns:
        Sample rate in Hz, falling back to `DEFAULT_SAMPLE_RATE`
    """
    import sys

    rate = device_sample_rate(output_device)
    if rate is None:
        print(
            '[AUDIO] {} did not report a sample rate — using {} Hz'.format(
                describe_device(output_device), DEFAULT_SAMPLE_RATE),
            file=sys.stderr,
        )
        return DEFAULT_SAMPLE_RATE

    # Probe the microphone rather than discovering the problem when the stream
    # fails to open, which is silent for the rest of the session.
    try:
        import sounddevice as sd
        sd.check_input_settings(
            device=input_device, samplerate=rate, channels=DEFAULT_CHANNELS,
            dtype='float32',
        )
    except Exception:
        # Both are named, and so is the fix, because nothing downstream can
        # resolve this: the capture and playback ends of one ring buffer cannot
        # run at two rates without resampling, which this does not do. Playing
        # 48 kHz samples out of a 44.1 kHz endpoint is a pitch shift and a
        # steady drift, so the mismatch is reported rather than absorbed.
        print(
            '[AUDIO] {} wants {} Hz but the microphone ({}) will not open at '
            'that rate. Audio may not start. Set both devices to the same '
            'rate in the system sound settings.'.format(
                describe_device(output_device), rate,
                describe_device(input_device)),
            file=sys.stderr,
        )

    return rate


def _stream_latency_ns(stream: Any) -> int:
    """
    What a PortAudio stream holds, in nanoseconds, or 0 if it will not say.

    Both ends of the path need it and neither can assume it: the same VB-Audio
    cable reports 2ms on WASAPI and 90ms on MME. `find_virtual_output` already
    prefers the low-latency instance; this accounts for whatever is left, so
    the presented delay is D rather than D plus an unknown device.
    """
    try:
        latency = getattr(stream, 'latency', 0.0)
        # An OutputStream on a duplex device can report a pair.
        if isinstance(latency, (tuple, list)):
            latency = latency[-1] if latency else 0.0
        return max(0, int(float(latency or 0.0) * 1_000_000_000))
    except Exception:
        return 0


def describe_device(index: Optional[int]) -> str:
    """Human-readable name for a device index, for logs and status lines."""
    if index is None:
        return 'system default'
    try:
        import sounddevice as sd
        return str(sd.query_devices(index).get('name', index))
    except Exception:
        return str(index)


class AudioRingBuffer:
    """Thread-safe ring buffer of timestamped PCM audio chunks.

    Uses collections.deque with maxlen for automatic eviction of old chunks.
    CPython GIL guarantees atomic append/popleft on deque, so no explicit
    lock is needed for single-producer / single-consumer access patterns.

    For multi-consumer scenarios (Phase 4: playback reads while capture writes),
    the GIL still protects individual operations, but iteration requires a
    snapshot to avoid RuntimeError from concurrent mutation.
    """

    def __init__(
        self,
        max_chunks: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """
        Args:
            max_chunks: Maximum number of chunks to retain
            sample_rate: Audio sample rate (for duration calculations)
        """
        self._buf: Deque[AudioChunk] = collections.deque(maxlen=max_chunks)
        self.sample_rate = sample_rate

    def append(self, capture_ts: int, pcm: npt.NDArray[Any]) -> None:
        """Add a new audio chunk to the buffer.

        Args:
            capture_ts: Capture timestamp in nanoseconds (time.perf_counter_ns)
            pcm: PCM audio data as float32 numpy array, shape (frames, channels)
        """
        self._buf.append((capture_ts, pcm))

    def peek_oldest(self) -> Optional[AudioChunk]:
        """Return the oldest chunk without removing it, or None if empty."""
        if self._buf:
            return self._buf[0]
        return None

    def popleft(self) -> Optional[AudioChunk]:
        """Remove and return the oldest chunk, or None if empty."""
        try:
            return self._buf.popleft()
        except IndexError:
            return None

    def snapshot(self) -> List[Any]:
        """Return a shallow copy of all chunks for safe iteration."""
        return list(self._buf)

    def clear(self) -> None:
        """Discard all buffered audio."""
        self._buf.clear()

    @property
    def count(self) -> int:
        """Number of chunks currently buffered."""
        return len(self._buf)

    @property
    def empty(self) -> bool:
        return len(self._buf) == 0

    def duration_ns(self) -> int:
        """Time span covered by the buffer in nanoseconds.

        Returns 0 if fewer than 2 chunks are buffered.
        """
        if len(self._buf) < 2:
            return 0
        return self._buf[-1][0] - self._buf[0][0]


class AudioCapture:
    """Captures audio from the local microphone using sounddevice.

    Each captured block is timestamped with time.perf_counter_ns() — the same
    clock used by the webcam capture thread in bridge.py — and appended to an
    AudioRingBuffer.

    Lifecycle is tied to the webcam: start when streaming begins, stop when
    streaming ends.

    Example:
        capture = AudioCapture()
        capture.start()
        # ... later ...
        chunk = capture.ring_buffer.popleft()
        capture.stop()
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        buffer_seconds: int = DEFAULT_BUFFER_SECONDS,
    ) -> None:
        """
        Args:
            device: Audio input device index (None = system default)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1 = mono)
            block_size: Frames per callback block
            buffer_seconds: How many seconds of audio to retain
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size

        max_chunks = int(buffer_seconds * sample_rate / block_size)
        self.ring_buffer = AudioRingBuffer(max_chunks, sample_rate)

        self._stream: Optional[Any] = None
        self._running = False

        # What the device holds before handing a block over. Read from
        # PortAudio once the stream is open, and subtracted from every chunk's
        # timestamp — a block delivered now was recorded before now.
        self._input_latency_ns: int = 0

        # Voice transformer (set externally via set_voice_transformer)
        self._voice_transformer: Optional['VoiceTransformer'] = None

        # Clock drift monitoring — tracks expected vs actual sample count.
        # 0 means "no baseline yet"; it is taken at the first delivered block,
        # never at `start()` — see `_audio_callback`.
        self._drift_start_ns: int = 0
        self._drift_samples: int = 0
        # Threshold in seconds: warn if audio clock drifts more than this
        self._drift_warn_threshold: float = 0.05  # 50 ms

    def _audio_callback(
        self,
        indata: npt.NDArray[Any],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """sounddevice InputStream callback — called from a dedicated audio thread.

        Args:
            indata: Recorded audio data, shape (frames, channels), float32
            frames: Number of frames in this block
            time_info: PortAudio time info (not used — we use our own clock)
            status: PortAudio status flags
        """
        if status:
            print(f'[AUDIO] capture status: {status}', file=sys.stderr)

        delivered_ts = time.perf_counter_ns()

        # **A chunk's timestamp is its first sample, not its delivery.** `_seek`
        # computes its offset into a chunk as `(playback_point - chunk_ts)`, so
        # it reads the stamp as the moment the first sample was recorded. This
        # callback runs once the block is complete and the device has handed it
        # over, which is a block's duration plus the input latency later. Both
        # were previously inside the stamp, and both pushed audio later than it
        # happened — the direction that makes the sound trail the picture.
        span_ns = int(frames / self.sample_rate * 1_000_000_000)
        capture_ts = delivered_ts - span_ns - self._input_latency_ns

        # **The drift baseline starts at the first block, not at `start()`.**
        # Opening a device takes time — tens to hundreds of milliseconds on
        # Windows — and measuring from before the stream ran counted that
        # interval as samples which failed to arrive. `drift_ns` therefore sat
        # permanently negative by the device-open cost, for the whole session,
        # and `check_health` only rebaselines above 200ms so anything under
        # that was warned about every two seconds and never corrected.
        if self._drift_start_ns == 0:
            self._drift_start_ns = capture_ts
        else:
            self._drift_samples += frames

        # Copy the data — sounddevice reuses the buffer after callback returns
        pcm = indata.copy()
        if self._voice_transformer is not None:
            pcm = self._voice_transformer.process(pcm)
        self.ring_buffer.append(capture_ts, pcm)

    def set_voice_transformer(
        self, transformer: Optional['VoiceTransformer']
    ) -> None:
        """Attach or detach a VoiceTransformer for real-time pitch shifting."""
        self._voice_transformer = transformer

    def start(self) -> None:
        """Open the audio input stream and begin capturing."""
        if self._running:
            return

        try:
            import sounddevice as sd
        except ImportError:
            import sys
            print(
                '[AUDIO] sounddevice not installed — audio capture disabled. '
                'Install with: pip install sounddevice',
                file=sys.stderr,
            )
            return

        self.ring_buffer.clear()
        self._drift_start_ns = 0
        self._drift_samples = 0
        self._input_latency_ns = 0

        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                dtype='float32',
                callback=self._audio_callback,
            )
            self._stream.start()
            self._input_latency_ns = _stream_latency_ns(self._stream)
            self._running = True
        except Exception as e:
            import sys
            # Name the rate we asked for and the rate the device wants. Without
            # both, this is "audio does not work" and the next step is guessing
            # at devices — which is exactly how the 44.1/48 mismatch came to
            # cost a debugging session on a machine it should have explained
            # itself on.
            wanted = device_sample_rate(self.device)
            print(
                '[AUDIO] Failed to start audio capture at {} Hz on {}: {}{}'
                .format(
                    self.sample_rate, describe_device(self.device), e,
                    '' if wanted in (None, self.sample_rate)
                    else ' — the device reports {} Hz'.format(wanted),
                ),
                file=sys.stderr,
            )
            self._stream = None

    def stop(self) -> None:
        """Stop capturing and close the audio stream."""
        if not self._running:
            return

        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                import sys
                print(f'[AUDIO] Error stopping audio capture: {e}', file=sys.stderr)
            self._stream = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def drift_ns(self) -> int:
        """Signed clock drift in nanoseconds (positive = audio ahead of wall clock).

        Used by AudioPlayback to compensate the playback point so audio stays
        aligned with the video playout offset despite hardware clock skew.
        """
        if self._drift_start_ns == 0:
            return 0
        elapsed_ns = time.perf_counter_ns() - self._drift_start_ns
        if elapsed_ns <= 0 or self._drift_samples <= 0:
            return 0
        expected_samples = elapsed_ns / 1_000_000_000 * self.sample_rate
        drift_samples = self._drift_samples - expected_samples
        return int(drift_samples / self.sample_rate * 1_000_000_000)

    def reset_drift(self) -> None:
        """Reset drift counters to prevent unbounded accumulation."""
        self._drift_start_ns = time.perf_counter_ns()
        self._drift_samples = 0

    def check_health(self) -> Dict[str, Any]:
        """Check capture stream health and clock drift.

        Returns a dict with:
            active: bool — whether the underlying PortAudio stream is alive
            drift_ms: float — divergence between wall clock and audio clock
            drift_warning: bool — True if drift exceeds the warning threshold
        """
        active = False
        if self._stream is not None:
            try:
                active = self._stream.active
            except Exception:
                active = False

        drift_ms = 0.0
        drift_warning = False
        elapsed_ns = (0 if self._drift_start_ns == 0
                      else time.perf_counter_ns() - self._drift_start_ns)
        if elapsed_ns > 0 and self._drift_samples > 0:
            expected_samples = elapsed_ns / 1_000_000_000 * self.sample_rate
            drift_s = abs(self._drift_samples - expected_samples) / self.sample_rate
            drift_ms = drift_s * 1000.0
            drift_warning = drift_s > self._drift_warn_threshold

        return {
            'active': active,
            'drift_ms': round(drift_ms, 2),
            'drift_warning': drift_warning,
        }

    def try_recover(self) -> bool:
        """Attempt to restart the audio stream after a failure.

        Returns True if the stream was successfully restarted.
        """
        if self._running and self._stream is not None:
            try:
                if self._stream.active:
                    return True  # still alive, nothing to do
            except Exception:
                pass

        # Stream died — close and reopen
        print('[AUDIO] Attempting capture stream recovery...', file=sys.stderr)
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._running = False
        self.start()
        return self._running


# ── Phase 3: Adaptive playout ─────────────────────────────────────────────


class RTTTracker:
    """Sliding-window RTT estimator for adaptive playout delay.

    Tracks round-trip latency of video frames (desktop → GPU → desktop) and
    computes a smoothed target delay:

        target_delay = median(rtt) + 1 * stddev(rtt)

    Uses median instead of mean for robustness against outlier spikes.
    Clamped to [FLOOR_NS, CEILING_NS].

    **This is the largest piece of latency the application itself controls**,
    so the constants matter and each is a trade rather than a tuning knob:

    - `INITIAL_DELAY_NS` is what every session pays before a single RTT sample
      exists. It was 400ms, which meant even a nearby pod felt heavily delayed
      for the first seconds and then improved — read by an operator as "it is
      sluggish", since first impressions are formed exactly then. It now starts
      low and adapts *up*, which costs some jitter in the first half-second
      instead of costing latency in every session.

    - Smoothing is **asymmetric**, which is standard for a playout buffer and
      was not what this did. Growing late causes visible glitches, so the
      target rises quickly; shrinking early causes underruns, so it falls
      slowly. One symmetric alpha has to choose which failure to accept, and
      0.2 chose to be slow in both directions.

    - `FLOOR_NS` is added even on a perfect link. One frame interval is the
      defensible floor — below that the buffer cannot absorb ordinary
      frame-interval quantisation — and 20fps makes that 50ms.

    None of this touches the network's own delay, which on a remote pod
    dominates everything here. It only stops the client adding more than it
    needs to.
    """

    WINDOW_SIZE: int = 60         # ~2 seconds at 30 fps (larger window for stability)
    UPDATE_INTERVAL: int = 5      # recalculate every N samples
    SMOOTHING_ALPHA_UP: float = 0.5    # rise fast: a late buffer glitches
    SMOOTHING_ALPHA_DOWN: float = 0.1  # fall slow: an early buffer underruns
    FLOOR_NS: int = 50_000_000            # 50 ms — one frame interval at 20fps
    CEILING_NS: int = 2_000_000_000      # 2 s  (accommodates a bad remote link)
    INITIAL_DELAY_NS: int = 120_000_000  # 120 ms, then adapt from measurement
    WARMUP_SAMPLES: int = 5             # min samples before adapting

    def __init__(self) -> None:
        self._samples: 'collections.deque[Any]' = collections.deque(maxlen=self.WINDOW_SIZE)
        self._target_delay_ns: int = self.INITIAL_DELAY_NS
        self._count: int = 0

    def record(self, capture_ts_ns: int, arrival_ts_ns: int) -> None:
        """Record one RTT sample.

        Args:
            capture_ts_ns: perf_counter_ns when the frame was captured locally
            arrival_ts_ns: perf_counter_ns when the processed frame arrived back
        """
        if capture_ts_ns <= 0:
            return
        rtt = arrival_ts_ns - capture_ts_ns
        if rtt < 0:
            return  # clock anomaly, skip
        self._samples.append(rtt)
        self._count += 1

        if (self._count % self.UPDATE_INTERVAL == 0
                and len(self._samples) >= self.WARMUP_SAMPLES):
            self._recompute()

    def _recompute(self) -> None:
        """Recompute target delay from the current sample window."""
        arr = np.array(self._samples, dtype=np.float64)
        raw = int(float(np.median(arr)) + 1.0 * float(np.std(arr)))
        clamped = max(self.FLOOR_NS, min(self.CEILING_NS, raw))
        # Asymmetric: rise fast, fall slow. Under-buffering shows as a visible
        # glitch the moment the link hiccups, while over-buffering only costs
        # latency — so the direction that risks a glitch is the one to take
        # quickly, and the direction that only costs delay can be cautious.
        alpha = (self.SMOOTHING_ALPHA_UP if clamped > self._target_delay_ns
                 else self.SMOOTHING_ALPHA_DOWN)
        self._target_delay_ns = int(
            alpha * clamped + (1.0 - alpha) * self._target_delay_ns
        )

    @property
    def target_delay_ns(self) -> int:
        """Current adaptive playout delay in nanoseconds."""
        return self._target_delay_ns

    def reset(self) -> None:
        """Clear all samples and revert to the initial warmup delay."""
        self._samples.clear()
        self._target_delay_ns = self.INITIAL_DELAY_NS
        self._count = 0


class JitterBuffer:
    """FIFO buffer for processed video frames with adaptive timed release.

    Frames are pushed by the WebSocket receive thread and popped by the Qt
    render timer when they become eligible for display.

    A frame is eligible when::

        now_ns - capture_ts >= target_delay_ns

    If multiple frames are eligible (e.g. after a UI stall), only the most
    recent is returned — intermediate frames are dropped to stay current.

    The embedded RTTTracker computes target_delay_ns adaptively from observed
    round-trip latencies.
    """

    # Has to exceed what D holds in flight: `(D - rtt) * fps` frames are
    # waiting to age into their deadline at any moment, and a frame evicted
    # before it gets there is one the operator never sees. 60 was 2s at 30fps,
    # exactly the escalation ceiling and therefore no headroom at all.
    MAX_FRAMES: int = 120  # ~4 seconds at 30 fps

    def __init__(
        self,
        fixed_delay_ns: int = DEFAULT_PLAYOUT_DELAY_NS,
        calibrate: bool = True,
    ) -> None:
        """
        Args:
            fixed_delay_ns: The delay both streams are held to. 0 means adapt,
                which is the previous behaviour and still reachable
            calibrate: Treat `fixed_delay_ns` as a provisional value and replace
                it, once, with one measured from the link. False pins it
        """
        self._buf: 'collections.deque[Any]' = collections.deque(maxlen=self.MAX_FRAMES)
        self._rtt = RTTTracker()
        # 0 means adapt, which is the previous behaviour and still reachable.
        self._fixed_delay_ns = int(fixed_delay_ns)
        # What `clear()` returns to, since a new pod is a new link and has to
        # be measured again rather than inheriting the last one's answer.
        self._provisional_delay_ns = int(fixed_delay_ns)
        self._calibrate = bool(calibrate) and self._fixed_delay_ns > 0
        self._calibrated = False
        # Bumped whenever D moves. Audio's read cursor is deliberately
        # continuous and will not follow a changing delay, so it watches this
        # instead and repositions once — see `AudioPlayback._output_callback`.
        self._delay_epoch = 0
        # The last frame actually shown. A slot with nothing eligible repeats
        # it rather than slipping the schedule — see `next_for_slot`.
        self._last_shown: Optional[Tuple[int, bytes]] = None
        # Age of the picture presented at the last slot. Half of the A/V skew
        # measurement; `AudioPlayback` owns the other half. Nothing measured
        # this before, which is how video and audio came to be held to two
        # different delays without any readout disagreeing.
        self._last_video_age_ns = 0
        self._slots = 0
        self._repeats = 0
        self._repeats_total = 0
        # Slots with nothing to show *and nothing waiting*. The repeat count
        # includes every slot the stream simply had no new frame for, which on
        # a 20fps stream against a 30fps tick is a third of them on a perfect
        # link — so it is the badge's number, not the escalator's.
        self._starved = 0
        self._pushes = 0
        self._forced = 0
        self._late_dropped = 0
        self._escalations = 0
        self._calibrate_seen = 0
        self._calibrate_next = _CALIBRATE_SKIP + _CALIBRATE_SAMPLES
        self._calibrate_attempts = 0

    def push(self, capture_ts: int, jpeg_bytes: bytes) -> None:
        """Enqueue a processed frame. Called from the WS receive thread.

        Args:
            capture_ts: Original capture timestamp (perf_counter_ns), or 0
            jpeg_bytes: JPEG-encoded processed frame
        """
        arrival_ts = time.perf_counter_ns()
        self._pushes += 1
        self._rtt.record(capture_ts, arrival_ts)
        self._maybe_calibrate()
        self._buf.append((capture_ts, jpeg_bytes))
        self._drop_overflow()

    def pop_eligible(self) -> Optional[Tuple[int, bytes]]:
        """Return the most recent eligible frame, or None.

        If several frames have passed their playout time, all but the newest
        eligible frame are silently dropped — this keeps the display current
        after transient stalls.

        Called from the Qt render timer (main thread).
        """
        if not self._buf:
            return None

        now = time.perf_counter_ns()
        # `self.target_delay_ns`, never `self._rtt.target_delay_ns`. Audio
        # positions itself against the property, so reading the adaptive
        # estimate here held the two streams to two different delays: video was
        # released on arrival while audio waited the fixed D, and the sound
        # trailed the lips by the difference. It also made escalation actively
        # harmful — raising D moved audio further back and never touched the
        # video it was raised to steady.
        target = self.target_delay_ns
        result: Optional[Tuple[int, bytes]] = None

        # The buffer is full, so the next push evicts the frame at the front
        # before it reaches its deadline — and the one after that, and so on.
        # That is a picture which never moves again, from a delay the buffer
        # cannot hold. Showing it early is wrong by less than a frozen face is,
        # and it is counted so the cause is not mistaken for a network fault.
        forced = len(self._buf) >= self.MAX_FRAMES

        while self._buf:
            capture_ts, jpeg = self._buf[0]
            # Legacy frame without timestamp — display immediately
            if capture_ts <= 0:
                result = self._buf.popleft()
                continue
            age = now - capture_ts
            if age >= target:
                result = self._buf.popleft()
            elif forced and result is None:
                self._forced += 1
                result = self._buf.popleft()
                break
            else:
                break

        return result

    def _drop_overflow(self) -> None:
        """Discard frames that are catastrophically stale (older than 2× ceiling)."""
        now = time.perf_counter_ns()
        discard_threshold = self._rtt.CEILING_NS * 2
        while self._buf:
            capture_ts = self._buf[0][0]
            if capture_ts <= 0:
                break
            if now - capture_ts > discard_threshold:
                self._buf.popleft()
            else:
                break

    @property
    def target_delay_ns(self) -> int:
        """
        The playout delay both streams are held to.

        Fixed unless `fixed_delay_ns` was 0. The RTT tracker keeps measuring
        either way — that telemetry says whether the fixed value is still the
        right one — it simply stops steering.
        """
        if self._fixed_delay_ns > 0:
            return self._fixed_delay_ns
        return self._rtt.target_delay_ns

    @property
    def delay_epoch(self) -> int:
        """Increments whenever `target_delay_ns` moves. Audio watches this."""
        return self._delay_epoch

    @property
    def calibrated(self) -> bool:
        """True once D has been measured from this link and frozen."""
        return self._calibrated

    @property
    def video_age_ns(self) -> int:
        """Age of the picture presented at the last slot."""
        return self._last_video_age_ns

    def _maybe_calibrate(self) -> None:
        """
        Replace the provisional delay with one measured from this link, once.

        D cannot be smaller than what video actually costs — a frame cannot be
        shown before it arrives — so audio necessarily waits as long as video
        does. What it must not do is wait *longer*, which is what a constant
        tuned against one datacenter guarantees on every other one.

        p95 rather than the median, because a D that only covers half the
        frames repeats the other half. Plus the spread and a margin, with the
        spread floored at one frame interval so a suspiciously steady window
        cannot produce a delay carrying no headroom at all.

        Committed exactly once per session, about three seconds in, and then
        frozen: a delay that keeps moving is what made speech break up, and
        this keeps the fixed-delay property while dropping the guessed number.
        """
        if not self._calibrate or self._calibrated:
            return

        self._calibrate_seen += 1
        # Discard the warm-up, then look once per window rather than per frame.
        if self._calibrate_seen < self._calibrate_next:
            return
        self._calibrate_next = self._calibrate_seen + _CALIBRATE_SAMPLES

        samples = self._rtt._samples
        if len(samples) < _CALIBRATE_SAMPLES:
            return

        arr = np.array(samples, dtype=np.float64)
        p50 = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))

        # Refuse a window that is not describing a steady link. A tail far
        # above the middle means the window straddles a change — model warm-up
        # ending is the usual one — and its p95 describes the thing that ended
        # rather than the thing that continues.
        if p95 > p50 * _CALIBRATE_SPREAD_RATIO or p95 > _CALIBRATE_MAX_NS:
            self._calibrate_attempts += 1
            if self._calibrate_attempts < _CALIBRATE_ATTEMPTS:
                return
            self._calibrated = True
            print(
                '[SYNC] playout delay left at {:.0f}ms — the round trip never '
                'settled (p50 {:.0f} / p95 {:.0f}ms). Escalation will raise it '
                'if slots actually starve.'.format(
                    self._fixed_delay_ns / 1_000_000,
                    p50 / 1_000_000, p95 / 1_000_000),
                file=sys.stderr,
            )
            return

        spread = max(float(np.std(arr)), float(_CALIBRATE_SPREAD_FLOOR_NS))
        raw = p95 + spread + _CALIBRATE_MARGIN_NS
        quantised = int(round(raw / _DELAY_QUANTUM_NS)) * _DELAY_QUANTUM_NS
        delay = max(_CALIBRATE_FLOOR_NS, min(_CALIBRATE_MAX_NS, quantised))

        self._calibrated = True
        if delay == self._fixed_delay_ns:
            return

        self._fixed_delay_ns = delay
        self._delay_epoch += 1
        print(
            '[SYNC] playout delay calibrated to {:.0f}ms '
            '(rtt p50 {:.0f} / p95 {:.0f} over {} samples, was {:.0f}ms)'
            .format(delay / 1_000_000, p50 / 1_000_000, p95 / 1_000_000,
                    len(samples), self._provisional_delay_ns / 1_000_000),
            file=sys.stderr,
        )

    def next_for_slot(self) -> Optional[Tuple[int, bytes]]:
        """
        The frame to display for this slot, holding the schedule.

        Called once per display tick. Three rules, and the second is the one
        that is easy to get wrong:

        1. **The slot fires on time, always.** Waiting for a late frame would
           slip the schedule, and audio is locked to the same clock — so a
           stall is either a gap in speech or a drift out of sync, and both
           defeat the point of a fixed delay.
        2. **A frame that missed its slot is discarded, not shown late.**
           Playing the straggler shifts everything one slot later and the
           pattern never recovers; the next frame to show is the next one that
           is *on time*. `pop_eligible` already drops all but the newest.
        3. **A slot with nothing eligible repeats the last shown frame.** One
           repeat is 33-50ms of an already mostly-still face and is invisible,
           which is exactly why video is the cheap place to absorb jitter and
           audio is not. It is always the last *swapped* frame — never the raw
           camera, never black.

        Returns:
            (capture_ts, jpeg) to display, or None before the first frame ever
            arrives, when there is nothing to repeat.
        """
        self._slots += 1
        now = time.perf_counter_ns()

        eligible = self.pop_eligible()
        if eligible is not None:
            self._last_shown = eligible
        else:
            # Nothing eligible. Whether that is a fault depends entirely on
            # whether anything was *waiting*: an empty buffer means frames are
            # arriving already past their deadline, which more headroom fixes,
            # and a full one means D is generous, which it does not.
            if not self._buf:
                self._starved += 1
            if self._last_shown is not None:
                self._repeats += 1
                self._repeats_total += 1

        # Measured on the frame actually presented, repeats included: a held
        # frame really is getting older on screen, and pretending otherwise
        # would hide the one case where the skew grows.
        if self._last_shown is not None and self._last_shown[0] > 0:
            self._last_video_age_ns = now - self._last_shown[0]

        self._maybe_escalate()
        return self._last_shown

    def _maybe_escalate(self) -> None:
        """
        Raise the fixed delay when repeats have been sustained.

        One repeat is invisible; a sustained rate is a frozen face while audio
        keeps going, which reads as a broken swap rather than a slow network.
        That is evidence D is too small for this link — so it steps, once, on a
        window of evidence, rather than being chased per frame the way the
        adaptive version was.
        """
        if self._fixed_delay_ns <= 0 or self._slots < _ESCALATE_AFTER_SLOTS:
            return

        ratio = self._starved / float(self._slots)
        arriving = self._pushes > 0
        self._slots = 0
        self._repeats = 0
        self._starved = 0
        self._pushes = 0

        # Nothing arrived at all, so there is nothing a bigger buffer would
        # have held. A dead pipeline is not a slow link, and stepping D for it
        # only delays the picture that does eventually come back.
        if not arriving:
            return
        if ratio < _ESCALATE_STARVED_RATIO:
            return
        if self._fixed_delay_ns >= _ESCALATE_CEILING_NS:
            return

        self._fixed_delay_ns = min(
            _ESCALATE_CEILING_NS, self._fixed_delay_ns + _ESCALATE_STEP_NS)
        self._escalations += 1
        self._delay_epoch += 1
        print(
            '[SYNC] {:.0f}% of slots starved — raising playout delay to '
            '{:.0f}ms. Frames are arriving past their deadline, so the link '
            'needs more headroom than the current delay allows.'.format(
                ratio * 100, self._fixed_delay_ns / 1_000_000),
            file=sys.stderr,
        )

    def clear(self) -> None:
        """Discard all buffered frames and reset RTT statistics.

        The held frame goes with them: it belongs to the session that just
        ended, and repeating it into a new one would show a stale face.
        """
        self._buf.clear()
        self._rtt.reset()
        self._last_shown = None
        self._last_video_age_ns = 0
        self._slots = 0
        self._repeats = 0
        self._starved = 0
        self._pushes = 0
        self._calibrate_seen = 0
        self._calibrate_next = _CALIBRATE_SKIP + _CALIBRATE_SAMPLES
        self._calibrate_attempts = 0
        # A new session is a new link, so D is measured again rather than
        # inheriting the last one's answer. The epoch moves with it: audio may
        # still be running — `clear()` is called on reconnect — and has to be
        # told the number under it changed.
        if self._calibrate and (
                self._calibrated
                or self._fixed_delay_ns != self._provisional_delay_ns):
            self._calibrated = False
            self._fixed_delay_ns = self._provisional_delay_ns
            self._delay_epoch += 1

    @property
    def depth(self) -> int:
        """Number of frames currently buffered."""
        return len(self._buf)

    def sync_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics for debugging A/V sync.

        `rtt_*` is **glass to glass** — capture on this machine to the
        processed frame arriving back — so it already contains the uplink, the
        pipeline and the downlink. Read it against the pipeline's own latency
        report: whatever the two differ by is network and encode, and on a
        remote pod that difference is usually most of the total.

        Returns a dict with:
            target_delay_ms: current adaptive playout delay this client adds
            buffer_depth: number of frames waiting in the jitter buffer
            rtt_samples: number of RTT samples in the current window
            rtt_p50_ms: median RTT over the sliding window (0 if empty)
            rtt_p95_ms: 95th-percentile RTT (0 if empty)
            rtt_stddev_ms: stddev of RTT over the sliding window (0 if empty)
        """
        samples = self._rtt._samples
        if len(samples) >= 2:
            arr = np.array(samples, dtype=np.float64)
            p50_ms = float(np.median(arr)) / 1_000_000
            p95_ms = float(np.percentile(arr, 95)) / 1_000_000
            std_ms = float(np.std(arr)) / 1_000_000
        elif len(samples) == 1:
            p50_ms = p95_ms = samples[0] / 1_000_000
            std_ms = 0.0
        else:
            p50_ms = p95_ms = std_ms = 0.0

        return {
            # The delay actually in force. This used to report the adaptive
            # estimate, so the `[SYNC] delay=` line and the viewport badge never
            # showed the number either stream was held to, and never moved when
            # escalation fired — which is most of why a several-hundred-
            # millisecond A/V split went unnoticed for so long.
            'target_delay_ms': round(self.target_delay_ns / 1_000_000, 1),
            'rtt_target_ms': round(self._rtt.target_delay_ns / 1_000_000, 1),
            'calibrated': self._calibrated,
            'video_age_ms': round(self._last_video_age_ns / 1_000_000, 1),
            'buffer_depth': len(self._buf),
            'rtt_samples': len(samples),
            'rtt_p50_ms': round(p50_ms, 1),
            'rtt_p95_ms': round(p95_ms, 1),
            'rtt_stddev_ms': round(std_ms, 1),
            'fixed_delay': self._fixed_delay_ns > 0,
            'repeats': self._repeats_total,
            'escalations': self._escalations,
            # A frame shown before its deadline because the buffer was about to
            # evict it. Non-zero means D is larger than the buffer can hold.
            'forced': self._forced,
        }


# ── Phase 4: Synchronized audio playback ──────────────────────────────────


class AudioPlayback:
    """Plays captured audio in sync with the jitter-buffered video.

    Uses a sounddevice OutputStream whose callback pulls PCM chunks from the
    AudioRingBuffer at ``playback_point = now - target_delay``. The
    ``target_delay`` is read from the JitterBuffer's RTTTracker so that audio
    and video share the exact same playout offset.

    Chunks that have fallen entirely behind the playback point are silently
    discarded (the listener would hear them as stale). Gaps are filled with
    silence.

    Example::

        playback = AudioPlayback(capture.ring_buffer, jitter_buffer)
        playback.start()
        # ... later ...
        playback.stop()
    """

    def __init__(
        self,
        ring_buffer: AudioRingBuffer,
        jitter_buffer: JitterBuffer,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        audio_capture: Optional['AudioCapture'] = None,
        device: Optional[int] = None,
    ) -> None:
        """
        Args:
            ring_buffer: AudioRingBuffer filled by AudioCapture
            jitter_buffer: JitterBuffer whose target_delay_ns drives sync
            sample_rate: Must match the capture sample rate
            channels: Must match the capture channel count
            block_size: OutputStream block size (frames per callback)
            audio_capture: Optional AudioCapture for clock drift compensation
            device: Output device index. None means the system default, which
                is almost certainly wrong here — see `start`.
        """
        self._ring = ring_buffer
        self._jitter = jitter_buffer
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self._audio_capture = audio_capture

        self._stream: Optional[Any] = None
        self._running = False
        # Leftover samples from a partially consumed chunk
        self._leftover: Optional[npt.NDArray[Any]] = None
        # Set when playback has no cursor yet — at start, and after a
        # correction large enough that continuing would be meaningless.
        self._needs_seek = True
        # Capture instant of the next sample to be written. Advanced by what is
        # actually consumed, re-anchored to a chunk's own stamp whenever a block
        # starts on one, so it survives a gap in capture. This is what makes the
        # audio half of the A/V skew measurable at all.
        self._cursor_ts = 0
        self._last_audio_age_ns = 0
        # What the output device holds after this callback returns. Read from
        # PortAudio once the stream is open.
        self._output_latency_ns = 0
        # The delay generation this cursor was positioned against. -1 so the
        # first callback always establishes it.
        self._delay_epoch = -1
        # Diagnosis. Audio faults are audible but invisible: "it sounds broken"
        # could be an underrun, a trim, or the device itself, and until these
        # were counted there was no way to tell which.
        self._underruns = 0
        self._trims = 0
        self._resyncs = 0

    # How far the buffer may run ahead of the target delay before the oldest
    # audio is dropped to catch up. Generous on purpose: a trim is audible, so
    # it should answer real drift rather than ordinary scheduling jitter.
    _TRIM_TOLERANCE_NS = 250_000_000

    # Past this the cursor is meaningless — the stream was paused, the device
    # stalled, or capture stopped — and continuing would play something very
    # old. Start again from the correct point instead.
    _RESYNC_NS = 1_500_000_000

    def _buffered_ns(self) -> int:
        """Duration of audio currently queued, including any leftover."""
        total = 0
        if self._leftover is not None:
            total += self._leftover.shape[0]
        for _ts, pcm in self._ring.snapshot():
            total += pcm.shape[0]
        return int(total / self.sample_rate * 1_000_000_000)

    def _seek(self, playback_point: int) -> None:
        """
        Drop audio older than `playback_point`, seeking *into* the chunk that
        straddles it.

        The seek within the chunk is the part that was missing. Chunks are
        ~23ms, and a straddling chunk used to be played from its first sample
        regardless of where the playback point fell inside it — so the position
        quantised to a chunk boundary and moved by a whole chunk whenever the
        boundary drifted past. That is what a listener hears as a stutter with
        pieces missing.
        """
        self._leftover = None
        self._cursor_ts = playback_point

        while True:
            chunk = self._ring.peek_oldest()
            if chunk is None:
                return

            chunk_ts, pcm = chunk
            chunk_dur = int(pcm.shape[0] / self.sample_rate * 1_000_000_000)

            if chunk_ts + chunk_dur <= playback_point:
                self._ring.popleft()          # entirely in the past
                continue

            if chunk_ts >= playback_point:
                # Future audio; playback starts at its head, which is later
                # than asked for. Say so, rather than reporting the age we
                # wanted — that gap is the underrun about to be counted.
                self._cursor_ts = chunk_ts
                return

            self._ring.popleft()
            offset = int((playback_point - chunk_ts)
                         / 1_000_000_000 * self.sample_rate)
            offset = max(0, min(pcm.shape[0], offset))
            if offset < pcm.shape[0]:
                self._leftover = pcm[offset:]
            return

    def _output_callback(
        self,
        outdata: npt.NDArray[Any],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """sounddevice OutputStream callback — runs on a dedicated audio thread.

        Fills *outdata* from a **continuous cursor**: leftover first, then whole
        chunks in order. It does not re-derive its read position from the clock
        on every block, which is what it used to do — `now - target_delay`,
        against a `target_delay` that the video RTT tracker moves continuously.
        Every move of that estimate discarded a chunk or inserted a block of
        silence, and audio has no tolerance for either: video can drop a frame
        unnoticed, a 23ms hole in speech is a click.

        Latency is corrected instead by *depth*, rarely and outside the fill:
        if more audio is queued than the target delay wants, the oldest is
        trimmed. Otherwise playback simply continues, which is what makes it
        sound continuous.
        """
        if status:
            import sys
            print(f'[AUDIO] playback status: {status}', file=sys.stderr)

        now = time.perf_counter_ns()
        target = self._jitter.target_delay_ns

        # The block written here is not audible now — it is audible once the
        # device has played what it already holds. So the sample to write is
        # the one whose presentation instant is `now + output_latency`, not
        # `now`. Without this the delay is D *plus* the device, and the device
        # is 2ms on WASAPI and 90ms on MME for the same cable.
        #
        # The capture clock's drift is deliberately **not** added here any
        # more. The cursor below is continuous, so a drift term only moves the
        # seek point — and a rate mismatch cannot be repaired by seeking
        # somewhere the samples are not; that needs a resampler, which
        # docs/ACCEPTED_RISKS.md records as not being carried. What it did do
        # was fold the input device's open latency into the position for the
        # whole session. It is still measured, and now only reported.
        playback_point = now - target + self._output_latency_ns

        # D moved: calibration committed, or the buffer escalated. This is the
        # only thing besides a trim or a resync permitted to move the cursor.
        # The cursor is continuous precisely so it does not chase an estimate;
        # calibrating is what turns the estimate into a decision, once, and a
        # decision is worth one reposition.
        epoch = self._jitter.delay_epoch
        if epoch != self._delay_epoch:
            self._delay_epoch = epoch
            self._needs_seek = True

        # Establish or re-establish the cursor. After this the fill below is
        # pure sequential reading.
        if self._needs_seek:
            self._seek(playback_point)
            self._needs_seek = False
        else:
            oldest = self._ring.peek_oldest()
            if (self._leftover is None and oldest is not None
                    and playback_point - oldest[0] > self._RESYNC_NS):
                self._resyncs += 1
                self._seek(playback_point)
            elif self._buffered_ns() > target + self._TRIM_TOLERANCE_NS:
                # Drifted behind: capture has outrun playback. Trim rather than
                # let the delay grow without bound.
                self._trims += 1
                self._seek(playback_point)

        written = 0
        cursor_ts = self._cursor_ts

        if self._leftover is not None and self._leftover.shape[0] > 0:
            n = min(self._leftover.shape[0], frames - written)
            outdata[written:written + n] = self._leftover[:n]
            written += n
            self._leftover = (self._leftover[n:]
                              if n < self._leftover.shape[0] else None)

        while written < frames:
            chunk = self._ring.popleft()
            if chunk is None:
                break
            chunk_ts, pcm = chunk
            if written == 0:
                # Nothing carried over, so this block begins exactly at this
                # chunk. Re-anchoring on its own stamp keeps the reported age
                # true across a gap in capture, which a running count would
                # quietly absorb.
                cursor_ts = chunk_ts
            n = min(pcm.shape[0], frames - written)
            outdata[written:written + n] = pcm[:n]
            written += n
            if n < pcm.shape[0]:
                self._leftover = pcm[n:]

        if written > 0:
            self._last_audio_age_ns = now - cursor_ts
            self._cursor_ts = cursor_ts + int(
                written / self.sample_rate * 1_000_000_000)

        if written < frames:
            # Underrun: nothing captured yet for this block. Silence is the
            # only option, but the next block continues from here rather than
            # from a recomputed clock position, so one gap stays one gap.
            self._underruns += 1
            outdata[written:] = 0.0

    def stats(self) -> Dict[str, Any]:
        """Playback health, for the periodic sync log.

        Returns:
            buffered_ms, and counts of underruns, trims and resyncs since start
        """
        drift_ms = 0.0
        if self._audio_capture is not None:
            drift_ms = round(self._audio_capture.drift_ns / 1_000_000, 1)
        return {
            'buffered_ms': round(self._buffered_ns() / 1_000_000, 1),
            # Age of the audio this callback emitted. The counterpart of
            # `JitterBuffer.video_age_ms`; the difference between the two is
            # the A/V skew, and nothing measured it before.
            'audio_age_ms': round(self._last_audio_age_ns / 1_000_000, 1),
            'underruns': self._underruns,
            'trims': self._trims,
            'resyncs': self._resyncs,
            'drift_ms': drift_ms,
            'out_latency_ms': round(self._output_latency_ns / 1_000_000, 1),
            'device': describe_device(self.device),
            'virtual': self.device is not None,
        }

    def start(self) -> None:
        """Open the audio output stream and begin playback."""
        if self._running:
            return

        try:
            import sounddevice as sd
        except ImportError:
            import sys
            print(
                '[AUDIO] sounddevice not installed — audio playback disabled.',
                file=sys.stderr,
            )
            return

        self._leftover = None
        self._needs_seek = True
        self._cursor_ts = 0
        self._output_latency_ns = 0

        # This audio is delayed to match the swapped video, so where it goes
        # decides whether the delay is useful or harmful. Into a virtual output
        # the conferencing app has selected as its microphone, it is the point
        # of the whole subsystem. Into the system default it is the operator
        # hearing themselves half a second late, while the call still receives
        # their real voice undelayed from the real microphone.
        if self.device is None:
            self.device = find_virtual_output()

        import sys
        if self.device is None:
            print(
                '[AUDIO] No virtual audio output found — playing to the system '
                'default. The call will NOT receive time-aligned audio: your '
                'microphone still reaches it undelayed, ahead of the swapped '
                'video. Install a virtual audio cable (VB-Audio on Windows, '
                'BlackHole on macOS) and select it as the microphone in your '
                'conferencing app.',
                file=sys.stderr,
            )
        else:
            print('[AUDIO] Playing to virtual output: {}'.format(
                describe_device(self.device)), file=sys.stderr)

        try:
            self._stream = sd.OutputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                dtype='float32',
                callback=self._output_callback,
            )
            self._stream.start()
            self._output_latency_ns = _stream_latency_ns(self._stream)
            self._running = True
        except Exception as e:
            wanted = device_sample_rate(self.device)
            print(
                '[AUDIO] Failed to start audio playback at {} Hz on {}: {}{}'
                .format(
                    self.sample_rate, describe_device(self.device), e,
                    '' if wanted in (None, self.sample_rate)
                    else ' — the device reports {} Hz'.format(wanted),
                ),
                file=sys.stderr,
            )
            self._stream = None

    def stop(self) -> None:
        """Stop playback and close the output stream."""
        if not self._running:
            return

        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                import sys
                print(f'[AUDIO] Error stopping audio playback: {e}', file=sys.stderr)
            self._stream = None
        self._leftover = None

    @property
    def is_running(self) -> bool:
        return self._running

    def try_recover(self) -> bool:
        """Attempt to restart the output stream after a failure.

        Returns True if the stream was successfully restarted.
        """
        if self._running and self._stream is not None:
            try:
                if self._stream.active:
                    return True
            except Exception:
                pass

        print('[AUDIO] Attempting playback stream recovery...', file=sys.stderr)
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._running = False
        self._leftover = None
        self.start()
        return self._running
