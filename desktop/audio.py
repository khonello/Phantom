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

# Default audio parameters
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_BLOCK_SIZE = 1024  # ~23ms at 44100 Hz
DEFAULT_BUFFER_SECONDS = 10  # ring buffer capacity


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

    Returns:
        sounddevice device index, or None when nothing suitable is installed
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception:
        return None

    for hint in _VIRTUAL_OUTPUT_HINTS:
        for index, device in enumerate(devices):
            try:
                if int(device.get('max_output_channels', 0)) <= 0:
                    continue
                if hint in str(device.get('name', '')).lower():
                    return index
            except Exception:
                continue
    return None


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

        # Voice transformer (set externally via set_voice_transformer)
        self._voice_transformer: Optional['VoiceTransformer'] = None

        # Clock drift monitoring — tracks expected vs actual sample count
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

        capture_ts = time.perf_counter_ns()
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
        self._drift_start_ns = time.perf_counter_ns()
        self._drift_samples = 0

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
            self._running = True
        except Exception as e:
            import sys
            print(f'[AUDIO] Failed to start audio capture: {e}', file=sys.stderr)
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
        elapsed_ns = time.perf_counter_ns() - self._drift_start_ns
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
    CEILING_NS: int = 2_000_000_000      # 2 s  (accommodates RunPod RTT)
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

    MAX_FRAMES: int = 60  # ~2 seconds at 30 fps

    def __init__(self) -> None:
        self._buf: 'collections.deque[Any]' = collections.deque(maxlen=self.MAX_FRAMES)
        self._rtt = RTTTracker()

    def push(self, capture_ts: int, jpeg_bytes: bytes) -> None:
        """Enqueue a processed frame. Called from the WS receive thread.

        Args:
            capture_ts: Original capture timestamp (perf_counter_ns), or 0
            jpeg_bytes: JPEG-encoded processed frame
        """
        arrival_ts = time.perf_counter_ns()
        self._rtt.record(capture_ts, arrival_ts)
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
        target = self._rtt.target_delay_ns
        result: Optional[Tuple[int, bytes]] = None

        while self._buf:
            capture_ts, jpeg = self._buf[0]
            # Legacy frame without timestamp — display immediately
            if capture_ts <= 0:
                result = self._buf.popleft()
                continue
            age = now - capture_ts
            if age >= target:
                result = self._buf.popleft()
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
        """Current adaptive playout delay in nanoseconds."""
        return self._rtt.target_delay_ns

    def clear(self) -> None:
        """Discard all buffered frames and reset RTT statistics."""
        self._buf.clear()
        self._rtt.reset()

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
            'target_delay_ms': round(self._rtt.target_delay_ns / 1_000_000, 1),
            'buffer_depth': len(self._buf),
            'rtt_samples': len(samples),
            'rtt_p50_ms': round(p50_ms, 1),
            'rtt_p95_ms': round(p95_ms, 1),
            'rtt_stddev_ms': round(std_ms, 1),
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
                return                        # future audio; start at its head

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
        drift_ns = 0
        if self._audio_capture is not None:
            drift_ns = self._audio_capture.drift_ns
        playback_point = now - target + drift_ns

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
            _chunk_ts, pcm = chunk
            n = min(pcm.shape[0], frames - written)
            outdata[written:written + n] = pcm[:n]
            written += n
            if n < pcm.shape[0]:
                self._leftover = pcm[n:]

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
        return {
            'buffered_ms': round(self._buffered_ns() / 1_000_000, 1),
            'underruns': self._underruns,
            'trims': self._trims,
            'resyncs': self._resyncs,
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
            self._running = True
        except Exception as e:
            import sys
            print(f'[AUDIO] Failed to start audio playback: {e}', file=sys.stderr)
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
