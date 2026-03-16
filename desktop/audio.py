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
import threading
import time
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

# PCM chunk: (capture_ts_ns, pcm_data as float32 numpy array)
AudioChunk = Tuple[int, np.ndarray]

# Default audio parameters
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_BLOCK_SIZE = 1024  # ~23ms at 44100 Hz
DEFAULT_BUFFER_SECONDS = 10  # ring buffer capacity


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

    def append(self, capture_ts: int, pcm: np.ndarray) -> None:
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

    def snapshot(self) -> list:
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

        self._stream: Optional[object] = None
        self._running = False

        # Clock drift monitoring — tracks expected vs actual sample count
        self._drift_start_ns: int = 0
        self._drift_samples: int = 0
        # Threshold in seconds: warn if audio clock drifts more than this
        self._drift_warn_threshold: float = 0.05  # 50 ms

    def _audio_callback(
        self,
        indata: np.ndarray,
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
        self.ring_buffer.append(capture_ts, indata.copy())

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
                active = self._stream.active  # type: ignore[union-attr]
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
                if self._stream.active:  # type: ignore[union-attr]
                    return True  # still alive, nothing to do
            except Exception:
                pass

        # Stream died — close and reopen
        print('[AUDIO] Attempting capture stream recovery...', file=sys.stderr)
        if self._stream is not None:
            try:
                self._stream.close()  # type: ignore[union-attr]
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

        target_delay = mean(rtt) + 2 * stddev(rtt)

    Clamped to [FLOOR_NS, CEILING_NS] and updated via exponential smoothing
    every UPDATE_INTERVAL frames to prevent oscillation.
    """

    WINDOW_SIZE: int = 30         # ~1 second at 30 fps
    UPDATE_INTERVAL: int = 10     # recalculate every N samples
    SMOOTHING_ALPHA: float = 0.1  # exponential smoothing factor
    FLOOR_NS: int = 80_000_000            # 80 ms
    CEILING_NS: int = 2_000_000_000      # 2 s  (accommodates RunPod RTT)
    INITIAL_DELAY_NS: int = 400_000_000  # 400 ms (session warmup for remote GPU)
    WARMUP_SAMPLES: int = 10            # min samples before adapting

    def __init__(self) -> None:
        self._samples: collections.deque = collections.deque(maxlen=self.WINDOW_SIZE)
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
        raw = int(float(np.mean(arr)) + 2.0 * float(np.std(arr)))
        clamped = max(self.FLOOR_NS, min(self.CEILING_NS, raw))
        # Exponential smoothing prevents sudden jumps
        self._target_delay_ns = int(
            self.SMOOTHING_ALPHA * clamped
            + (1.0 - self.SMOOTHING_ALPHA) * self._target_delay_ns
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
        self._buf: collections.deque = collections.deque(maxlen=self.MAX_FRAMES)
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

        Returns a dict with:
            target_delay_ms: current adaptive playout delay
            buffer_depth: number of frames waiting in the jitter buffer
            rtt_samples: number of RTT samples in the current window
            rtt_mean_ms: mean RTT over the sliding window (0 if empty)
            rtt_stddev_ms: stddev of RTT over the sliding window (0 if empty)
        """
        samples = self._rtt._samples
        if len(samples) >= 2:
            arr = np.array(samples, dtype=np.float64)
            mean_ms = float(np.mean(arr)) / 1_000_000
            std_ms = float(np.std(arr)) / 1_000_000
        elif len(samples) == 1:
            mean_ms = samples[0] / 1_000_000
            std_ms = 0.0
        else:
            mean_ms = 0.0
            std_ms = 0.0

        return {
            'target_delay_ms': round(self._rtt.target_delay_ns / 1_000_000, 1),
            'buffer_depth': len(self._buf),
            'rtt_samples': len(samples),
            'rtt_mean_ms': round(mean_ms, 1),
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
    ) -> None:
        """
        Args:
            ring_buffer: AudioRingBuffer filled by AudioCapture
            jitter_buffer: JitterBuffer whose target_delay_ns drives sync
            sample_rate: Must match the capture sample rate
            channels: Must match the capture channel count
            block_size: OutputStream block size (frames per callback)
        """
        self._ring = ring_buffer
        self._jitter = jitter_buffer
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size

        self._stream: Optional[object] = None
        self._running = False
        # Leftover samples from a partially consumed chunk
        self._leftover: Optional[np.ndarray] = None

    def _output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """sounddevice OutputStream callback — runs on a dedicated audio thread.

        Computes the current playback point, discards stale audio, and fills
        *outdata* with the PCM samples that correspond to the video being
        displayed right now. Any remaining space is zero-filled (silence).
        """
        if status:
            import sys
            print(f'[AUDIO] playback status: {status}', file=sys.stderr)

        now = time.perf_counter_ns()
        target = self._jitter.target_delay_ns
        playback_point = now - target

        written = 0

        # 1. Drain leftover from a previous partially-consumed chunk
        if self._leftover is not None and self._leftover.shape[0] > 0:
            n = min(self._leftover.shape[0], frames - written)
            outdata[written:written + n] = self._leftover[:n]
            written += n
            if n < self._leftover.shape[0]:
                self._leftover = self._leftover[n:]
            else:
                self._leftover = None

        # 2. Consume chunks from the ring buffer
        while written < frames:
            chunk = self._ring.peek_oldest()
            if chunk is None:
                break

            chunk_ts, pcm = chunk
            chunk_dur_ns = int(pcm.shape[0] / self.sample_rate * 1_000_000_000)

            # Chunk ended before playback point — too old, discard
            if chunk_ts + chunk_dur_ns < playback_point:
                self._ring.popleft()
                continue

            # Chunk starts after playback point — too early, wait
            if chunk_ts > playback_point:
                break

            # Chunk overlaps playback point — consume it
            self._ring.popleft()
            n = min(pcm.shape[0], frames - written)
            outdata[written:written + n] = pcm[:n]
            written += n
            if n < pcm.shape[0]:
                self._leftover = pcm[n:]

        # 3. Fill remainder with silence
        if written < frames:
            outdata[written:] = 0.0

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

        try:
            self._stream = sd.OutputStream(
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
                if self._stream.active:  # type: ignore[union-attr]
                    return True
            except Exception:
                pass

        print('[AUDIO] Attempting playback stream recovery...', file=sys.stderr)
        if self._stream is not None:
            try:
                self._stream.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._stream = None

        self._running = False
        self._leftover = None
        self.start()
        return self._running
