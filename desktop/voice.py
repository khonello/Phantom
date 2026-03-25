"""Voice transformation with formant-preserving pitch shifting.

Applies real-time pitch and formant adjustments to PCM audio chunks
to make the speaker sound like a different voice type (female, male,
child, deep). All processing is CPU-based — no GPU required.

Requires: parselmouth (pip install praat-parselmouth)
"""

import sys
from typing import Dict, Optional, Tuple

import numpy as np


# Preset: (pitch_shift_semitones, formant_shift_ratio)
# pitch_shift > 0 = higher pitch, formant_shift > 1.0 = higher formants
VOICE_PRESETS: Dict[str, Tuple[float, float]] = {
    'female': (4.0, 1.15),
    'male': (-3.5, 0.85),
    'child': (6.0, 1.25),
    'deep': (-5.0, 0.78),
}


class VoiceTransformer:
    """Transforms PCM audio chunks using formant-preserving pitch shift.

    Designed to sit between AudioCapture and AudioRingBuffer:
        mic → AudioCapture callback → VoiceTransformer.process() → ring buffer

    Thread-safety: instances are not shared across threads. The audio
    capture callback runs on a single dedicated thread.
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        self._sample_rate = sample_rate
        self._preset: Optional[str] = None
        self._pitch_semitones: float = 0.0
        self._formant_ratio: float = 1.0
        self._available = False

        try:
            import parselmouth  # noqa: F401
            self._available = True
        except ImportError:
            print(
                '[VOICE] praat-parselmouth not installed — voice transform '
                'disabled. Install with: pip install praat-parselmouth',
                file=sys.stderr,
            )

    @property
    def preset(self) -> Optional[str]:
        """Current voice preset name, or None if disabled."""
        return self._preset

    def set_preset(self, name: Optional[str]) -> None:
        """Set the active voice preset.

        Args:
            name: Preset name ('female', 'male', 'child', 'deep') or
                  None / 'none' to disable transformation.
        """
        if name is None or name.lower() == 'none':
            self._preset = None
            self._pitch_semitones = 0.0
            self._formant_ratio = 1.0
            return

        key = name.lower()
        if key not in VOICE_PRESETS:
            print(f'[VOICE] Unknown preset: {name!r}', file=sys.stderr)
            return

        self._preset = key
        self._pitch_semitones, self._formant_ratio = VOICE_PRESETS[key]

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Transform a PCM audio chunk.

        Args:
            pcm: float32 array, shape (frames,) or (frames, channels).

        Returns:
            Transformed float32 array with the same shape. Returns the
            input unchanged if no preset is active or parselmouth is
            unavailable.
        """
        if self._preset is None or not self._available:
            return pcm

        import parselmouth
        from parselmouth.praat import call

        original_shape = pcm.shape
        # parselmouth expects mono 1-D float64
        if pcm.ndim == 2:
            mono = pcm[:, 0].astype(np.float64)
        else:
            mono = pcm.astype(np.float64)

        # Avoid processing silence / near-silence
        if np.max(np.abs(mono)) < 1e-6:
            return pcm

        try:
            snd = parselmouth.Sound(mono, sampling_frequency=self._sample_rate)

            # Praat needs >= 3 periods of the lowest pitch to analyse.
            # For short chunks, raise min_pitch so 3 periods fit the duration.
            duration = len(mono) / self._sample_rate
            min_pitch = max(75.0, 3.0 / duration) if duration > 0 else 75.0

            # 1. Shift formants by resampling the spectral envelope
            if self._formant_ratio != 1.0:
                snd = call(
                    snd, "Change gender",
                    min_pitch,   # min pitch (Hz) for pitch detection
                    600.0,  # max pitch (Hz) for pitch detection
                    self._formant_ratio,  # formant shift ratio
                    0.0,    # new pitch median (0 = use factor instead)
                    2 ** (self._pitch_semitones / 12.0),  # pitch factor
                    1.0,    # duration factor
                )
            else:
                # Pure pitch shift without formant change
                snd = call(
                    snd, "Change gender",
                    min_pitch, 600.0,
                    1.0,
                    0.0,
                    2 ** (self._pitch_semitones / 12.0),
                    1.0,
                )

            result = snd.values[0].astype(np.float32)

            # Match original length (Praat may slightly change it)
            target_len = original_shape[0]
            if len(result) > target_len:
                result = result[:target_len]
            elif len(result) < target_len:
                result = np.pad(result, (0, target_len - len(result)))

            # Restore original shape
            if pcm.ndim == 2:
                result = result.reshape(-1, 1)
                if original_shape[1] > 1:
                    result = np.broadcast_to(
                        result, (target_len, original_shape[1])
                    ).copy()

            return result

        except Exception as e:
            print(f'[VOICE] Transform error: {e}', file=sys.stderr)
            return pcm
