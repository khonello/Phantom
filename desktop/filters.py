"""
Look filters — a decorative layer composited over a finished frame.

Filters are the **last layer**, applied after the swap has fully composited.
That ordering is not cosmetic: `FaceCompositor` matches the swapped face's
colour and grain to the frame around it, so grading before the swap would mean
matching the face to an already-graded frame and then grading it a second time.
Applied last, a filter can change how the picture reads without ever being able
to break the swap underneath it.

They run **on the desktop**, not in the pipeline, for three reasons that all
point the same way: they need nothing the face models provide, they must not
compete for a latency budget the swap has not yet been measured against, and
changing one should be a local variable rather than a round trip to a rented
GPU.

Everything here is a lookup table or a cached multiply. Per-pixel Python would
not hold 30fps at 960x540; `cv2.LUT` and a precomputed mask do, and the cost
when no filter is enabled is exactly zero because the caller keeps its original
path.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


def _curve(shadows: float, midtones: float, highlights: float) -> npt.NDArray[Any]:
    """
    Build a 256-entry tone curve from three control points.

    Args:
        shadows: Output for input 0, in [0, 1]
        midtones: Output for input 0.5, in [0, 1]
        highlights: Output for input 1, in [0, 1]

    Returns:
        uint8 lookup table
    """
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([shadows, midtones, highlights])
    ramp = np.linspace(0.0, 1.0, 256)
    return np.clip(np.interp(ramp, xs, ys) * 255.0, 0, 255).astype(np.uint8)


def _channel_lut(blue: npt.NDArray[Any], green: npt.NDArray[Any], red: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Stack three per-channel curves into one BGR lookup table."""
    return np.dstack([blue, green, red]).astype(np.uint8)


# Per-channel curves, built once. BGR order throughout — this is OpenCV.
_WARM = _channel_lut(
    blue=_curve(0.0, 0.44, 0.94),
    green=_curve(0.0, 0.50, 0.99),
    red=_curve(0.02, 0.57, 1.0),
)

_COOL = _channel_lut(
    blue=_curve(0.03, 0.57, 1.0),
    green=_curve(0.0, 0.51, 0.99),
    red=_curve(0.0, 0.45, 0.95),
)

_SEPIA = _channel_lut(
    blue=_curve(0.05, 0.40, 0.83),
    green=_curve(0.05, 0.50, 0.93),
    red=_curve(0.08, 0.60, 1.0),
)

# Lifted blacks and pulled highlights — the faded-film look.
_FADE = _channel_lut(
    blue=_curve(0.13, 0.53, 0.92),
    green=_curve(0.11, 0.51, 0.91),
    red=_curve(0.10, 0.52, 0.93),
)

# A straight S-free contrast stretch about mid-grey, for the looks that want
# punch rather than a colour cast.
_CONTRAST = np.clip(
    ((np.arange(256, dtype=np.float32) / 255.0 - 0.5) * 1.28 + 0.5) * 255.0, 0, 255
).astype(np.uint8)

# Vignette masks are shape-dependent, so they are built on first use and kept.
_VIGNETTE_CACHE: Dict[Tuple[int, int, int], npt.NDArray[Any]] = {}


def _vignette_mask(shape: Tuple[int, int], strength: int) -> npt.NDArray[Any]:
    """
    A radial falloff mask for the given frame shape, cached.

    Args:
        shape: (height, width)
        strength: 0-100, how dark the corners go

    Returns:
        float32 mask in [0, 1], shaped (height, width, 3)
    """
    height, width = shape
    key = (height, width, strength)
    cached = _VIGNETTE_CACHE.get(key)
    if cached is not None:
        return cached

    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs * xs + ys * ys) / np.sqrt(2.0)
    falloff = 1.0 - (strength / 100.0) * np.clip(radius - 0.35, 0.0, None) ** 1.6 * 2.2
    mask = np.repeat(np.clip(falloff, 0.0, 1.0)[:, :, None], 3, axis=2)

    # One mask per (shape, strength) is a few hundred KB; a handful is fine, an
    # unbounded cache after many resizes is not.
    if len(_VIGNETTE_CACHE) > 8:
        _VIGNETTE_CACHE.clear()
    _VIGNETTE_CACHE[key] = mask
    return mask


def _apply_vignette(frame: npt.NDArray[Any], strength: int) -> npt.NDArray[Any]:
    """
    Darken the edges.

    `cv2.multiply` rather than a numpy `astype` round trip: the numpy form
    allocates two float32 copies of the whole frame and measured ~13ms at
    960x540, which is half the budget of the timer that drives the display.
    """
    mask = _vignette_mask((frame.shape[0], frame.shape[1]), strength)
    return cv2.multiply(frame, mask, dtype=cv2.CV_8U)


def _saturate(frame: npt.NDArray[Any], factor: float) -> npt.NDArray[Any]:
    """
    Scale saturation by pushing the frame away from its own greyscale.

    Equivalent in effect to scaling S in HSV, without the two colour-space
    conversions that made that form ~10ms a frame. Greyscale is the zero-
    saturation version of the image, so extrapolating past it saturates.
    """
    grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, factor, grey, 1.0 - factor, 0)


def _mono(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def _noir(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.LUT(grey, _CONTRAST)
    return _apply_vignette(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), 55)


def _vivid(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return cv2.LUT(_saturate(frame, 1.35), _CONTRAST)


def _soft(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    A gentle diffusion — bloom, not blur.

    The blurred copy is mixed *under* the original rather than replacing it, so
    edges stay where they were and only the light spreads. A straight blur here
    would undo the detail matching the compositor works to preserve.
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), 6.0)
    return cv2.addWeighted(frame, 0.78, blurred, 0.22, 0)


class Filter:
    """One named look, and the function that applies it."""

    def __init__(self, key: str, name: str, apply: Optional[Callable[[npt.NDArray[Any]], npt.NDArray[Any]]]) -> None:
        """
        Args:
            key: Stable identifier used by the UI and stored in settings
            name: Label shown to the operator
            apply: Transform on a BGR frame, or None for the identity
        """
        self.key = key
        self.name = name
        self._apply = apply

    def __call__(self, frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self._apply is None:
            return frame
        return self._apply(frame)


FILTERS: List[Filter] = [
    Filter('none', 'None', None),
    Filter('warm', 'Warm', lambda f: cv2.LUT(f, _WARM)),
    Filter('cool', 'Cool', lambda f: cv2.LUT(f, _COOL)),
    Filter('vivid', 'Vivid', _vivid),
    Filter('fade', 'Fade', lambda f: cv2.LUT(f, _FADE)),
    Filter('sepia', 'Sepia', lambda f: cv2.LUT(f, _SEPIA)),
    Filter('mono', 'Mono', _mono),
    Filter('noir', 'Noir', _noir),
    Filter('soft', 'Soft', _soft),
    Filter('vignette', 'Vignette', lambda f: _apply_vignette(f, 60)),
]

_BY_KEY: Dict[str, Filter] = {f.key: f for f in FILTERS}


def get(key: str) -> Optional[Filter]:
    """One filter by key, or None if there is no such filter."""
    return _BY_KEY.get(key)


def apply(frame: npt.NDArray[Any], key: str) -> npt.NDArray[Any]:
    """
    Apply a filter by key, returning the frame unchanged if it does not apply.

    An unknown key is not an error: a filter removed in a later version must
    degrade to no filter rather than failing a live call.

    Args:
        frame: BGR frame to grade
        key: Filter key, possibly empty or unknown

    Returns:
        The graded frame, or the input unchanged
    """
    if not key or key == 'none':
        return frame
    chosen = _BY_KEY.get(key)
    if chosen is None:
        return frame
    try:
        return chosen(frame)
    except Exception:
        # A filter is decorative. It must never be the reason a frame is lost.
        return frame


def names() -> List[Dict[str, str]]:
    """Every filter as {key, name}, for the picker."""
    return [{'key': f.key, 'name': f.name} for f in FILTERS]
