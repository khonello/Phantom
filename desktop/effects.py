"""
Overlay effects — confetti and friends, drawn on top of everything.

The second of two decorative layers. A **filter** (`desktop/filters.py`) regrades
the whole picture; an **effect** draws moving things over it. Order is filter
first, effect second, because an effect is meant to sit on the picture rather
than be part of it — grading confetti would tint it to match a look it is
supposed to be separate from.

**Animation is a function of time, not of call count.** Every particle's
position is computed from a timestamp, so nothing here holds mutable state
between frames. That matters more than it looks: the same effect is rendered
from two different places at two different rates — the webcam thread's local
preview and the display timer's pipeline frames — and a `step()`-style animator
advanced by both would run at the sum of their rates and jitter with either.
Sampling a clock instead means any number of callers can render the same instant
and agree, with no locking and nothing to race.

Particles wrap rather than respawn, so there is no lifecycle to manage and the
loop is seamless. Nothing is loaded from disk: these are drawn, not decoded, so
there are no assets to bundle and no GIF to keep in step with a frame rate.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

# Particle counts are per effect. Chosen against the measured cost of drawing
# them — see the benchmark note in the module docstring of `filters.py`; the
# budget is the same 33ms display tick, shared with a filter.
_FIELD_CACHE: Dict[Tuple[str, int], Dict[str, npt.NDArray[Any]]] = {}


def _field(key: str, count: int) -> Dict[str, npt.NDArray[Any]]:
    """
    The fixed random properties of one effect's particles.

    Generated once from a seed derived from the effect's name, so a given
    effect always looks the same and two renderers of the same instant draw
    identical frames.

    Args:
        key: Effect key, used as the seed
        count: How many particles

    Returns:
        Dict of per-particle arrays
    """
    cache_key = (key, count)
    cached = _FIELD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
    field: Dict[str, npt.NDArray[Any]] = {
        'x': rng.random(count).astype(np.float32),
        'y': rng.random(count).astype(np.float32),
        'speed': (0.10 + rng.random(count) * 0.35).astype(np.float32),
        'drift': ((rng.random(count) - 0.5) * 0.22).astype(np.float32),
        'size': (rng.random(count)).astype(np.float32),
        'phase': (rng.random(count) * math.tau).astype(np.float32),
        'hue': rng.integers(0, 180, count).astype(np.uint8),
    }
    _FIELD_CACHE[cache_key] = field
    return field


def _positions(field: Dict[str, npt.NDArray[Any]], t: float, width: int, height: int,
               sway: float = 0.0) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Where every particle is at time `t`, in pixels.

    Vertical motion wraps with a modulo, which is what makes the loop seamless
    and removes any need to respawn.
    """
    y = np.mod(field['y'] + field['speed'] * t, 1.0)
    x = np.mod(
        field['x'] + field['drift'] * t
        + (np.sin(field['phase'] + t * 2.0) * sway if sway else 0.0),
        1.0,
    )
    return (x * width).astype(np.int32), (y * height).astype(np.int32)


def _hsv_colours(hue: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Saturated BGR colours from a hue array, built once per call."""
    hsv = np.stack([hue, np.full_like(hue, 220), np.full_like(hue, 255)], axis=1)
    return cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)


def _confetti(frame: npt.NDArray[Any], t: float) -> npt.NDArray[Any]:
    """Falling coloured rectangles, tumbling as they go."""
    height, width = frame.shape[:2]
    field = _field('confetti', 130)
    xs, ys = _positions(field, t, width, height, sway=0.03)
    colours = _hsv_colours(field['hue'])
    base = max(3, int(min(width, height) * 0.012))

    out = frame.copy()
    for i in range(len(xs)):
        w = base + int(field['size'][i] * base)
        # Tumble: the drawn height collapses and reopens, which reads as a flat
        # piece of paper turning over.
        h = max(1, int(abs(math.cos(field['phase'][i] + t * 3.0)) * w * 1.6))
        x, y = int(xs[i]), int(ys[i])
        colour = (int(colours[i][0]), int(colours[i][1]), int(colours[i][2]))
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, -1)
    return out


def _snow(frame: npt.NDArray[Any], t: float) -> npt.NDArray[Any]:
    """Soft white flakes, blended rather than painted on."""
    height, width = frame.shape[:2]
    field = _field('snow', 150)
    xs, ys = _positions(field, t, width, height, sway=0.05)
    base = max(2, int(min(width, height) * 0.006))

    layer = np.zeros_like(frame)
    for i in range(len(xs)):
        radius = base + int(field['size'][i] * base)
        cv2.circle(layer, (int(xs[i]), int(ys[i])), radius, (255, 255, 255), -1)
    return cv2.addWeighted(frame, 1.0, layer, 0.75, 0)


def _hearts(frame: npt.NDArray[Any], t: float) -> npt.NDArray[Any]:
    """Rising hearts, drawn as two circles and a triangle."""
    height, width = frame.shape[:2]
    field = _field('hearts', 60)
    xs, ys = _positions(field, t, width, height, sway=0.06)
    ys = height - ys  # rise instead of fall
    base = max(4, int(min(width, height) * 0.014))

    layer = np.zeros_like(frame)
    for i in range(len(xs)):
        s = base + int(field['size'][i] * base)
        x, y = int(xs[i]), int(ys[i])
        colour = (90, 60, 235)
        cv2.circle(layer, (x - s // 2, y), s // 2, colour, -1)
        cv2.circle(layer, (x + s // 2, y), s // 2, colour, -1)
        cv2.drawContours(
            layer,
            [np.array([[x - s, y], [x + s, y], [x, y + s + s // 2]], dtype=np.int32)],
            0, colour, -1,
        )
    return cv2.addWeighted(frame, 1.0, layer, 0.85, 0)


def _bubbles(frame: npt.NDArray[Any], t: float) -> npt.NDArray[Any]:
    """Rising outlined bubbles — hollow, so the picture shows through."""
    height, width = frame.shape[:2]
    field = _field('bubbles', 70)
    xs, ys = _positions(field, t, width, height, sway=0.07)
    ys = height - ys
    base = max(4, int(min(width, height) * 0.018))

    layer = np.zeros_like(frame)
    for i in range(len(xs)):
        radius = base + int(field['size'][i] * base)
        cv2.circle(layer, (int(xs[i]), int(ys[i])), radius, (255, 235, 200), 2)
    return cv2.addWeighted(frame, 1.0, layer, 0.7, 0)


def _sparkle(frame: npt.NDArray[Any], t: float) -> npt.NDArray[Any]:
    """Twinkling points that fade in and out rather than travelling far."""
    height, width = frame.shape[:2]
    field = _field('sparkle', 110)
    xs = (field['x'] * width).astype(np.int32)
    ys = (field['y'] * height).astype(np.int32)
    twinkle = (np.sin(field['phase'] + t * 4.0) + 1.0) * 0.5
    base = max(2, int(min(width, height) * 0.005))

    layer = np.zeros_like(frame)
    for i in range(len(xs)):
        if twinkle[i] < 0.45:
            continue
        radius = max(1, int(base * twinkle[i] * 1.8))
        value = int(200 + 55 * twinkle[i])
        cv2.circle(layer, (int(xs[i]), int(ys[i])), radius, (value, value, value), -1)
    return cv2.addWeighted(frame, 1.0, layer, 0.9, 0)


class Effect:
    """One named overlay, and the function that draws it at a moment in time."""

    def __init__(self, key: str, name: str,
                 draw: Optional[Callable[[npt.NDArray[Any], float], npt.NDArray[Any]]]) -> None:
        """
        Args:
            key: Stable identifier used by the UI
            name: Label shown to the operator
            draw: Renderer taking (frame, seconds), or None for no overlay
        """
        self.key = key
        self.name = name
        self._draw = draw

    def __call__(self, frame: npt.NDArray[Any], t: float) -> npt.NDArray[Any]:
        if self._draw is None:
            return frame
        return self._draw(frame, t)


EFFECTS: List[Effect] = [
    Effect('none', 'None', None),
    Effect('confetti', 'Confetti', _confetti),
    Effect('snow', 'Snow', _snow),
    Effect('hearts', 'Hearts', _hearts),
    Effect('bubbles', 'Bubbles', _bubbles),
    Effect('sparkle', 'Sparkle', _sparkle),
]

_BY_KEY: Dict[str, Effect] = {e.key: e for e in EFFECTS}


def get(key: str) -> Optional[Effect]:
    """One effect by key, or None if there is no such effect."""
    return _BY_KEY.get(key)


def render(frame: npt.NDArray[Any], key: str, t: float) -> Any:
    """
    Draw an effect over a frame, returning it unchanged if it does not apply.

    Args:
        frame: BGR frame, already swapped and graded
        key: Effect key, possibly empty or unknown
        t: Seconds — any monotonic clock, since only differences matter

    Returns:
        The frame with the overlay drawn, or the input unchanged
    """
    if not key or key == 'none':
        return frame
    chosen = _BY_KEY.get(key)
    if chosen is None:
        return frame
    try:
        return chosen(frame, t)
    except Exception:
        # Decoration must never be the reason a frame is lost.
        return frame


def names() -> List[Dict[str, str]]:
    """Every effect as {key, name}, for the picker."""
    return [{'key': e.key, 'name': e.name} for e in EFFECTS]
