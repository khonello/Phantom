"""
Skin segmentation — where on the frame the person's skin is, face and body.

Built for RESEMBLANCE.md §3.2. Three things want it and none of them had it:
the complexion reading needs a neck to say whether the face and the rest of
the person agree (`complexion_neck`, `complexion_seam`); Route A grades every
visible skin pixel toward the source's complexion and cannot grade what it
cannot find; and Route C composites a reenacted head onto the target's body
along exactly this boundary.

`SkinSegmenter.segment` returns two masks, frame-sized, float32 in [0, 1]:

    face   the landmark hull minus eyes, nostrils and mouth — the texture
           layer's own `skin_mask`, warped back into frame space, so every
           stage that says "skin" means the same pixels
    body   classified skin OUTSIDE the face: neck, ears, chest, hands

A registry of backends, in the shape of `backgrounds.MODELS` and for the same
reason — the backend owns the facts about itself. One is built:

**`seeded`** (default). No model file. The face's own skin pixels are the
sample — a real face, this person, under this light — and a colour model is
fitted to them every frame: the median and spread in the a/b plane of LAB,
with a wide lightness gate because a neck in shadow and a forehead in light
are the same pigment. Every pixel in the frame is scored against it. It adapts
to the lighting for free and costs a few milliseconds at half resolution.

What it cannot do, stated up front: it fires on skin-coloured backgrounds —
a pine door, a beige wall, a wooden desk — because it knows colour and not
anatomy. Component filtering removes specks; it does not remove a wall. The
answer to that is the second backend, a parsing model with face-skin and
body-skin classes (MediaPipe's multiclass selfie export is the candidate),
which is registered as a spec below and **not yet built**: its pre- and
post-processing are not verified, and a wrong convention there produces a
washed-out matte that reads as a weak model rather than a wrong input. The
registry exists so that it lands as an entry, not a fork.

Smoothing is on the **parameters**, not the matte. An EMA on a median and a
spread cannot crawl; an EMA on a boundary can, and the neck is the longest
boundary in the picture. A light EMA on the mask sits on top for the edge.
Reset on face loss and on source change, with the compositor's temporal state.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.processing.geometry import canonical_from_frame
from pipeline.processing.texture import skin_mask
from pipeline.types import Face, Frame, Mask


# Canonical edge the face skin mask is drawn at before being warped back.
_FACE_MASK_SIZE = 256

# Classification runs at this fraction of frame size. A skin boundary is a
# slowly varying thing and the mask is feathered afterwards; a 640x360 frame
# at half resolution is 57,600 pixels, which is what makes this affordable
# every frame rather than every fifth.
_WORK_SCALE = 0.5

# Too few sample pixels and the colour model is a guess — a face at the frame
# edge, or one small enough that the feature exclusions ate it.
_MIN_SAMPLE = 200

# How far from the face's median chroma still counts as skin, in units of the
# sample's own spread (median absolute deviation). Three MADs covers the
# pigment variation across one person's face and neck; a wall of the same hue
# is usually outside it and sometimes not.
_CHROMA_TOLERANCE = 3.5

# Floor on the spread, in 8-bit LAB units, so a very flat sample (a plain
# face under soft light) does not produce a model so tight the neck fails it.
_MIN_SPREAD = 2.5

# Lightness gate, as multiples of the face's median L. The neck sits in the
# jaw's shadow and the chest below it; hands catch a different light again.
# Wide on purpose — chroma is what separates skin from not-skin here. Widened
# after the first footage run (2026-09-14): a face lit by a phone torch with
# the neck in room light put the neck under 0.45 of the face's L, and the
# seam that produced was measured at 6.3 units.
_L_LOW = 0.30
_L_HIGH = 1.60

# How far the face hull is grown before the body mask is taken outside it, as
# a fraction of the face's extent. The ring this leaves is the jaw feather —
# neither face nor body, and neither reading should sample it.
_FACE_GROW = 0.08

# A body component smaller than this fraction of the face's area is a speck —
# a bright spot on a wall, an earring — and is dropped. Hands are far larger.
_MIN_COMPONENT = 0.05

# The second seed: a corridor directly under the face hull, where the pixels
# are the person's neck and chest under THEIR OWN light — which on a call is
# routinely not the face's light (a lamp, a window, a phone torch held at the
# face). A model fitted from the face alone gates on the face's lightness and
# loses whatever sits in a different light; measured 2026-09-14, that was the
# shoulders. Width in face widths either side of centre, depth in face heights
# below the hull, and a loose admission test so the corridor sample is skin
# and not a collar.
_NECK_WIDTH = 0.55
_NECK_DEPTH = 0.7
_NECK_ADMIT_CHROMA = 6.0     # MADs from the FACE model, loose on purpose
_NECK_ADMIT_L_LOW = 0.12     # of the face's L — a neck in deep shadow still counts
_NECK_ADMIT_L_HIGH = 1.6

# Smoothing. The parameters are medians already, but under a hunting
# auto-exposure the face's median moves every frame and a fast EMA follows
# it; 0.25 is roughly four frames at 15fps. The mask edge a little slower.
_PARAM_ALPHA = 0.25
_MASK_ALPHA = 0.6

# Feather at the body mask's edge, in pixels at working scale.
_FEATHER = 2.0


@dataclass(frozen=True)
class SkinBackend:
    """
    One way of finding skin, and the facts about it.

    Attributes:
        key: Name selected by `skin_model`
        needs_file: Whether a weight file has to be present
        filename: The weight file, if any
        built: Whether this backend runs today. A registered spec that is not
            built is refused with a message, never silently substituted
        notes: What is easy to get wrong about it
    """

    key: str
    needs_file: bool
    filename: str
    built: bool
    notes: str


MODELS: List[SkinBackend] = [
    SkinBackend(
        key='seeded',
        needs_file=False,
        filename='',
        built=True,
        notes=('Colour model fitted per frame from the face\'s own skin. Knows '
               'colour, not anatomy: fires on skin-coloured backgrounds.'),
    ),
    SkinBackend(
        key='mediapipe_multiclass',
        needs_file=True,
        filename='selfie_multiclass_256x256.onnx',
        built=False,
        notes=('Parsing model with face-skin and body-skin classes. Pre- and '
               'post-processing unverified; register the conventions before '
               'trusting a matte from it.'),
    ),
]

_BY_KEY: Dict[str, SkinBackend] = {m.key: m for m in MODELS}
DEFAULT = 'seeded'


def names() -> List[str]:
    """Registered backend keys, built or not."""
    return [m.key for m in MODELS]


def resolve(key: Optional[str]) -> SkinBackend:
    """
    The backend for a config value, falling back to the default.

    An unknown or unbuilt name lands on the default rather than raising —
    this is a measurement service and a decorative stage's input, and a
    typo in `.env` must not refuse a session. The gap between requested
    and resolved is reported by `tools/stats.py`, the same as the model
    registries.
    """
    backend = _BY_KEY.get((key or '').strip().lower())
    if backend is None or not backend.built:
        return _BY_KEY[DEFAULT]
    return backend


@dataclass
class SkinMasks:
    """The result for one frame. Both masks frame-sized, float32 in [0, 1]."""

    face: Mask
    body: Mask

    @property
    def skin(self) -> Mask:
        """Face and body together."""
        return np.maximum(self.face, self.body)


class SkinSegmenter:
    """
    Find the person's skin in a frame, given their detected face.

    Example:
        segmenter = SkinSegmenter()
        masks = segmenter.segment(frame, face)
        if masks is not None:
            neck_and_hands = masks.body
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        self.backend = resolve(backend)
        self.last_ms: float = 0.0
        self._centre: Optional[np.ndarray] = None
        self._spread: Optional[np.ndarray] = None
        self._lightness: Optional[float] = None
        self._body: Optional[Mask] = None
        # The neck model, when the corridor under the face yields one.
        self._neck_centre: Optional[np.ndarray] = None
        self._neck_spread: Optional[np.ndarray] = None
        self._neck_lightness: Optional[float] = None
        # Whether the last frame found a neck model — for the readings.
        self.last_neck_seeded: bool = False

    def reset(self) -> None:
        """Drop the smoothed model. Face lost, source changed, stream restart."""
        self._centre = None
        self._spread = None
        self._lightness = None
        self._body = None
        self._neck_centre = None
        self._neck_spread = None
        self._neck_lightness = None

    @property
    def sample_lab(self) -> Optional[np.ndarray]:
        """
        The smoothed (L, a, b) of the face skin the model was last fitted to.

        The same sample the classifier uses, already EMA'd, so a stage that
        needs the target's complexion reads one number rather than measuring
        the face again and smoothing the result a second time.
        """
        if self._centre is None or self._lightness is None:
            return None
        return np.array(
            [self._lightness, float(self._centre[0]), float(self._centre[1])],
            dtype=np.float64)

    def segment(self, frame: Frame, face: Face) -> Optional[SkinMasks]:
        """
        Face and body skin masks for one frame.

        Args:
            frame: BGR frame
            face: The detection made on it

        Returns:
            The masks, or None when no face skin could be sampled — in which
            case the smoothed model is left alone rather than reset, since one
            bad frame is not a lost face
        """
        started = time.perf_counter()
        result = self._seeded(frame, face)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return result

    # ------------------------------------------------------------------
    # The seeded backend
    # ------------------------------------------------------------------

    def _face_masks(
        self, face: Face, full: Tuple[int, int], work: Tuple[int, int],
    ) -> Optional[Tuple[Mask, Mask]]:
        """
        The texture layer's skin mask, warped into frame space at two sizes.

        One canonical mask, two warps — cheaper than warping once and
        resizing, and the working-scale one is exact rather than an
        area-average of the full one.

        Args:
            face: The detection
            full: (width, height) of the frame
            work: (width, height) at working scale

        Returns:
            (full-size mask, working-size mask), or None without geometry
        """
        matrix = canonical_from_frame(face, _FACE_MASK_SIZE)
        if matrix is None:
            return None
        canonical = skin_mask(face, matrix, _FACE_MASK_SIZE)

        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        at_full = cv2.warpAffine(
            canonical, matrix, full, flags=flags,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # frame -> canonical is `matrix`; working -> canonical is the same map
        # with the working scale undone on the input side: the linear part
        # divided by the scale, the translation untouched.
        scaled = matrix.copy()
        scaled[:, :2] /= _WORK_SCALE
        at_work = cv2.warpAffine(
            canonical, scaled, work, flags=flags,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return (np.asarray(at_full, dtype=np.float32),
                np.asarray(at_work, dtype=np.float32))

    def _seeded(self, frame: Frame, face: Face) -> Optional[SkinMasks]:
        height, width = frame.shape[:2]
        work_w = max(8, int(round(width * _WORK_SCALE)))
        work_h = max(8, int(round(height * _WORK_SCALE)))

        masks = self._face_masks(face, (width, height), (work_w, work_h))
        if masks is None:
            return None
        face_full, face_small = masks

        small = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0]
        chroma = lab[:, :, 1:]

        inside = face_small > 0.5
        sample = chroma[inside]
        if sample.shape[0] < _MIN_SAMPLE:
            return None

        # Fit this frame's model from the face, then smooth the *parameters*.
        centre = np.median(sample, axis=0).astype(np.float32)
        spread = np.maximum(
            np.median(np.abs(sample - centre), axis=0) * 1.4826,
            _MIN_SPREAD,
        ).astype(np.float32)
        lightness = float(np.median(light[inside]))
        if self._centre is None or self._spread is None or self._lightness is None:
            self._centre, self._spread, self._lightness = centre, spread, lightness
        else:
            a = _PARAM_ALPHA
            self._centre = (a * centre + (1.0 - a) * self._centre).astype(np.float32)
            self._spread = (a * spread + (1.0 - a) * self._spread).astype(np.float32)
            self._lightness = a * lightness + (1.0 - a) * self._lightness

        score = _score(chroma, light, self._centre, self._spread, self._lightness)

        # Body is skin outside the grown face hull. The ring between the hull
        # and its growth is the jaw feather and belongs to neither.
        extent = _face_extent(face) * _WORK_SCALE
        grow = max(1, int(round(extent * _FACE_GROW)))
        hull = _hull_mask(face, (work_h, work_w), _WORK_SCALE)
        grown = cv2.dilate(hull, self._kernel(grow))

        # The second seed. Skin in the corridor under the face, admitted by a
        # loose test against the FACE model, then fitted as its own model so
        # the neck's light gates the neck. A pixel is skin if EITHER model
        # says so.
        self.last_neck_seeded = False
        corridor = _neck_corridor(face, hull, (work_h, work_w), _WORK_SCALE)
        if corridor is not None:
            admit = corridor & (grown <= 0.5)
            if int(admit.sum()) >= _MIN_SAMPLE:
                # Scored on the corridor's rows and columns only — the loose
                # test is consulted nowhere else, and the corridor is a
                # rectangle, so this is the same answer over a fifth of the
                # pixels.
                c_rows = np.flatnonzero(corridor.any(axis=1))
                c_cols = np.flatnonzero(corridor.any(axis=0))
                r0, r1 = int(c_rows[0]), int(c_rows[-1]) + 1
                k0, k1 = int(c_cols[0]), int(c_cols[-1]) + 1
                loose = np.zeros(chroma.shape[:2], dtype=np.float32)
                loose[r0:r1, k0:k1] = _score(
                    chroma[r0:r1, k0:k1], light[r0:r1, k0:k1], self._centre,
                    self._spread * (_NECK_ADMIT_CHROMA / _CHROMA_TOLERANCE),
                    self._lightness, low=_NECK_ADMIT_L_LOW, high=_NECK_ADMIT_L_HIGH)
                seed = admit & (loose > 0.5)
                if int(seed.sum()) >= _MIN_SAMPLE:
                    neck_sample = chroma[seed]
                    n_centre = np.median(neck_sample, axis=0).astype(np.float32)
                    n_spread = np.maximum(
                        np.median(np.abs(neck_sample - n_centre), axis=0) * 1.4826,
                        _MIN_SPREAD).astype(np.float32)
                    n_light = float(np.median(light[seed]))
                    if (self._neck_centre is None or self._neck_spread is None
                            or self._neck_lightness is None):
                        self._neck_centre, self._neck_spread, self._neck_lightness = (
                            n_centre, n_spread, n_light)
                    else:
                        a = _PARAM_ALPHA
                        self._neck_centre = (a * n_centre + (1.0 - a) * self._neck_centre).astype(np.float32)
                        self._neck_spread = (a * n_spread + (1.0 - a) * self._neck_spread).astype(np.float32)
                        self._neck_lightness = a * n_light + (1.0 - a) * self._neck_lightness
                    self.last_neck_seeded = True
        if (self.last_neck_seeded and self._neck_centre is not None
                and self._neck_spread is not None and self._neck_lightness is not None):
            neck_score = _score(chroma, light, self._neck_centre,
                                self._neck_spread, self._neck_lightness)
            np.maximum(score, neck_score, out=score)

        score[grown > 0.5] = 0.0

        body = self._clean(score, face_area=float(np.count_nonzero(hull > 0.5)))

        if self._body is not None and self._body.shape == body.shape:
            cv2.addWeighted(body, _MASK_ALPHA, self._body, 1.0 - _MASK_ALPHA, 0.0, dst=body)
        self._body = body

        body_full = cv2.resize(body, (width, height), interpolation=cv2.INTER_LINEAR)
        return SkinMasks(face=face_full, body=np.asarray(body_full, dtype=np.float32))

    _kernels: Dict[int, np.ndarray] = {}

    @classmethod
    def _kernel(cls, grow: int) -> np.ndarray:
        """Structuring element for a growth radius, built once per radius."""
        kernel = cls._kernels.get(grow)
        if kernel is None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
            cls._kernels[grow] = kernel
        return kernel

    @staticmethod
    def _clean(body: Mask, face_area: float) -> Mask:
        """Open, drop specks, feather."""
        binary = np.asarray(cv2.morphologyEx(
            (body > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)), dtype=np.uint8)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        floor = max(16.0, face_area * _MIN_COMPONENT)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= floor:
                keep[labels == index] = 1

        # Feather the kept components. Blurring the binary rather than the
        # score means a removed speck cannot bleed back in at the edge.
        soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), _FEATHER)
        return np.asarray(np.clip(soft, 0.0, 1.0), dtype=np.float32)


def _score(
    chroma: np.ndarray, light: np.ndarray,
    centre: np.ndarray, spread: np.ndarray, lightness: float,
    low: float = _L_LOW, high: float = _L_HIGH,
) -> np.ndarray:
    """
    Skin score per pixel against one colour model.

    Chroma distance in units of the model's spread, ramped from 1 at one
    spread to 0 at `_CHROMA_TOLERANCE`, gated on lightness as multiples of the
    model's own L. One function for the face model and the neck model, so the
    two cannot disagree about what "matches" means.
    """
    delta = chroma.astype(np.float32)
    delta -= centre
    delta /= spread
    np.multiply(delta, delta, out=delta)
    distance2 = delta[:, :, 0]
    distance2 += delta[:, :, 1]
    score = np.sqrt(distance2, out=distance2)
    score -= 1.0
    score /= (_CHROMA_TOLERANCE - 1.0)
    np.subtract(1.0, score, out=score)
    np.clip(score, 0.0, 1.0, out=score)
    score[(light <= lightness * low) | (light >= lightness * high)] = 0.0
    return np.asarray(score, dtype=np.float32)


def _neck_corridor(
    face: Face, hull: Mask, shape: Tuple[int, int], scale: float,
) -> Optional[np.ndarray]:
    """
    The region directly under the face hull, at working scale, as a boolean.

    Where the neck and upper chest are on anyone facing a camera. Bounded by
    face widths either side of the face's centre and face heights below the
    hull's lowest point, so a raised hand or a wall never lands in it.
    """
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return None
    box = np.asarray(bbox, dtype=np.float64).reshape(-1) * scale
    if box.size < 4:
        return None
    rows = np.flatnonzero(hull.max(axis=1) > 0.5)
    if rows.size == 0:
        return None
    width = max(box[2] - box[0], 1.0)
    height = max(box[3] - box[1], 1.0)
    cx = (box[0] + box[2]) / 2.0
    top = int(rows[-1]) + 1
    bottom = min(shape[0], int(top + height * _NECK_DEPTH))
    left = max(0, int(cx - width * _NECK_WIDTH))
    right = min(shape[1], int(cx + width * _NECK_WIDTH))
    if bottom <= top or right <= left:
        return None
    region = np.zeros(shape, dtype=bool)
    region[top:bottom, left:right] = True
    return region


def _face_extent(face: Face) -> float:
    """Longer side of the face box, in frame pixels."""
    bbox = getattr(face, 'bbox', None)
    if bbox is not None:
        box = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if box.size >= 4:
            return float(max(box[2] - box[0], box[3] - box[1]))
    kps = getattr(face, 'kps', None)
    if kps is not None:
        pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
        return float(np.ptp(pts, axis=0).max() * 2.0)
    return 100.0


def _hull_mask(face: Face, shape: Tuple[int, int], scale: float) -> Mask:
    """The face hull as a binary mask at working scale."""
    mask = np.zeros(shape, dtype=np.float32)
    landmarks = getattr(face, 'landmark_2d_106', None)
    if landmarks is not None and len(landmarks) >= 3:
        points = (np.asarray(landmarks, dtype=np.float32) * scale).reshape(-1, 1, 2)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), (1.0,))
        return mask

    bbox = getattr(face, 'bbox', None)
    if bbox is not None:
        box = np.asarray(bbox, dtype=np.float64).reshape(-1) * scale
        if box.size >= 4:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            ax, ay = (box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0
            cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)),
                        0, 0, 360, (1.0,), -1)
    return mask
