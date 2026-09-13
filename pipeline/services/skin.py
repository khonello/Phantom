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
_CHROMA_TOLERANCE = 3.0

# Floor on the spread, in 8-bit LAB units, so a very flat sample (a plain
# face under soft light) does not produce a model so tight the neck fails it.
_MIN_SPREAD = 2.5

# Lightness gate, as multiples of the face's median L. The neck sits in the
# jaw's shadow and the chest below it; hands catch a different light again.
# Wide on purpose — chroma is what separates skin from not-skin here.
_L_LOW = 0.45
_L_HIGH = 1.45

# How far the face hull is grown before the body mask is taken outside it, as
# a fraction of the face's extent. The ring this leaves is the jaw feather —
# neither face nor body, and neither reading should sample it.
_FACE_GROW = 0.08

# A body component smaller than this fraction of the face's area is a speck —
# a bright spot on a wall, an earring — and is dropped. Hands are far larger.
_MIN_COMPONENT = 0.05

# Smoothing. Parameters converge fast (they are already medians), the mask
# edge a little slower.
_PARAM_ALPHA = 0.5
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

    def reset(self) -> None:
        """Drop the smoothed model. Face lost, source changed, stream restart."""
        self._centre = None
        self._spread = None
        self._lightness = None
        self._body = None

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

        # Score every pixel: chroma distance in units of the spread, ramped
        # from 1 at one spread to 0 at the tolerance, gated on lightness.
        # Squared distance against a squared ramp avoids a sqrt over the frame.
        delta = chroma.astype(np.float32)
        delta -= self._centre
        delta /= self._spread
        np.multiply(delta, delta, out=delta)
        distance2 = delta[:, :, 0]
        distance2 += delta[:, :, 1]
        # Ramp in distance: 1 - (d - 1) / (T - 1). Written on d^2 through the
        # identity d = sqrt(d2) only where it matters, which is the ramp band.
        score = np.sqrt(distance2, out=distance2)
        score -= 1.0
        score /= (_CHROMA_TOLERANCE - 1.0)
        np.subtract(1.0, score, out=score)
        np.clip(score, 0.0, 1.0, out=score)
        low = self._lightness * _L_LOW
        high = self._lightness * _L_HIGH
        score[(light <= low) | (light >= high)] = 0.0

        # Body is skin outside the grown face hull. The ring between the hull
        # and its growth is the jaw feather and belongs to neither.
        extent = _face_extent(face) * _WORK_SCALE
        grow = max(1, int(round(extent * _FACE_GROW)))
        hull = _hull_mask(face, (work_h, work_w), _WORK_SCALE)
        grown = cv2.dilate(hull, self._kernel(grow))
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
