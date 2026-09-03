"""
Source skin texture: extracted once per identity, reprojected every frame.

The swapped face carries **58% of the frame's high-frequency energy** on the one
clip this has been measured on, against an ideal of 1.00. That deficit is what
reads as plastic, and restoration does not close it — turning restoration off
entirely moves the number by 0.03. The detail was never there: the swapper
generates at 128 or 256 native and everything downstream resamples that.

So the detail is taken from where it does exist — the operator's own source
photograph — high-passed, stored in canonical face framing, and warped onto the
face every frame.

Three properties decide the design, and all three are recorded in
docs/TEXTURE_PIPELINE.md:

**Extraction runs once.** The map is a function of the source image and nothing
else, so it is built when the source is set and cached for the session. Only the
warp runs per frame, which is one `warpAffine` of a small single-channel image.

**Canonical space is the FFHQ framing the compositor already uses.**
`estimate_similarity` fits the same five keypoints into the same `FFHQ_TEMPLATE`
that `_ffhq_geometry` fits, so no new geometry is introduced. Note what this does
and does not buy: because both are *similarity* transforms, composing
source->canonical->target has identical error to source->target in one step. The
win is the caching, not the accuracy. It also means yaw is **not** corrected — a
source shot at an angle yields a foreshortened map, which is why source selection
scores frontality and why the confidence mask (not built yet) has to fall off
with pose distance.

**The band is chosen at the size it will be displayed at.** This is the part that
is easy to get wrong and expensive to get wrong. A high-frequency field built at
512 and warped down onto a 101px face is decimated — the pores land under the
Nyquist limit and average to nothing, which is the same mechanism that makes
CodeFormer's 512 crop worth +0.03 to the detail ratio. So the *crop* is cached at
high resolution and the *map* is derived per working size, memoised. The high-pass
sigma scales with that size against the same 256px reference `_match_detail` uses,
so both stages mean the same physical detail by "texture".

The map is monochrome for the reason `_add_grain` is monochrome: independent
per-channel high-frequency detail reads as coloured speckle, nothing like skin.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from pipeline.types import Frame, Face, Mask, Matrix
from pipeline.logging import emit_warning
from pipeline.processing.geometry import (
    ALIGNED_STEPS,
    DETAIL_SIGMA,
    DETAIL_SIGMA_REFERENCE,
    FFHQ_TEMPLATE,
    canonical_from_frame,
)

# Re-exported so a caller working with textures has one import rather than two.
__all__ = ['SourceTexture', 'CANONICAL_SIZE', 'TEXTURE_MAX', 'map_size',
           'canonical_from_frame', 'extract']


# Resolution the source crop is cached at. Above any working size the compositor
# will ask for, so every derived map is a downsample of real pixels rather than
# an upsample of a small one.
CANONICAL_SIZE = 512

# Working sizes a map may be derived at. The same ladder `_aligned_size` steps
# through, so a face that moves between two compositing resolutions moves between
# two texture resolutions at the same points.
_MAP_STEPS = ALIGNED_STEPS

# Ceiling on how much detail `texture_strength` may add, in 8-bit units of
# standard deviation. The map is normalised to unit deviation inside the skin
# mask, so `texture_strength * _TEXTURE_MAX` is literally the standard deviation
# of what gets added to the picture. Sits just above `_GRAIN_MAX` (6.0): pore
# contrast on a webcam face is a few units, and detail that overwhelms the grain
# it sits under is a texture the camera could not have recorded.
TEXTURE_MAX = 8.0

# Feature exclusions, as radii in canonical units (the FFHQ template is
# normalised, so these are constants rather than landmark lookups). Eyes, nostrils
# and the mouth line carry the strongest high-frequency energy in any face crop,
# and it is *structure*, not skin — reprojecting it paints a second set of
# eyelashes over the swap's own.
_EYE_RADIUS = 0.085
_MOUTH_RADIUS = 0.105
_NOSE_RADIUS = 0.070

# Erode and feather on the skin mask, as fractions of the canonical edge. The
# same fraction-of-crop treatment `_FFHQ_ERODE` and `_FFHQ_FEATHER` get, and for
# the same reason: the map is built at several sizes and a blur in pixels would
# mean a different mask at each one.
_SKIN_ERODE = 6.0 / 512.0
_SKIN_FEATHER = 10.0 / 512.0


@dataclass
class SourceTexture:
    """
    Skin detail lifted from one source image, in canonical face framing.

    Holds the *crop*, not the map. Maps are derived per working size by
    `detail_for` and memoised, because the band has to be chosen at the
    resolution it will be displayed at — see the module docstring.
    """

    path: str
    crop: Frame                      # canonical FFHQ-framed source face, BGR
    skin: Mask                       # canonical skin mask, float32 [0, 1]
    native_px: int                   # face extent in the source image
    yaw: Optional[float] = None      # source pose, for the confidence mask
    _maps: Dict[int, Frame] = field(default_factory=dict, repr=False)

    @property
    def upsampled(self) -> bool:
        """
        Whether the canonical crop is larger than the detail actually present.

        Not fatal — a map derived below `native_px` is still real detail — but it
        bounds what any working size above it can contain, and it is the first
        thing to check when the layer looks weak.
        """
        return self.native_px < CANONICAL_SIZE

    def detail_for(self, size: int) -> Optional[Frame]:
        """
        Zero-mean, unit-deviation skin detail at a working resolution.

        Args:
            size: Edge length of the map, snapped by `map_size` to the ladder

        Returns:
            Single-channel float32 map, masked to skin and normalised so that
            multiplying by a strength in 8-bit units gives that deviation. None
            if the crop holds no usable detail at this size.
        """
        cached = self._maps.get(size)
        if cached is not None:
            return cached

        built = self._build_map(size)
        # Cached either way: a size that produced nothing will produce nothing
        # again, and the alternative is rebuilding it every frame.
        self._maps[size] = built if built is not None else _EMPTY
        return built

    def _build_map(self, size: int) -> Optional[Frame]:
        """Derive the high-frequency band at `size`. See `detail_for`."""
        # INTER_AREA on the way down: it is the resampling filter that does not
        # alias, which matters more here than anywhere else in the pipeline —
        # aliased high frequencies are exactly the shimmer this layer must not
        # introduce.
        interpolation = (
            cv2.INTER_AREA if size < self.crop.shape[0] else cv2.INTER_CUBIC
        )
        crop = cv2.resize(self.crop, (size, size), interpolation=interpolation)
        skin = cv2.resize(self.skin, (size, size), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # The same band split `_match_detail` uses, at the same 256px reference,
        # so the additive layer here and the multiplicative one there describe
        # the same physical detail rather than fighting over adjacent bands.
        sigma = DETAIL_SIGMA * size / DETAIL_SIGMA_REFERENCE
        high = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), sigma))

        binary = (skin > 0.5).astype(np.uint8)
        if cv2.countNonZero(binary) < 64:
            return None

        # Normalise inside the mask only. Deviation over the whole square would
        # be dominated by the excluded features and the crop's border, so the
        # strength knob would mean something different for every source image.
        _, deviation = cv2.meanStdDev(high, mask=binary)
        energy = float(np.sqrt(np.mean(np.square(deviation))))
        if energy < 1e-3:
            return None

        detail: Frame = (high / energy) * skin
        return np.ascontiguousarray(detail, dtype=np.float32)


# Sentinel for "this size yielded nothing", so a failed build is not retried per
# frame. Never returned — `detail_for` maps it back to None.
_EMPTY: Frame = np.zeros((1, 1), dtype=np.float32)


def map_size(extent: float) -> int:
    """
    Working resolution for a face of `extent` pixels in the frame.

    Snapped to the compositing ladder rather than used raw, so the map is rebuilt
    on a step change instead of on every pixel of head movement — the memoisation
    in `detail_for` is only worth having if the key is stable.

    Args:
        extent: The face's extent in frame pixels

    Returns:
        A size from `_MAP_STEPS`
    """
    return min(_MAP_STEPS, key=lambda s: abs(s - extent))


def extract(
    image_path: str,
    face: Face,
) -> Optional[SourceTexture]:
    """
    Build the canonical texture record for one source image.

    Runs once per identity, at `set_source` time, off the live path — so it reads
    the file from disk and pays a full-resolution warp without apology.

    Args:
        image_path: Source photograph the operator uploaded
        face: The detection already made for it, from `FaceDatabase`'s cache

    Returns:
        A `SourceTexture`, or None if the image or its geometry is unusable.
        None is not an error worth stopping for — the swap runs without the
        layer, which is what it did before this existed.
    """
    try:
        image = cv2.imread(image_path)
    except Exception as e:
        emit_warning(
            f'Texture source unreadable: {type(e).__name__}: {e}', scope='TEXTURE',
        )
        return None
    if image is None:
        return None

    matrix = canonical_from_frame(face, CANONICAL_SIZE)
    if matrix is None:
        return None

    crop = cv2.warpAffine(
        image,
        matrix,
        (CANONICAL_SIZE, CANONICAL_SIZE),
        borderMode=cv2.BORDER_REPLICATE,
    )
    skin = _skin_mask(face, matrix, CANONICAL_SIZE)

    return SourceTexture(
        path=image_path,
        crop=crop,
        skin=skin,
        native_px=_face_extent(face),
        yaw=_face_yaw(face),
    )


def _face_yaw(face: Face) -> Optional[float]:
    """
    Source pose in degrees, or None when the model pack does not carry it.

    Deliberately reads `face.pose` only, rather than falling back to
    `guards.measure_yaw`'s keypoint approximation. This value is metadata for
    the confidence mask that is not built yet, and the two estimators are not on
    the same scale — recording a number without recording which scale it is on
    is how a threshold gets calibrated against the wrong one.
    """
    pose = getattr(face, 'pose', None)
    if pose is None:
        return None
    try:
        values = np.asarray(pose, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    # InsightFace orders this (pitch, yaw, roll).
    if values.size < 2 or not np.isfinite(values[1]):
        return None
    return float(values[1])


def _face_extent(face: Face) -> int:
    """Shorter side of the detection's bounding box, in source pixels."""
    bbox = getattr(face, 'bbox', None)
    if bbox is None or len(bbox) < 4:
        return 0
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return 0
    return int(min(abs(x2 - x1), abs(y2 - y1)))


def _skin_mask(face: Face, matrix: Matrix, size: int) -> Mask:
    """
    Where on the canonical crop the detail is skin rather than structure.

    Two terms. The landmark hull bounds the face against hair and background —
    without it the map carries the source's hairline, which lands somewhere else
    entirely on the target's head. The feature exclusions remove eyes, nostrils
    and mouth, which are the strongest high-frequency energy in any face crop and
    are structure the swap generates for itself.

    This is deliberately *not* BiSeNet. A parsing net would be better here and is
    affordable here — this runs once — but the per-frame side of this layer must
    reuse masks that already exist, and keeping both sides on the same definition
    of skin is worth more than the accuracy. See docs/TEXTURE_PIPELINE.md §8.

    Args:
        face: Source detection, for its 106 landmarks
        matrix: frame -> canonical affine for the source image
        size: Edge length of canonical space

    Returns:
        Float32 mask in [0, 1], eroded and feathered.
    """
    mask = np.zeros((size, size), dtype=np.float32)

    landmarks = getattr(face, 'landmark_2d_106', None)
    if landmarks is not None and len(landmarks) >= 3:
        points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(cv2.transform(points, matrix).reshape(-1, 2))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), (1.0,))
    else:
        # No landmarks: fall back to the template's own footprint. Safe for the
        # same reason `_template_ellipse` is — canonical space is normalised, so
        # an ellipse here is at least always centred on the face.
        cv2.ellipse(
            mask,
            (size // 2, int(size * 0.52)),
            (int(size * 0.36), int(size * 0.44)),
            0, 0, 360, (1.0,), -1,
        )

    # Features come out after the hull, in canonical coordinates. The template is
    # normalised and ordered (left eye, right eye, nose, left mouth, right
    # mouth), so these are constants — no landmark lookup, and identical for
    # every source image.
    for index, radius in (
        (0, _EYE_RADIUS), (1, _EYE_RADIUS), (2, _NOSE_RADIUS),
        (3, _MOUTH_RADIUS), (4, _MOUTH_RADIUS),
    ):
        centre = (FFHQ_TEMPLATE[index] * size).astype(np.int32)
        cv2.circle(mask, (int(centre[0]), int(centre[1])), int(radius * size), (0.0,), -1)

    erode_px = max(3, int(round(size * _SKIN_ERODE))) | 1
    eroded = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    feathered = cv2.GaussianBlur(eroded, (0, 0), size * _SKIN_FEATHER)
    result: Mask = np.clip(feathered, 0.0, 1.0).astype(np.float32)
    return result
