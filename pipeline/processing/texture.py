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

**The band is wider than one octave, because skin is more than pores.** The
sigma is `DETAIL_SIGMA * texture_band`, and at the default that lands on
`_SCATTER_SIGMA` — so the layer owns everything finer than the distance light
diffuses under skin, and the scatter pass owns everything coarser. That split is
physical rather than arbitrary: what sits above the diffusion length is *shading*,
and what sits below it is *surface* — pores, freckles, moles, spots, scars,
stubble shadow, fine creases. At `DETAIL_SIGMA` alone the high-pass keeps roughly
the finest two pixels, which holds pore noise and cuts the low half off every mark
larger than that. The visible result is a face that measures as textured and reads
as smooth, because the marks a viewer actually names were subtracted at extraction.

**Energy is redistributed toward structure, because RMS is the wrong statistic
for a mark.** The map is normalised to unit deviation and spent against a budget
denominated in deviation, and a *sparse* field spends that budget badly: a dozen
spots on an otherwise flat cheek contribute little to a second moment, so matching
second moments scales them down to the level of the dense pore noise they sit in.
`texture_contrast` expands the amplitude distribution before renormalising — same
total energy, more of it in the marks and less in the filler. It is a general
statement about localised skin features, not about any one kind.

The map is monochrome for the reason `_add_grain` is monochrome: independent
per-channel high-frequency detail reads as coloured speckle, nothing like skin.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
__all__ = ['SourceTexture', 'CANONICAL_SIZE', 'TEXTURE_MAX', 'STRENGTH_MAX',
           'CONTRAST_RANGE', 'BAND_RANGE', 'RELIEF_RANGE', 'map_size',
           'canonical_from_frame',
           'extract']


# Resolution the source crop is cached at. Above any working size the compositor
# will ask for, so every derived map is a downsample of real pixels rather than
# an upsample of a small one.
CANONICAL_SIZE = 512

# Working sizes a map may be derived at. The same ladder `_aligned_size` steps
# through, so a face that moves between two compositing resolutions moves between
# two texture resolutions at the same points.
_MAP_STEPS = ALIGNED_STEPS

# Ceiling on how much detail the layer may add, in 8-bit units of standard
# deviation. The map is normalised to unit deviation inside the skin mask, so the
# amount the compositor computes is literally the standard deviation of what gets
# added to the picture.
#
# Raised from 8.0 with the band. This is a *backstop against a broken estimate*,
# not the control — the measured headroom is the control — and the band it now
# bounds runs from the finest octave up to the subsurface diffusion length rather
# than the finest octave alone, so the deviation a real face legitimately carries
# across it is correspondingly larger. Left at 8.0 it would start binding on a
# large, well-lit face and would silently cap the layer at the moment it has most
# to say.
TEXTURE_MAX = 12.0

# Ceiling on `texture_strength`. Above 1.0 the layer deliberately exceeds the
# measured parity bound — the point at which the face carries as much
# high-frequency energy as the camera recorded of the real one. That is not a
# shipping value and the desktop slider does not reach it; it exists because
# separating "the map is weak" from "the budget is small" takes one run with the
# bound lifted, and without it that diagnosis needs a code change.
STRENGTH_MAX = 2.0

# Bounds on the two shaping knobs. Both are clamped rather than rejected, for the
# reason every `set_realism` field is: these get typed at a prompt between takes.
CONTRAST_RANGE = (1.0, 3.0)
BAND_RANGE = (1.0, 4.0)
RELIEF_RANGE = (0.0, 0.95)

# Deviation, in 8-bit units inside the skin mask, below which an octave is not a
# signal. Each octave is normalised to unit deviation before it is weighted, so a
# band holding nothing real would be amplified to parity with one that does — and
# what is in that band on a clean, well-lit photograph is sensor noise and JPEG
# ringing. Dropped rather than scaled, because an octave of amplified ringing
# reprojected onto a moving face is exactly the crawl this layer must not add.
_OCTAVE_FLOOR = 0.35

# Working size the diagnostic octave reading is taken at. Fixed rather than
# following the live config so the number means the same thing between sessions
# and can be compared across source photographs.
_REPORT_SIZE = 256

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
    _maps: Dict['_MapKey', Frame] = field(default_factory=dict, repr=False)
    # Raw deviation of each octave in the chosen photograph, in 8-bit units, at
    # a fixed reference size. Measured once at extraction and never used by the
    # compositor — this is the diagnostic that answers "is my source photo good
    # enough", which was otherwise a guess. (pores, marks).
    octaves: Tuple[float, float] = (0.0, 0.0)

    @property
    def upsampled(self) -> bool:
        """
        Whether the canonical crop is larger than the detail actually present.

        Not fatal — a map derived below `native_px` is still real detail — but it
        bounds what any working size above it can contain, and it is the first
        thing to check when the layer looks weak.
        """
        return self.native_px < CANONICAL_SIZE

    def detail_for(
        self,
        size: int,
        contrast: float = 1.0,
        band: float = 1.0,
        relief: float = 0.0,
    ) -> Optional[Frame]:
        """
        Zero-mean, unit-deviation skin detail at a working resolution.

        Args:
            size: Edge length of the map, snapped by `map_size` to the ladder
            contrast: Amplitude shaping on the mark octave; 1.0 is none
            band: How far up in scale the layer reaches, as a multiple of
                `DETAIL_SIGMA`; 1.0 is the finest octave alone
            relief: The mark octave's share of the amplitude, in quadrature
                against the pore octave. 0.0 is pores only

        Returns:
            Single-channel float32 map, masked to skin and normalised so that
            multiplying by a strength in 8-bit units gives that deviation. None
            if the crop holds no usable detail at this size.
        """
        # Keyed on the shaping, not on the size alone. These are live A/B knobs
        # — `set_realism` moves them between takes — and a cache keyed on size
        # would hand back the map built under the previous setting and make the
        # comparison a measurement of nothing.
        key = (size, round(float(contrast), 3), round(float(band), 3),
               round(float(relief), 3))
        cached = self._maps.get(key)
        if cached is not None:
            return None if cached is _EMPTY else cached

        # A sweep across several values leaves one map per combination. They are
        # small (64 KB at 128, 1 MB at 512) but not free, and the working set is
        # one or two sizes — so drop the lot rather than grow without bound.
        if len(self._maps) >= _MAP_CACHE_MAX:
            self._maps.clear()

        built = self._build_map(size, contrast, band, relief)
        # Cached either way: a combination that produced nothing will produce
        # nothing again, and the alternative is rebuilding it every frame.
        self._maps[key] = built if built is not None else _EMPTY
        return built

    def _build_map(
        self,
        size: int,
        contrast: float,
        band: float,
        relief: float,
    ) -> Optional[Frame]:
        """Derive the skin-detail map at `size`. See `detail_for`."""
        # INTER_AREA on the way down: it is the resampling filter that does not
        # alias, which matters more here than anywhere else in the pipeline —
        # aliased high frequencies are exactly the shimmer this layer must not
        # introduce.
        interpolation = (
            cv2.INTER_AREA if size < self.crop.shape[0] else cv2.INTER_CUBIC
        )
        crop = cv2.resize(self.crop, (size, size), interpolation=interpolation)
        skin = cv2.resize(self.skin, (size, size), interpolation=cv2.INTER_LINEAR)

        binary = (skin > 0.5).astype(np.uint8)
        if cv2.countNonZero(binary) < 64:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        contrast = float(np.clip(contrast, *CONTRAST_RANGE))
        band = float(np.clip(band, *BAND_RANGE))
        relief = float(np.clip(relief, *RELIEF_RANGE))

        # Two cuts, giving two octaves. The fine one starts at the same
        # `DETAIL_SIGMA` `_match_detail` splits on, at the same 256px reference,
        # so the two stages still agree about where texture begins. The coarse
        # one runs up to `band` times that, which at the default is the
        # subsurface diffusion length — the scale above which a variation in the
        # picture is shading rather than something on the skin.
        fine_sigma = DETAIL_SIGMA * size / DETAIL_SIGMA_REFERENCE
        coarse_sigma = fine_sigma * band

        fine_low = cv2.GaussianBlur(gray, (0, 0), fine_sigma)
        pores = cv2.subtract(gray, fine_low)

        # Everything between the two cuts. This is where a mark of any kind
        # lives — a freckle, a spot, a mole, a scar, a crease, the shadow of
        # stubble — because what separates those from pore noise is that they
        # are *larger*, not that they are a different kind of thing. It is also
        # the half of every mark the single-octave version threw away: a 4px
        # spot has most of its energy below the fine cut, so what survived was
        # its rim.
        marks = (
            cv2.subtract(fine_low, cv2.GaussianBlur(gray, (0, 0), coarse_sigma))
            if band > 1.0 else None
        )

        # Amplitude shaping, on the marks only. A second moment is the wrong
        # statistic for a sparse field: a dozen spots on a flat cheek barely
        # move it, so a budget denominated in deviation scales them down to the
        # level of the dense noise they sit among, and the face measures as
        # textured while reading as smooth. Expanding the amplitude
        # distribution and renormalising moves the same energy into the marks.
        #
        # Deliberately not applied to `pores`. That octave is dense filler by
        # nature, and expanding its amplitude distribution is how a pore field
        # becomes speckle — the exact artefact the monochrome rule exists to
        # avoid, arrived at from the other direction.
        if marks is not None and contrast > 1.0:
            marks = np.sign(marks) * (np.abs(marks) ** contrast)

        # Each octave normalised on its own before it is weighted, so `relief`
        # means "share of the budget" rather than "share of whatever this
        # photograph happened to have most of". An octave holding no signal is
        # dropped rather than amplified — see `_OCTAVE_FLOOR`.
        levels: List[Tuple[Frame, float]] = []
        pore_level = _normalise(pores, binary, floor=_OCTAVE_FLOOR)
        mark_level = (
            _normalise(marks, binary, floor=_OCTAVE_FLOOR)
            if marks is not None else None
        )

        # Quadrature, like every other mix in this pipeline: independent
        # zero-mean fields add that way, so these weights compose with the
        # headroom arithmetic in `_texture_headroom` instead of fighting it.
        if pore_level is not None:
            levels.append((pore_level, float(np.sqrt(max(0.0, 1.0 - relief ** 2)))))
        if mark_level is not None:
            levels.append((mark_level, relief if pore_level is not None else 1.0))

        levels = [(field_, weight) for field_, weight in levels if weight > 1e-3]
        if not levels:
            return None


        combined = levels[0][0] * levels[0][1]
        for field_, weight in levels[1:]:
            combined = combined + field_ * weight

        # One last normalisation over the sum. The octaves are not perfectly
        # independent — they were cut from one image — so their weighted sum
        # does not land at unit deviation on its own, and the compositor's
        # arithmetic depends on it doing exactly that.
        unit = _normalise(combined, binary)
        if unit is None:
            return None

        return np.ascontiguousarray(unit * skin, dtype=np.float32)


# Sentinel for "this combination yielded nothing", so a failed build is not
# retried per frame. Never returned — `detail_for` maps it back to None.
_EMPTY: Frame = np.zeros((1, 1), dtype=np.float32)

# (size, contrast, band, relief) — everything the map is a function of.
_MapKey = Tuple[int, float, float, float]

# Distinct maps held before the cache is dropped wholesale. The working set is
# one or two sizes; anything beyond that is a sweep walking through values it
# will not come back to.
_MAP_CACHE_MAX = 16


def _normalise(
    field_: Frame,
    binary: Frame,
    floor: float = 0.0,
) -> Optional[Frame]:
    """
    Centre and scale a field to zero mean and unit deviation inside a mask.

    Inside the mask only, and that is the whole point of taking a mask at all:
    deviation over the entire square would be dominated by the excluded features
    and by the crop's replicated border, so a strength knob calibrated against it
    would mean something different for every source photograph.

    Args:
        field_: Single-channel float32 field over the canonical crop
        binary: Where to measure, as an 8-bit 0/1 mask
        floor: Deviation below which the field is not a signal. Zero accepts
            anything measurable; callers separating real detail from sensor
            noise pass `_OCTAVE_FLOOR`.

    Returns:
        The normalised field, or None when it holds nothing worth keeping.
    """
    mean, deviation = cv2.meanStdDev(field_, mask=binary)
    energy = float(np.sqrt(np.mean(np.square(deviation))))
    if energy < max(floor, 1e-3):
        return None
    result: Frame = (field_ - float(mean[0][0])) / energy
    return result.astype(np.float32)


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
        octaves=_measure_octaves(crop, skin),
    )


def _measure_octaves(crop: Frame, skin: Mask) -> Tuple[float, float]:
    """
    Raw deviation of each octave in the source crop, in 8-bit units.

    A diagnostic, not an input: nothing in the compositor reads it. It exists
    because "is the chosen photograph sharp enough" was a question this layer
    could not answer about itself, and the visible symptom of a soft donor is
    identical to the visible symptom of a starved budget — the operator sees a
    face that is nearly smooth either way. One number per octave separates them
    in the log, before anyone spends a pod session on it.

    Fixed at `_REPORT_SIZE` with the default band split rather than following
    the live config, so the reading means the same thing between sessions and
    can be compared across source photographs.

    Args:
        crop: Canonical FFHQ-framed source face, BGR
        skin: Canonical skin mask

    Returns:
        (pore deviation, mark deviation), both zero if unmeasurable
    """
    size = _REPORT_SIZE
    small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    binary = (
        cv2.resize(skin, (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
    ).astype(np.uint8)
    if cv2.countNonZero(binary) < 64:
        return (0.0, 0.0)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine_sigma = DETAIL_SIGMA * size / DETAIL_SIGMA_REFERENCE
    fine_low = cv2.GaussianBlur(gray, (0, 0), fine_sigma)

    def deviation(field_: Frame) -> float:
        _, dev = cv2.meanStdDev(field_, mask=binary)
        return round(float(dev[0][0]), 2)

    return (
        deviation(cv2.subtract(gray, fine_low)),
        deviation(cv2.subtract(
            fine_low, cv2.GaussianBlur(gray, (0, 0), fine_sigma * 2.0))),
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
