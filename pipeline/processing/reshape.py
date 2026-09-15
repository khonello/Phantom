"""
Move the head's outline toward the source's, in frame space, after the paste.

**The one thing nothing else here can do.** Every swap model generates into a
crop framed by the *target's* five keypoints, and the mask is a convex hull of
the *target's* landmarks — so the silhouette is the target's by construction,
and measurement bears that out: `outline_shift` reads ~0 on every registered
LIVE model at every mask setting. The head still reads as the source's, but
entirely through appearance at fixed landmark positions. The literal boundary
never moves.

This moves it, by deforming the finished picture rather than by asking a
generator to produce a different one.

**Why after the paste and not before.** Anything applied inside the compositor
is bounded by the mask, and the silhouette is *outside* the mask — that is the
whole reason it cannot move. Only a deformation of the frame itself reaches it.
Running here also means the surrounding pixels come along: narrow a jaw and the
background just outside moves inward too, so there is no hole to inpaint and no
torn edge. That is the difference between a warp and a cut-out.

**Why the displacement is fixed per pairing rather than recomputed per frame.**
The shape difference between two people is a property of the two people; only
pose and expression change frame to frame. Recomputing it every frame would
track expression — correcting a smile toward the source's neutral mouth — and
would shimmer, because a landmark-driven field inherits every landmark wobble,
on the longest boundary in the picture. `LandmarkStabilizer` cannot help: that
is *correct* landmark motion, the same argument texture swimming rests on.

So the delta is measured once and thereafter only *placed*: each frame fits the
reference landmarks onto the current ones with a similarity, which carries the
head's rotation and scale, and the stored vectors are rotated by the same fit.
No canonical intermediate, no transform to get backwards.

**What it is not.** A deformation, not identity transfer. It moves where the
boundary sits; it does not generate the source's head. Past a few percent of
face extent it stops reading as a different head and starts reading as a
face-slimming filter, which is why the magnitude is bounded rather than left to
a strength knob alone.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.processing.geometry import estimate_similarity
from pipeline.types import Frame, Points

# Resolution the dense field is interpolated at before being resized to the
# frame. The field is a sum of Gaussians over 106 points and has no structure
# worth resolving finer — 64x64x106 is a third of a million operations, where
# the same at frame resolution would be tens of millions per frame.
_GRID = 64

# Interpolation width for the scattered landmark displacements, as a fraction
# of face radius. Wide enough that the field is smooth between neighbouring
# landmarks, narrow enough that the jaw does not drag the brow with it.
_SPREAD = 0.35

# How far past the face the deformation reaches before it is gone, as a
# fraction of face radius. This is the part that stops the picture tearing: the
# boundary needs the pixels just outside it to move too, or a narrowed jaw
# shears against static background. Half a radius keeps the drag local enough
# that straight lines behind the head do not visibly bend.
_FALLOFF = 0.5

# Hard ceiling on displacement, as a fraction of the face's radius in frame.
#
# A *safety* ceiling, not the operating point — it must not bind in normal use
# or `shape_warp` stops being a control. Measured on a fixture matching the real
# pairing's `shape_mismatch` of 0.14, face radius 59px:
#
#     ceiling   outline_shift at strength 0.25 / 0.5 / 1.0   binds?
#     0.08      +0.106  +0.205  +0.334                       yes, at 1.0
#     0.15      +0.106  +0.206  +0.376                       no
#     0.30      +0.106  +0.206  +0.375                       no
#
# 0.08 clipped the top of the range; anything from 0.15 up is identical, which
# means the field itself is the limit rather than the cap. 0.15 is 8.8px on a
# 59px radius — about 7% of face width, the conservative end of what a
# face-slimming filter does — and it still catches an outlier pairing.
#
# The warp does not fold at any of these: the Jacobian determinant stays in
# 0.57 to 1.34, never negative.
MAX_SHIFT = 0.15

# Below this many landmarks there is not enough to interpolate a field from.
_MIN_POINTS = 8

# How much of the delta's *antisymmetric* part to keep, across the face's own
# midline.
#
# **This is the fix for a one-sided warp, and it was found on footage.** The
# delta is `fitted_source - target`, and a similarity fit cannot correct pose —
# so if either face is turned, one side is foreshortened and the residual is
# one-sided. Applied, that pulls a single cheek in and gets visibly worse with
# strength, which is not a head-shape correction at all.
#
# The head-shape difference actually worth transferring — face width, jaw
# width, face length — is very nearly symmetric. Pose contamination is
# antisymmetric. Dropping the antisymmetric part therefore keeps the signal and
# rejects the artefact, and it does so whatever the cause: a turned source, a
# turned build frame, or landmark noise on one side.
#
# **Zero for now**, deliberately. Real faces are genuinely a little asymmetric
# and that is part of a likeness, so there is a case for keeping a share — but
# the observed failure was a badly one-sided warp, nothing here can distinguish
# genuine asymmetry from pose contamination, and this is a deformation rather
# than identity transfer. Raise it once the symmetric version has been seen
# working on footage; the correction it would add is subtle and the artefact it
# risks is not.
#
# Verified on a clean mirror-symmetric fixture: a constant sideways push — what
# pose contamination looks like — is removed entirely, while an outward push
# from the midline, which is a genuine width difference, survives at 4.83 of 5.
_ASYMMETRY_KEEP = 0.0


_GRIDS: Dict[Tuple[int, int], Tuple[npt.NDArray[Any], npt.NDArray[Any]]] = {}


def _grid(width: int, height: int) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Pixel-coordinate meshes for a region, built once per size."""
    key = (width, height)
    cached = _GRIDS.get(key)
    if cached is None:
        cached = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32))
        if len(_GRIDS) > 8:
            _GRIDS.clear()
        _GRIDS[key] = cached
    return cached


def _midline(points: Points) -> npt.NDArray[Any]:
    """
    The horizontal normal of the face's own vertical axis.

    From the landmark cloud's principal direction rather than from named
    points, so it does not depend on the pack's index layout — the same reason
    the shape metric ranks by hull distance instead of assuming "0-32 is the
    jaw". A face is markedly taller than it is wide, so the first principal
    component is its vertical.

    Args:
        points: (N, 2) landmarks

    Returns:
        (2,) unit vector perpendicular to the face's vertical axis
    """
    centred = np.asarray(points, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    axis = right[0]
    return np.array([-axis[1], axis[0]], dtype=np.float64)


def _symmetrise(
    reference: Points,
    delta: Points,
    keep: float = _ASYMMETRY_KEEP,
) -> Points:
    """
    Drop the part of the delta that is not mirrored across the face's midline.

    Each landmark is paired with whichever landmark lands nearest its own
    reflection, and the pair's displacements are averaged after reflecting the
    partner's. Pairing by geometry rather than by index keeps this independent
    of the model pack.

    Args:
        reference: (N, 2) the landmarks the delta was measured on
        delta: (N, 2) the displacements
        keep: Share of the antisymmetric part to retain, [0, 1]

    Returns:
        (N, 2) displacements with the antisymmetric part attenuated
    """
    points = np.asarray(reference, dtype=np.float64)
    shift = np.asarray(delta, dtype=np.float64)

    normal = _midline(points)
    centre = points.mean(axis=0)
    centred = points - centre

    # Reflect the positions and pair each landmark with its nearest mirror.
    mirrored = centred - 2.0 * (centred @ normal)[:, None] * normal[None, :]
    distance = ((mirrored[:, None, :] - centred[None, :, :]) ** 2).sum(axis=2)
    partner = np.argmin(distance, axis=1)

    # Reflect the partner's *vector* too — a displacement pointing left on one
    # cheek corresponds to one pointing right on the other.
    flipped = shift - 2.0 * (shift @ normal)[:, None] * normal[None, :]
    symmetric = 0.5 * (shift + flipped[partner])

    share = float(np.clip(keep, 0.0, 1.0))
    return symmetric + (shift - symmetric) * share


def _radius(points: Points) -> float:
    """RMS distance of a point set from its own centroid."""
    centred = points - points.mean(axis=0)
    return float(np.sqrt(float((centred ** 2).sum(axis=1).mean())))


class ShapeWarp:
    """
    A fixed per-landmark displacement, measured once for one source/target pair.

    Example:
        warp = ShapeWarp.between(source_landmarks, target_landmarks)
        if warp is not None:
            frame = warp.apply(frame, current_landmarks, strength=0.5)
    """

    def __init__(self, reference: Points, delta: Points) -> None:
        """
        Args:
            reference: (N, 2) the target's landmarks the delta was measured on
            delta: (N, 2) displacement from each of those toward the source's
                shape, in the same units
        """
        self.reference = reference
        self.delta = delta
        self._span = _radius(reference)

    @classmethod
    def between(cls, source: Points, target: Points) -> Optional['ShapeWarp']:
        """
        Measure the shape difference between two landmark sets.

        The source is fitted onto the target with a similarity first, so what
        survives is *shape* rather than the fact that two photographs were taken
        at different distances and angles. Without that the field would try to
        move the target's head to the source photograph's scale and position,
        which is not a shape correction at all.

        Args:
            source: (N, 2) landmarks on the source photograph
            target: (N, 2) landmarks on the target

        Returns:
            A warp, or None if the two sets are not comparable
        """
        first = np.asarray(source, dtype=np.float64)
        second = np.asarray(target, dtype=np.float64)
        if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2:
            return None
        if first.shape[0] < _MIN_POINTS or _radius(second) < 1e-6:
            return None

        matrix = estimate_similarity(first, second)
        if matrix is None:
            return None

        fitted = first @ matrix[:, :2].T + matrix[:, 2]
        # Symmetrised before it is stored, so every frame afterwards uses the
        # cleaned field — see `_ASYMMETRY_KEEP` for why a raw delta warps one
        # side of the face.
        return cls(reference=second,
                   delta=_symmetrise(second, fitted - second))

    @property
    def magnitude(self) -> float:
        """RMS displacement as a fraction of face radius, before any strength."""
        if self._span < 1e-6:
            return 0.0
        return float(np.sqrt((self.delta ** 2).sum(axis=1).mean()) / self._span)

    def _placed(
        self,
        current: Points,
    ) -> Optional[Tuple[Points, Points, float]]:
        """
        Rotate and scale the stored delta onto this frame's face.

        Args:
            current: (N, 2) the target's landmarks in this frame

        Returns:
            (anchors, displacements, radius) in frame coordinates, or None
        """
        points = np.asarray(current, dtype=np.float64)
        if points.shape != self.reference.shape:
            return None

        span = _radius(points)
        if span < 1e-6:
            return None

        # The fit carries the head's rotation and its size in this frame. The
        # delta is a set of *vectors*, so only the linear part applies — adding
        # the translation would move the whole head rather than reshape it.
        matrix = estimate_similarity(self.reference, points)
        if matrix is None:
            return None

        return points, self.delta @ matrix[:, :2].T, span

    def field(
        self,
        current: Points,
        shape: Tuple[int, int],
        strength: float,
    ) -> Optional[Tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """
        The dense displacement field for one frame.

        Args:
            current: (N, 2) the target's landmarks in this frame
            shape: (height, width) of the frame
            strength: Fraction of the measured difference to apply, [0, 1]

        Returns:
            (dx, dy) at frame resolution, or None if it cannot be built
        """
        placed = self._placed(current)
        if placed is None:
            return None

        anchors, vectors, span = placed
        vectors = vectors * float(np.clip(strength, 0.0, 1.0))

        height, width = shape
        # Interpolate coarsely over the whole frame. The face occupies a small
        # part of it, but the falloff below is what bounds the field, and a grid
        # that only covered the face would have to be pasted back with its own
        # edge handling.
        xs = (np.arange(_GRID, dtype=np.float64) + 0.5) * (width / _GRID)
        ys = (np.arange(_GRID, dtype=np.float64) + 0.5) * (height / _GRID)
        gx, gy = np.meshgrid(xs, ys)
        flat = np.stack([gx.ravel(), gy.ravel()], axis=1)

        # Only grid points within reach of the head are computed. The falloff
        # below multiplies the field by exp(-beyond^2 / 2(span*_FALLOFF)^2),
        # which at six falloff widths is exp(-18) ~ 1e-8 of a displacement
        # already bounded to a few pixels — a millionth of a pixel, below the
        # float32 the field is cast to. Those points are now exactly zero
        # rather than immeasurably small, and the interpolation, exp and
        # weighting run over the half of the grid that matters, in float32.
        # Measured: 35ms -> 10ms on a laptop, field within 5e-7 px of the
        # float64 whole-grid version, and the rendered frame within 1 LSB on
        # ~0.01% of pixels from rounding at the remap coordinates.
        centre = anchors.mean(axis=0)
        distance_to_centre = np.linalg.norm(flat - centre, axis=1)
        near = distance_to_centre <= span * (1.0 + 6.0 * _FALLOFF)
        if not bool(np.any(near)):
            return None

        sigma = max(1e-6, span * _SPREAD)
        points = flat[near].astype(np.float32)
        anchors32 = anchors.astype(np.float32)
        diff = points[:, None, :] - anchors32[None, :, :]
        distance2 = np.einsum('ijk,ijk->ij', diff, diff)
        weight = np.exp(distance2 * np.float32(-1.0 / (2.0 * sigma * sigma)))

        total = weight.sum(axis=1, keepdims=True)
        # A grid point beyond every landmark's reach gets no displacement rather
        # than a divide-by-zero, which is also the right answer there.
        safe = np.where(total > 1e-12, total, 1.0)
        near_field = (weight @ vectors.astype(np.float32)) / safe
        near_field[total[:, 0] <= 1e-12] = 0.0

        # Fall off outside the head so the background is not dragged with it:
        # one inside the face, decaying over `_FALLOFF` of a radius past it.
        beyond = np.maximum(0.0, distance_to_centre[near] - span)
        near_field *= np.exp(
            -(beyond ** 2) / (2.0 * (span * _FALLOFF) ** 2))[:, None].astype(np.float32)

        # Bounded in *magnitude*, not per axis — clipping x and y separately
        # would change the direction a point moves in wherever it bound.
        limit = MAX_SHIFT * span
        length = np.linalg.norm(near_field, axis=1)
        over = length > limit
        if bool(np.any(over)):
            near_field[over] *= (limit / np.maximum(length[over], 1e-6))[:, None]

        field = np.zeros((_GRID * _GRID, 2), dtype=np.float32)
        field[near] = near_field

        coarse = np.stack([
            field[:, 0].reshape(_GRID, _GRID),
            field[:, 1].reshape(_GRID, _GRID),
        ], axis=2).astype(np.float32)

        dense = cv2.resize(coarse, (width, height),
                           interpolation=cv2.INTER_LINEAR)
        return dense[:, :, 0], dense[:, :, 1]

    def apply(
        self,
        frame: Frame,
        current: Points,
        strength: float,
    ) -> Frame:
        """
        Deform `frame` so the head moves toward the source's shape.

        Args:
            frame: The finished frame, BGR
            current: (N, 2) the target's landmarks in this frame
            strength: Fraction of the measured difference to apply, [0, 1]

        Returns:
            The deformed frame, or `frame` unchanged when the warp cannot run
        """
        if strength <= 0.0 or frame is None or frame.ndim != 3:
            return frame

        built = self.field(current, frame.shape[:2], strength)
        if built is None:
            return frame

        shift_x, shift_y = built
        height, width = frame.shape[:2]

        # Only the field's support needs remapping. Outside it the shift is
        # exactly zero, and `remap` at integer coordinates with INTER_LINEAR
        # returns the pixel unchanged — so pixels outside the region are the
        # input's own, and inside it the result matches a whole-frame remap to
        # within 1 LSB on a handful of pixels: `remap` quantises its float32
        # coordinates to 1/32 px, and the region's small local coordinates
        # round more precisely than the frame's large ones. The margin is the
        # field's largest displacement plus the bilinear footprint: a pixel
        # inside the region can sample from that far beyond it, and the sample
        # has to land on the real neighbour rather than a replicated border.
        moving = (np.abs(shift_x) > 1e-6) | (np.abs(shift_y) > 1e-6)
        rows = np.flatnonzero(moving.any(axis=1))
        cols = np.flatnonzero(moving.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return frame
        margin = int(np.ceil(max(float(np.abs(shift_x).max()), float(np.abs(shift_y).max())))) + 2
        y0, y1 = max(0, int(rows[0]) - margin), min(height, int(rows[-1]) + 1 + margin)
        x0, x1 = max(0, int(cols[0]) - margin), min(width, int(cols[-1]) + 1 + margin)

        grid_x, grid_y = _grid(x1 - x0, y1 - y0)
        # `remap` reads *from* these coordinates, so moving the picture by +d
        # means sampling from -d. The maps are in the ROI's own coordinates,
        # so the ROI is what gets sampled — a displacement that reaches past
        # its edge is answered by BORDER_REPLICATE, as it was on the whole
        # frame's edge before.
        map_x = grid_x - shift_x[y0:y1, x0:x1]
        map_y = grid_y - shift_y[y0:y1, x0:x1]
        warped = frame.copy()
        warped[y0:y1, x0:x1] = cv2.remap(
            frame[y0:y1, x0:x1],
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped
