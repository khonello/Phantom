"""
Does the output have the source's head shape, or the target's?

`IdentityProbe` answers "is this the right person" with an ArcFace cosine, and
that instrument is deliberately **blind to the question this one asks**.
Recognition models are trained to be invariant to a great deal of geometry, so a
swap can move the jawline visibly and shift the cosine by almost nothing. The
one channel a viewer reads most directly for identity — the outline of the head
— is the channel the existing measurement cannot see.

It is also the channel this pipeline structurally loses, in three separate
places:

    alignment   a similarity fit has 4 degrees of freedom, so the crop handed
                to the swapper is framed by the TARGET's five points and no
                amount of source identity reshapes it
    generator   most models repaint the interior and leave the contour where
                they found it; only 3D-supervised ones move it
    mask        the silhouette is a convex hull of the TARGET's landmarks, so
                whatever contour movement survived is clipped back off

Each has a different lever behind it, and until now none of the three had a
number. `mask_shape_growth` in particular has been shipped, defaulted to zero,
and never measured — because measuring it needed exactly this.

**What is measured.** Landmarks from the pack's own 106-point model on three
faces — the source photograph, the target, and the finished output — reduced to
pure *shape* by fitting away the similarity transform that separates them. Two
readings come out of that reduction:

    mismatch    how far apart the source's and the target's shapes are. The
                quantity behind "swaps look better when the heads match"
    shift       how far the output moved off the target's shape toward the
                source's. 0.0 is the target's outline exactly, 1.0 is the
                source's, negative is further away than the target was

Reported twice: over all 106 points, and over the **outline** subset alone,
which is the silhouette and the part that carries identity. The two come apart
in exactly the case that matters — a model that repaints features toward the
source while leaving the contour at the target's scores well on the first and
near zero on the second, which is `inswapper` and `alphaface` by construction.

**Read `shift`, not the absolutes.** Out-of-plane pose changes a face's apparent
2D shape, so a source photographed at an angle inflates `mismatch` for a reason
that is not head shape. It inflates `gap` by nearly the same factor, and `shift`
is their ratio — so the ratio survives pose far better than either term does.
The same argument the identity probe makes for reading its drops rather than its
levels.

**No new model.** The 106-point landmark model is already loaded by `buffalo_l`
and already run per frame by the shape-following mask. This borrows it, exactly
as `IdentityProbe` borrows the recognition model rather than loading a second
copy onto the GPU.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.processing.geometry import estimate_similarity
from pipeline.types import Face, Frame, Matrix, Points

# Share of the landmarks counted as outline, taken by distance to the convex
# hull rather than by index.
#
# Deliberately not an index range. The 106-point layout is a property of the
# model pack — the scatter stage already builds its exclusions from `face.kps`
# rather than the 106 for this reason, and the guards already treat a missing
# `face.pose` as a capability gap. A hard-coded "0-32 is the jaw" would be right
# for `buffalo_l` and silently wrong for the next pack, producing a plausible
# number measured on the wrong points.
#
# A third of a 106-point face is ~35 points and lands on the jaw, chin and
# temples: the silhouette plus its immediate surround. Widening it would start
# taking in the nose.
_OUTLINE_SHARE = 1.0 / 3.0

# Below this many points on either side of the split, the subset is too small to
# fit a transform on or to average over, and the outline readings are withheld
# rather than reported noisily.
_MIN_POINTS = 8

# `shift` is a ratio against `mismatch`, so it is meaningless when the two head
# shapes already agree — dividing a small residual by a smaller one describes
# estimator noise rather than a swap. Below this the pair has no shape
# disagreement to resolve, which is itself the useful answer.
_MIN_MISMATCH = 0.008


@dataclass(frozen=True)
class ShapeReading:
    """
    One frame's shape comparison, as fractions of the face's own size.

    `mismatch` and `gap` are RMS residuals measured after each set has been
    scaled to unit radius, so 0.02 means "two percent of the face's size" and
    holds whatever image the landmarks came off. `shift` is their ratio,
    dimensionless, and is the reading to act on.
    """

    mismatch: float
    gap: float
    shift: Optional[float]
    outline_mismatch: Optional[float]
    outline_gap: Optional[float]
    outline_shift: Optional[float]
    interior_mismatch: Optional[float]
    interior_shift: Optional[float]

    def as_readings(self) -> Dict[str, float]:
        """
        The named scalars, omitting any that could not be measured.

        Absent entries are left out rather than zeroed: a missing reading and a
        reading of zero mean opposite things here — "not measurable" against
        "measured, and the output kept the target's shape exactly".

        Returns:
            Reading name to value
        """
        out: Dict[str, float] = {
            'shape_mismatch': self.mismatch,
            'shape_gap': self.gap,
        }
        if self.shift is not None:
            out['shape_shift'] = self.shift
        if self.outline_mismatch is not None:
            out['outline_mismatch'] = self.outline_mismatch
        if self.outline_shift is not None:
            out['outline_shift'] = self.outline_shift
        if self.interior_mismatch is not None:
            out['interior_mismatch'] = self.interior_mismatch
        if self.interior_shift is not None:
            out['interior_shift'] = self.interior_shift
        return out


def _normalised(points: Points) -> Optional[Points]:
    """
    Centre a point set on the origin and scale it to unit RMS radius.

    Done *before* the fit rather than dividing the residual afterwards, and the
    difference is not cosmetic. Normalising each set by its own size makes every
    reading invariant to a similarity transform of any input — which is required
    here, because the three landmark sets are read off three different images at
    three different scales. Dividing at the end instead ties the answer to
    whichever frame supplied the denominator, and an output whose face measures
    slightly larger than the target's then reports a shape difference for that
    reason alone.

    It also puts every residual in one unit — a fraction of the face's own
    radius — so `gap` and `mismatch` are directly comparable and their ratio
    means something.

    Args:
        points: (N, 2) landmarks

    Returns:
        (N, 2) centred, unit-radius points, or None if degenerate
    """
    array = np.asarray(points, dtype=np.float64)
    centred = array - array.mean(axis=0)
    radius = float(np.sqrt(float((centred ** 2).sum(axis=1).mean())))
    if radius < 1e-9:
        return None
    return centred / radius


def residual(
    source: Points,
    target: Points,
    fit_on: Optional[npt.NDArray[Any]] = None,
    measure_on: Optional[npt.NDArray[Any]] = None,
) -> Optional[float]:
    """
    RMS shape difference between two landmark sets, as a fraction of face size.

    Both sets are reduced to unit size first, then the similarity transform
    between them is fitted and removed — which is what leaves *shape*: scale,
    rotation and position are exactly the four degrees of freedom alignment
    already normalises away, so what survives is the part `estimate_similarity`
    cannot represent. Using that same estimator is deliberate: the residual
    reported here is literally the residual the pipeline's own alignment leaves
    behind.

    `fit_on` and `measure_on` come apart for the outline reading, which uses the
    whole-face fit and measures only at the silhouette. Anchoring that fit on
    the *interior* features instead reads better on paper — "given the eyes,
    nose and mouth are lined up, how far off is the outline?" — and measured
    worse on both halves of the test that matters: on a fixture where only the
    contour moved to the source it read 0.841 against the whole-face fit's
    0.891, and on one where only the interior moved it leaked 0.117 against
    0.069. The reason is that an interior-anchored fit is itself recomputed
    between the two residuals, so an interior change moves the frame the outline
    is measured in. A fit spread over every point is the more stable frame.

    Note both sets are normalised over **all** their points even when the fit
    and the measurement use a subset, so the outline reading and the whole-face
    reading are denominated in the same unit and can be printed side by side.

    Args:
        source: (N, 2) reference shape
        target: (N, 2) shape to measure against, in any coordinate frame
        fit_on: Indices to fit the transform on. All points if None
        measure_on: Indices to measure the residual over. All points if None

    Returns:
        Residual as a fraction of the face's RMS radius, or None if degenerate
    """
    if np.asarray(source).shape != np.asarray(target).shape:
        return None

    first = _normalised(source)
    second = _normalised(target)
    if first is None or second is None:
        return None
    if first.ndim != 2 or first.shape[1] != 2 or first.shape[0] < 3:
        return None

    fit = slice(None) if fit_on is None else fit_on
    matrix = estimate_similarity(first[fit], second[fit])
    if matrix is None:
        return None

    warped = first @ matrix[:, :2].T + matrix[:, 2]

    measure = slice(None) if measure_on is None else measure_on
    error = warped[measure] - second[measure]

    return float(np.sqrt(float((error ** 2).sum(axis=1).mean())))


def outline_indices(
    points: Points,
) -> Optional[Tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
    """
    Split a landmark set into outline and interior, by distance to the hull.

    The silhouette is the convex hull of the landmarks — not by analogy but
    literally: that is what `FaceMasker` fills to decide which pixels the swap
    is allowed to occupy, and therefore what clips a shape-aware model's contour
    back off. Ranking every point by how close it sits to that boundary and
    taking the nearest third names the silhouette without knowing anything about
    the pack's index layout.

    A radial ranking from the centroid was tried first and is wrong in a way
    worth recording: a face is taller than it is wide, so the outer brow ends
    outrank the chin on distance-from-centre, and the subset drifts off the jaw
    — which is the one part of the contour a shape-aware model actually moves.

    Taken from the **source's** shape rather than from a mean of all three,
    because the source is the reference every residual is fitted onto: using one
    set for both keeps the subset selection consistent with the measurement, and
    avoids having to align three point clouds before averaging them.

    Args:
        points: (N, 2) landmarks

    Returns:
        (outline, interior) index arrays, or None if either side would be too
        small to fit or to average
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < _MIN_POINTS * 2:
        return None

    count = max(_MIN_POINTS, int(round(array.shape[0] * _OUTLINE_SHARE)))
    if array.shape[0] - count < _MIN_POINTS:
        return None

    # Scale-free before measuring distances, so the ranking does not depend on
    # whether the landmarks arrived in a 112px crop or a 640px frame.
    scaled = _normalised(array)
    if scaled is None:
        return None

    hull = cv2.convexHull(scaled.astype(np.float32))
    # Negative outside, positive inside, zero on the boundary. The absolute
    # value is what "on the silhouette" means — a hull vertex reads 0.
    depth = np.array([
        abs(float(cv2.pointPolygonTest(hull, (float(x), float(y)), True)))
        for x, y in scaled
    ], dtype=np.float64)

    order = np.argsort(depth)
    return order[:count], order[count:]


def compare(
    source: Points,
    target: Points,
    output: Points,
) -> Optional[ShapeReading]:
    """
    How far the output's head shape moved off the target's toward the source's.

    Args:
        source: (N, 2) landmarks on the source photograph
        target: (N, 2) landmarks on the target, before the swap
        output: (N, 2) landmarks on the finished frame

    Returns:
        A `ShapeReading`, or None if the three sets are not comparable
    """
    first = np.asarray(source, dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 2:
        return None
    if (np.asarray(target).shape != first.shape
            or np.asarray(output).shape != first.shape):
        return None

    mismatch = residual(source, target)
    gap = residual(source, output)
    if mismatch is None or gap is None:
        return None

    shift = None if mismatch < _MIN_MISMATCH else 1.0 - (gap / mismatch)

    outline_mismatch: Optional[float] = None
    outline_gap: Optional[float] = None
    outline_shift: Optional[float] = None
    interior_mismatch: Optional[float] = None
    interior_shift: Optional[float] = None

    split = outline_indices(first)
    if split is not None:
        outline, interior = split
        # **Each subset is fitted on itself**, which is what makes the two
        # readings independent. Measured on a fixture where only the contour
        # moved against one where only the interior did, separation on the
        # outline reading came to 0.983 fitting on the outline, against 0.821
        # fitting on all points and 0.724 fitting on the interior. Fitting on
        # anything the subset does not contain lets the *other* half's movement
        # drag the frame the subset is measured in, which is the contamination
        # that made a contour-only change read as 0.470 of interior movement.
        outline_mismatch = residual(
            source, target, fit_on=outline, measure_on=outline)
        outline_gap = residual(
            source, output, fit_on=outline, measure_on=outline)
        if (outline_mismatch is not None and outline_gap is not None
                and outline_mismatch >= _MIN_MISMATCH):
            outline_shift = 1.0 - (outline_gap / outline_mismatch)

        # **The half that was missing, and it is the half that moves.**
        #
        # A swap model generates into a crop framed by the *target's* five
        # keypoints and the mask is a hull of the target's landmarks, so the
        # silhouette cannot move — `outline_shift` correctly reads ~0 and it
        # was read as "the head shape did not change". On footage the head
        # plainly does read as the source's, because what a viewer takes for
        # head shape is carried by the *interior* contours: where the cheekbone
        # sits, how the jaw shadow falls, the width between the cheek lines.
        #
        # Reporting only the outline made a true statement answer the wrong
        # question. Both halves, always, so the pair says which channel moved.
        # **The interior reading is the weaker of the two, and knowingly so.**
        # A similarity fitted on the interior alone can absorb a uniform
        # scaling of the features, so a change that is scale-like — the whole
        # feature set wider or narrower — is partly fitted away and understated.
        # The outline does not have that problem, because normalisation has
        # already removed global scale and what is left at the silhouette is
        # proportion. Read the interior as "did it move at all", not as a
        # calibrated fraction.
        interior_mismatch = residual(
            source, target, fit_on=interior, measure_on=interior)
        interior_gap = residual(
            source, output, fit_on=interior, measure_on=interior)
        if (interior_mismatch is not None and interior_gap is not None
                and interior_mismatch >= _MIN_MISMATCH):
            interior_shift = 1.0 - (interior_gap / interior_mismatch)

    return ShapeReading(
        mismatch=mismatch,
        gap=gap,
        shift=shift,
        outline_mismatch=outline_mismatch,
        outline_gap=outline_gap,
        outline_shift=outline_shift,
        interior_mismatch=interior_mismatch,
        interior_shift=interior_shift,
    )


class ShapeProbe:
    """
    Reads 106-point landmarks with the detector's own model.

    Shares the detector for the same reason `IdentityProbe` does: the pack has
    `landmark_2d_106` loaded already, the shape-following mask already runs it
    per frame, and a private session would double it for a diagnostic.

    Example:
        probe = ShapeProbe(detector)
        target = probe.landmarks(frame, face.bbox)
        output = probe.landmarks(pasted, face.bbox)
        reading = compare(source_landmarks, target, output)
    """

    def __init__(self, detector: Any) -> None:
        """
        Args:
            detector: A `FaceDetector`, consulted lazily for its landmark model
                so constructing a probe never loads anything
        """
        self._detector = detector
        self._model: Optional[Any] = None
        self._checked = False

    @property
    def available(self) -> bool:
        """Whether a 106-point landmark model could be resolved."""
        return self._landmark() is not None

    def _landmark(self) -> Optional[Any]:
        """
        The landmark model, or None on a pack that does not carry one.

        Resolved once, and a miss is a capability gap rather than a fault: the
        readings simply do not appear, exactly as the guards go quiet without
        `face.pose`.
        """
        if self._checked:
            return self._model

        self._checked = True
        try:
            self._model = self._detector.landmark_model()
        except Exception:
            self._model = None

        return self._model

    def landmarks(
        self,
        frame: Frame,
        bbox: Optional[Sequence[float]],
    ) -> Optional[Points]:
        """
        The 106 landmarks of the face in `bbox`, in `frame` coordinates.

        Run against a box rather than a fresh detection so the *output* frame is
        measured on the same face the target was. Re-detecting could pick a
        different face, or fail on a composite the detector likes less than the
        original, and either would silently compare two different people.

        Args:
            frame: The image to read, BGR
            bbox: (x1, y1, x2, y2) in frame coordinates

        Returns:
            (106, 2) points, or None if this could not be done
        """
        model = self._landmark()
        if model is None or frame is None or bbox is None or len(bbox) < 4:
            return None

        box = np.asarray(bbox, dtype=np.float32).ravel()[:4]
        if float(box[2] - box[0]) < 8.0 or float(box[3] - box[1]) < 8.0:
            return None

        try:
            points = model.get(frame, Face(bbox=box, det_score=1.0))
        except Exception:
            # A diagnostic on a path that may be live. It declines rather than
            # raising, for the same reason the shape mask's own landmark probe
            # does: a per-frame exception here would cost the call.
            return None

        if points is None or len(points) < _MIN_POINTS * 2:
            return None

        return np.asarray(points, dtype=np.float64)[:, :2]

    def landmarks_aligned(
        self,
        crop: Frame,
        bbox: Optional[Sequence[float]],
        matrix: Matrix,
    ) -> Optional[Points]:
        """
        The 106 landmarks of a face in an aligned crop, in crop coordinates.

        The aligned-space sibling of `landmarks`, and the reason attribution is
        possible at all: the geometry a swap model produced exists **only** in
        its own crop. The frame holds the target's face and the detection holds
        the target's landmarks, so a contour the generator moved and the mask
        then clipped off leaves no trace anywhere else.

        Nothing has to be re-derived to compare the result against the other
        two: `compare` normalises each set independently before fitting, so a
        crop in aligned space and a frame in frame space are directly
        comparable. That invariance is the property the tests pin.

        The box handed to the landmark model is the target's own, warped into
        aligned space — exact and free, since the crop was built from the
        target's keypoints and the generated face occupies the same region of it
        by construction.

        Args:
            crop: The aligned crop, BGR
            bbox: (x1, y1, x2, y2) in frame coordinates
            matrix: 2x3 affine, frame space -> aligned space

        Returns:
            (106, 2) points in crop coordinates, or None if this could not be
            done
        """
        if crop is None or bbox is None or len(bbox) < 4 or crop.ndim != 3:
            return None

        edge = float(crop.shape[0])
        box = np.asarray(bbox, dtype=np.float32).ravel()[:4]
        corners = np.array([
            [box[0], box[1]], [box[2], box[1]],
            [box[2], box[3]], [box[0], box[3]],
        ], dtype=np.float32).reshape(-1, 1, 2)

        moved = cv2.transform(
            corners, np.asarray(matrix, dtype=np.float32)).reshape(-1, 2)
        low = np.clip(moved.min(axis=0), 0.0, edge)
        high = np.clip(moved.max(axis=0), 0.0, edge)

        return self.landmarks(crop, [low[0], low[1], high[0], high[1]])
