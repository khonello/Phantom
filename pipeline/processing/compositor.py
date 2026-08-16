"""
Aligned-space face compositing for the Phantom pipeline.

The swapper and the enhancer both ship their own crop-and-paste. Letting them
each do it independently is what produced the classic tells: a face pasted with
stale alignment, a hard elliptical seam, colour that pulses under Poisson
blending, and a sharp poreless face sitting on a soft noisy body.

This module takes ownership of that step. Everything happens in *aligned* face
space — the normalized crop the swapper works in — because alignment removes
rigid head motion, which makes temporal smoothing safe, and it lets colour and
detail statistics compare like against like.

Per face:

    1. restore      (in FFHQ space, blended back at partial strength)
    2. temporal EMA (motion-gated, single subject only)
    3. colour match (LAB, sampled inside the mask, continuous ramp)
    4. detail match (high-frequency band scaling)
    5. warp back    (into a region of interest, not the whole frame)
    6. composite    (soft alpha)
    7. grain        (frame space, monochrome, matched to the source noise)
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face, Mask, Matrix, Points
from pipeline.services.enhancement import Enhancer, CROP_SIZE
from pipeline.services.masking import FaceMasker
from pipeline.logging import emit_warning

# FFHQ 5-point alignment template, normalized to [0, 1]. Restoration models
# (both CodeFormer and GFPGAN) are trained on crops framed this way and have
# strong priors about where features sit, so the crop handed to them must use
# this template rather than the swapper's tighter arcface framing.
_FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float64)


def estimate_similarity(source: Points, target: Points) -> Optional[Matrix]:
    """
    Least-squares similarity transform between two point sets (Umeyama).

    Deliberately not `cv2.estimateAffinePartial2D`, whose RANSAC and LMEDS
    methods are randomized and so carry no guarantee of returning the same
    matrix for the same input. Anything that varies frame to frame here feeds
    straight back into the shimmer the rest of this module exists to remove.
    For five points and a similarity transform the least-squares solution is
    closed-form, exact and reproducible — and it is the same estimator
    InsightFace uses for its own alignment, so the two agree by construction.

    Note this is a similarity (4 degrees of freedom: scale, rotation,
    translation). It cannot reshape a face, so landmarks whose proportions
    differ from the target template land with a small residual. That is
    expected and correct.

    Args:
        source: (N, 2) source points
        target: (N, 2) target points

    Returns:
        2x3 affine mapping source onto target, or None if degenerate.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 2:
        return None

    count = source.shape[0]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_demean = source - source_mean
    target_demean = target - target_mean

    covariance = (target_demean.T @ source_demean) / count
    reflection = np.ones(2)
    if np.linalg.det(covariance) < 0:
        reflection[1] = -1.0

    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    if np.linalg.matrix_rank(covariance) == 0:
        return None

    rotation = u_matrix @ np.diag(reflection) @ vt_matrix

    variance = float(source_demean.var(axis=0).sum())
    scale = 1.0 if variance < 1e-9 else float(singular @ reflection) / variance

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_mean - scale * (rotation @ source_mean)
    return matrix


def compose_affine(outer: Matrix, inner: Matrix) -> Matrix:
    """
    Compose two 2x3 affines: apply `inner` first, then `outer`.

    Args:
        outer: Transform applied second
        inner: Transform applied first

    Returns:
        2x3 affine equivalent to outer(inner(x))
    """
    outer_h = np.vstack([outer, [0.0, 0.0, 1.0]])
    inner_h = np.vstack([inner, [0.0, 0.0, 1.0]])
    composed: Matrix = (outer_h @ inner_h)[:2, :]
    return composed


class FaceCompositor:
    """
    Composites a swapped face crop back into its frame.

    Holds the temporal state for one tracked subject. Callers must invoke
    `reset()` when the face is lost or the source identity changes.

    Example:
        compositor = FaceCompositor(CONFIG, enhancer, masker)
        frame = compositor.composite(frame, face, bgr_fake, matrix)
    """

    # Bounds on the aligned working resolution (`config.aligned_size`). Higher
    # than the swapper's 128 because the restorer genuinely produces 512 of
    # detail, and because mask edges and detail statistics both degrade at 128.
    _ALIGNED_MIN = 128
    _ALIGNED_MAX = 512

    # Gaussian sigma separating "texture" from "shape", specified at 256 and
    # scaled with the working resolution so detail matching behaves the same
    # across quality presets.
    _DETAIL_SIGMA = 1.5
    _DETAIL_SIGMA_REFERENCE = 256.0
    # Bounds on the detail-matching ratio. Unbounded correction turns a flat
    # region into blotches.
    _DETAIL_RATIO = (0.6, 1.6)

    # Colour transfer: LAB distance below _COLOR_FLOOR needs no correction,
    # above _COLOR_FLOOR + _COLOR_RANGE gets full correction. A ramp rather
    # than a threshold, so correction never snaps on and off between frames.
    _COLOR_FLOOR = 4.0
    _COLOR_RANGE = 12.0
    _COLOR_RATIO = (0.7, 1.4)
    # Damping on the L channel's standard deviation. Matching L *mean* fixes
    # brightness and matters; forcing L *std* flattens facial contrast.
    _LUMA_STD_DAMP = 0.5

    # Mean absolute difference between consecutive real crops, in 0-255, at
    # which temporal smoothing is fully released. Below the floor the subject
    # is effectively still; above the ceiling they are talking or turning and
    # smoothing would ghost.
    _MOTION_FLOOR = 2.0
    _MOTION_CEIL = 8.0

    # Noise sigma is clamped to this range before grain is applied.
    _GRAIN_MAX = 6.0
    # Laplacian (4-neighbour) amplifies noise variance by the sum of its
    # squared coefficients: 4*(1^2) + (-4)^2 = 20.
    _LAPLACIAN_GAIN = np.sqrt(20.0)

    def __init__(
        self,
        config: FaceSwapConfig,
        enhancer: Enhancer,
        masker: FaceMasker,
    ) -> None:
        """
        Initialize the compositor.

        Args:
            config: Configuration object
            enhancer: Enhancer service (may be unavailable)
            masker: FaceMasker service
        """
        self.config = config
        self.enhancer = enhancer
        self.masker = masker

        self._prev_fake: Optional[Frame] = None
        self._prev_real: Optional[Frame] = None

    def reset(self) -> None:
        """Drop temporal state (face lost, source changed, pipeline restart)."""
        self._prev_fake = None
        self._prev_real = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def composite(
        self,
        frame: Frame,
        face: Face,
        swapped: Frame,
        matrix: Matrix,
    ) -> Frame:
        """
        Composite a swapped face crop into the frame.

        Args:
            frame: Original frame (not modified)
            face: Detection the swap was generated for
            swapped: The swapper's aligned output crop
            matrix: 2x3 affine mapping frame space -> the swapper's crop

        Returns:
            A new frame with the face composited in. Returns `frame`
            unchanged if compositing fails.
        """
        try:
            return self._composite_impl(frame, face, swapped, matrix)
        except Exception as e:
            emit_warning(
                f'Compositing failed: {type(e).__name__}: {e}',
                scope='COMPOSITOR',
            )
            return frame

    def _composite_impl(
        self,
        frame: Frame,
        face: Face,
        swapped: Frame,
        matrix: Matrix,
    ) -> Frame:
        """Implementation of `composite`."""
        size = self._aligned_size()

        # Rescale the swapper's affine to our working resolution. Scaling the
        # output canvas by k scales all six entries by k, exactly.
        scale = size / float(swapped.shape[0])
        aligned_matrix = (matrix.astype(np.float32) * scale)

        real = cv2.warpAffine(frame, aligned_matrix, (size, size))
        fake = cv2.resize(swapped, (size, size), interpolation=cv2.INTER_CUBIC)

        if self.config.enhance:
            fake = self._enhance(fake, frame, face, aligned_matrix)

        # Temporal smoothing needs a stable subject identity. With multiple
        # faces the per-frame detection order is not stable, so smoothing
        # would blend between different people.
        if not self.config.many_faces:
            fake = self._smooth(fake, real)

        mask = self.masker.build(
            face, aligned_matrix, real, (frame.shape[0], frame.shape[1]),
        )

        if self.config.color_correction:
            fake = self._match_color(fake, real, mask)

        fake = self._match_detail(fake, real, mask)

        return self._paste(frame, fake, mask, aligned_matrix)

    # ------------------------------------------------------------------
    # Aligned-space stages
    # ------------------------------------------------------------------

    def _aligned_size(self) -> int:
        """Working resolution for aligned space, clamped to a sane range."""
        requested = int(getattr(self.config, 'aligned_size', 256) or 256)
        clamped = max(self._ALIGNED_MIN, min(self._ALIGNED_MAX, requested))
        return clamped - (clamped % 2)

    def _enhance(
        self,
        fake: Frame,
        frame: Frame,
        face: Face,
        aligned_matrix: Matrix,
    ) -> Frame:
        """
        Restore the face, blended back at partial strength.

        Restoration happens in **FFHQ space**, not the swapper's arcface
        space. Both restorers are trained on FFHQ-framed crops and rely on
        features sitting where FFHQ puts them; handing them the tighter
        arcface framing measurably degrades the result.

        The crop given to the restorer is the real frame in FFHQ framing with
        the swapped face composited over it. That matters because FFHQ framing
        is wider than arcface — roughly 18% more of the head — so warping the
        swap alone would leave an empty ring. Filling it from the frame gives
        the model a complete, plausible face to work on. Only the region the
        swap actually covers survives the mask downstream, so the real face at
        the edges never reaches the output.

        Full-strength restoration is what makes a swap read as AI: skin comes
        back poreless and perfectly sharp. Blending keeps some of the input's
        imperfection, which is what a believable webcam looks like.
        """
        if not self.enhancer.available:
            return fake

        size = fake.shape[0]
        geometry = self._ffhq_geometry(face, aligned_matrix)
        if geometry is None:
            return fake
        aligned_to_ffhq, ffhq_from_frame = geometry

        crop = self._build_ffhq_crop(fake, frame, aligned_to_ffhq, ffhq_from_frame, size)

        restored = self.enhancer.restore(crop)
        if restored is None:
            return fake
        if restored.shape[0] != CROP_SIZE:
            restored = cv2.resize(restored, (CROP_SIZE, CROP_SIZE))

        back = cv2.warpAffine(
            restored,
            cv2.invertAffineTransform(aligned_to_ffhq),
            (size, size),
            borderMode=cv2.BORDER_REPLICATE,
        )

        strength = float(np.clip(self.config.enhance_strength, 0.0, 1.0))
        if strength >= 1.0:
            return back
        if strength <= 0.0:
            return fake

        return cv2.addWeighted(back, strength, fake, 1.0 - strength, 0.0)

    @staticmethod
    def _ffhq_geometry(
        face: Face,
        aligned_matrix: Matrix,
    ) -> Optional[Tuple[Matrix, Matrix]]:
        """
        Affines linking aligned space, frame space and FFHQ space.

        Returns:
            (aligned -> ffhq, frame -> ffhq), or None if the landmarks are
            unusable.
        """
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) != len(_FFHQ_TEMPLATE):
            return None

        ffhq_from_frame = estimate_similarity(
            np.asarray(kps, dtype=np.float64),
            _FFHQ_TEMPLATE * CROP_SIZE,
        )
        if ffhq_from_frame is None:
            return None

        frame_from_aligned = cv2.invertAffineTransform(aligned_matrix)
        aligned_to_ffhq = compose_affine(ffhq_from_frame, frame_from_aligned)
        return (
            aligned_to_ffhq.astype(np.float32),
            ffhq_from_frame.astype(np.float32),
        )

    @staticmethod
    def _build_ffhq_crop(
        fake: Frame,
        frame: Frame,
        aligned_to_ffhq: Matrix,
        ffhq_from_frame: Matrix,
        size: int,
    ) -> Frame:
        """
        FFHQ-framed crop: the real frame with the swapped face laid over it.

        The seam between the two is feathered so the restorer does not see a
        hard rectangular edge and try to reconstruct it as a feature.
        """
        base = cv2.warpAffine(
            frame,
            ffhq_from_frame,
            (CROP_SIZE, CROP_SIZE),
            borderMode=cv2.BORDER_REPLICATE,
        )
        overlay = cv2.warpAffine(
            fake,
            aligned_to_ffhq,
            (CROP_SIZE, CROP_SIZE),
            borderMode=cv2.BORDER_REPLICATE,
        )

        coverage = cv2.warpAffine(
            np.full((size, size), 255, dtype=np.uint8),
            aligned_to_ffhq,
            (CROP_SIZE, CROP_SIZE),
            flags=cv2.INTER_NEAREST,
        )
        coverage = cv2.erode(coverage, np.ones((5, 5), np.uint8), iterations=1)
        alpha = cv2.GaussianBlur(coverage.astype(np.float32) / 255.0, (0, 0), 6.0)
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

        merged: Frame = (
            overlay.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        return merged

    def _smooth(self, fake: Frame, real: Frame) -> Frame:
        """
        Temporal EMA in aligned space, released under motion.

        Alignment has already removed translation, scale and rotation, so
        what remains between frames is expression change plus the generator's
        own frame-to-frame instability. Smoothing kills the latter, which is
        the shimmer that reads as fake.

        The motion gate is measured on the *real* crops, not the fakes: the
        question is whether the subject actually changed, and the fakes carry
        generator noise that would confuse that signal.
        """
        alpha = float(np.clip(self.config.temporal_alpha, 0.0, 1.0))
        if alpha >= 1.0:
            return fake

        prev_fake, prev_real = self._prev_fake, self._prev_real
        if (
            prev_fake is None
            or prev_real is None
            or prev_fake.shape != fake.shape
            or prev_real.shape != real.shape
        ):
            self._prev_fake = fake.astype(np.float32)
            self._prev_real = real.astype(np.float32)
            return fake

        current_real = real.astype(np.float32)
        change = float(np.abs(current_real - prev_real).mean())
        gate = np.clip(
            (change - self._MOTION_FLOOR) / (self._MOTION_CEIL - self._MOTION_FLOOR),
            0.0,
            1.0,
        )
        effective = alpha + (1.0 - alpha) * gate

        smoothed = effective * fake.astype(np.float32) + (1.0 - effective) * prev_fake

        self._prev_fake = smoothed
        self._prev_real = current_real

        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def _match_color(self, fake: Frame, real: Frame, mask: Mask) -> Frame:
        """
        Match the swap's colour distribution to the target's, in LAB.

        Statistics are sampled *inside the mask only*. Sampling a bounding
        box instead pulls in hair and background, which is how a bright
        window behind someone ends up shifting their skin tone.
        """
        binary = (mask > 0.5).astype(np.uint8)
        if int(binary.sum()) < 64:
            return fake

        fake_lab = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB)
        real_lab = cv2.cvtColor(real, cv2.COLOR_BGR2LAB)

        fake_mean, fake_std = cv2.meanStdDev(fake_lab, mask=binary)
        real_mean, real_std = cv2.meanStdDev(real_lab, mask=binary)

        delta = float(np.linalg.norm(real_mean - fake_mean))
        strength = float(np.clip(
            (delta - self._COLOR_FLOOR) / self._COLOR_RANGE, 0.0, 1.0,
        ))
        strength *= float(np.clip(self.config.color_strength, 0.0, 1.0))
        if strength <= 0.0:
            return fake

        result = fake_lab.astype(np.float32)
        for channel in range(3):
            f_mean = float(fake_mean[channel][0])
            f_std = float(fake_std[channel][0])
            r_mean = float(real_mean[channel][0])
            r_std = float(real_std[channel][0])

            if f_std < 1e-3:
                ratio = 1.0
            else:
                ratio = float(np.clip(r_std / f_std, *self._COLOR_RATIO))
            if channel == 0:
                ratio = 1.0 + (ratio - 1.0) * self._LUMA_STD_DAMP

            target = (result[:, :, channel] - f_mean) * ratio + r_mean
            result[:, :, channel] += (target - result[:, :, channel]) * strength

        corrected = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    def _match_detail(self, fake: Frame, real: Frame, mask: Mask) -> Frame:
        """
        Scale the swap's high-frequency band to match the target's.

        Solving for a blur radius is unstable and only corrects in one
        direction. Scaling the high band handles both cases, which is
        necessary here: the swap is softer than the frame before enhancement
        and sharper after it.
        """
        selection = mask > 0.5
        if int(selection.sum()) < 64:
            return fake

        fake_f = fake.astype(np.float32)
        real_f = real.astype(np.float32)

        # Scale the band split with the working resolution, so "texture" means
        # the same physical detail at every quality preset.
        sigma = self._DETAIL_SIGMA * fake.shape[0] / self._DETAIL_SIGMA_REFERENCE

        fake_low = cv2.GaussianBlur(fake_f, (0, 0), sigma)
        real_low = cv2.GaussianBlur(real_f, (0, 0), sigma)

        fake_high = fake_f - fake_low
        real_high = real_f - real_low

        fake_energy = float(fake_high[selection].std())
        real_energy = float(real_high[selection].std())
        if fake_energy < 1e-3:
            return fake

        ratio = float(np.clip(real_energy / fake_energy, *self._DETAIL_RATIO))
        matched = fake_low + fake_high * ratio

        return np.clip(matched, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Frame-space compositing
    # ------------------------------------------------------------------

    def _paste(
        self,
        frame: Frame,
        fake: Frame,
        mask: Mask,
        matrix: Matrix,
    ) -> Frame:
        """
        Warp the finished crop back and alpha-composite it.

        Work is confined to a region of interest around the face, so cost
        scales with face size rather than frame size.
        """
        height, width = frame.shape[:2]
        inverse = cv2.invertAffineTransform(matrix)

        roi = self._region_of_interest(inverse, fake.shape[0], width, height)
        if roi is None:
            return frame
        x0, y0, x1, y1 = roi
        roi_w, roi_h = x1 - x0, y1 - y0

        # Shift the destination origin to the ROI instead of warping full frame.
        local = inverse.copy()
        local[0, 2] -= x0
        local[1, 2] -= y0

        warped_fake = cv2.warpAffine(fake, local, (roi_w, roi_h))
        warped_mask = cv2.warpAffine(mask, local, (roi_w, roi_h))

        # Second, smaller feather in frame space. Blurring only at the aligned
        # resolution leaves a stair-stepped seam once the face in the frame is
        # larger than the aligned crop.
        sigma = max(1.0, roi_w * 0.01)
        warped_mask = cv2.GaussianBlur(warped_mask, (0, 0), sigma)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)

        target = frame[y0:y1, x0:x1].astype(np.float32)
        alpha = warped_mask[:, :, None]
        blended = warped_fake.astype(np.float32) * alpha + target * (1.0 - alpha)

        if self.config.grain:
            blended = self._add_grain(blended, target, warped_mask)

        result = frame.copy()
        result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _region_of_interest(
        inverse: Matrix,
        size: int,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Frame-space bounding box of the aligned crop, clamped to the frame.

        Padded so the frame-space feather has room to fall off.
        """
        corners = np.array(
            [[0, 0], [size, 0], [size, size], [0, size]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        projected = cv2.transform(corners, inverse).reshape(-1, 2)

        pad = 4
        x0 = max(0, int(np.floor(projected[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(projected[:, 1].min())) - pad)
        x1 = min(width, int(np.ceil(projected[:, 0].max())) + pad)
        y1 = min(height, int(np.ceil(projected[:, 1].max())) + pad)

        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return x0, y0, x1, y1

    def _add_grain(
        self,
        blended: Frame,
        target: Frame,
        mask: Mask,
    ) -> Frame:
        """
        Add sensor-matched grain over the composited region.

        The generated face is noise-free while the rest of the frame carries
        sensor noise and JPEG artefacts. That mismatch is read instantly even
        when a viewer cannot name it.

        Grain is added here rather than in aligned space because warping a
        crop down to the face's size in frame would filter the noise into
        blobs. It is monochrome — one luma field added to all three channels
        — because independent per-channel noise looks like coloured confetti,
        nothing like a camera.
        """
        sigma = self._estimate_noise(target)
        if sigma <= 0.1:
            return blended

        noise = np.random.normal(0.0, sigma, mask.shape).astype(np.float32)
        return blended + noise[:, :, None] * mask[:, :, None]

    def _estimate_noise(self, region: Frame) -> float:
        """
        Robust noise sigma of a region, via the MAD of its Laplacian.

        A median-based estimator is used rather than a mean-based one so that
        genuine facial detail and edges do not inflate the result.
        """
        gray = cv2.cvtColor(np.clip(region, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)

        median = float(np.median(laplacian))
        mad = float(np.median(np.abs(laplacian - median)))
        sigma = 1.4826 * mad / self._LAPLACIAN_GAIN

        return float(np.clip(sigma, 0.0, self._GRAIN_MAX))
