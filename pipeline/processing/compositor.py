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
from pipeline.services import guards
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
    # `config.aligned_size` is a *ceiling*; the size actually used is chosen per
    # face from how many pixels it occupies in the frame. Compositing a face
    # that covers 130px of frame at 320 upsamples the swapper's 128 output more
    # than twice over and then does every downstream stage on six times the
    # pixels, for detail that was never in the source. Sitting near the face's
    # own resolution is both cheaper and more honest.
    _ALIGNED_STEPS = (128, 192, 256, 320, 384, 448, 512)
    # Fractional change required before switching step. Without it a face
    # hovering on a boundary would flip size frame to frame, and each flip
    # discards the temporal state (the smoothed buffer is the wrong shape).
    _ALIGNED_HYSTERESIS = 0.18

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

    # Illumination matching. A global mean/std shift is correct only when the
    # light is flat; a video call almost never is — there is a window or a lamp
    # on one side. Under directional light the real face carries a brightness
    # gradient the swap does not, and no single shift can match both ends of it,
    # so it lands correct on average and visibly wrong at one edge.
    #
    # This corrects the *low-frequency* difference that survives the global
    # match. The scale factor is what keeps it from copying the target's face
    # onto the swap: the residual is computed at 1/8 resolution, so only
    # illumination survives it, never features.
    _ILLUM_DOWNSCALE = 8
    _ILLUM_SIGMA = 4.0    # at the downscaled resolution, so 32px at full size
    _ILLUM_LIMIT = 12.0   # max correction in LAB units, per channel
    _ILLUM_SCALE = 0.7    # match most of the gradient, not all of it

    # Whole-crop motion, as mean absolute difference between consecutive real
    # crops in 0-255, at which temporal smoothing is fully released. Below the
    # floor the subject is effectively still; above the ceiling they are turning
    # and smoothing would ghost.
    #
    # These sit lower than they would need to for a raw per-pixel difference.
    # The change map is area-reduced first (see _MOTION_DOWNSCALE), which takes
    # sensor noise out of the measurement — so the floor no longer has to clear
    # a noise baseline of its own, and the same numeric reading now means
    # strictly more real motion than it used to.
    _MOTION_FLOOR = 1.0
    _MOTION_CEIL = 6.0
    # Per-region motion, measured as excess over the crop's own still baseline.
    # This is what catches a mouth moving while the head is still — motion
    # confined to a small part of the crop that a whole-crop average cannot see.
    # Relative to the baseline rather than absolute, so it self-calibrates to
    # however noisy the camera is.
    _MOTION_LOCAL_FLOOR = 1.5
    _MOTION_LOCAL_RANGE = 6.0
    # Blur applied to the change map before gating, in pixels at 256. Without it
    # the gate would respond to sensor noise, which is per-pixel; with it the
    # gate asks whether a *region* moved.
    _MOTION_SIGMA = 6.0
    # The change map is built at 1/N resolution. This is not only for speed: the
    # area-average suppresses sensor noise, which is spatially incoherent, while
    # leaving real motion, which is not. The measure therefore reports motion
    # rather than noise, and a genuinely still subject is smoothed properly
    # instead of being held part-way open by whatever grain the camera has.
    _MOTION_DOWNSCALE = 4

    # Noise sigma is clamped to this range before grain is applied.
    _GRAIN_MAX = 6.0
    # Subsampling stride for the noise estimate.
    _NOISE_STRIDE = 2
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
        self._working_size: Optional[int] = None

    def reset(self) -> None:
        """Drop temporal state (face lost, source changed, pipeline restart)."""
        self._prev_fake = None
        self._prev_real = None
        self._working_size = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def composite(
        self,
        frame: Frame,
        face: Face,
        swapped: Frame,
        matrix: Matrix,
    ) -> Optional[Frame]:
        """
        Composite a swapped face crop into the frame.

        Args:
            frame: Original frame (not modified)
            face: Detection the swap was generated for
            swapped: The swapper's aligned output crop
            matrix: 2x3 affine mapping frame space -> the swapper's crop

        Returns:
            A new frame with the face composited in, or **None** if no swapped
            frame could be produced — the occlusion guard refused it, or
            compositing failed.

            None rather than the untouched frame, because on the live path the
            untouched frame is the operator's real face, and emitting that is the
            exact exposure the guards exist to prevent. What to show instead is
            the caller's decision: live holds the last good frame, batch passes
            the original through.
        """
        try:
            return self._composite_impl(frame, face, swapped, matrix)
        except Exception as e:
            emit_warning(
                f'Compositing failed: {type(e).__name__}: {e}',
                scope='COMPOSITOR',
            )
            # Fail closed. Temporal state may be half-updated, so drop it rather
            # than smooth the next frame against it.
            self.reset()
            return None

    def _composite_impl(
        self,
        frame: Frame,
        face: Face,
        swapped: Frame,
        matrix: Matrix,
    ) -> Optional[Frame]:
        """Implementation of `composite`."""
        size = self._aligned_size(matrix, swapped.shape[0])

        # Rescale the swapper's affine to our working resolution. Scaling the
        # output canvas by k scales all six entries by k, exactly.
        scale = size / float(swapped.shape[0])
        aligned_matrix = (matrix.astype(np.float32) * scale)

        real = cv2.warpAffine(frame, aligned_matrix, (size, size))

        # Built before restoration and smoothing, not after, so the occlusion
        # guard can refuse the frame before anything mutates temporal state. The
        # mask depends only on the face, the affine and the real crop, so this is
        # the same mask it was when it came later.
        mask = self.masker.build(
            face, aligned_matrix, real, (frame.shape[0], frame.shape[1]),
        )

        if not guards.coverage_ok(self.config, self.masker.last_coverage):
            return None

        fake = cv2.resize(swapped, (size, size), interpolation=cv2.INTER_CUBIC)

        if self.config.enhance:
            fake = self._enhance(fake, frame, face, aligned_matrix)

        # Temporal smoothing needs a stable subject identity. With multiple
        # faces the per-frame detection order is not stable, so smoothing
        # would blend between different people.
        if not self.config.many_faces:
            fake = self._smooth(fake, real)

        if self.config.color_correction:
            fake = self._match_color(fake, real, mask)

        fake = self._match_detail(fake, real, mask)

        return self._paste(frame, fake, mask, aligned_matrix)

    # ------------------------------------------------------------------
    # Aligned-space stages
    # ------------------------------------------------------------------

    def _aligned_size(self, matrix: Matrix, crop_size: int) -> int:
        """
        Working resolution for aligned space, chosen from the face's size.

        `config.aligned_size` sets the ceiling. Within it, the size follows how
        many frame pixels the face actually covers, so someone sitting back from
        the camera is not composited at the same cost as someone filling it —
        and, more importantly, is not upsampled to a detail level their webcam
        never captured.

        Args:
            matrix: 2x3 affine mapping frame space -> the swapper's crop
            crop_size: Edge length of the swapper's output crop

        Returns:
            Even working resolution, within [_ALIGNED_MIN, config ceiling].
        """
        requested = int(getattr(self.config, 'aligned_size', 256) or 256)
        # The floor comes from the model: compositing a face below the swapper's
        # native output size discards detail the model already produced. A 128px
        # model can legitimately drop to 128 for a distant face; a 256px one
        # cannot without wasting half of what it generated.
        floor = max(
            self._ALIGNED_MIN,
            int(getattr(self.config, 'aligned_min', self._ALIGNED_MIN) or self._ALIGNED_MIN),
        )
        floor = min(floor, self._ALIGNED_MAX)
        ceiling = max(floor, min(self._ALIGNED_MAX, requested))

        # A similarity transform's linear part scales area by its determinant,
        # so the face's extent in frame is the crop size divided by that scale.
        scale = float(np.sqrt(abs(float(np.linalg.det(matrix[:, :2])))))
        extent = (crop_size / scale) if scale > 1e-6 else float(ceiling)
        target = float(np.clip(extent, floor, ceiling))

        current = self._working_size
        if current is not None and floor <= current <= ceiling:
            if abs(target - current) <= current * self._ALIGNED_HYSTERESIS:
                return current

        steps = [s for s in self._ALIGNED_STEPS if floor <= s <= ceiling]
        chosen = min(steps or [floor], key=lambda s: abs(s - target))

        self._working_size = chosen
        return chosen

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
        gate = self._motion_gate(current_real, prev_real)

        # prev + (fake - prev) * effective, with the gate merged to three
        # channels up front. The equivalent broadcast form builds a full-size
        # temporary for every term, which on this path is the single most
        # expensive thing the compositor does.
        effective = alpha + (1.0 - alpha) * cv2.merge([gate, gate, gate])

        smoothed = cv2.add(
            prev_fake,
            cv2.multiply(cv2.subtract(fake.astype(np.float32), prev_fake), effective),
        )

        self._prev_fake = smoothed
        self._prev_real = current_real

        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def _motion_gate(self, current_real: Frame, prev_real: Frame) -> Frame:
        """
        Per-pixel release factor for temporal smoothing: 0 = smooth fully,
        1 = pass the current frame through untouched.

        Combines two measures, and takes whichever releases more:

        - **Whole-crop** — the subject turned or moved. Unchanged from the
          original behaviour, so its tuning still means what it did.
        - **Per-region** — this part of the face moved more than the rest of
          it. This is the one that matters on a call. Someone talking with a
          still head changes only the mouth, perhaps 15% of the crop; averaged
          over the whole crop that lands below the floor, the gate reads
          "still", and smoothing stays fully on over the exact region a viewer
          is watching. Lips then blend across frames and smear.

        Taking the maximum means this can only ever smooth *less* than before,
        never more. That is the safe direction: under-smoothing costs a little
        shimmer, over-smoothing ghosts the mouth.

        Args:
            current_real: Aligned real crop for this frame (float32)
            prev_real: Aligned real crop for the previous frame (float32)

        Returns:
            float32 gate with shape (H, W); the caller expands it to three
            channels.
        """
        size = current_real.shape[0]
        small = max(16, size // self._MOTION_DOWNSCALE)

        current_small = cv2.resize(current_real, (small, small), interpolation=cv2.INTER_AREA)
        prev_small = cv2.resize(prev_real, (small, small), interpolation=cv2.INTER_AREA)

        difference = cv2.absdiff(current_small, prev_small)
        change: Frame = (
            difference[:, :, 0] + difference[:, :, 1] + difference[:, :, 2]
        ).astype(np.float32)
        change *= 1.0 / 3.0

        sigma = (
            self._MOTION_SIGMA * size / self._DETAIL_SIGMA_REFERENCE
        ) / self._MOTION_DOWNSCALE
        blurred: Frame = cv2.GaussianBlur(change, (0, 0), max(sigma, 0.6))

        global_change = float(blurred.mean())
        global_gate = float(np.clip(
            (global_change - self._MOTION_FLOOR)
            / (self._MOTION_CEIL - self._MOTION_FLOOR),
            0.0,
            1.0,
        ))

        # The median is dominated by still skin even mid-sentence, which makes
        # it a good estimate of this camera's no-motion level — including its
        # residual noise, so the local term does not need to know the sensor.
        baseline = float(np.median(blurred))
        local_gate = np.clip(
            (blurred - baseline - self._MOTION_LOCAL_FLOOR) / self._MOTION_LOCAL_RANGE,
            0.0,
            1.0,
        )

        np.maximum(local_gate, global_gate, out=local_gate)
        gate: Frame = cv2.resize(
            local_gate, (size, size), interpolation=cv2.INTER_LINEAR,
        )
        return gate

    def _match_color(self, fake: Frame, real: Frame, mask: Mask) -> Frame:
        """
        Match the swap's colour distribution to the target's, in LAB.

        Statistics are sampled *inside the mask only*. Sampling a bounding
        box instead pulls in hair and background, which is how a bright
        window behind someone ends up shifting their skin tone.

        Two passes, deliberately gated differently. The global mean/std
        transfer only engages once the overall colours actually differ, since
        correcting a match is pure risk. The illumination pass always runs,
        because the case it exists for — a face lit from one side — is one
        where the global means already agree and only their *distribution*
        across the face differs. Gating it on the same distance would switch
        it off in precisely the situation it was written for.
        """
        binary = (mask > 0.5).astype(np.uint8)
        if int(binary.sum()) < 64:
            return fake

        color_strength = float(np.clip(self.config.color_strength, 0.0, 1.0))
        if color_strength <= 0.0:
            return fake

        fake_lab = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB)
        real_lab = cv2.cvtColor(real, cv2.COLOR_BGR2LAB)

        fake_mean, fake_std = cv2.meanStdDev(fake_lab, mask=binary)
        real_mean, real_std = cv2.meanStdDev(real_lab, mask=binary)

        # Ramp rather than a threshold, so correction never snaps on and off
        # between frames.
        delta = float(np.linalg.norm(real_mean - fake_mean))
        global_strength = color_strength * float(np.clip(
            (delta - self._COLOR_FLOOR) / self._COLOR_RANGE, 0.0, 1.0,
        ))

        result = fake_lab.astype(np.float32)

        if global_strength > 0.0:
            # Per channel the transfer is affine — scale then shift — so the
            # coefficients can be solved as three scalars and applied in one
            # vectorised pass. Written out channel by channel this was nine
            # full-resolution array operations per frame.
            #
            #   new = x + ((x - f_mean) * ratio + r_mean - x) * s
            #       = x * (1 - s + s * ratio) + s * (r_mean - f_mean * ratio)
            gain = np.ones(3, dtype=np.float32)
            offset = np.zeros(3, dtype=np.float32)

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

                gain[channel] = 1.0 - global_strength + global_strength * ratio
                offset[channel] = global_strength * (r_mean - f_mean * ratio)

            result *= gain
            result += offset

        result = self._match_illumination(
            result, real_lab.astype(np.float32), mask, color_strength,
        )

        corrected = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    def _match_illumination(
        self,
        fake_lab: Frame,
        real_lab: Frame,
        mask: Mask,
        strength: float,
    ) -> Frame:
        """
        Match the low-frequency lighting gradient the global transfer misses.

        Works on a heavily downscaled residual, which does three things at once:
        it is cheap, it guarantees only illumination survives (facial features
        cannot outlive an 8x reduction), and the bilinear upsample gives a
        smooth correction field with no edges of its own.

        The blur is *normalized by the mask* — each output pixel is divided by
        the blurred mask weight — so only in-mask pixels contribute. Blurring
        the raw residual instead would pull hair and background into the
        correction, which is the same mistake sampling a bounding box makes.

        Args:
            fake_lab: Globally colour-matched swap, LAB float32
            real_lab: Target crop, LAB float32
            mask: Soft compositing mask in [0, 1]
            strength: Same ramped strength the global transfer used

        Returns:
            fake_lab with the illumination residual added.
        """
        size = fake_lab.shape[0]
        half = max(16, size // 2)
        small = max(8, size // self._ILLUM_DOWNSCALE)

        # Everything below half resolution, so only the final add touches a
        # full-size array. Subtraction is linear, so downscaling the two crops
        # before differencing is exact; the mask weighting must still happen
        # before the reduction to `small`, or out-of-mask pixels would leak in.
        real_half = cv2.resize(real_lab, (half, half), interpolation=cv2.INTER_AREA)
        fake_half = cv2.resize(fake_lab, (half, half), interpolation=cv2.INTER_AREA)
        weight_half = cv2.resize(mask, (half, half), interpolation=cv2.INTER_AREA)

        residual = cv2.multiply(
            cv2.subtract(real_half, fake_half),
            cv2.merge([weight_half, weight_half, weight_half]),
        )

        # INTER_AREA is a box filter, so the downsample is itself most of the
        # smoothing; the Gaussian below only has to finish the job.
        residual_small = cv2.resize(residual, (small, small), interpolation=cv2.INTER_AREA)
        weight_small = cv2.resize(weight_half, (small, small), interpolation=cv2.INTER_AREA)

        sigma = self._ILLUM_SIGMA * small / 32.0
        residual_small = cv2.GaussianBlur(residual_small, (0, 0), max(sigma, 0.8))

        # Two different uses of the mask, which must not be the same array:
        # `denominator` is the blurred weight the residual was accumulated
        # against, and dividing by it is what makes this a normalized
        # convolution. `weight_small` is the fade applied afterwards. Reusing
        # one for both would cancel the fade out entirely.
        denominator = cv2.GaussianBlur(weight_small, (0, 0), max(sigma, 0.8))

        scaled: Frame = residual_small
        scaled /= np.maximum(denominator, 1e-3)[:, :, None]
        np.clip(scaled, -self._ILLUM_LIMIT, self._ILLUM_LIMIT, out=scaled)

        # Fade the correction out with the mask, so it cannot introduce an edge
        # of its own where the composite is already handing back to the frame.
        # Folded in here rather than after the upsample: the fade is inherently
        # low-frequency, and doing it at full size would cost more than the rest
        # of this method put together.
        scaled *= (strength * self._ILLUM_SCALE * weight_small)[:, :, None]

        correction = cv2.resize(
            scaled, (size, size), interpolation=cv2.INTER_LINEAR,
        )

        corrected: Frame = cv2.add(fake_lab, correction)
        return corrected

    def _match_detail(self, fake: Frame, real: Frame, mask: Mask) -> Frame:
        """
        Scale the swap's high-frequency band to match the target's.

        Solving for a blur radius is unstable and only corrects in one
        direction. Scaling the high band handles both cases, which is
        necessary here: the swap is softer than the frame before enhancement
        and sharper after it.
        """
        binary = (mask > 0.5).astype(np.uint8)
        if cv2.countNonZero(binary) < 64:
            return fake

        fake_f = fake.astype(np.float32)
        real_f = real.astype(np.float32)

        # Scale the band split with the working resolution, so "texture" means
        # the same physical detail at every quality preset.
        sigma = self._DETAIL_SIGMA * fake.shape[0] / self._DETAIL_SIGMA_REFERENCE

        fake_low = cv2.GaussianBlur(fake_f, (0, 0), sigma)
        real_low = cv2.GaussianBlur(real_f, (0, 0), sigma)

        fake_high = cv2.subtract(fake_f, fake_low)
        real_high = cv2.subtract(real_f, real_low)

        # meanStdDev with a mask rather than `array[boolean]`, which would
        # allocate a copy of every selected pixel on every frame. The high band
        # has ~zero mean per channel, so pooling the per-channel deviations
        # reproduces the previous single-population figure.
        _, fake_dev = cv2.meanStdDev(fake_high, mask=binary)
        _, real_dev = cv2.meanStdDev(real_high, mask=binary)
        fake_energy = float(np.sqrt(np.mean(np.square(fake_dev))))
        real_energy = float(np.sqrt(np.mean(np.square(real_dev))))
        if fake_energy < 1e-3:
            return fake

        ratio = float(np.clip(real_energy / fake_energy, *self._DETAIL_RATIO))

        # fake_low + (fake - fake_low) * ratio, rearranged so it is one fused
        # pass rather than a multiply and an add over separate temporaries.
        matched = cv2.addWeighted(fake_f, ratio, fake_low, 1.0 - ratio, 0.0)

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

        # target + (fake - target) * alpha. Written with an explicit 3-channel
        # alpha and cv2 ops rather than `mask[:, :, None]` broadcasting, which
        # numpy expands into a temporary the size of the ROI on every frame.
        alpha = cv2.merge([warped_mask, warped_mask, warped_mask])
        blended = cv2.add(
            target,
            cv2.multiply(cv2.subtract(warped_fake.astype(np.float32), target), alpha),
        )

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

        # Estimate from a subsample. Both medians are sorts, and a noise sigma
        # converges on far fewer than the ~35k pixels a face ROI carries; the
        # stride costs nothing and removes two full-size sorts per frame.
        sample: Frame = np.ascontiguousarray(
            laplacian[::self._NOISE_STRIDE, ::self._NOISE_STRIDE], dtype=np.float32,
        )
        if sample.size < 64:
            sample = laplacian.astype(np.float32)

        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        sigma = 1.4826 * mad / self._LAPLACIAN_GAIN

        return float(np.clip(sigma, 0.0, self._GRAIN_MAX))
