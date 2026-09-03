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

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face, Mask, Matrix
from pipeline.services.enhancement import Enhancer
from pipeline.services.masking import FaceMasker
from pipeline.services import guards
from pipeline.logging import emit_warning
from pipeline.processing import texture
from pipeline.processing.texture import SourceTexture
from pipeline.processing.geometry import (
    ALIGNED_STEPS,
    DETAIL_SIGMA,
    DETAIL_SIGMA_REFERENCE,
    FFHQ_TEMPLATE as _FFHQ_TEMPLATE,
    canonical_from_frame,
    compose_affine,
    estimate_similarity,
)

# Re-exported. `estimate_similarity` and `compose_affine` moved to
# `geometry` once the texture extractor needed the same fit against a source
# photograph, but they read as compositing vocabulary and callers already import
# them from here.
__all__ = ['FaceCompositor', 'estimate_similarity', 'compose_affine']

# Seam feathering for the FFHQ crop, as fractions of its edge length rather
# than absolute pixels — the crop size is now a variable, and a blur measured
# in pixels would mean a different seam at each one. These reproduce the
# previous 5px erode and 6.0 sigma exactly at 512.
_FFHQ_ERODE = 5.0 / 512.0
_FFHQ_FEATHER = 6.0 / 512.0


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
    _ALIGNED_STEPS = ALIGNED_STEPS
    # Fractional change required before switching step. Without it a face
    # hovering on a boundary would flip size frame to frame, and each flip
    # discards the temporal state (the smoothed buffer is the wrong shape).
    _ALIGNED_HYSTERESIS = 0.18

    # Gaussian sigma separating "texture" from "shape", specified at 256 and
    # scaled with the working resolution so detail matching behaves the same
    # across quality presets. Defined in `geometry` because the texture layer
    # *adds* to the band this stage *scales*, and two stages describing
    # adjacent-but-different bands would fight.
    _DETAIL_SIGMA = DETAIL_SIGMA
    _DETAIL_SIGMA_REFERENCE = DETAIL_SIGMA_REFERENCE
    # Bounds on the detail-matching ratio. Unbounded correction turns a flat
    # region into blotches.
    _DETAIL_RATIO = (0.6, 1.6)

    # Colour transfer: LAB distance below _COLOR_FLOOR needs no correction,
    # above _COLOR_FLOOR + _COLOR_RANGE gets full correction. A ramp rather
    # than a threshold, so correction never snaps on and off between frames.
    #
    # The floor was 4.0 with a range of 12.0, which left a sub-4-unit LAB mean
    # difference corrected by *nothing* and a 10-unit one only half corrected.
    # `_match_illumination` recovers about 70% of what that leaves, so a 3.9-unit
    # difference still landed ~1.2 units uncorrected -- across a transition a
    # couple of pixels wide, at a boundary, which is where the eye compares
    # hardest. The floor's stated purpose was that correcting a match is pure
    # risk; the property it was protecting (no snapping between frames) is
    # delivered by the *ramp*, so the floor only has to clear estimator noise.
    _COLOR_FLOOR = 1.5
    _COLOR_RANGE = 8.0
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

    # Floor on the frame-space feather, in pixels. A face small enough that
    # 4% of it is under two pixels has no transition to speak of otherwise.
    _FEATHER_FLOOR = 2.0

    # Edge of the window the headroom statistics are taken over. A standard
    # deviation converges on a few thousand pixels, so measuring every pixel of
    # a large face buys nothing and costs everything: at a 500px region the full
    # measurement was 16ms, which is more than the whole rest of the frame and
    # was paid even when the answer was "no headroom, add nothing". The region
    # is centred on the face, so a centred window is face pixels.
    _HEADROOM_SAMPLE = 160

    # Headroom below which the texture layer does not bother. Adding a fraction
    # of an 8-bit unit costs a warp and changes nothing anyone can see.
    _TEXTURE_FLOOR = 0.25

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

        # Skin detail lifted from the operator's source photograph, set by the
        # pipeline when the source changes. Deliberately not built here: it is a
        # property of the source images, which the compositor never sees.
        #
        # Survives `reset()`. Every other field here is temporal state that a
        # face loss or a resolution change invalidates; this one is a function
        # of the identity, and dropping it on a lost face would mean rebuilding
        # it — a file read and a full-resolution warp — on the live path.
        self.source_texture: Optional[SourceTexture] = None

        # High-band deviation the texture layer was allowed on the last frame,
        # in 8-bit units, or None when it did not run. Same pattern as
        # `masker.last_coverage`: the stage that measures a thing owns the
        # number. Worth reading while judging -- a headroom that is routinely
        # zero means detail matching has already taken the face to the target's
        # texture level and the texture layer has nothing left to add.
        self.last_texture_headroom: Optional[float] = None

        # Per-stage milliseconds for the frame just composited, read by the
        # pipeline's latency budget. Same pattern as `masker.last_coverage`:
        # the stage that measures a thing owns the number, and whoever needs it
        # reads it afterwards rather than having a timer threaded through.
        #
        # Cheap enough to leave on — a `perf_counter` either side of calls that
        # already cost milliseconds — and the alternative is a debug-only path
        # that is never on when the question comes up.
        self.last_stage_ms: Dict[str, float] = {}

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
        stages = self.last_stage_ms
        stages.clear()
        mark = time.perf_counter()

        def elapsed(name: str) -> None:
            """Record milliseconds since the last mark, under `name`."""
            nonlocal mark
            now = time.perf_counter()
            stages[name] = (now - mark) * 1000.0
            mark = now

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
        elapsed('mask')

        if not guards.coverage_ok(self.config, self.masker.last_coverage):
            return None

        fake = cv2.resize(swapped, (size, size), interpolation=cv2.INTER_CUBIC)

        if self.config.enhance and self._restore_worthwhile(face):
            fake = self._enhance(fake, frame, face, aligned_matrix)
            elapsed('restore')

        # Temporal smoothing needs a stable subject identity. With multiple
        # faces the per-frame detection order is not stable, so smoothing
        # would blend between different people.
        if not self.config.many_faces:
            fake = self._smooth(fake, real)
            elapsed('smooth')

        if self.config.color_correction:
            fake = self._match_color(fake, real, mask)
            elapsed('colour')

        fake = self._match_detail(fake, real, mask)
        elapsed('detail')

        pasted = self._paste(frame, fake, mask, aligned_matrix, face, size, scale)
        elapsed('paste')
        return pasted

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

    def _restore_worthwhile(self, face: Face) -> bool:
        """
        Whether this face is big enough to be worth restoring.

        Restoration is the most expensive stage in the pipeline and the only
        one whose cost does *not* follow the face's size in frame — it runs on
        a fixed FFHQ crop either way. Below `restore_min_face` the swap being
        restored came out of a generator smaller than the crop it is being
        upsampled into, so the model is reconstructing detail that was never
        in the source.

        Measured on the shorter side of the bounding box, the same as
        `guard_min_frame_px`, so the two thresholds are set in one unit.

        Args:
            face: Detection the swap was generated for

        Returns:
            True to restore. A missing or unreadable bbox restores, since the
            threshold cannot be evaluated and silently skipping would change
            what the operator sees for a reason nobody could see.
        """
        threshold = int(getattr(self.config, 'restore_min_face', 0) or 0)
        if threshold <= 0:
            return True

        bbox = getattr(face, 'bbox', None)
        if bbox is None or len(bbox) < 4:
            return True
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            return True

        return min(abs(x2 - x1), abs(y2 - y1)) >= threshold

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
        # The enhancer owns this, not the compositor: the request is
        # `config.restore_size`, but a model with fixed spatial dims overrides
        # it, and only the loaded backend knows which it is.
        ffhq_size = self.enhancer.crop_size
        geometry = self._ffhq_geometry(face, aligned_matrix, ffhq_size)
        if geometry is None:
            return fake
        aligned_to_ffhq, ffhq_from_frame = geometry

        crop = self._build_ffhq_crop(
            fake, frame, aligned_to_ffhq, ffhq_from_frame, size, ffhq_size,
        )

        restored = self.enhancer.restore(crop)
        if restored is None:
            return fake
        if restored.shape[0] != ffhq_size:
            restored = cv2.resize(restored, (ffhq_size, ffhq_size))

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
        ffhq_size: int,
    ) -> Optional[Tuple[Matrix, Matrix]]:
        """
        Affines linking aligned space, frame space and FFHQ space.

        Args:
            face: Detection the swap was generated for
            aligned_matrix: 2x3 affine mapping frame space -> aligned space
            ffhq_size: Edge length of the FFHQ crop. The template is
                normalised, so scaling it is all a smaller crop needs — the
                framing is identical, only the sampling rate changes.

        Returns:
            (aligned -> ffhq, frame -> ffhq), or None if the landmarks are
            unusable.
        """
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) != len(_FFHQ_TEMPLATE):
            return None

        ffhq_from_frame = estimate_similarity(
            np.asarray(kps, dtype=np.float64),
            _FFHQ_TEMPLATE * ffhq_size,
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
        ffhq_size: int,
    ) -> Frame:
        """
        FFHQ-framed crop: the real frame with the swapped face laid over it.

        The seam between the two is feathered so the restorer does not see a
        hard rectangular edge and try to reconstruct it as a feature. The
        feather is expressed as a fraction of the crop rather than in absolute
        pixels, so it stays the same *seam* at any `ffhq_size` — a fixed 6px
        blur on a 512 crop is a 3px blur's worth of softness on a 256 one.
        """
        base = cv2.warpAffine(
            frame,
            ffhq_from_frame,
            (ffhq_size, ffhq_size),
            borderMode=cv2.BORDER_REPLICATE,
        )
        overlay = cv2.warpAffine(
            fake,
            aligned_to_ffhq,
            (ffhq_size, ffhq_size),
            borderMode=cv2.BORDER_REPLICATE,
        )

        erode_px = max(3, int(round(ffhq_size * _FFHQ_ERODE))) | 1
        coverage = cv2.warpAffine(
            np.full((size, size), 255, dtype=np.uint8),
            aligned_to_ffhq,
            (ffhq_size, ffhq_size),
            flags=cv2.INTER_NEAREST,
        )
        coverage = cv2.erode(coverage, np.ones((erode_px, erode_px), np.uint8), iterations=1)
        alpha = cv2.GaussianBlur(
            coverage.astype(np.float32) / 255.0, (0, 0), ffhq_size * _FFHQ_FEATHER,
        )
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
        face: Face,
        aligned_size: int,
        scale: float,
    ) -> Frame:
        """
        Warp the finished crop back and alpha-composite it.

        Work is confined to a region of interest around the face, so cost
        scales with face size rather than frame size.

        Skin texture and grain are both added **here**, in frame space, after
        the crop has been warped down to the face's real size. That placement is
        the whole reason either of them survives — see `_add_texture`.
        """
        height, width = frame.shape[:2]
        inverse = cv2.invertAffineTransform(matrix)

        # The face's extent in frame, from the affine rather than from the ROI:
        # the ROI's own size depends on the padding, and the padding depends on
        # the feather, which depends on this. Same formula `_aligned_size` uses.
        extent = (aligned_size / scale) if scale > 1e-6 else float(aligned_size)

        # Frame-space feather, and enough padding for it to fall off inside the
        # region. A blur wider than the pad would be reflected back off the ROI
        # border and the mask would never reach zero there.
        feather = float(np.clip(getattr(self.config, 'mask_feather', 0.04), 0.0, 0.25))
        sigma = max(self._FEATHER_FLOOR, extent * feather)
        pad = int(np.ceil(sigma * 3.0)) + 4

        roi = self._region_of_interest(inverse, fake.shape[0], width, height, pad)
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

        # Second feather, in frame space, and the one that matters. Blurring
        # only at the aligned resolution leaves a stair-stepped seam once the
        # face in the frame is larger than the aligned crop — and, worse, an
        # aligned-space width is a fraction of a *crop* rather than of the face,
        # so it shrinks by the warp's scale factor on the way here and nothing
        # was looking at the product. This one is measured against the face's own
        # extent, which is the only length the eye is comparing against.
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

        # Texture before grain, and both after the composite. Texture is the
        # skin the source person actually has; grain is the sensor the target
        # camera actually has. In that order they layer the way the real thing
        # does — the camera photographs the skin, not the other way round.
        blended = self._add_texture(
            blended, target, warped_mask, face, extent, (x0, y0, roi_w, roi_h),
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
        pad: int = 4,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Frame-space bounding box of the aligned crop, clamped to the frame.

        Padded so the frame-space feather has room to fall off. The caller sizes
        the padding from its own blur radius rather than assuming a constant —
        a feather wider than the pad reflects off the border and leaves the mask
        never reaching zero along it, which is a seam made by the fix for one.
        """
        corners = np.array(
            [[0, 0], [size, 0], [size, size], [0, size]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        projected = cv2.transform(corners, inverse).reshape(-1, 2)

        x0 = max(0, int(np.floor(projected[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(projected[:, 1].min())) - pad)
        x1 = min(width, int(np.ceil(projected[:, 0].max())) + pad)
        y1 = min(height, int(np.ceil(projected[:, 1].max())) + pad)

        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return x0, y0, x1, y1

    def _add_texture(
        self,
        blended: Frame,
        target: Frame,
        mask: Mask,
        face: Face,
        extent: float,
        roi: Tuple[int, int, int, int],
    ) -> Frame:
        """
        Add the source's own skin detail, reprojected onto this frame's face.

        **Why this runs in frame space and not in aligned space**, which is where
        every other appearance stage runs. The compositor works at 128-320 and
        `_paste` warps the finished crop down onto a face that is often ~100px.
        A high-frequency field added before that warp is decimated by it — pores
        land under the Nyquist limit of the destination and average to nothing.
        That is the same mechanism that leaves 86% of the restorer's 512 crop on
        the floor one operation after it is made, and it is why `_add_grain`
        already says grain "would filter into blobs" if it were added earlier.
        Pores are grain with structure. They belong in the same place.

        **How much is added is measured, not set.** The map carries unit
        deviation, so a raw strength would be an open-ended gain: nothing
        downstream of here looks at the result, `_match_detail` ran in aligned
        space before it, and past parity the face becomes noisier than the camera
        that supposedly shot it — failure mode 1 approached from the other side.
        So the *headroom* is measured first: how much high-frequency energy the
        real face in this frame has, less what the swap already carries, less
        what grain is about to add. `texture_strength` is the fraction of that
        gap to close, which makes one setting mean the same thing on any clip.

        The reference is the operator's own face in the same pixels — not the
        frame at large. It is a real face, at the right size, through the right
        lens, under the right light, which is a better statement of "what skin
        looks like here" than anything the background could offer.

        Two masks apply and both are needed. The skin mask is baked into the map
        at extraction, keeping detail off eyes, nostrils and mouth. The
        compositing alpha is applied here, keeping it inside the swap — and
        because that alpha is the feathered one, texture fades out exactly where
        the swap does rather than stopping at a hard edge of its own.

        Args:
            blended: The composited ROI, float32
            target: The original frame's pixels over the same ROI, float32
            mask: Compositing alpha over the ROI, [0, 1]
            face: Detection this frame was swapped for
            extent: The face's extent in frame pixels
            roi: (x0, y0, width, height) of the region in frame space

        Returns:
            `blended` with detail added, or unchanged when the layer is off, no
            source texture is loaded, the geometry is unusable, or there is no
            headroom left. Every failure here is silent and non-fatal: this is a
            decorative layer, and a frame without it is the output this project
            shipped before it existed.
        """
        self.last_texture_headroom = None

        strength = float(np.clip(getattr(self.config, 'texture_strength', 0.0), 0.0, 1.0))
        if strength <= 0.0 or self.source_texture is None:
            return blended

        started = time.perf_counter()

        size = texture.map_size(extent)
        detail = self.source_texture.detail_for(size)
        if detail is None:
            return blended

        canonical = canonical_from_frame(face, size)
        if canonical is None:
            return blended

        headroom = self._texture_headroom(blended, target, mask, extent)
        self.last_texture_headroom = headroom
        if headroom <= self._TEXTURE_FLOOR:
            return blended

        x0, y0, roi_w, roi_h = roi
        local = cv2.invertAffineTransform(canonical)
        local[0, 2] -= x0
        local[1, 2] -= y0

        # Zero outside canonical space, which is the correct border here: no map
        # means no detail, and replicating the edge would smear the outermost
        # row of pores across the rest of the ROI.
        warped = cv2.warpAffine(detail, local, (roi_w, roi_h), borderValue=(0.0,))

        # The map carries unit deviation inside its skin mask, so this product is
        # literally the standard deviation of what reaches the picture. Capped at
        # `TEXTURE_MAX` as a backstop rather than as the control: the measurement
        # is what sets the level, and the constant only catches an estimate that
        # has gone wrong — a busy background inside the ROI, say.
        amount = min(strength * headroom, texture.TEXTURE_MAX)
        result: Frame = blended + (warped * amount)[:, :, None] * mask[:, :, None]

        # Contained within the `paste` bucket rather than added to it — the frame
        # total does not change, this just says how much of paste it was.
        self.last_stage_ms['texture'] = (time.perf_counter() - started) * 1000.0
        return result

    def _texture_headroom(
        self,
        blended: Frame,
        target: Frame,
        mask: Mask,
        extent: float,
    ) -> float:
        """
        How much high-frequency deviation the face can still take, in 8-bit units.

        Independent zero-mean fields add in quadrature, so a face already
        carrying `f` and about to receive grain of `g` can accept `t` before it
        matches a real face carrying `r`:

            f^2 + g^2 + t^2  =  r^2

        Anything past that is a face with more texture than the camera recorded,
        which reads as noise rather than as skin.

        The band is the same one `_match_detail` scales and the extractor cuts,
        expressed here in frame pixels: `DETAIL_SIGMA` is specified at a 256px
        face, so at `extent` pixels it is that fraction of it.

        Args:
            blended: The composited ROI, float32
            target: The original frame over the same ROI, float32
            mask: Compositing alpha over the ROI
            extent: The face's extent in frame pixels

        Returns:
            Deviation still available, in 8-bit units. Zero when the swap has
            already reached or passed the real face's texture level.
        """
        # Sample a bounded window rather than the whole region — see
        # `_HEADROOM_SAMPLE`. Cropping rather than downscaling, because a
        # downscale is itself a low-pass and would measure a different band than
        # the one being added.
        cap = self._HEADROOM_SAMPLE
        height, width = mask.shape[:2]
        if height > cap or width > cap:
            top = max(0, (height - cap) // 2)
            left = max(0, (width - cap) // 2)
            rows = slice(top, top + min(height, cap))
            cols = slice(left, left + min(width, cap))
            mask = mask[rows, cols]
            blended = blended[rows, cols]
            target = target[rows, cols]

        binary = (mask > 0.5).astype(np.uint8)
        if cv2.countNonZero(binary) < 64:
            return 0.0

        sigma = max(0.6, DETAIL_SIGMA * extent / DETAIL_SIGMA_REFERENCE)

        def deviation(image: Frame) -> float:
            """High-band deviation of `image` inside the mask."""
            gray = cv2.cvtColor(
                np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY,
            ).astype(np.float32)
            high = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), sigma))
            _, dev = cv2.meanStdDev(high, mask=binary)
            return float(dev[0][0])

        real = deviation(target)
        fake = deviation(blended)

        # Grain has not been added yet and is about to be, over the same pixels.
        # Leaving it out of the sum would let the two layers each land at the
        # target's level and the pair overshoot it together.
        grain = self._estimate_noise(target) if self.config.grain else 0.0

        available = real * real - fake * fake - grain * grain
        return float(np.sqrt(available)) if available > 0.0 else 0.0

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
