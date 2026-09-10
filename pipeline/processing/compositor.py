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
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face, Mask, Matrix
from pipeline.services.enhancement import Enhancer
from pipeline.services.masking import FaceMasker
from pipeline.services import guards
from pipeline.services import identity
from pipeline.services import shape as shape_metric
from pipeline.services import swapper_models
from pipeline.services.identity import IdentityProbe
from pipeline.services.shape import ShapeProbe
from pipeline.logging import emit_warning
from pipeline.processing import texture
from pipeline.processing.texture import SourceTexture
# `alignment_template` imported by name rather than through the module, because
# `_enhance` binds a local called `geometry` for its FFHQ affines and a module
# of that name would be shadowed inside it.
from pipeline.processing.geometry import (
    ALIGNED_STEPS,
    alignment_template,
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
    # Ceiling on the a/b difference `complexion_keep` may leave uncorrected, in
    # LAB units of chroma distance. This is the bound that makes the feature
    # safe rather than the knob: withholding a fixed *fraction* would leave a
    # step proportional to how different the two people are, which is largest
    # exactly where it is least affordable.
    #
    # 3.0 is a starting point chosen on the design target, not measured. For
    # scale: `_COLOR_FLOOR` is 1.5 and treats anything under it as estimator
    # noise, and a just-noticeable chroma difference across a *hard* edge is
    # around 2.3 — this boundary is feathered and 4:2:0-subsampled, so it
    # tolerates more. Sweep it against footage before trusting it.
    _COMPLEXION_RESIDUAL = 3.0

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

    # A feather wider than this many pixels is blurred at reduced resolution;
    # the reduction is the ratio, capped at 4. Below it the blur is cheap and
    # the radius is too small to survive a resample — and losing a tight feather
    # would restore the hard edge the feather exists to remove.
    _FEATHER_MIN_SIGMA = 4.0
    # Never reduce a region below this, or the mask loses its shape rather than
    # just its resolution.
    _FEATHER_MIN_SIZE = 32

    # Floor on the frame-space feather, in pixels. A face small enough that
    # 4% of it is under two pixels has no transition to speak of otherwise.
    _FEATHER_FLOOR = 2.0

    # Subsurface scattering. Sigma is specified at a 256px face and scales with
    # the working resolution, like every other spatial constant here, so "soft"
    # means the same physical distance whether the operator is close to the
    # camera or sitting back — otherwise the look changes when they lean.
    #
    # 3.0 at 256 is roughly 2mm on a real face, which is the order of skin's
    # actual diffusion length. Deliberately above `_DETAIL_SIGMA` (1.5): this is
    # a shading effect and that is a texture one, and a pass that reached into
    # the texture band would be undoing the stage that comes after it.
    _SCATTER_SIGMA = 3.0
    _SCATTER_REFERENCE = 256.0
    # Feature exclusion radii, as fractions of the inter-ocular distance in
    # aligned space. Derived from `face.kps` rather than the 106 landmarks, so
    # this does not depend on a layout that varies between model packs.
    _SCATTER_EYE = 0.42
    _SCATTER_MOUTH = 0.40
    _SCATTER_NOSE = 0.30

    # The feature-exclusion weight is built at 1/N and scaled back up, with a
    # floor so a small crop does not lose the features entirely. It is a smooth
    # field and the blur is most of its cost, so building it at sixteen times
    # the pixels it needs was waste: 0.43ms against 0.08ms at 256, 1.10 against
    # 0.13 at 320. The two agree to within 0.17 on a [0, 1] weight.
    _WEIGHT_DOWNSCALE = 4
    _WEIGHT_MIN = 48

    # Pose agreement between the source photograph and the frame, in degrees of
    # yaw. Below the first the map is used at full strength; above the second it
    # is not used at all.
    #
    # This is the term that answers a real limitation rather than a hypothetical
    # one. Canonical space is a *similarity* transform, so it does not correct
    # pose at any strength — a source shot at an angle yields a map whose pores
    # are foreshortened on one cheek, and warping it onto a target at a
    # different angle stretches that error rather than removing it. The only
    # honest responses are to attenuate, which is this, or to fit in 3D, which
    # is docs/TEXTURE_PIPELINE.md phase D.
    #
    # It is also the one lever against texture swimming that is not "turn the
    # strength down": swimming is worst exactly where the pose has moved
    # furthest from the source, so a term that falls off with pose distance
    # takes the detail away at the moment it would start to crawl.
    _POSE_FULL = 12.0
    _POSE_LIMIT = 45.0

    # Edge of the window every distribution statistic is taken over. A standard
    # deviation and a median both converge on a few thousand pixels, so
    # measuring every pixel of a large face buys nothing and costs everything:
    # the headroom measurement was 16ms at a 500px region, more than the whole
    # rest of the frame, and was paid even when the answer was "no headroom, add
    # nothing"; the noise estimate behind grain was another 4.6ms there. The
    # region is centred on the face, so a centred window is face pixels.
    _STAT_WINDOW = 160

    # Headroom below which the texture layer does not bother. Adding a fraction
    # of an 8-bit unit costs a warp and changes nothing anyone can see.
    _TEXTURE_FLOOR = 0.25

    # Ceiling on the correction for what the warp's bilinear resampling took
    # out of the texture map. Measured retention was 0.431, needing a gain of
    # 2.3, so 4.0 leaves room for a harder warp — a small face, a turned head —
    # without letting a degenerate one through. Past this the map has been
    # destroyed rather than attenuated, and multiplying up what survived would
    # amplify interpolation artefacts instead of restoring pores. The shortfall
    # is then visible in `texture_delivered` against `texture_headroom` rather
    # than silently corrected.
    _WARP_GAIN_MAX = 4.0

    # Ceiling on the share of the target's high band `_match_detail` will hold
    # back for the texture layer. See `_texture_reserve` — this is the fix for
    # the two stages competing over one budget, where the one that ran first
    # took all of it.
    #
    # 0.8 leaves the swap 60% of the target's band amplitude. Past that the
    # swap's own high frequencies are being gutted in favour of a reprojected
    # map — and that map is fixed content warped per frame, so it cannot follow
    # an expression the way the swap's own band does. A face whose surface
    # detail is almost entirely reprojected is a different failure from a
    # plastic one, not an improvement on it.
    #
    # Note this is no longer set by `_DETAIL_RATIO`'s floor: that clamp now
    # scales with the reservation, since it bounds deviation from the target and
    # reserving moves the target. It was, and the floor quietly kept a quarter
    # of the room the reservation had promised.
    _RESERVE_MAX = 0.8

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

        # Unit-variance noise, reused across frames at a random offset rather
        # than regenerated. `np.random.normal` at region size cost 1.5ms a frame
        # at 256 and scales with area - around 5.9ms on a face filling a 500px
        # region - which is a large slice of the compositor for a field whose
        # only requirement is to look like sensor noise.
        #
        # A cache, not temporal state: it survives `reset()` deliberately, and
        # the per-frame offset is what stops consecutive frames sharing a
        # pattern. Fixed-pattern noise is a worse artefact than none.
        self._noise: Optional[Frame] = None

        # Skin detail lifted from the operator's source photograph, set by the
        # pipeline when the source changes. Deliberately not built here: it is a
        # property of the source images, which the compositor never sees.
        #
        # Survives `reset()`. Every other field here is temporal state that a
        # face loss or a resolution change invalidates; this one is a function
        # of the identity, and dropping it on a lost face would mean rebuilding
        # it — a file read and a full-resolution warp — on the live path.
        self.source_texture: Optional[SourceTexture] = None

        # Said once, when the operator has asked for texture and there is none
        # to apply. Every other way this layer declines is visible in the
        # readings — `texture_headroom` and `texture_confidence` are recorded
        # whenever it gets past its guards, so their absence from a REALISM
        # block already means "it did not run". This case is the one that looks
        # identical to a working layer set too low: the slider says 0.4, the
        # frame is unchanged, and nothing anywhere disagrees.
        self._warned_no_texture = False

        # Said once, when a reserve was made and the headroom to fill it was not
        # there — the state that leaves a face softer than with the layer off.
        self._warned_no_headroom = False

        # High-band deviation the texture layer was allowed on the last frame,
        # in 8-bit units, or None when it did not run. Same pattern as
        # `masker.last_coverage`: the stage that measures a thing owns the
        # number. Worth reading while judging -- a headroom that is routinely
        # zero means detail matching has already taken the face to the target's
        # texture level and the texture layer has nothing left to add.
        self.last_texture_headroom: Optional[float] = None

        # And what the layer *actually* put on the face, in the same 8-bit
        # units, so the two can be read against each other. `headroom` is the
        # budget and this is the spend; they are only equal if the map still
        # carries unit deviation after being warped into frame space and
        # multiplied by the compositing alpha, which was assumed and never
        # measured. A spend well under the budget is the state that leaves the
        # face softer than with texture switched off, because `_match_detail`
        # has already stood down by `detail_reserve` to make the room.
        self.last_texture_delivered: Optional[float] = None

        # The share of the compositing alpha the map's support actually covers.
        # Read with `last_texture_delivered`: a field that is unit-deviation on
        # a fraction `c` of the region it is measured over reads `sqrt(c)`, so
        # this is what separates "the map is weak" from "the map covers less of
        # the face than the reserve assumed".
        self.last_texture_coverage: Optional[float] = None

        # The correction `_match_detail` *wanted* on the last frame, before its
        # clamp, or None when the stage did not run. This is the reading that
        # decides how much of the texture work was necessary: if the clamp is
        # binding, part of the face/frame detail gap is the stage not being
        # allowed to correct far enough, and raising a constant is a cheaper
        # lever than any of it. Percentiles of the *clamped* value cannot answer
        # that, because they cannot exceed the clamp.
        self.last_detail_ratio: Optional[float] = None

        # The share of the target's high band that was held back for the texture
        # layer on the last frame. Published because the two stages now share
        # one budget and a reader has to be able to see the split: a reserve
        # that is routinely zero while `texture_strength` is set means the layer
        # is declining somewhere — no map at this working size, or a pose too
        # far from the source photograph — and the visible symptom of that is
        # identical to a strength set too low.
        self.last_detail_reserve: Optional[float] = None

        # Pose agreement on the last frame, in [0, 1], or None when it could not
        # be measured. Published for the same reason as the others: a texture
        # layer that quietly does nothing because every frame is off-pose looks
        # identical to one whose strength is set too low.
        self.last_texture_confidence: Optional[float] = None

        # How many LAB units of chroma difference the colour match was allowed
        # to leave on the face, or None when the stage did not run. Published
        # for the reason `last_texture_headroom` is: a keep that is routinely
        # zero means the two complexions already agreed and the knob had nothing
        # to spend, which looks identical to a knob set too low.
        self.last_complexion_kept: Optional[float] = None

        # Per-stage milliseconds for the frame just composited, read by the
        # pipeline's latency budget. Same pattern as `masker.last_coverage`:
        # the stage that measures a thing owns the number, and whoever needs it
        # reads it afterwards rather than having a timer threaded through.
        #
        # Cheap enough to leave on — a `perf_counter` either side of calls that
        # already cost milliseconds — and the alternative is a debug-only path
        # that is never on when the question comes up.
        self.last_stage_ms: Dict[str, float] = {}

        # Identity, and the two things needed to measure it. Both are set by the
        # pipeline: the probe when services are built, the embedding when the
        # source changes. Absent either, every `id_*` reading is simply missing
        # from the report, which is the honest rendering of "not measured".
        self.identity: Optional[IdentityProbe] = None
        self.source_identity: Optional[Any] = None

        # Shape, and the two things needed to measure it. Set by the pipeline
        # exactly as the identity pair is. The source landmarks are the *one
        # best* photograph's rather than an average of all of them, for the
        # reason the texture layer picks one: identity is distributed and
        # averages soundly, geometry is not � averaging landmarks taken at
        # different angles produces a face nobody has.
        self.shape: Optional[ShapeProbe] = None
        self.source_shape: Optional[Any] = None

        # Cosine similarities from the last measured frame, by stage name. Same
        # pattern as `last_detail_ratio`: the stage that measures a thing owns
        # the number and whoever needs it reads it afterwards.
        self.last_identity: Dict[str, float] = {}

        # Shape readings from the last measured frame. Same ownership rule.
        self.last_shape: Dict[str, float] = {}

        # Frames since the last identity measurement. Measuring is several
        # ArcFace inferences, so it runs every Nth frame rather than on all of
        # them — a distribution over a run is what the reading is for, and it
        # does not need every sample to have one.
        self._identity_tick = 0

    def _template(self) -> Any:
        """
        The five-point template the current swap model aligns to.

        Read per frame rather than cached, because `set_realism` can switch the
        swap model mid-stream and a stale template would silently mis-frame
        every recognition crop and put the shape ramp's eye line 6% out.

        Returns:
            Normalised 5x2 template
        """
        return alignment_template(
            swapper_models.resolve(self.config.swapper_model).template)

    def _identity_due(self) -> bool:
        """
        Whether this frame should be measured for identity.

        Returns:
            True on every Nth frame while `identity_probe` is set and a probe
            and a source embedding are both available
        """
        interval = int(getattr(self.config, 'identity_probe', 0) or 0)
        if interval <= 0:
            return False
        if self.identity is None or self.source_identity is None:
            return False

        self._identity_tick += 1
        if self._identity_tick < interval:
            return False

        self._identity_tick = 0
        return True

    def _measure_identity(self, name: str, crop: Frame, template: Any) -> None:
        """
        Record how much of the source's identity an aligned crop carries.

        Args:
            name: Reading name — `id_swap`, `id_restore`, `id_final`
            crop: The aligned crop as it stands at this stage
            template: The template that crop is framed by
        """
        if self.identity is None:
            return

        embedding = self.identity.embed_aligned(crop, template)
        score = identity.cosine(self.source_identity, embedding)
        if score is not None:
            self.last_identity[name] = score

    def _measure_output(self, pasted: Optional[Frame], face: Face) -> None:
        """
        Record identity on the finished frame, against source *and* target.

        Both, because they answer different failures and one of them is
        invisible on its own. `id_out` falling says the source is not coming
        through. `id_target` rising says the *target* is — that the output is
        drifting back toward the person who was already there, which is what a
        swap failing to take looks like and what a mask that hands most of the
        face back to the frame produces. A run where `id_out` is mediocre and
        `id_target` is low is a weak swap; one where both are middling is a
        swap being diluted after the fact.

        Args:
            pasted: The finished frame
            face: The target detection, for its keypoints and identity
        """
        if self.identity is None or pasted is None:
            return

        kps = getattr(face, 'kps', None)
        embedding = self.identity.embed_frame(pasted, kps)
        if embedding is None:
            return

        score = identity.cosine(self.source_identity, embedding)
        if score is not None:
            self.last_identity['id_out'] = score

        against = identity.cosine(
            getattr(face, 'normed_embedding', None), embedding)
        if against is not None:
            self.last_identity['id_target'] = against

    def _measure_shape(self, pasted: Optional[Frame], face: Face) -> None:
        """
        Record whether the output took the source's head shape or the target's.

        The one thing `_measure_output` cannot see. ArcFace is trained to be
        invariant to a great deal of geometry, so a swap can move the jawline
        visibly and shift the cosine by almost nothing — and the head's outline
        is among the strongest identity cues a viewer actually reads.

        The target's landmarks come from the detection rather than from the
        frame, which is both free and safer: they were computed before anything
        was composited, so there is no question of reading them back off a frame
        that has since been pasted into.

        Args:
            pasted: The finished frame
            face: The target detection, for its landmarks and box
        """
        if self.shape is None or pasted is None or self.source_shape is None:
            return

        target = getattr(face, 'landmark_2d_106', None)
        if target is None:
            return

        output = self.shape.landmarks(pasted, getattr(face, 'bbox', None))
        if output is None:
            return

        reading = shape_metric.compare(self.source_shape, target, output)
        if reading is not None:
            # `update`, not assignment: the per-stage readings below are
            # recorded earlier in the frame and assigning here would wipe them,
            # leaving the attribution silently empty.
            self.last_shape.update(reading.as_readings())

    def _measure_shape_stage(
        self,
        name: str,
        crop: Optional[Frame],
        face: Face,
        matrix: Matrix,
    ) -> None:
        """
        Record the outline shift of an aligned crop, mid-chain.

        This is what makes the shape reading *actionable* rather than merely
        true. `outline_shift` on the finished frame says whether the source's
        head shape survived; it cannot say where it was lost, and the two causes
        have opposite fixes — a contour the generator moved and the mask then
        clipped calls for `mask_shape_growth`, while a contour the generator
        never moved calls for a different model and leaves the mask innocent.

        Only the outline is recorded, not the whole-face figure. At an
        intermediate stage the whole-face residual is dominated by the interior
        features that every stage repaints, and no lever is attached to it; the
        silhouette is the channel with `mask_shape_growth` behind it.

        Args:
            name: Reading name — `outline_swap`, `outline_final`
            crop: The aligned crop as it stands at this stage
            face: The target detection, for its landmarks and box
            matrix: 2x3 affine, frame space -> aligned space
        """
        if self.shape is None or crop is None or self.source_shape is None:
            return

        target = getattr(face, 'landmark_2d_106', None)
        if target is None:
            return

        points = self.shape.landmarks_aligned(
            crop, getattr(face, 'bbox', None), matrix)
        if points is None:
            return

        reading = shape_metric.compare(self.source_shape, target, points)
        if reading is None:
            return
        if reading.outline_shift is not None:
            self.last_shape[name] = reading.outline_shift
        if reading.interior_shift is not None:
            self.last_shape[name.replace('outline', 'interior')] = (
                reading.interior_shift)

    def clear_readings(self) -> None:
        """
        Drop what the last frame measured, without touching temporal state.

        Called at the top of every stream frame, so a frame that never reaches
        compositing leaves *nothing* behind rather than the previous frame's
        numbers. Guarded frames are the case: `check_frame` refuses, the
        compositor is never called, and the per-frame readings would otherwise
        still be sitting there to be recorded a second time — inflating every
        distribution by however often the guards fired, which on a real call is
        often.

        Deliberately separate from `reset()`. That one drops the smoothing
        buffers and must not be called per frame; this one only drops
        measurements, and must be.
        """
        self.last_stage_ms.clear()
        self.last_identity.clear()
        self.last_shape.clear()
        self.last_detail_ratio = None
        self.last_detail_reserve = None
        self.last_texture_headroom = None
        self.last_texture_confidence = None
        self.last_texture_delivered = None
        self.last_texture_coverage = None
        self.last_complexion_kept = None

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

        # Hoisted above the mask, which used to build first. The shape term
        # needs the generated crop to find the outline of the face that was
        # actually produced, and this resize depends on nothing but `swapped`
        # and `size`. Nothing else moved: the mask is still built before
        # restoration and smoothing, so the occlusion guard can still refuse the
        # frame before anything mutates temporal state.
        fake = cv2.resize(swapped, (size, size), interpolation=cv2.INTER_CUBIC)

        template = self._template()
        mask = self.masker.build(
            face, aligned_matrix, real, (frame.shape[0], frame.shape[1]),
            swapped=fake, template=template,
        )
        elapsed('mask')

        if not guards.coverage_ok(self.config, self.masker.last_coverage):
            return None

        measure = self._identity_due()
        if measure:
            self._measure_identity('id_swap', fake, template)
            # The contour the generator actually produced. It exists nowhere
            # else — the frame holds the target's face — so if this is not read
            # here, a contour the mask later clips off leaves no evidence that
            # it was ever generated.
            self._measure_shape_stage(
                'outline_swap', fake, face, aligned_matrix)

        if self.config.enhance and self._restore_worthwhile(face):
            fake = self._enhance(fake, frame, face, aligned_matrix)
            elapsed('restore')
            if measure:
                self._measure_identity('id_restore', fake, template)

        # Temporal smoothing needs a stable subject identity. With multiple
        # faces the per-frame detection order is not stable, so smoothing
        # would blend between different people.
        if not self.config.many_faces:
            fake = self._smooth(fake, real)
            elapsed('smooth')

        # Before colour matching, and that ordering is the point. This is a
        # low-frequency change to the luminance channel, so it has to happen
        # where the colour stages can still reconcile it against the target
        # rather than landing on top of a finished match — see
        # docs/TEXTURE_PIPELINE.md section 3.2.
        scatter_on = float(getattr(self.config, 'diffuse_strength', 0.0)) > 0.0
        if scatter_on or self.config.color_correction:
            # One conversion for both stages rather than one each. A round trip
            # is 1.9ms at 256, which was more than everything `_scatter` does
            # with it, and paying it twice bought only a tidier signature.
            fake_lab = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB).astype(np.float32)

            if scatter_on:
                fake_lab = self._scatter(fake_lab, mask, face, aligned_matrix)
            elapsed('scatter')

            if self.config.color_correction:
                fake_lab = self._match_color(fake_lab, real, mask)
            elapsed('colour')

            fake = cv2.cvtColor(
                np.clip(fake_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR,
            )

        # The face's extent in frame, from the affine's own determinant. Note
        # this is *not* `scale` above: that one is the ratio between the
        # swapper's crop and the working resolution, and passing it here read
        # 128 for a 400px face — which set the seam feather to a third of what
        # it should be and had the texture layer build its map at 128 for a face
        # four times that, which is the decimation it exists to avoid.
        #
        # Computed here rather than after detail matching because the reserve
        # below needs it: the texture layer's working size follows the face, and
        # whether a map exists at that size decides whether reserving is safe.
        geometric = float(np.sqrt(abs(float(np.linalg.det(aligned_matrix[:, :2])))))
        extent = (size / geometric) if geometric > 1e-6 else float(size)

        fake = self._match_detail(
            fake, real, mask,
            reserve=self._texture_reserve(face, extent),
            band=self._texture_shaping()[2],
            # Only used while reserving, to measure on the same skin the
            # frame-space stage will measure. Passed rather than derived so
            # this stage stays callable without a detection.
            face=face,
            matrix=aligned_matrix,
        )
        elapsed('detail')

        if measure:
            self._measure_identity('id_final', fake, template)
            # Still aligned space, so the difference from `outline_swap` is
            # restoration and the aligned-space stages — and the difference
            # from `outline_shift` is the mask and the paste. Restoration is
            # the one nobody would suspect of moving a contour: it regresses a
            # face toward its training manifold at `enhance_strength`, which is
            # a plausible way to lose a widened jaw.
            self._measure_shape_stage(
                'outline_final', fake, face, aligned_matrix)

        pasted = self._paste(frame, fake, mask, aligned_matrix, face, extent)
        elapsed('paste')

        if measure:
            # The one that counts. Everything before it is measured in aligned
            # space at the compositor's working resolution; this is the face at
            # the size it leaves the pipeline, through the mask, in the frame —
            # which is where the silhouette, the paste and the frame-space
            # feather finally get a vote.
            self._measure_output(pasted, face)
            # And the axis the cosine above is blind to, measured in the same
            # place and for the same reason: the mask is the last stage that
            # can take the source's outline away, so shape has to be read after
            # it rather than on the aligned crop.
            self._measure_shape(pasted, face)

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

    def _scatter(
        self,
        fake_lab: Frame,
        mask: Mask,
        face: Face,
        matrix: Matrix,
    ) -> Frame:
        """
        Soften the shading the way light under skin does.

        Real skin is translucent. Light enters, scatters through a millimetre or
        two of tissue and leaves somewhere slightly else, which blurs the
        *shading* while leaving the texture sitting on top of it sharp. A
        generated face has none of that, and the result reads hard — a different
        complaint from plastic, in a different band, and `texture_strength`
        does not answer it.

        **Luminance only.** The pass runs on LAB's L channel and leaves a and b
        untouched. It takes and returns LAB rather than BGR because
        `_match_color` needs the same space immediately afterwards, and a second
        round trip cost more than everything this does put together. Applied to full RGB it would drift skin tone as well as
        shading, and applied without the feature exclusions below it would
        soften eyes and mouth — which is precisely the identity-softening
        failure this whole pipeline exists to route around in full-strength
        restoration. Scoping it is not a refinement; it is the difference
        between the pass and the thing it is imitating.

        **Before `_match_detail`, deliberately.** A blur at `_SCATTER_SIGMA`
        does attenuate some of the texture band on its way past, and detail
        matching runs afterwards and scales that band back up against the real
        crop. So the ordering is self-correcting: what scatter takes out of the
        texture band, the next stage puts back, and what it takes out of the
        shading band stays out — which is the split that was wanted.

        Args:
            fake_lab: The swap in aligned space, LAB float32
            mask: Compositing mask, so nothing outside the swap is touched
            face: Detection, for the five keypoints the exclusions are built on
            matrix: frame -> aligned affine

        Returns:
            `fake_lab` with softened shading, or unchanged when the layer
            is off or has nothing to act on.
        """
        strength = float(np.clip(getattr(self.config, 'diffuse_strength', 0.0), 0.0, 1.0))
        if strength <= 0.0:
            return fake_lab

        size = fake_lab.shape[0]
        weight = self._scatter_weight(mask, face, matrix, size)
        if weight is None or not weight.any():
            return fake_lab

        luma = fake_lab[:, :, 0]

        sigma = self._SCATTER_SIGMA * size / self._SCATTER_REFERENCE
        soft = cv2.GaussianBlur(luma, (0, 0), max(sigma, 0.6))

        # luma + (soft - luma) * w, which moves the shading toward its blurred
        # self by `w` and leaves it alone where `w` is zero.
        fake_lab[:, :, 0] = luma + (soft - luma) * (weight * strength)
        return fake_lab

    def _scatter_weight(
        self,
        mask: Mask,
        face: Face,
        matrix: Matrix,
        size: int,
    ) -> Optional[Mask]:
        """
        Where scattering may apply: inside the swap, on skin, off the features.

        Built from the five keypoints rather than the 106 landmarks, because the
        five are the same points the swap itself is aligned on and their meaning
        does not change with the model pack. Radii scale with the inter-ocular
        distance, so the exclusions track the face rather than the crop.

        Args:
            mask: Compositing mask in aligned space
            face: Detection for this frame
            matrix: frame -> aligned affine
            size: Aligned edge length

        Returns:
            Weight in [0, 1], or None when the keypoints are unusable — in which
            case the caller declines to run rather than softening blind.
        """
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 5:
            return None

        points = cv2.transform(
            np.asarray(kps, dtype=np.float32).reshape(-1, 1, 2),
            matrix.astype(np.float32),
        ).reshape(-1, 2)

        span = float(np.linalg.norm(points[1] - points[0]))
        if span < 1e-3:
            return None

        small = max(self._WEIGHT_MIN, size // self._WEIGHT_DOWNSCALE)
        ratio = small / float(size)
        weight = cv2.resize(mask, (small, small), interpolation=cv2.INTER_AREA)

        for index, scale in (
            (0, self._SCATTER_EYE), (1, self._SCATTER_EYE),
            (2, self._SCATTER_NOSE),
            (3, self._SCATTER_MOUTH), (4, self._SCATTER_MOUTH),
        ):
            centre = points[index] * ratio
            cv2.circle(
                weight,
                (int(round(centre[0])), int(round(centre[1]))),
                max(1, int(round(span * scale * ratio))),
                (0.0,),
                -1,
            )

        # Feather the exclusions, or each one is a visible disc of "sharp"
        # sitting in softened skin — a seam of its own, at the eyes.
        weight = cv2.GaussianBlur(weight, (0, 0), max(1.0, small * 0.02))
        resized: Mask = cv2.resize(
            weight, (size, size), interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def _match_color(self, fake_lab: Frame, real: Frame, mask: Mask) -> Frame:
        """
        Match the swap's colour distribution to the target's, in LAB.

        Takes and returns LAB float32; the conversion belongs to the caller,
        because the scatter pass needs the same space immediately before this
        one and a round trip between them cost more than either stage's work.

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
            return fake_lab

        color_strength = float(np.clip(self.config.color_strength, 0.0, 1.0))
        if color_strength <= 0.0:
            return fake_lab

        real_lab = cv2.cvtColor(real, cv2.COLOR_BGR2LAB)

        fake_mean, fake_std = cv2.meanStdDev(fake_lab, mask=binary)
        real_mean, real_std = cv2.meanStdDev(real_lab, mask=binary)

        # Ramp rather than a threshold, so correction never snaps on and off
        # between frames.
        delta = float(np.linalg.norm(real_mean - fake_mean))
        global_strength = color_strength * float(np.clip(
            (delta - self._COLOR_FLOOR) / self._COLOR_RANGE, 0.0, 1.0,
        ))

        chroma_scale = self._complexion_scale(fake_mean, real_mean)

        result = fake_lab

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

                # L is corrected in full and a/b may be held back. `chroma_scale`
                # is exactly 1.0 unless `complexion_keep` is set, and `x * 1.0`
                # is bit-exact, so the default path is unchanged.
                strength = (global_strength if channel == 0
                            else global_strength * chroma_scale)

                gain[channel] = 1.0 - strength + strength * ratio
                offset[channel] = strength * (r_mean - f_mean * ratio)

            result *= gain
            result += offset

        return self._match_illumination(
            result, real_lab.astype(np.float32), mask, color_strength,
            chroma_scale,
        )

    def _complexion_scale(self, fake_mean: Frame, real_mean: Frame) -> float:
        """
        How much of the a/b correction to apply, so some source skin tone stays.

        Complexion is an identity cue and `_match_color` spends it: the global
        mean transfer moves the swapped face onto the *target's* skin tone,
        which is most of what "it doesn't quite look like me" is in colour
        terms. It is not a bug — a face whose tone does not match the neck it
        sits on is failure mode 2 — but the correction is applied to luminance
        and chroma alike, and those are not equally costly to relax.

        Luminance is corrected in full and always will be: a brightness step at
        the jaw is the most visible seam there is. Chroma is where the pigment
        lives, and it is the cheaper channel to leave slightly uncorrected here
        for a reason specific to this pipeline — every frame is JPEG-encoded at
        OpenCV's default 4:2:0, so chroma is already subsampled 2x in both axes
        before it reaches the call, and a chroma step at the seam is blurred by
        the transport in a way a luminance step is not.

        **Bounded by the difference, not by the knob**, which is the same shape
        as `_texture_headroom`. What is withheld is capped at
        `_COMPLEXION_RESIDUAL` LAB units of a/b distance, so:

            tones already close   -> keep the whole fraction, costs nothing
            tones far apart       -> the cap binds and this gives way

        The second case is the one that matters. CLAUDE.md records the hard
        footage: a fair, well-lit source against a dark, under-lit target.
        Keeping the source's complexion *there* is a light face on a dark neck,
        which is not "matching the source" — it is a broken composite. So this
        withdraws exactly as the gap grows.

        Note it withholds correction rather than importing the source
        photograph's colour. The swap already carries some of the source's
        complexion; steering toward the donor image's measured tone would carry
        that photograph's white balance with it, which is a different and worse
        thing to be right about.

        Args:
            fake_mean: Per-channel LAB mean of the swap, inside the mask
            real_mean: Per-channel LAB mean of the target, inside the mask

        Returns:
            Multiplier for the a/b correction, in [0, 1]. Exactly 1.0 — full
            correction, today's behaviour — when the feature is off
        """
        self.last_complexion_kept = None

        keep = float(np.clip(
            getattr(self.config, 'complexion_keep', 0.0) or 0.0, 0.0, 1.0))
        if keep <= 0.0:
            return 1.0

        gap = float(np.hypot(
            float(real_mean[1][0]) - float(fake_mean[1][0]),
            float(real_mean[2][0]) - float(fake_mean[2][0]),
        ))
        if gap < 1e-3:
            # Nothing to keep: the tones already agree, so withholding
            # correction and applying it are the same operation.
            self.last_complexion_kept = 0.0
            return 1.0

        withheld = min(gap * keep, self._COMPLEXION_RESIDUAL)
        self.last_complexion_kept = withheld
        return float(np.clip(1.0 - withheld / gap, 0.0, 1.0))

    def _match_illumination(
        self,
        fake_lab: Frame,
        real_lab: Frame,
        mask: Mask,
        strength: float,
        chroma_scale: float = 1.0,
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
            chroma_scale: The same a/b factor `_complexion_scale` handed the
                global pass. **This is not optional bookkeeping.** A held-back
                chroma mean is a low-frequency difference, which is exactly what
                this stage is built to find and correct — so without it, ~70% of
                whatever the global pass withheld would be silently put back
                here and `complexion_keep` would look like a knob that does
                almost nothing

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

        # Per channel, and applied as a second multiply rather than folded into
        # the line above so the default is bit-exact: `chroma_scale` is 1.0
        # unless `complexion_keep` is set, and `x * 1.0` is exact for every
        # finite float.
        if chroma_scale != 1.0:
            scaled *= np.array(
                [1.0, chroma_scale, chroma_scale], dtype=np.float32)

        correction = cv2.resize(
            scaled, (size, size), interpolation=cv2.INTER_LINEAR,
        )

        corrected: Frame = cv2.add(fake_lab, correction)
        return corrected

    def _match_detail(
        self,
        fake: Frame,
        real: Frame,
        mask: Mask,
        reserve: float = 0.0,
        band: float = 1.0,
        face: Optional[Face] = None,
        matrix: Optional[Matrix] = None,
    ) -> Frame:
        """
        Scale the swap's high-frequency band to match the target's.

        Solving for a blur radius is unstable and only corrects in one
        direction. Scaling the high band handles both cases, which is
        necessary here: the swap is softer than the frame before enhancement
        and sharper after it.

        **`reserve` is the fix for two stages competing over one budget.** This
        stage and `_add_texture` both aim at the same quantity — the target
        face's high-frequency energy — and this one runs first, in aligned
        space, while the other runs in frame space inside `_paste`. So this one
        took all of it: the clamp was measured binding on 0% of 2267 frames,
        meaning parity was not merely attempted but reached, every frame, and
        the headroom the texture layer then measured was whatever the warp down
        to frame space happened to lose. That was p50 **0.78** of an 8-bit unit,
        against a real face carrying several, which is why the layer was
        invisible at every strength and why 0.3 and 0.5 read identically on
        every metric.

        Worse than the amount is what the amount was spent on. This stage can
        only amplify the band the swap already has, and that band is upsampled
        128-native output with no structure in it. The face arrived at the
        correct energy and the wrong content: textured by measurement, smooth to
        look at.

        So a share is held back. Reserving `r` aims at `sqrt(1 - r^2)` of the
        target's energy, leaving `r` for real skin — quadrature, because the two
        fields are independent, which is the same arithmetic
        `_texture_headroom` uses to decide it may spend it. Total band energy
        still lands at parity and overshoot is still impossible; what changes is
        the *composition*, from amplified mush to detail a camera recorded.

        **`band` comes with it, and has to.** Reserving is only coherent if both
        stages mean the same band by "texture". Left at the narrowest octave
        while the texture layer spans up to the subsurface diffusion length,
        the reservation has almost no leverage on what the layer actually
        measures: attenuating one octave of a field measured over three barely
        moves the total, and the headroom stays where it was. Measured on a
        synthetic face in exactly the starved regime, reserving 0.4 over the
        narrow band moved the headroom 0.30 -> 0.31, which is nothing.

        So when a reserve is requested, this stage scales the same span the map
        will fill. That widens what it touches — mid-frequency content sits
        closer to facial structure than pore noise does — and the trade is
        deliberate: it is scaled *down*, and real recorded detail is put back in
        its place. With `reserve` at 0.0 the split is the original
        `DETAIL_SIGMA` and the stage is bit-identical to what it was.

        The reservation is still approximate — the two stages work in different
        spaces and the warp between them loses some — and it does not need to be
        exact. The frame-space measurement remains the authority on what is
        actually added. This only stops the authority from being handed an
        already-empty budget.

        Args:
            fake: The swapped crop in aligned space
            real: The target's own crop over the same geometry
            mask: Where to measure, and where the result applies
            reserve: Share of the target's band amplitude to leave unfilled,
                for the texture layer to fill with real detail. 0.0 restores
                the original behaviour exactly.
            band: Span of that band, as a multiple of `DETAIL_SIGMA`. Read only
                when reserving, so a run without the texture layer splits where
                it always did.
        """
        self.last_detail_ratio = None
        self.last_detail_reserve = None

        binary = (mask > 0.5).astype(np.uint8)
        if cv2.countNonZero(binary) < 64:
            return fake

        fake_f = fake.astype(np.float32)
        real_f = real.astype(np.float32)

        # Scale the band split with the working resolution, so "texture" means
        # the same physical detail at every quality preset. The `band` widening
        # applies only while reserving — see the docstring; without it the two
        # stages would be reserving and spending across different spans.
        reserve = float(np.clip(reserve, 0.0, self._RESERVE_MAX))
        span = float(band) if reserve > 0.0 else 1.0
        sigma = (
            self._DETAIL_SIGMA * span * fake.shape[0]
            / self._DETAIL_SIGMA_REFERENCE
        )

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

        # **The target's band is skin *and* sensor noise, and grain supplies the
        # noise separately.** Aiming at the total therefore spends that budget
        # twice — once here, amplifying the swap's own structureless band until
        # it carries as much energy as skin plus noise, and again in `_add_grain`
        # a moment later — after which `_texture_headroom` correctly reports
        # there is nothing left to add. Measured on a freckled source at aligned
        # 256: band 5.50, noise 2.32, so 42% of the amplitude was double
        # counted, and `real^2 - fake^2` stayed under `grain^2` for every
        # reserve below 0.5. The headroom came out at **exactly zero** across
        # the whole 0.3-0.5 range this layer documents as its working range: the
        # reservation was made, this stage undershot to honour it, and nothing
        # filled the gap. Softer than no texture layer at all, which is the one
        # outcome `_texture_reserve` exists to prevent.
        #
        # Discounted only while reserving, like the band above. With no reserve
        # nothing downstream is waiting for the room, so the stage stays
        # bit-identical to what every existing measurement was taken against —
        # and the pre-existing overshoot (matched to skin+noise, then given
        # grain on top) stays where it is rather than being changed silently in
        # the same edit. It is worth revisiting on its own.
        #
        # The estimate is taken in aligned space while grain lands in frame
        # space, which is approximate — but the reservation is approximate by
        # construction, and `_texture_headroom` re-measures in frame space and
        # remains the authority on what is actually spent. This only stops that
        # authority being handed an already-empty budget.
        discount = 1.0
        if reserve > 0.0:
            # **Measured exactly as `_texture_headroom` will measure it**, and
            # that correspondence is the whole point rather than a tidiness.
            # The two stages were reading the same face through three different
            # instruments: pooled per-channel deviation against grayscale, the
            # full compositing mask against the skin-weighted one with the
            # features cut out, and a broadband noise sigma subtracted from
            # both. Each difference is small — together they left this stage
            # overshooting its own aim by ~2.5%, which is nothing at a large
            # reserve and is the *entire* reservation at a small one. Traced on
            # a 275px face: reserve 0.2 asked for 1.06 units of room and the
            # measured headroom came back 0.00, because fake landed at 5.34
            # where the reservation wanted 5.21.
            #
            # Reading the same pixels the same way makes the arithmetic close:
            # headroom is then `reserve * skin_only`, which is positive for
            # every reserve above zero, so the knob is linear instead of having
            # a dead zone whose edge nobody can predict.
            weight = (
                self._scatter_weight(mask, face, matrix, fake.shape[0])
                if face is not None and matrix is not None else None
            )
            measured = (
                (weight > 0.5).astype(np.uint8) if weight is not None else binary
            )
            if cv2.countNonZero(measured) < 64:
                measured = binary

            # The same centred window too, for the same reason: skin at the
            # nose and inner cheek is not the skin at the jaw and forehead, and
            # measuring the whole crop here against a 160px centre there is the
            # last of the three ways these stages were reading different faces.
            srows, scols = self._stat_window(mask.shape)
            measured = measured[srows, scols]
            if cv2.countNonZero(measured) < 64:
                measured = binary[srows, scols]

            def band_dev(image: Frame) -> float:
                """Grayscale high-band deviation over the skin, as headroom takes it."""
                gray = cv2.cvtColor(
                    np.clip(image[srows, scols], 0, 255).astype(np.uint8),
                    cv2.COLOR_BGR2GRAY,
                ).astype(np.float32)
                high = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), sigma))
                return float(cv2.meanStdDev(high, mask=measured)[1][0][0])

            real_energy = band_dev(real)
            fake_energy = band_dev(fake)
            if real_energy < 1e-3 or fake_energy < 1e-3:
                return fake

            # Grain is about to supply the sensor noise separately, so the swap
            # must not also be amplified to cover it — see the note above.
            if self.config.grain:
                noise = self._estimate_noise(real)
                skin_only = float(np.sqrt(
                    max(0.0, real_energy * real_energy - noise * noise),
                ))
                if skin_only < 1e-3:
                    return fake
                discount = skin_only / real_energy
                real_energy = skin_only

        # Quadrature: leaving `reserve` of the amplitude for another field means
        # aiming this one at the rest, not at all of it.
        self.last_detail_reserve = reserve
        held_back = float(np.sqrt(max(0.0, 1.0 - reserve * reserve)))

        wanted = (real_energy * held_back) / fake_energy
        self.last_detail_ratio = float(wanted)

        # The clamp moves with the target it is clamping. `_DETAIL_RATIO` bounds
        # how far this stage may deviate from what it is aiming at, and both the
        # reserve and the noise discount lower what it is aiming at — so a fixed
        # floor would refuse the very attenuation they asked for. Measured: at
        # reserve 0.8 the wanted ratio was 0.54 against a floor of 0.60, so the
        # clamp silently kept a quarter of the room that had just been promised
        # to the texture layer. Both factors have to appear here for the same
        # reason, and `attenuation` is deliberately one number so a third one
        # cannot be added to the aim and forgotten here.
        attenuation = held_back * discount
        low, high = self._DETAIL_RATIO
        ratio = float(np.clip(wanted, low * attenuation, high * attenuation))

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
        extent: float,
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
        warped_mask = np.clip(self._feather(warped_mask, sigma), 0.0, 1.0)

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

    def _feather(self, mask: Mask, sigma: float) -> Mask:
        """
        Blur a mask, at a resolution matched to how wide the blur is.

        A feathered mask is a smooth field by construction, so blurring it at
        full resolution is work for nothing once the radius is large: at a 400px
        face the frame-space feather is 16px, and that Gaussian cost **4.10ms**
        over a 410px region — the single largest item in the compositor. Done at
        a quarter and scaled back it is 0.93ms, and the two agree to within
        **0.008** on a mask that runs 0 to 1.

        The reduction follows the radius rather than being fixed. A tight
        feather is cheap already and must not be resampled, because a sigma of a
        pixel or two does not survive a downscale — and losing it would put back
        the hard edge this whole path exists to remove.

        Args:
            mask: Soft mask over the region
            sigma: Blur radius in pixels at full resolution

        Returns:
            The blurred mask, at the same size it came in
        """
        height, width = mask.shape[:2]
        factor = int(np.clip(sigma / self._FEATHER_MIN_SIGMA, 1, 4))
        if factor <= 1 or min(height, width) < self._FEATHER_MIN_SIZE * factor:
            return cv2.GaussianBlur(mask, (0, 0), sigma)

        small = (max(1, width // factor), max(1, height // factor))
        reduced = cv2.resize(mask, small, interpolation=cv2.INTER_AREA)
        reduced = cv2.GaussianBlur(reduced, (0, 0), max(1.0, sigma / factor))
        restored: Mask = cv2.resize(
            reduced, (width, height), interpolation=cv2.INTER_LINEAR,
        )
        return restored

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

        Three things gate it, and none of them is new machinery. The skin mask is
        baked into the map at extraction, keeping detail off eyes, nostrils and
        mouth. The compositing alpha is applied here, keeping it inside the swap
        — and because that alpha is the feathered one, and already carries the
        XSeg occlusion term the masker applied, texture fades out exactly where
        the swap does and never lands on a hand or a microphone. Pose agreement
        scales the whole thing, because the map is only as good as the angle it
        was taken at matches the angle it is being used at.

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
        self.last_texture_confidence = None
        self.last_texture_delivered = None
        self.last_texture_coverage = None

        strength, contrast, band, relief = self._texture_shaping()
        if strength <= 0.0:
            return blended

        if self.source_texture is None:
            # Once per compositor, not per frame: this is the live path at
            # 15-20fps, and a warning repeated thirty times a second is a
            # denial of service on its own log.
            if not self._warned_no_texture:
                self._warned_no_texture = True
                emit_warning(
                    'texture_strength is {:.2f} but no source texture was '
                    'extracted, so the layer is doing nothing. A source set of '
                    'only .npy embeddings has no pixels to take skin detail '
                    'from; otherwise the chosen photo had unusable keypoints. '
                    'Look for a "Texture source:" line at source load.'.format(
                        strength),
                    scope='TEXTURE',
                )
            return blended

        confidence = self._pose_confidence(face)
        self.last_texture_confidence = confidence
        if confidence <= 0.0:
            return blended

        started = time.perf_counter()

        size = texture.map_size(extent)
        detail = self.source_texture.detail_for(size, contrast, band, relief)
        if detail is None:
            return blended

        canonical = canonical_from_frame(face, size)
        if canonical is None:
            return blended

        # `band` goes to both, and it has to: the headroom is what the map is
        # allowed to spend, so measuring a narrower band than the map carries
        # would understate the energy about to be added and overshoot by the
        # difference — the one failure this whole measured-budget design exists
        # to make impossible.
        headroom = self._texture_headroom(
            blended, target, mask, extent, face, roi, band,
        )
        self.last_texture_headroom = headroom
        if headroom <= self._TEXTURE_FLOOR:
            # A reservation was made and nothing filled it: `_match_detail`
            # undershot by `reserve` for this layer, and this layer then found
            # no room. That leaves the face softer than with the layer switched
            # off, and it is invisible without being said — the readings carry
            # the zero, but only a stream reports them, and this is exactly the
            # state a still render lands in. Once per compositor, like the
            # missing-source warning beside it.
            if not self._warned_no_headroom:
                self._warned_no_headroom = True
                emit_warning(
                    'texture_strength is {:.2f} but the measured headroom is '
                    '{:.2f}, under the {:.2f} floor — the layer added nothing '
                    'while detail matching had already held back {:.2f} of the '
                    'band for it, so this face is softer than with texture off. '
                    'The target may already carry as much high-frequency energy '
                    'as it can, or the region may be too small to '
                    'measure.'.format(
                        strength, headroom, self._TEXTURE_FLOOR,
                        self.last_detail_reserve or 0.0),
                    scope='TEXTURE',
                )
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
        #
        # **`strength` is spent once, in `_texture_reserve`, not again here.**
        # It used to appear in both places: the reserve is `strength` and this
        # was `strength * headroom`, so the delivered amplitude went as the
        # *square* — 0.4 asked `_match_detail` to stand back by 40% and then
        # filled 40% of what that freed, delivering 16%. The knob is documented
        # as "the fraction of the measured gap to close", so closing the gap it
        # opened is what it has to do. Below 1.0 that means spending the whole
        # measured headroom: the reservation decides the share, the frame-space
        # measurement decides the amount, and `f^2 + g^2 + t^2 = r^2` still puts
        # the total exactly at parity — overshoot remains impossible.
        #
        # Above 1.0 is the diagnostic overshoot and keeps multiplying, since
        # past parity there is no reservation left to be the control.
        # **The map is normalised before the warp and spent after it.**
        #
        # `detail_for` returns unit deviation in *canonical* space. `warpAffine`
        # then resamples it bilinearly, and bilinear interpolation is a low-pass
        # filter whose response falls to zero at Nyquist — while this map's
        # finest octave sits at the resolution limit by construction, since
        # `DETAIL_SIGMA` is 1.5. Any rotation, non-unit scale or sub-pixel
        # offset therefore attenuates exactly the content the layer exists to
        # add, and `amount` was being spent as though it had not.
        #
        # Measured 2026-09-10: the map retained **0.431** of its deviation, so
        # the layer delivered 41% of a budget `_match_detail` had already stood
        # down to make room for. Coverage was 0.920, which ruled out the area
        # explanation — this is amplitude, and it is recoverable because it is
        # measurable.
        #
        # So measure what survived and scale by it, rather than assuming. Same
        # principle as the headroom itself: the arithmetic that decides how much
        # to add is only sound if every term in it is measured in the space it
        # is spent in.
        inside = mask > 0.5
        committed = int(np.count_nonzero(inside))
        support = np.abs(warped) > 1e-6
        landed = inside & support
        realised = (float(warped[landed].std())
                    if int(np.count_nonzero(landed)) > 64 else 0.0)

        # `amount` stays what it always was — the deviation this layer intends
        # to *deliver*, bounded by `TEXTURE_MAX`. The gain is what the map has
        # to be multiplied by for that to arrive, so the cap keeps meaning what
        # it says rather than silently becoming a bound on the pre-warp scale.
        amount = min(
            max(strength, 1.0) * headroom * confidence, texture.TEXTURE_MAX,
        )
        gain = float(np.clip(
            1.0 / realised if realised > 1e-6 else 1.0,
            1.0, self._WARP_GAIN_MAX,
        ))
        added = (warped * (amount * gain)) * mask

        # **What actually reached the picture, against what was budgeted.**
        #
        # With the gain applied this is a *check* rather than an estimate: it
        # should now land on `amount`, and a shortfall that survives the
        # correction means the gain hit `_WARP_GAIN_MAX` — which is the case
        # that ceiling exists for, and the one worth seeing rather than
        # silently correcting.
        #
        # Measured on the field itself rather than by differencing the picture,
        # so it is exactly this layer's contribution and not the sum of
        # everything else that touched the ROI. Over the pixels the alpha
        # actually commits, since the deviation of a field that is mostly zeros
        # outside the mask says nothing about what landed on the face.
        if committed > 64:
            self.last_texture_delivered = float(added[inside].std())

            # And the area term, which is what separated the two possible
            # causes of a short spend before the gain existed: a field that is
            # unit-deviation on a fraction `c` of the region it is measured over
            # reads `sqrt(c)`, not 1. Kept because it still distinguishes
            # "attenuated by the warp" — now corrected — from "arriving over
            # less of the face than the reserve assumed", which is a separate
            # defect the gain does not touch: `_match_detail` stands down
            # uniformly across the whole face while the fill is skin-only by
            # construction, so the excluded features lose detail with nothing
            # replacing it.
            self.last_texture_coverage = (
                float(int(np.count_nonzero(landed))) / float(committed))
        else:
            self.last_texture_delivered = None
            self.last_texture_coverage = None

        result: Frame = blended + added[:, :, None]

        # Contained within the `paste` bucket rather than added to it — the frame
        # total does not change, this just says how much of paste it was.
        self.last_stage_ms['texture'] = (time.perf_counter() - started) * 1000.0
        return result

    def _texture_shaping(self) -> Tuple[float, float, float, float]:
        """
        The four texture knobs, read and clamped in one place.

        `_add_texture` and `_texture_reserve` both need them and must not
        disagree: a reserve computed against one strength and an amount spent
        against another would leave the face short by the difference, softer
        than it was before any of this existed.

        Returns:
            (strength, contrast, band, relief), each clamped to its range
        """
        return (
            float(np.clip(
                getattr(self.config, 'texture_strength', 0.0),
                0.0, texture.STRENGTH_MAX)),
            float(np.clip(
                getattr(self.config, 'texture_contrast', 1.0),
                *texture.CONTRAST_RANGE)),
            float(np.clip(
                getattr(self.config, 'texture_band', 1.0),
                *texture.BAND_RANGE)),
            float(np.clip(
                getattr(self.config, 'texture_relief', 0.0),
                *texture.RELIEF_RANGE)),
        )

    def _texture_reserve(self, face: Face, extent: float) -> float:
        """
        Share of the target's high band to leave for real skin detail.

        Non-zero only when the texture layer will actually run on this frame,
        and that condition is the whole safety argument. A reservation is a
        deliberate *undershoot* by `_match_detail`; if the layer then declines —
        no source photograph, a pose too far from it, no map at this working
        size — nothing fills the gap and the face comes out softer than it would
        have with no texture layer at all. So every gate `_add_texture` applies
        before it commits is applied here first, against the same clamped
        values.

        The one gate that cannot be checked in advance is the headroom itself,
        which is measured in frame space after the warp. That direction is safe:
        reserving makes headroom *more* likely to exist, not less.

        Args:
            face: Detection for this frame
            extent: The face's extent in frame pixels

        Returns:
            Amplitude share in [0, `_RESERVE_MAX`]; 0.0 when the layer is off or
            cannot run.
        """
        if self.source_texture is None:
            return 0.0

        strength, contrast, band, relief = self._texture_shaping()
        if strength <= 0.0:
            return 0.0

        confidence = self._pose_confidence(face)
        if confidence <= 0.0:
            return 0.0

        # Builds the map if this is the first frame at this size, which `_paste`
        # would have done moments later anyway — `detail_for` memoises, so the
        # work is paid once either way rather than twice.
        if self.source_texture.detail_for(
                texture.map_size(extent), contrast, band, relief) is None:
            return 0.0

        # `strength` may exceed 1.0 to overshoot parity deliberately; a reserve
        # cannot. Above 1.0 the extra is spent on top of a full reservation.
        return float(np.clip(
            min(strength, 1.0) * confidence, 0.0, self._RESERVE_MAX,
        ))

    def _pose_confidence(self, face: Face) -> float:
        """
        How far the frame's pose agrees with the source photograph's, in [0, 1].

        Ramped rather than thresholded, for the reason every other ramp in this
        module exists: a switch would snap on and off between frames as a head
        drifts across the boundary, and a texture layer appearing and vanishing
        is more visible than one that is slightly wrong.

        **A missing reading means full confidence, not none.** A model pack
        without `pose` cannot answer this, and silently attenuating to zero
        there would turn "we cannot measure the angle" into "the layer does
        nothing", which is a behaviour change hiding inside a capability gap.
        `guards.probe_capabilities` already reports that pack difference at
        startup; this must not also encode it.

        Only the *magnitude* of the disagreement is used. The direction matters
        too — at yaw it is the cheek turning away whose pores are stretched, so
        the correct term falls off across the face rather than uniformly — but
        that needs the sign convention of `face.pose` pinned against real
        footage, and a directional term applied with the sign backwards would
        attenuate the half of the face that is still good. Uniform is the
        conservative version of the same idea; see docs/TEXTURE_PIPELINE.md.

        Args:
            face: Detection for this frame

        Returns:
            1.0 when the poses agree or cannot be compared, falling to 0.0 as
            the disagreement reaches `_POSE_LIMIT`.
        """
        assert self.source_texture is not None
        source_yaw = self.source_texture.yaw
        if source_yaw is None:
            return 1.0

        pose = getattr(face, 'pose', None)
        if pose is None:
            return 1.0
        try:
            values = np.asarray(pose, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return 1.0
        if values.size < 2 or not np.isfinite(values[1]):
            return 1.0

        # InsightFace orders this (pitch, yaw, roll) — the same field and index
        # `guards.measure_yaw` prefers, so the two cannot disagree about what
        # the number means.
        delta = abs(float(values[1]) - source_yaw)
        if delta <= self._POSE_FULL:
            return 1.0

        span = self._POSE_LIMIT - self._POSE_FULL
        if span <= 1e-6:
            return 0.0
        return float(np.clip(1.0 - (delta - self._POSE_FULL) / span, 0.0, 1.0))

    def _stat_window(
        self,
        shape: Tuple[int, ...],
    ) -> Tuple[slice, slice]:
        """
        A centred window of at most `_STAT_WINDOW` on a side.

        Cropping rather than downscaling, deliberately: a downscale is itself a
        low-pass, and both callers are measuring high-frequency content, so
        shrinking the image first would measure a different band than the one in
        question.

        Args:
            shape: (height, width) of the region

        Returns:
            Row and column slices to apply to anything over that region
        """
        cap = self._STAT_WINDOW
        height, width = int(shape[0]), int(shape[1])
        top = max(0, (height - cap) // 2)
        left = max(0, (width - cap) // 2)
        return (
            slice(top, top + min(height, cap)),
            slice(left, left + min(width, cap)),
        )

    def _texture_headroom(
        self,
        blended: Frame,
        target: Frame,
        mask: Mask,
        extent: float,
        face: Face,
        roi: Tuple[int, int, int, int],
        band: float = 1.0,
    ) -> float:
        """
        How much high-frequency deviation the face can still take, in 8-bit units.

        Independent zero-mean fields add in quadrature, so a face already
        carrying `f` and about to receive grain of `g` can accept `t` before it
        matches a real face carrying `r`:

            f^2 + g^2 + t^2  =  r^2

        Anything past that is a face with more texture than the camera recorded,
        which reads as noise rather than as skin.

        The band is the one the extractor cut, expressed here in frame pixels:
        `DETAIL_SIGMA` is specified at a 256px face, so at `extent` pixels it is
        that fraction of it, times `band`. That multiplier is not optional —
        the map spans up to the subsurface diffusion length, and measuring the
        finest octave alone would report a fraction of the energy the layer is
        about to add and let it overshoot by the rest.

        **Measured on skin, not over everything inside the mask.** The window
        below sits at the centre of the region, and for a large face that centre
        is nose, mouth and eyes — whose high-frequency content is *structure*
        rather than skin texture. Reading that as "what a real face carries
        here" overstates the headroom and lets the layer add more than the
        cheeks justify. The same exclusions the scatter pass uses, built after
        the crop so their cost is flat in face size rather than growing with it.

        Args:
            blended: The composited ROI, float32
            target: The original frame over the same ROI, float32
            mask: Compositing alpha over the ROI
            extent: The face's extent in frame pixels
            face: Detection, for the keypoints the exclusions are placed on
            roi: (x0, y0, width, height) of the region in frame space
            band: The texture band's span, as a multiple of `DETAIL_SIGMA` —
                the same value the map was built with

        Returns:
            Deviation still available, in 8-bit units. Zero when the swap has
            already reached or passed the real face's texture level.
        """
        # Sample a bounded window rather than the whole region — see
        # `_HEADROOM_SAMPLE`. Cropping rather than downscaling, because a
        # downscale is itself a low-pass and would measure a different band than
        # the one being added.
        rows, cols = self._stat_window(mask.shape)
        mask = mask[rows, cols]
        blended = blended[rows, cols]
        target = target[rows, cols]

        # Frame keypoints into window coordinates: the region's own origin, plus
        # wherever the window starts inside it.
        x0, y0 = roi[0], roi[1]
        offset = np.array([
            [1.0, 0.0, -float(x0 + cols.start)],
            [0.0, 1.0, -float(y0 + rows.start)],
        ], dtype=np.float32)
        skin = self._scatter_weight(mask, face, offset, max(mask.shape))
        if skin is not None:
            mask = skin

        binary = (mask > 0.5).astype(np.uint8)
        if cv2.countNonZero(binary) < 64:
            return 0.0

        sigma = max(
            0.6,
            DETAIL_SIGMA * float(band) * extent / DETAIL_SIGMA_REFERENCE,
        )

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

        # `cv2.merge` and `cv2.add` rather than `mask[:, :, None]`
        # broadcasting, for the same reason `_paste` spells its own composite
        # out: numpy expands the broadcast into a full-size temporary on every
        # frame, and at a 500px region that was 4.2ms against 2.4ms.
        field = cv2.multiply(self._noise_field(mask.shape), mask) * sigma
        return cv2.add(blended, cv2.merge([field, field, field]))

    def _noise_field(self, shape: Tuple[int, ...]) -> Frame:
        """
        A unit-variance noise window, cut from a cached tile at a random offset.

        The tile is grown to twice the largest region asked for, so there are
        many distinct windows and consecutive frames do not share one. Only the
        offset changes per frame, which is what keeps this from reading as
        fixed-pattern noise - a static grain overlay is a worse tell than no
        grain at all.

        Args:
            shape: (height, width) of the region needing noise

        Returns:
            A float32 view of that size, unit standard deviation
        """
        height, width = int(shape[0]), int(shape[1])
        tile = self._noise
        if tile is None or tile.shape[0] < height * 2 or tile.shape[1] < width * 2:
            tile = np.random.normal(
                0.0, 1.0, (height * 2, width * 2),
            ).astype(np.float32)
            self._noise = tile

        top = int(np.random.randint(0, tile.shape[0] - height + 1))
        left = int(np.random.randint(0, tile.shape[1] - width + 1))
        window: Frame = tile[top:top + height, left:left + width]
        return window

    def _estimate_noise(self, region: Frame) -> float:
        """
        Robust noise sigma of a region, via the MAD of its Laplacian.

        A median-based estimator is used rather than a mean-based one so that
        genuine facial detail and edges do not inflate the result.
        """
        # Bounded before the transform, not after. The stride below already
        # said a sigma converges on far fewer pixels than a face region carries,
        # but it was applied to the *result* — so the colour conversion and the
        # Laplacian still ran over every pixel, which was 4.6ms at a 500px
        # region. Cropping first makes the whole estimate flat in region size.
        rows, cols = self._stat_window(region.shape)
        sample_region = region[rows, cols]

        gray = cv2.cvtColor(
            np.clip(sample_region, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY,
        )
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
