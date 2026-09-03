"""
Configuration management for the Phantom face-swapping pipeline.

Replaces pipeline.globals with a typed, observable configuration object.
Supports change notifications via callbacks for reactive updates.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict, Tuple


@dataclass
class FaceSwapConfig:
    """
    Centralized runtime configuration for the Phantom pipeline.

    All fields are typed and have sensible defaults. Change listeners can be
    registered via on_change() to react to configuration updates.
    """

    # Input/output paths
    source_path: Optional[str] = None
    source_paths: List[str] = field(default_factory=list)
    target_path: Optional[str] = None
    # Photo mode: several image targets, each swapped independently. Kept
    # separate from `target_path` so a photo job cannot be mistaken for a
    # single-file batch, and vice versa.
    target_paths: List[str] = field(default_factory=list)
    # Bundled template in use, and the face within it the template's author
    # chose. The point is normalised (x, y) so it survives a resize, and it is
    # what answers the multi-face guard: the ambiguity was resolved offline.
    template_id: Optional[str] = None
    target_face_point: Optional[Tuple[float, float]] = None
    # The same answer, given by the operator instead of by a manifest, once per
    # uploaded photo. A list rather than a second scalar because photo mode
    # carries up to four targets and each asks the question separately; index
    # aligns with `target_paths`, and None means "nobody chose for this one".
    target_face_points: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    # RGBA layer drawn over the finished swap, for hair, glasses or a hand that
    # belongs in front of the face. Carried as a path rather than as a Template
    # so the pipeline stays unaware of the library.
    target_foreground: Optional[str] = None
    # Where derived photo outputs go. Set for a template job, whose target
    # lives in the shared library and must not be written next to.
    output_dir: Optional[str] = None
    output_path: Optional[str] = None
    save_embedding_path: Optional[str] = None

    # Processing pipeline
    frame_processors: List[str] = field(default_factory=lambda: ['face_swapper'])
    # Hand back what was handed in: preserve the source rate unless asked to
    # retime. `pipeline/core.py` carries the reasoning and the matching flag.
    keep_fps: bool = True
    keep_audio: bool = True
    keep_frames: bool = False
    many_faces: bool = False
    video_encoder: str = 'libx264'
    video_quality: int = 18
    max_memory: int = 16
    execution_providers: List[str] = field(default_factory=lambda: ['CPUExecutionProvider'])
    execution_threads: int = 8

    # Inference speed levers. All default off, so the out-of-the-box path is
    # bit-identical to what it was before they existed — each one is opted into
    # and measured rather than assumed.
    #
    # `fp16` loads a converted half-precision copy of the weights when one has
    # been produced (tools/convert_fp16.py). It is the largest single win and
    # the only one of these that can change what the output looks like, so it
    # is A/B'd on real footage before it ships, not judged on a latency number.
    #
    # `cuda_graphs` captures the kernel launch sequence once and replays it,
    # which is worth having on batch-1 models made of many small kernels. It
    # needs IOBinding with stable buffer addresses and static input shapes, so
    # it only applies to models that declare themselves static.
    #
    # `trt` routes through the TensorRT provider. Engines are built once per
    # GPU architecture and cached on the network volume; `trt_gpus` names the
    # architectures worth spending a multi-minute build on, since that build
    # comes out of a paid hour with an operator waiting.
    fp16: bool = False
    cuda_graphs: bool = False
    # Copies on a separate CUDA stream, so a transfer can overlap compute.
    # Ignored where CUDA graphs are active — capture assumes one stream.
    cuda_streams: bool = False
    trt: bool = False
    trt_gpus: str = 'RTX 4090,RTX 5090,H100,H200,L40S,RTX 6000 Ada,RTX 4080'

    # Move the JPEG encode off the pipeline thread and onto the sender thread
    # that was already there waiting on the network. Two effects: the encode
    # leaves the critical path, and because the sender drains to the newest
    # frame before encoding, frames that were going to be dropped are no longer
    # encoded at all. Costs memory — the queue then holds raw frames rather
    # than JPEGs — which is why it is bounded at two and opt-in.
    async_encode: bool = False

    # Stream pipeline - quality presets
    quality: str = 'optimal'
    alpha: float = 0.6  # EMA on face landmarks (0.0 = max smoothing, 1.0 = off)
    enhance: bool = True
    color_correction: bool = True
    preprocessing: bool = False
    buffer_size: int = 4
    warmup_frames: int = 5

    # Capture and encode, driven by the quality preset. Shared by the pipeline's
    # own VideoCapture loop and the desktop's webcam thread, so local and push
    # mode capture identically.
    capture_width: int = 640
    capture_height: int = 360
    capture_fps: int = 20
    jpeg_quality: int = 70

    # Detector input size. Detection runs on every frame, so this is the single
    # largest per-frame cost — 320 is a quarter the pixels of 640.
    det_size: int = 448

    # Debug frame capture. When `debug_frames_dir` is set, the stream writes
    # (input, output) pairs so realism changes can be measured against a fixed
    # clip rather than re-recorded each time. Written on a background thread and
    # dropped under pressure, so enabling it does not change the latency of the
    # thing being measured. Off and free by default.
    #
    # Written as lossless PNG, because these frames get measured and a lossy
    # encode would add its own artefacts to the very statistics being compared.
    # That costs disk: roughly 1.4 MB per pair at 640x360, so about 27 MB/s at
    # 20fps. Raise the stride for anything longer than a short clip.
    debug_frames_dir: Optional[str] = None
    debug_frames_stride: int = 1   # keep every Nth frame
    debug_frames_limit: int = 0    # stop after N pairs; 0 = unlimited

    # Face-swap model. Selects both the weights and the realism profile tuned
    # for them — see pipeline/services/swapper_models.py. Changing this without
    # changing the profile is what would make a better model look worse.
    swapper_model: str = 'inswapper_128'

    # Realism / compositing
    # Restoration model. See pipeline/services/enhancer_models.py — the
    # registry owns the crop size, whether the model has a fidelity weight,
    # and where to fetch it. Default is the 256-crop model: CodeFormer's extra
    # 24ms buys +0.03 of face/frame detail on a 101px webcam face, because the
    # 512 result is warped straight back down into a 128-192 aligned space.
    enhancer_model: str = 'gpen_bfr_256'
    enhancer_weight: float = 0.7    # CodeFormer fidelity: 0 = heaviest restoration
                                    # and most hallucination, 1 = closest to input.
                                    # Read only by models that declare a weight
                                    # input — GPEN has none, so it does nothing
                                    # there and `enhance_strength` is the only
                                    # remaining control.
    enhance_strength: float = 0.7   # how much of the restored face to keep

    # Named restoration strength — see schema.RESTORATION_PRESETS. `auto`
    # defers to the swap model's profile, which is the behaviour that existed
    # before this was a control. Anything else is an explicit operator choice
    # and outranks the profile, so switching models cannot silently undo it.
    restoration_preset: str = 'auto'
    aligned_size: int = 256         # compositing ceiling (128-512), from the preset
    # Compositing floor, from the model profile. Compositing below a model's
    # native output size throws away detail it already generated, so a 256 model
    # should never composite a distant face at 128 the way a 128 model can.
    aligned_min: int = 128
    temporal_alpha: float = 0.6     # EMA on aligned pixels (1.0 = off)

    # Restoration resolution, and when to spend it at all.
    #
    # Both restorers are trained on FFHQ 512x512 crops, so 512 is the default
    # and the out-of-the-box path is unchanged. It is also the single largest
    # cost in the pipeline and the one stage that ignores how big the face
    # actually is: a 101px face in a 640x360 frame is upsampled ~20x in pixel
    # count for it, and conv cost scales with pixel count. `_aligned_size`
    # already applies the opposite principle one stage earlier.
    #
    # `restore_size` only takes effect if the loaded model accepts the shape.
    # A graph exported with fixed 512 spatial dims is honoured over this, and
    # `Enhancer.crop_size` says so once rather than throwing per frame.
    restore_size: int = 512
    # Skip restoration entirely below this face size (px, shorter side of the
    # bounding box — the same measure as `guard_min_frame_px`). 0 never skips.
    # A small face's swap came out of a 128px generator; whether 512-space
    # restoration visibly improves it is a footage question, and this is what
    # makes it askable without a new model.
    restore_min_face: int = 0
    color_strength: float = 1.0     # scales the LAB colour transfer

    # Seam. Both are the direct levers on failure mode 2 and both were reported
    # wrong on footage -- "like the face pasted on target" -- so they are knobs
    # rather than constants: this is exactly the kind of question only an A/B on
    # a real call can settle. See docs/TEXTURE_PIPELINE.md section 13.
    #
    # `mask_feather` is the frame-space transition width as a fraction of the
    # face's extent in frame. It was 0.01 of the region of interest, which on a
    # 101px face is 1.1px -- combined with the aligned-space feather scaled down,
    # a total transition of ~1.4% of the face's width. That is a hard edge.
    #
    # `mask_erode` pulls the mask in, in aligned space, *before* it is feathered.
    # Without it the hull is expanded 10% radially and then blurred, so the
    # midpoint of the transition sits outside the landmark silhouette -- on neck
    # at the chin, on hair at the temples, which is where the material either
    # side differs most. Eroding first puts the soft part on skin. Deliberately
    # not "extend the mask": growing the *coverage* puts swapped skin where hair
    # should be, which docs/ENHANCEMENT.md records as the worse tell.
    mask_feather: float = 0.04
    mask_erode: float = 0.03

    # Subsurface scattering, approximated. Real skin is translucent: light
    # enters, bounces around under the surface and leaves somewhere slightly
    # else, which softens the *shading* over a millimetre or two without
    # touching the texture on top of it. A generated face has none of that, and
    # the result reads as hard rather than as plastic — a different complaint
    # from the one `texture_strength` answers, in a different frequency band.
    #
    # Scoped to the luminance channel, and to skin with eyes, nose and mouth
    # cut out. Unscoped it would soften facial structure, which is the exact
    # failure this whole pipeline routes around in full-strength restoration.
    #
    # **Defaults off, and never judged on footage.** It is the highest-risk
    # layer here by the design's own admission; 0.2-0.4 is the expected working
    # range if it turns out to be wanted at all.
    diffuse_strength: float = 0.0

    # Skin detail lifted from the operator's own source photograph and warped
    # onto the face every frame. The swapped face carries 58% of the frame's
    # high-frequency energy against an ideal of 1.00, and restoration was
    # measured to close 7% of that gap — this is the layer aimed at the rest.
    #
    # **The fraction of the measured gap to close**, not a gain. The compositor
    # measures how much high-frequency energy the operator's real face has in
    # these very pixels, subtracts what the swap already carries and what grain
    # is about to add, and this is the share of what remains. So 1.0 lands at
    # parity with the real face and cannot overshoot it, and one setting means
    # the same thing on any clip rather than tracking whichever had most
    # contrast. `texture.TEXTURE_MAX` survives only as a backstop against an
    # estimate gone wrong.
    #
    # **Defaults off.** It has never been judged on footage, and a realism knob
    # that arrives switched on is one nobody chose. 0.3-0.5 is the expected
    # working range; see docs/TEXTURE_PIPELINE.md.
    texture_strength: float = 0.0
    grain: bool = True              # match sensor noise on the composited face
    occluder: bool = True           # XSeg mask so hands/mics are not overpainted

    # Input guards. Refusing an input beats swapping it badly: a frame with no
    # face is obviously broken and gets fixed, while a frame with a *stranger's*
    # face swapped in looks like it worked. See docs/INPUT_GUARDS.md.
    guards: bool = True             # master switch for runtime guards
    guard_multi_face: bool = True   # reject multi-face sources, guard multi-face frames
    guard_min_source_px: int = 110  # minimum source face size, shorter side
    guard_min_frame_px: int = 80    # minimum runtime face size, shorter side
    guard_max_yaw: float = 35.0     # degrees; ArcFace degrades sharply toward profile
    guard_min_confidence: float = 0.5   # detection score floor (detect thresh is 0.35)
    guard_min_coverage: float = 0.4     # fraction of the hull left unoccluded

    # Cosine below which consecutive frames are treated as different people, so
    # `LandmarkStabilizer` drops its smoothing. On config rather than as a
    # constant because it is the guard threshold most able to *cost* realism: set
    # too high, the stabilizer resets on ordinary motion blur and the shimmer it
    # exists to remove comes back.
    #
    # 0.35 rather than 0.5 because the two populations are far apart and the
    # threshold should sit in the gap with margin on both sides. ArcFace
    # embeddings of *different* people are close to orthogonal (~0.0-0.2); the
    # same person on consecutive frames normally sits above 0.9 and degrades only
    # under blur or extreme pose. 0.5 leaves almost no room beneath a degraded
    # same-person reading; 0.35 clears both.
    #
    # The threshold is only half the protection — see `_IDENTITY_CONFIRM`, which
    # requires the reading to stay low across several frames before believing it.
    # Set this to -1.0 to disable identity-based resetting entirely, leaving only
    # the centroid-jump test.
    guard_identity_sim: float = 0.35

    # The two thresholds the design left open, because neither can be set
    # honestly without real uploads to calibrate against. Both start permissive:
    # a guard that rejects a usable photo is friction at the exact moment a new
    # customer is deciding whether this works at all, so the failure to prefer
    # is letting a marginal image through, not turning a good one away.
    guard_min_sharpness: float = 40.0   # Laplacian variance floor on source images
    guard_outlier_sim: float = 0.35     # cosine floor against the group mean

    # Calibration. In observe mode every guard is evaluated and recorded but
    # none of them act, so a single session measures what all the thresholds
    # would have done without any of them affecting the footage being measured.
    # This is how the numbers above stop being guesses.
    guard_observe: bool = False
    guard_report: Optional[str] = None  # write the telemetry JSON here on stop

    # Vestigial — kept so `apply_preset` and the desktop's `set_quality`
    # round-trip keeps working. No longer read by the pipeline: face tracking
    # was replaced by per-frame detection plus landmark EMA, and blending is
    # now handled by FaceCompositor's mask rather than a global alpha.
    tracker: str = 'csrt'
    blend: float = 0.65
    luminance_blend: bool = True
    redetect_interval: int = 30

    # I/O configuration
    input_url: Optional[str] = None
    stream_url: Optional[str] = None
    preview_url: Optional[str] = None
    virtual_cam: bool = False
    control_port: int = 9000

    # Logging & status
    log_level: str = 'error'
    status_message: str = ''
    embedding_ready: bool = False
    headless: bool = False
    stream_mode: bool = False

    # Internal state
    shutdown_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    _listeners: List[Callable[[str, Any], None]] = field(default_factory=list, repr=False, compare=False)

    def on_change(self, cb: Callable[[str, Any], None]) -> None:
        """
        Register a callback to be invoked when configuration changes.

        Callback signature: cb(field_name: str, new_value: Any)
        """
        self._listeners.append(cb)

    def off_change(self, cb: Callable[[str, Any], None]) -> None:
        """Unregister a previously registered change listener."""
        if cb in self._listeners:
            self._listeners.remove(cb)

    def set(self, field: str, value: Any) -> None:
        """
        Set a configuration field and notify all listeners.

        Args:
            field: Name of the configuration field (must exist)
            value: New value for the field

        Raises:
            AttributeError: If field doesn't exist
            TypeError: If value type is incompatible
        """
        if not hasattr(self, field):
            raise AttributeError(f"FaceSwapConfig has no field '{field}'")

        setattr(self, field, value)

        # Notify all listeners
        for cb in self._listeners:
            try:
                cb(field, value)
            except Exception as e:
                # Log but don't crash if callback fails
                import sys
                print(f"Warning: config change listener failed: {e}", file=sys.stderr)

    def apply_model_profile(self, model_name: Optional[str] = None) -> None:
        """
        Apply the realism profile belonging to a swap model.

        Ordering matters: this runs *after* `apply_preset`, because the preset
        owns compute and the model owns appearance. Explicit CLI, env and
        `set_realism` values are applied after both and win over each.

        Args:
            model_name: Registry key. None uses the configured `swapper_model`
        """
        from pipeline.services.swapper_models import resolve

        model = resolve(model_name or self.swapper_model)
        self.set('swapper_model', model.name)

        # An explicit restoration choice outranks the model's profile. Without
        # this, picking a strength and then changing swapper — which the
        # desktop does on start, and `set_realism` does live — would revert it
        # with nothing said. The same shape of bug as `startPipeline` firing
        # `set_enhance` over a pipeline launched with `--no-enhance`.
        pinned = (self.restoration_preset or 'auto') != 'auto'

        for key, value in model.look().items():
            if pinned and key in ('enhance_strength', 'enhance'):
                continue
            self.set(key, int(value) if key == 'aligned_min' else value)

    def apply_restoration_preset(self, name: str) -> bool:
        """
        Apply a named restoration strength.

        Args:
            name: A key of `schema.RESTORATION_PRESETS`

        Returns:
            True if applied. `auto` re-applies the current model's profile,
            since "follow the model" is only meaningful as the model's value.
        """
        from pipeline.api.schema import RESTORATION_PRESETS

        key = (name or '').strip().lower()
        if key not in RESTORATION_PRESETS:
            return False

        self.set('restoration_preset', key)

        values = RESTORATION_PRESETS[key]
        if values is None:
            self.apply_model_profile()
            return True

        for name_, value in values.items():
            self.set(name_, value)
        return True

    def apply_preset(self, preset_name: str) -> None:
        """
        Apply a named quality preset to the configuration.

        Available presets: 'fast', 'optimal', 'production'

        Args:
            preset_name: Name of the preset to apply

        Raises:
            ValueError: If preset name is not recognized
        """
        from pipeline.api.schema import PRESETS

        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")

        for key, value in PRESETS[preset_name].items():
            self.set(key, value)

    def get_preset_config(self) -> Dict[str, Any]:
        """
        Export current configuration as a dictionary.
        Useful for serialization or logging.
        """
        return {
            'source_path': self.source_path,
            'source_paths': self.source_paths,
            'target_path': self.target_path,
            'target_paths': self.target_paths,
            'template_id': self.template_id,
            'target_face_point': self.target_face_point,
            'target_face_points': self.target_face_points,
            'target_foreground': self.target_foreground,
            'output_dir': self.output_dir,
            'output_path': self.output_path,
            'frame_processors': self.frame_processors,
            'keep_fps': self.keep_fps,
            'keep_audio': self.keep_audio,
            'keep_frames': self.keep_frames,
            'many_faces': self.many_faces,
            'video_encoder': self.video_encoder,
            'video_quality': self.video_quality,
            'max_memory': self.max_memory,
            'execution_providers': self.execution_providers,
            'execution_threads': self.execution_threads,
            'quality': self.quality,
            'tracker': self.tracker,
            'alpha': self.alpha,
            'blend': self.blend,
            'luminance_blend': self.luminance_blend,
            'enhance': self.enhance,
            'enhancer_model': self.enhancer_model,
            'enhancer_weight': self.enhancer_weight,
            'enhance_strength': self.enhance_strength,
            'restoration_preset': self.restoration_preset,
            'aligned_size': self.aligned_size,
            'restore_size': self.restore_size,
            'restore_min_face': self.restore_min_face,
            'temporal_alpha': self.temporal_alpha,
            'color_correction': self.color_correction,
            'color_strength': self.color_strength,
            'mask_feather': self.mask_feather,
            'mask_erode': self.mask_erode,
            'diffuse_strength': self.diffuse_strength,
            'texture_strength': self.texture_strength,
            'grain': self.grain,
            'occluder': self.occluder,
            'buffer_size': self.buffer_size,
            'redetect_interval': self.redetect_interval,
            'warmup_frames': self.warmup_frames,
            'input_url': self.input_url,
            'control_port': self.control_port,
            'log_level': self.log_level,
            'headless': self.headless,
        }


# Global configuration singleton
CONFIG: FaceSwapConfig = FaceSwapConfig()
