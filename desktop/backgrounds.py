"""
Background replacement — segmentation applied to the frame that came back.

**Where this runs, and why it is here rather than in the pipeline.** The
pipeline returns a finished, swapped frame; this stage separates the person from
their room and replaces what is behind them, on the operator's machine, before
the frame reaches the viewport and the virtual camera. Three consequences follow
from that placement and all three are the reason for it:

- **Every conferencing app gets the same behaviour.** Some provide a background
  of their own, some do not, and the ones that do disagree about quality. Doing
  it at the virtual camera means one answer everywhere instead of one per app.
- **The pipeline's measurements are untouched.** `--debug-frames` are written on
  the pod, before this stage exists, so `tools/compare_frames.py` still divides
  the face's statistics by a real background. Segmenting *before* upload would
  have collapsed that denominator and made every realism reading taken after it
  incomparable with every reading taken before.
- **It costs the swap nothing.** The 50ms frame deadline is the pipeline's; this
  spends the desktop's 33ms display tick, which is the same budget filters and
  effects already spend.

What it does **not** buy, and should not be sold as: the operator's real room
still travels to the pod, because the frame is segmented after it comes back.
This is a feature for the call, not a privacy measure. Moving it before the
upload would make it one — and would break the measurement property above.

**Blur is the safe mode and solid colour is the dangerous one.** A matte error
under blur puts a few pixels of *blurred version of the same scene* against the
sharp person: low contrast, and nearly invisible. The same error against a
constant colour is a high-contrast fringe, and hair is where a cheap segmenter
is least certain. The order in `BACKGROUNDS` is deliberate — blur first, and
the colours are honestly named rather than presented as equals.

**No new dependency.** The desktop deliberately loads no face model, and
`requirements-desktop.txt` says so; `cv2.dnn` ships inside the opencv-python
that filters already require and reads ONNX directly, so this adds a model file
rather than a package. Absent that file the whole layer is a no-op that says so
once — the same way the pipeline degrades when the occluder is missing, rather
than failing a live call over a decorative stage.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

Frame = npt.NDArray[Any]
Mask = npt.NDArray[Any]

__all__ = [
    'Background', 'BACKGROUNDS', 'MODELS', 'Renderer',
    'SegmentationModel', 'Segmenter', 'get', 'names', 'model_path',
    'resolve_model', 'shared_segmenter',
]


# ── The model registry ───────────────────────────────────────────────────────
#
# Same shape as `pipeline/services/swapper_models.py` and `enhancer_models.py`,
# and for the same reason: **the model owns the facts about itself.** Input
# edge, normalisation and channel order are not preferences, they are properties
# of the weights, and hard-coding one model's answers is what makes a second
# model impossible to add. Getting any of them wrong produces a plausible-looking
# matte that is subtly wrong rather than an error — a normalisation mismatch
# reads as "the segmenter is bad", not as "the segmenter was fed the wrong
# numbers".
#
# `blobFromImage` computes `(pixel - mean) * scale`, so a model wanting
# `(p/255 - 0.5) / 0.5` is `mean=127.5, scale=1/127.5`, and one wanting `p/255`
# is `mean=0, scale=1/255`.


class SegmentationModel:
    """One segmentation model: how to feed it, and where it came from."""

    def __init__(
        self,
        key: str,
        filename: str,
        edge: int,
        scale: float,
        mean: float,
        swap_rb: bool,
        url: str = '',
        licence: str = '',
        notes: str = '',
    ) -> None:
        """
        Args:
            key: Registry key
            filename: Weight filename, as it is looked for on disk
            edge: Square input edge the model was exported at
            scale: `blobFromImage` scalefactor
            mean: Per-channel mean subtracted before scaling
            swap_rb: Whether the model wants RGB rather than OpenCV's BGR
            url: Where to fetch it
            licence: Licence, because this ships to customers
            notes: Anything a reader would otherwise have to rediscover
        """
        self.key = key
        self.filename = filename
        self.edge = edge
        self.scale = scale
        self.mean = mean
        self.swap_rb = swap_rb
        self.url = url
        self.licence = licence
        self.notes = notes

    def blob(self, frame: Frame) -> Any:
        """
        Preprocess one BGR frame into this model's input blob.

        Args:
            frame: BGR frame

        Returns:
            A 1x3xExE float32 blob
        """
        return cv2.dnn.blobFromImage(
            frame,
            scalefactor=self.scale,
            size=(self.edge, self.edge),
            mean=(self.mean, self.mean, self.mean),
            swapRB=self.swap_rb,
            crop=False,
        )


MODELS: List[SegmentationModel] = [
    # The shipped default. OpenCV's own zoo, exported for `cv2.dnn`, which is
    # why it needs no onnxruntime — and Apache 2.0, which matters because this
    # is a product with paying customers rather than a research checkout.
    SegmentationModel(
        key='pphumanseg',
        filename='human_segmentation_pphumanseg_2023mar.onnx',
        edge=192,
        scale=1.0 / 127.5,
        mean=127.5,
        swap_rb=True,
        url=('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
             'models/human_segmentation_pphumanseg/'
             'human_segmentation_pphumanseg_2023mar.onnx'),
        licence='Apache-2.0 (PaddlePaddle Authors)',
        # ASCII only: this string is printed by
        # tools/fetch_segmentation_model.py, and a Windows console at cp1252
        # renders an em-dash as a replacement character.
        notes=('Two-channel softmax over (background, person). Wants [-1, 1], '
               'not [0, 1]; feeding it [0, 1] gives a washed-out matte that '
               'looks like a weak model rather than a wrong input.'),
    ),
    # A MediaPipe-style export, for anyone who prefers one. Different edge and
    # different normalisation, which is the entire reason this is a registry.
    SegmentationModel(
        key='selfie',
        filename='selfie_segmentation.onnx',
        edge=256,
        scale=1.0 / 255.0,
        mean=0.0,
        swap_rb=True,
        licence='check the export you obtain',
        notes='Single-channel sigmoid in most exports of this family.',
    ),
]

_MODELS_BY_KEY: Dict[str, SegmentationModel] = {m.key: m for m in MODELS}

# Where weights are looked for, in order. The first is beside the desktop
# package so a checkout works without configuration; the second lets a frozen
# build ship one next to the executable. Kept as a list rather than one path for
# the reason `desktop/resources.py` exists: source and frozen layouts do not
# agree, and a stage that looks in exactly one place fails silently in exactly
# one of them.


def _search_roots() -> List[str]:
    """Directories a model may live in, most specific first."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, 'models'),
        os.path.join(os.path.dirname(here), 'models'),
        os.getcwd(),
    ]


# Feather on the upscaled mask, as a fraction of the frame's shorter side. The
# same reasoning as `mask_feather` in the compositor: a transition measured in
# absolute pixels is a different transition on every frame size, and the eye
# compares it against the subject, not against the sensor. Small — this is an
# edge that should soften, not a halo.
_FEATHER_FRACTION = 0.006

# Temporal smoothing on the mask. A segmentation mask is recomputed from
# scratch every frame and its boundary moves by a pixel or two even on a still
# subject, which reads as a crawling edge — failure mode 3, on the longest
# boundary in the picture. Low enough to not smear a fast turn into a trail.
_MASK_ALPHA = 0.6

# How often the matte is actually recomputed, in frames. The model is the whole
# cost of this layer — 28ms at 192x192 on a four-core laptop, against a 33ms
# display tick — and it is the one part that cannot be made cheaper by better
# code: threading it past two cores buys nothing (45.8 / 28.4 / 28.7 / 27.3ms at
# 1 / 2 / 4 / 8 threads), and the model's input edge is fixed by its export.
#
# So it is run every other frame and the smoothed mask carries the gap. That is
# defensible here in a way it would not be for the swap: a silhouette is a
# slowly-varying signal, the mask is already an EMA rather than a per-frame
# estimate, and the cost of being one frame late is a few pixels of background
# briefly on a shoulder — against a face that would visibly lag its own head.
# The bill for it is 66ms of matte lag at 15fps on the fastest movement, which
# is the trade to revisit if the edge is seen swimming during quick turns.
_MATTE_INTERVAL = 2

# Below this the previous mask is treated as unrelated to this one and dropped
# rather than blended. Covers a resolution change and the first frame after the
# layer is switched on, both of which would otherwise blend against a mask of a
# different shape.
_RESET_ON_SHAPE_CHANGE = True

# Background blur is done at a reduced resolution and scaled back up. A heavy
# Gaussian is a low-pass filter, so the octaves the downscale removes are ones
# the blur was about to remove anyway — the result is within a rounding error of
# blurring at full size and costs a sixteenth of it.
_BLUR_SCALE = 0.25


class Background:
    """One named background treatment, and how to build it from a frame."""

    def __init__(
        self,
        key: str,
        name: str,
        kind: str,
        colour: Optional[Tuple[int, int, int]] = None,
        sigma: float = 0.0,
    ) -> None:
        """
        Args:
            key: Stable identifier used by the UI and stored in settings
            name: Label shown to the operator
            kind: 'none', 'blur' or 'colour'
            colour: BGR fill for a 'colour' background
            sigma: Gaussian sigma for a 'blur' background, at full frame scale
        """
        self.key = key
        self.name = name
        self.kind = kind
        self.colour = colour
        self.sigma = sigma

    def build(self, frame: Frame) -> Optional[Frame]:
        """
        The picture to put behind the person.

        Args:
            frame: BGR frame, already swapped

        Returns:
            A frame-sized BGR image, or None when this background is a no-op
        """
        if self.kind == 'blur':
            return _blur_background(frame, self.sigma)
        if self.kind == 'colour' and self.colour is not None:
            filled: Frame = np.empty_like(frame)
            filled[:] = self.colour
            return filled
        return None


def _blur_background(frame: Frame, sigma: float) -> Frame:
    """
    A heavily blurred copy of the frame, computed at reduced resolution.

    Args:
        frame: BGR frame
        sigma: Gaussian sigma expressed at full frame scale

    Returns:
        A frame-sized blurred copy
    """
    height, width = frame.shape[:2]
    small_w = max(16, int(round(width * _BLUR_SCALE)))
    small_h = max(16, int(round(height * _BLUR_SCALE)))

    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    # Sigma scales with the resolution it is applied at, or the blur would be
    # four times weaker than asked for.
    small = cv2.GaussianBlur(small, (0, 0), max(0.6, sigma * _BLUR_SCALE))
    blurred: Frame = cv2.resize(
        small, (width, height), interpolation=cv2.INTER_LINEAR)
    return blurred


# Ordered deliberately: none, then the two forgiving modes, then the colours.
# See the module docstring — a matte error against a constant colour is the
# most visible failure this layer can produce, and putting the colours last is
# the only steer the picker itself can give.
BACKGROUNDS: List[Background] = [
    Background('none', 'None', 'none'),
    Background('blur', 'Blur', 'blur', sigma=14.0),
    Background('blur_max', 'Blur+', 'blur', sigma=30.0),
    Background('grey', 'Grey', 'colour', colour=(78, 74, 70)),
    Background('slate', 'Slate', 'colour', colour=(56, 46, 38)),
    Background('navy', 'Navy', 'colour', colour=(74, 44, 26)),
    Background('teal', 'Teal', 'colour', colour=(94, 82, 34)),
    Background('forest', 'Forest', 'colour', colour=(52, 68, 40)),
    Background('white', 'White', 'colour', colour=(238, 238, 238)),
]

_BY_KEY: Dict[str, Background] = {b.key: b for b in BACKGROUNDS}


def get(key: str) -> Optional[Background]:
    """One background by key, or None if there is no such background."""
    return _BY_KEY.get(key)


def names() -> List[Dict[str, str]]:
    """Every background as {key, name}, for the picker."""
    return [{'key': b.key, 'name': b.name} for b in BACKGROUNDS]


def resolve_model() -> Tuple[Optional[SegmentationModel], str]:
    """
    Which segmentation model to use, and where its weights are.

    `PHANTOM_SEGMENT_MODEL` may name a file directly; the registry entry is then
    chosen by filename, falling back to the default profile. That fallback is
    the one lossy case here and it is called out rather than hidden — a model
    fed the wrong normalisation produces a washed-out matte, which reads as a
    weak segmenter rather than as a misconfiguration.

    Returns:
        (model, path), with path empty when nothing was found
    """
    override = os.environ.get('PHANTOM_SEGMENT_MODEL', '').strip()
    if override:
        if not os.path.isfile(override):
            return None, ''
        name = os.path.basename(override)
        for model in MODELS:
            if model.filename == name:
                return model, override
        return MODELS[0], override

    for root in _search_roots():
        for model in MODELS:
            candidate = os.path.join(root, model.filename)
            if os.path.isfile(candidate):
                return model, candidate
    return None, ''


def model_path() -> str:
    """
    Where the segmentation model was found, or '' if it was not.

    Returns:
        An existing file path, or empty
    """
    return resolve_model()[1]


class Segmenter:
    """
    The person/background matte, from one ONNX model run through `cv2.dnn`.

    One instance is shared by every renderer: the net is the only expensive
    thing here and loading it twice would double the memory for no benefit.
    `cv2.dnn.Net.forward` is **not** re-entrant, and this is called from the
    webcam thread and the display timer both, so the inference is serialised
    behind a lock. That lock is the reason the renderers can stay lock-free.
    """

    def __init__(
        self,
        path: str = '',
        model: Optional[SegmentationModel] = None,
    ) -> None:
        """
        Args:
            path: Model file, or empty to resolve it from the usual locations
            model: Its profile; resolved alongside the path when omitted
        """
        if path and model is not None:
            self._model: Optional[SegmentationModel] = model
            self._path = path
        else:
            resolved, found = resolve_model()
            self._model = model if model is not None else resolved
            self._path = path or found
        self._net: Optional[Any] = None
        self._lock = threading.Lock()
        self._loaded = False
        self._failure = ''

    @property
    def model(self) -> Optional[SegmentationModel]:
        """The profile in use, or None when nothing resolved."""
        return self._model

    @property
    def available(self) -> bool:
        """Whether a model loaded, attempting the load once if needed."""
        self._ensure()
        return self._net is not None

    @property
    def failure(self) -> str:
        """Why the model is unavailable, or '' when it is fine."""
        self._ensure()
        return self._failure

    def _ensure(self) -> None:
        """Load the net once, recording why rather than raising."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._path or self._model is None:
                self._failure = (
                    'no segmentation model found. Run '
                    '`python tools/fetch_segmentation_model.py`, or put {} in '
                    'desktop/models/, or set PHANTOM_SEGMENT_MODEL to a '
                    'file.'.format(MODELS[0].filename)
                )
                return
            try:
                self._net = cv2.dnn.readNetFromONNX(self._path)
            except Exception as exc:                     # pragma: no cover
                self._net = None
                self._failure = '{} could not be loaded: {}: {}'.format(
                    self._path, type(exc).__name__, exc)

    def matte(self, frame: Frame) -> Optional[Mask]:
        """
        Person probability for every pixel of `frame`, in [0, 1].

        Args:
            frame: BGR frame

        Returns:
            float32 mask at the frame's own size, or None when unavailable
        """
        self._ensure()
        if self._net is None or self._model is None:
            return None

        blob = self._model.blob(frame)
        try:
            with self._lock:
                self._net.setInput(blob)
                output = self._net.forward()
        except Exception as exc:                          # pragma: no cover
            self._net = None
            self._failure = 'inference failed: {}: {}'.format(
                type(exc).__name__, exc)
            return None

        return _to_mask(output, frame.shape[1], frame.shape[0])


def _to_mask(output: Any, width: int, height: int) -> Optional[Mask]:
    """
    Reduce a segmentation model's output to one person-probability plane.

    The family this is written for is published in both shapes — one channel
    carrying a sigmoid, and two carrying a softmax over (background, person) —
    so the channel count is read rather than assumed. Guessing wrong produces an
    inverted matte, which composites the room over the person and looks like a
    broken model rather than a wrong assumption.

    Args:
        output: Raw network output
        width: Target width
        height: Target height

    Returns:
        float32 mask in [0, 1] at (height, width), or None if unreadable
    """
    array = np.asarray(output, dtype=np.float32)
    array = np.squeeze(array)

    if array.ndim == 3:
        # Channels-first if the leading axis is the small one, which it is for
        # every export of this shape; a 2-channel softmax takes its person plane.
        if array.shape[0] <= 2 and array.shape[0] < array.shape[-1]:
            plane = array[-1]
        elif array.shape[-1] <= 2:
            plane = array[..., -1]
        else:
            return None
    elif array.ndim == 2:
        plane = array
    else:
        return None

    lowest = float(plane.min())
    highest = float(plane.max())
    if not np.isfinite(lowest) or not np.isfinite(highest):
        return None
    # A logit output rather than a probability: squash it rather than clipping,
    # which would turn a perfectly good matte into a hard binary one.
    if lowest < 0.0 or highest > 1.0:
        plane = 1.0 / (1.0 + np.exp(-plane))

    resized: Mask = cv2.resize(
        plane.astype(np.float32), (width, height),
        interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def shared_segmenter() -> Segmenter:
    """
    The process-wide segmenter, created on first use.

    Returns:
        The shared instance
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = Segmenter()
    return _SHARED


_SHARED: Optional[Segmenter] = None


class Renderer:
    """
    One video stream's background layer, including its temporal state.

    **There is one of these per stream, and that is the whole design.** The mask
    is smoothed across frames, and smoothing is state; `Bridge._decorate` is
    called from the webcam thread and from the display timer, over two
    genuinely different videos — the local camera and the frames coming back
    from the pod. One shared EMA would be advanced by both and would blend the
    operator's own room against their swapped self at whatever the sum of the
    two rates happened to be. That is the same failure `desktop/effects.py`
    avoids by being a function of the clock; an overlay could be made stateless
    and a matte cannot, so the state is separated by stream instead.

    A still — a saved photo — should use a throwaway instance. There is nothing
    to smooth in one frame, and it must not disturb a live stream's mask.
    """

    def __init__(self, segmenter: Optional[Segmenter] = None) -> None:
        """
        Args:
            segmenter: Matte source; the shared one by default
        """
        self._segmenter = segmenter if segmenter is not None else shared_segmenter()
        self._previous: Optional[Mask] = None
        # The feathered mask handed to the compositor, kept separately from the
        # estimate above. `_previous` must stay un-feathered or each frame would
        # blur a copy of the last one and the subject would dissolve over a few
        # seconds; this is the presentable version, and it is also what a
        # decimated frame reuses.
        self._feathered: Optional[Mask] = None
        self._tick = 0

    @property
    def available(self) -> bool:
        """Whether a segmentation model is loaded and usable."""
        return self._segmenter.available

    @property
    def failure(self) -> str:
        """Why the layer cannot run, or '' when it can."""
        return self._segmenter.failure

    def reset(self) -> None:
        """Drop the smoothed mask, so the next frame starts clean."""
        self._previous = None
        self._feathered = None
        self._tick = 0

    def render(self, frame: Frame, key: str) -> Frame:
        """
        Replace what is behind the person.

        An unknown key is not an error, for the same reason it is not one in
        `desktop/filters.py`: a background removed in a later version must
        degrade to no background rather than failing a live call.

        Args:
            frame: BGR frame, already swapped
            key: Background key, possibly empty or unknown

        Returns:
            The frame with its background replaced, or the input unchanged
        """
        if not key or key == 'none':
            self._previous = None
            return frame

        chosen = _BY_KEY.get(key)
        if chosen is None or chosen.kind == 'none':
            self._previous = None
            return frame

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        matte = self._matte_for(frame)
        if matte is None:
            return frame

        replacement = chosen.build(frame)
        if replacement is None:
            return frame

        return _composite(frame, replacement, matte)

    def _matte_for(self, frame: Frame) -> Optional[Mask]:
        """
        The mask to composite with, recomputing it only when it is due.

        Two things force a recompute regardless of the schedule: having no
        cached mask at all, and a cached one of a different size. The second is
        not hypothetical — a quality preset change resizes the stream mid-call,
        and reusing a mask of the previous shape would either raise or, worse,
        broadcast into a plausible-looking wrong answer.

        Args:
            frame: BGR frame

        Returns:
            A feathered mask at the frame's size, or None when unavailable
        """
        height, width = frame.shape[:2]
        stale = (self._feathered is None
                 or self._feathered.shape != (height, width))

        if stale or self._tick % _MATTE_INTERVAL == 0:
            matte = self._segmenter.matte(frame)
            if matte is None:
                # **Give up rather than hold the last mask.** Reusing it looks
                # like the cheaper option and is the worse one: `Segmenter`
                # nulls its net on any inference failure, so a failure here is
                # permanent, and holding would composite a frozen silhouette
                # over moving video for the rest of the session. There is no
                # transient failure mode to ride out.
                self.reset()
                return None
            self._feathered = self._smooth(matte)

        self._tick += 1
        return self._feathered

    def _smooth(self, matte: Mask) -> Mask:
        """
        Blend this frame's matte with the last one, and feather the result.

        Args:
            matte: This frame's raw person probability

        Returns:
            The smoothed, feathered mask
        """
        previous = self._previous
        if (previous is not None and _RESET_ON_SHAPE_CHANGE
                and previous.shape != matte.shape):
            previous = None

        if previous is None:
            blended = matte
        else:
            blended = cv2.addWeighted(
                previous, _MASK_ALPHA, matte, 1.0 - _MASK_ALPHA, 0.0)

        self._previous = blended

        # Feathered after storing, so the stored mask is the estimate rather
        # than an increasingly-blurred copy of itself. Feathering the state
        # would compound the blur every frame and dissolve the subject.
        shorter = min(blended.shape[0], blended.shape[1])
        radius = max(1.0, shorter * _FEATHER_FRACTION)
        feathered: Mask = cv2.GaussianBlur(blended, (0, 0), radius)
        return feathered


def _composite(frame: Frame, background: Frame, matte: Mask) -> Frame:
    """
    Mix the person over the replacement background.

    **Integer throughout, and that is worth 9ms a frame.** The obvious version
    converts both pictures to float32, multiplies by a float alpha and adds:
    seven full-frame float passes, measured at 12.3ms on a 640x360 frame, which
    is more than a third of a 33ms display tick spent on a blend. `cv2.multiply`
    takes uint8 inputs with a scale factor and stays in 8-bit SIMD the whole
    way — 3.5ms for the same result, with a maximum difference of 2/255 from
    rounding, which is invisible and lands inside the grain the compositor
    already added upstream.

    cv2 ops rather than numpy broadcasting for the reason `_add_grain` in the
    pipeline's compositor moved to them: this runs on every frame on the
    operator's machine, which is not a fast one.

    Args:
        frame: BGR frame carrying the person
        background: BGR frame to put behind them
        matte: Person probability in [0, 1]

    Returns:
        The composited BGR frame, same dtype as the input
    """
    alpha = cv2.merge([matte, matte, matte])
    alpha8 = np.empty(alpha.shape, dtype=np.uint8)
    cv2.convertScaleAbs(alpha, dst=alpha8, alpha=255.0)

    person = cv2.multiply(frame, alpha8, scale=1.0 / 255.0)
    room = cv2.multiply(background, cv2.bitwise_not(alpha8), scale=1.0 / 255.0)
    out: Frame = cv2.add(person, room)
    return out
