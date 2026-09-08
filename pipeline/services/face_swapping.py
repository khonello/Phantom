"""
Face swapping service for the Phantom pipeline.

Three inference families, selected by `config.swapper_model` and described in
`pipeline/services/swapper_models.py`:

- **inswapper** — run through InsightFace's own `INSwapper`, which owns the
  alignment, the `emap` projection of the source embedding, and the crop. This
  is the incumbent path and is untouched.
- **hyperswap** — 256px native, run on a plain onnxruntime session here. Same
  ArcFace alignment template and the same *embedding* source contract, which is
  the whole reason it can be swapped in without the compositor, masker or guards
  changing.
- **hififace** — 256px native, trained with a 3DMM in the loop so the generated
  face follows the source's *contour* and not only its interior. Two departures
  from the others, both handled here rather than downstream: it aligns to
  `mtcnn_512` rather than arcface, and its identity vector goes through a small
  learned converter into the recognition space it was trained against.

All three return the aligned crop plus the affine that produced it, so
compositing stays where it belongs.

**One lever cuts across all of them: `identity_push`.** Every model lands
somewhere between the source and the target — that compromise is what "the
resemblance is good, but not very good" describes — and the vector each is
conditioned on can be moved along the line joining the two identities, away
from the target. See `_push`.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import insightface
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face
from pipeline.processing import geometry
from pipeline.services import swapper_models
from pipeline.logging import emit_status, emit_error, emit_warning

# The five-point templates now live in `processing/geometry.py`, beside FFHQ and
# the detail band, because three stages need to know what space a crop is in and
# only one of them makes the crop: the swapper builds it, `IdentityProbe`
# re-frames it for recognition, and the shape mask reads the eye line out of it.
#
# `_ARCFACE_TEMPLATE` is kept as the name this module has always used. It is
# exactly InsightFace's `arcface_dst` shifted by +8px in x and divided by 128 —
# the transform its own `estimate_norm` applies for a 128px crop — and equal to
# facefusion's `arcface_128` to eight decimal places, which is why inswapper and
# hyperswap produce crops in the same space.
_ARCFACE_TEMPLATE = geometry.ARCFACE_128_TEMPLATE

# Ceiling on `identity_push`. Past roughly this the conditioning vector leaves
# the region of the embedding space the generators were trained on, and the
# output degrades toward a face that is nobody rather than one that is more the
# source. Clamped rather than rejected, like every other live-swept knob.
PUSH_MAX = 0.6


@dataclass(frozen=True)
class _SourceIdentity:
    """
    A stand-in source face carrying a modified identity vector.

    Only what the models actually read: InSwapper consults `normed_embedding`
    and nothing else, and the converter path wants the raw vector's magnitude.
    Deliberately not a real `Face` — it must not be mistaken for a detection,
    and it has no landmarks, no box and no pose to offer.
    """

    normed_embedding: Any
    embedding: Any = None


class FaceSwapper:
    """
    Face swapping using InsightFace's inswapper model.

    This service is thread-safe and maintains an internal cache of the
    swap model. Configuration and model path are specified via constructor.

    Example:
        swapper = FaceSwapper(CONFIG)
        if swapper.pre_check():
            result_frame = swapper.swap(source_face, target_detection, frame)
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize the face swapper.

        Args:
            config: FaceSwapConfig with execution_providers and model_path
        """
        self.config = config
        self._swapper: Optional[Any] = None
        # Plain onnxruntime session for the non-inswapper families, cached
        # alongside the input names so they are introspected once rather than
        # assumed — the same approach masking.py takes with the occluder.
        self._session: Optional[Any] = None
        self._session_model: str = ''
        self._source_input: str = 'source'
        self._target_input: str = 'target'
        self._lock = threading.Lock()

        # The embedding converter some models need, and its results. The
        # conversion is keyed on the vector rather than on the source, so a
        # push that moves the vector correctly invalidates it.
        self._converter: Optional[Any] = None
        self._converter_model: str = ''
        self._converted: Dict[Any, Any] = {}

        # Said once, when a converter model is handed a unit vector because the
        # source has no raw embedding to rescale.
        self._warned_unit_source = False
        # Set once if this InsightFace build cannot return the unpasted swap,
        # so the fallback warning is not repeated on every frame.
        self._aligned_unsupported = False

    def _get_swapper(self) -> Any:
        """
        Get or create the face swap model (lazy initialization).

        Thread-safe. Model is cached after first access.

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If ONNX Runtime can't load the model
        """
        if self._swapper is None:
            with self._lock:
                if self._swapper is None:
                    model_path = self._resolve_model_path()
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model not found: {model_path}")

                    self._swapper = self._build_inswapper(model_path)
        return self._swapper

    def _build_inswapper(self, model_path: str) -> Any:
        """
        Load inswapper, giving it a session this pipeline configured.

        InsightFace's `get_model` builds its own `InferenceSession` and accepts
        only `providers` and `provider_options` — no `sess_options`, and no way
        to point at converted fp16 weights. That is how this code came to build
        a `SessionOptions`, set two fields on it and never pass it anywhere.

        `INSwapper` takes a prepared session, so we build one through the shared
        factory and hand it over. Note `model_file` stays the **fp32** path even
        when the session runs fp16: INSwapper reads the `emap` projection out of
        that file, and the source embedding is projected on the CPU in float32
        before it ever reaches the model.

        Falls back to InsightFace's own loader if anything about that shape has
        changed between versions — a swapper that loads the old way is much
        better than a session that does not load at all.

        Args:
            model_path: Path to the fp32 inswapper weights

        Returns:
            A live INSwapper
        """
        try:
            from insightface.model_zoo.inswapper import INSwapper
            from pipeline.services.onnx_session import create_session

            # Static shapes, but deliberately not `bound`: INSwapper calls
            # `session.run` itself, so there is no binding to capture a CUDA
            # graph against and asking for one fails at inference.
            session = create_session(
                self.config, model_path, 'inswapper', static_shapes=True,
            )
            return INSwapper(model_file=model_path, session=session)
        except Exception as e:
            emit_warning(
                f'Falling back to InsightFace session construction for the '
                f'swapper ({type(e).__name__}: {e}). fp16, CUDA graphs and '
                f'TensorRT do not apply to it.',
                scope='SWAPPER',
            )
            return insightface.model_zoo.get_model(
                model_path,
                providers=self.config.execution_providers,
            )

    def model(self) -> 'swapper_models.SwapperModel':
        """
        The registry entry for the configured model.

        Returns:
            Spec plus realism profile; falls back to the default on an unknown
            name rather than failing a session that has already been paid for
        """
        return swapper_models.resolve(self.config.swapper_model)

    # ------------------------------------------------------------------
    # The source vector: conversion, and pushing it away from the target
    # ------------------------------------------------------------------

    def identity_push(self) -> float:
        """
        How far to extrapolate the source identity away from the target's.

        Returns:
            The configured push, clamped to [0, PUSH_MAX]
        """
        value = float(getattr(self.config, 'identity_push', 0.0) or 0.0)
        return min(PUSH_MAX, max(0.0, value))

    def _push(self, normed: Any, target: Face) -> Any:
        """
        Move a source embedding directly away from the target's identity.

        Every swap model here is conditioned on an identity vector and every one
        of them lands somewhere *between* the source and the target — that is
        what "the resemblance is good but not very good" is, stated in the
        model's own terms. The generator's compromise is not reachable from
        outside, but the point it is asked to render is:

            pushed = normalise( source * (1 + k) - target * k )

        which is the source, moved along the line joining them, in the direction
        away from the target. At `k = 0` it is the source exactly and this is a
        no-op. It is the same lever facefusion exposes as `face_swapper_weight`,
        which interpolates over [0.35, -0.35] — the negative half of that range
        is this, and nothing here has ever used it.

        Two properties worth keeping. It is applied in **ArcFace space**, before
        any model-specific conversion, so the two vectors being combined are
        always from the same space — mixing a converted source with an
        unconverted target would be arithmetic on incommensurable things. And it
        is bounded: past roughly 0.5 the vector leaves the region of the
        embedding space the generator was trained on, and output degrades into
        a face that is nobody rather than a face that is more the source.

        Args:
            normed: L2-normalised source embedding, in ArcFace space
            target: The target face, for its own embedding

        Returns:
            A normalised, pushed embedding — or `normed` unchanged when the
            push is off or the target carries no embedding to push away from
        """
        push = self.identity_push()
        if push <= 0.0 or normed is None:
            return normed

        against = getattr(target, 'normed_embedding', None)
        if against is None:
            return normed

        source_vector = np.asarray(normed, dtype=np.float32).ravel()
        target_vector = np.asarray(against, dtype=np.float32).ravel()
        if source_vector.shape != target_vector.shape:
            return normed

        pushed = source_vector * (1.0 + push) - target_vector * push
        norm = float(np.linalg.norm(pushed))
        if norm < 1e-8:
            return normed

        return pushed / norm

    def pushed_source(self, source: Face, target: Face) -> Face:
        """
        The source face as the model should see it, with the push applied.

        A stand-in rather than a mutation: `source` is the cached identity for
        the whole session and the push depends on *this* target, so writing to
        it would make the next frame's push compound on the last one's.

        Args:
            source: Source face carrying the identity
            target: The face being swapped onto

        Returns:
            `source` itself when the push is off, else a light stand-in
            carrying the pushed vector. Only `normed_embedding` is consulted by
            InSwapper, which is the one consumer that receives a face rather
            than a vector.
        """
        if self.identity_push() <= 0.0:
            return source

        normed = getattr(source, 'normed_embedding', None)
        pushed = self._push(normed, target)
        if pushed is normed:
            return source

        return _SourceIdentity(
            normed_embedding=pushed,
            embedding=self._rescale(getattr(source, 'embedding', None), pushed),
        )

    @staticmethod
    def _rescale(raw: Any, direction: Any) -> Any:
        """
        Point a raw embedding along a new direction, keeping its magnitude.

        Identity is the *direction*; the norm is a magnitude the recognition
        model produced and that a converter may have been fitted against. So a
        push rotates and never rescales.

        Args:
            raw: The original unnormalised embedding, or None
            direction: Unit vector to point it along

        Returns:
            The rescaled vector, or None if there was no raw embedding
        """
        if raw is None:
            return None

        vector = np.asarray(raw, dtype=np.float32).ravel()
        return np.asarray(direction, dtype=np.float32).ravel() * float(
            np.linalg.norm(vector))

    def source_vector(
        self,
        model: 'swapper_models.SwapperModel',
        source: Face,
        target: Face,
    ) -> Optional[Any]:
        """
        The 512-d vector to condition a session-run model on.

        Three steps, in this order and for this reason: push in ArcFace space
        (both vectors from one space), convert into the model's own space if it
        needs one, then normalise. Converting first would leave the push mixing
        a converted source with an unconverted target.

        Args:
            model: Registry entry, which knows whether a converter is needed
            source: Source face
            target: Target face, for the push

        Returns:
            A float32 vector, or None when the source carries no embedding
        """
        normed = getattr(source, 'normed_embedding', None)
        if normed is None:
            emit_error(
                f'{model.name} needs a normalised source embedding and the '
                f'source face has none.',
                scope='SWAPPER',
            )
            return None

        normed = self._push(normed, target)

        if not model.converter_filename:
            return np.asarray(normed, dtype=np.float32)

        if model.converter_takes_raw:
            raw = getattr(source, 'embedding', None)
            if raw is None:
                # A `.npy` source, or an averaged one that dropped the raw
                # vector. Said once: the converter still produces a usable
                # identity from a unit vector, but it was fitted on vectors of
                # norm ~20 and this is a quieter, blander identity rather than
                # a failure — exactly the kind of degradation that gets blamed
                # on the model.
                if not self._warned_unit_source:
                    self._warned_unit_source = True
                    emit_warning(
                        f'{model.name} converts a raw ArcFace embedding and '
                        f'this source has only a normalised one (a .npy '
                        f'source, or an average taken before this mattered). '
                        f'Identity will be weaker than from the photographs.',
                        scope='SWAPPER',
                    )
                vector = np.asarray(normed, dtype=np.float32)
            else:
                vector = np.asarray(
                    self._rescale(raw, normed), dtype=np.float32)
        else:
            vector = np.asarray(normed, dtype=np.float32)

        converted = self._convert(model, vector)
        if converted is None:
            return None

        norm = float(np.linalg.norm(converted))
        if norm < 1e-8:
            return None

        return (converted / norm).astype(np.float32)

    def _convert(
        self,
        model: 'swapper_models.SwapperModel',
        vector: Any,
    ) -> Optional[Any]:
        """
        Map an ArcFace vector into the model's own recognition space.

        Cached on the vector's bytes, so this runs once per source rather than
        once per frame — the input only changes when the operator changes their
        photographs, or when the push moves it, and both are rare.

        Args:
            model: Registry entry naming the converter
            vector: The ArcFace-space vector to convert

        Returns:
            The converted vector, or None if the converter could not be loaded
        """
        array = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        key = (model.name, array.tobytes())
        cached = self._converted.get(key)
        if cached is not None:
            return cached

        session = self._get_converter(model)
        if session is None:
            return None

        try:
            name = session.get_inputs()[0].name
            result = session.run(None, {name: array})[0]
        except Exception as e:
            emit_error(
                f'{model.converter_filename} failed: {type(e).__name__}: {e}',
                exception=e, scope='SWAPPER',
            )
            return None

        converted = np.asarray(result, dtype=np.float32).ravel()

        # One entry per distinct source. Bounded because a session has one
        # source at a time and the push rarely moves; cleared with the model.
        if len(self._converted) > 8:
            self._converted.clear()
        self._converted[key] = converted
        return converted

    def _get_converter(
        self,
        model: 'swapper_models.SwapperModel',
    ) -> Optional[Any]:
        """
        Load (downloading if needed) the embedding converter for a model.

        Args:
            model: Registry entry naming the converter

        Returns:
            An InferenceSession, or None if the weights could not be obtained
        """
        if self._converter is not None and self._converter_model == model.name:
            return self._converter

        with self._lock:
            if self._converter is not None and self._converter_model == model.name:
                return self._converter

            path = self._resolve_named_model(model.converter_filename)
            if not os.path.isfile(path) and not self._fetch(
                model.converter_url, model.converter_filename,
                swapper_models.HIFIFACE_CONVERTER_SIZE_BYTES, path,
            ):
                return None

            try:
                from pipeline.services.onnx_session import create_session

                session = create_session(
                    self.config, path, f'{model.name}_converter',
                    static_shapes=True,
                )
            except Exception as e:
                emit_error(
                    f'Failed to load {model.converter_filename}: '
                    f'{type(e).__name__}: {e}',
                    exception=e, scope='SWAPPER',
                )
                return None

            self._converter = session
            self._converter_model = model.name
            return session

    # ------------------------------------------------------------------
    # Non-inswapper families (hyperswap, hififace): plain onnxruntime
    # ------------------------------------------------------------------

    def _get_session(self, model: 'swapper_models.SwapperModel') -> Optional[Any]:
        """
        Load (downloading if needed) the ONNX session for a non-inswapper model.

        Args:
            model: Registry entry to load

        Returns:
            An InferenceSession, or None if the weights could not be obtained
        """
        if self._session is not None and self._session_model == model.name:
            return self._session

        with self._lock:
            if self._session is not None and self._session_model == model.name:
                return self._session

            path = self._resolve_named_model(model.filename)
            if not os.path.isfile(path) and not self._download(model, path):
                return None

            try:
                from pipeline.services.onnx_session import create_session

                # Static shapes: the target crop is always `model.size` square
                # and the source is a 512-d embedding.
                session = create_session(
                    self.config, path, model.name, static_shapes=True,
                )

                # Introspected rather than assumed: exports differ in what they
                # name these, and a wrong key is a KeyError on every frame.
                names = [i.name for i in session.get_inputs()]
                self._source_input = next(
                    (n for n in names if 'source' in n.lower() or 'emb' in n.lower()),
                    names[0],
                )
                self._target_input = next(
                    (n for n in names if n != self._source_input), names[-1],
                )

                self._session = session
                self._session_model = model.name
                emit_status(
                    f'Swapper: {model.name} ({model.size}px native, inputs '
                    f'{self._source_input}/{self._target_input})',
                    scope='SWAPPER',
                )
                return session
            except Exception as e:
                emit_error(
                    f'Failed to load {model.name}: {type(e).__name__}: {e}',
                    exception=e, scope='SWAPPER',
                )
                return None

    @staticmethod
    def _download(model: 'swapper_models.SwapperModel', path: str) -> bool:
        """
        Fetch a model's weights.

        Args:
            model: Registry entry, supplying the URL
            path: Where the file should end up

        Returns:
            True if the file is present afterwards
        """
        return FaceSwapper._fetch(
            model.url, model.filename, model.size_bytes, path,
        )

    @staticmethod
    def _fetch(url: str, filename: str, size_bytes: int, path: str) -> bool:
        """
        Fetch one weight file.

        Separate from `_download` because a model can need more than one: an
        embedding converter is a second file with its own URL, and the size in
        the message is the model's own rather than a constant that happened to
        be right while every registered model was a hyperswap.

        Args:
            url: Where to fetch it from
            filename: For the log line
            size_bytes: Expected size, for the log line. 0 if unknown
            path: Where the file should end up

        Returns:
            True if the file is present afterwards
        """
        if not url:
            return False

        size = (' (~{} MB)'.format(size_bytes // (1024 * 1024))
                if size_bytes else '')
        emit_status(f'Downloading {filename}{size}...', scope='SWAPPER')
        try:
            from pipeline.io.ffmpeg import conditional_download

            os.makedirs(os.path.dirname(path), exist_ok=True)
            conditional_download(os.path.dirname(path), [url])
        except Exception as e:
            emit_error(
                f'Download failed for {filename}: {type(e).__name__}: {e}',
                exception=e, scope='SWAPPER',
            )
            return False

        return os.path.isfile(path)

    def _swap_session(
        self,
        model: 'swapper_models.SwapperModel',
        source: Face,
        target: Face,
        frame: Frame,
    ) -> Optional[Tuple[Frame, Any]]:
        """
        Run a non-inswapper model, returning the aligned crop and its affine.

        The source is the **L2-normalised embedding vector**. Worth naming
        precisely, because two libraries disagree about the word: facefusion's
        `embedding_norm` is that normalised 512-d vector, while InsightFace's
        attribute of the same name is a *scalar* magnitude and its vector is
        called `normed_embedding`. Passing the scalar would produce garbage
        rather than an error, so this reads the InsightFace name deliberately.

        Unlike inswapper there is no `emap` projection — the normalised
        embedding is fed straight in.

        Args:
            model: Registry entry
            source: Source face carrying the embedding
            target: Target face whose `kps` drive the alignment
            frame: Frame to sample the crop from

        Returns:
            (crop, matrix) in the same convention InsightFace returns, or None
        """
        session = self._get_session(model)
        if session is None:
            return None

        embedding = self.source_vector(model, source, target)
        if embedding is None:
            return None

        template = geometry.alignment_template(model.template)

        kps = getattr(target, 'kps', None)
        if kps is None or len(kps) != len(template):
            return None

        # Umeyama rather than cv2.estimateAffinePartial2D, which is what
        # facefusion uses here. The OpenCV estimators are randomized, and
        # anything that varies frame to frame feeds straight into the shimmer
        # the compositor exists to remove — same reasoning as
        # compositor.estimate_similarity, and the same function.
        from pipeline.processing.compositor import estimate_similarity

        matrix = estimate_similarity(
            np.asarray(kps, dtype=np.float64),
            template * model.size,
        )
        if matrix is None:
            return None

        matrix = matrix.astype(np.float32)
        crop = cv2.warpAffine(
            frame, matrix, (model.size, model.size),
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA,
        )

        mean = np.array(model.mean, dtype=np.float32)
        deviation = np.array(model.standard_deviation, dtype=np.float32)

        blob = crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = (blob - mean) / deviation
        blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)

        try:
            output = session.run(None, {
                self._source_input: np.asarray(
                    embedding, dtype=np.float32,
                ).reshape(1, -1),
                self._target_input: blob,
            })[0][0]
        except Exception as e:
            emit_error(
                f'{model.name} inference failed: {type(e).__name__}: {e}',
                exception=e, scope='SWAPPER',
            )
            return None

        result = output.transpose(1, 2, 0) * deviation + mean
        result = np.clip(result, 0.0, 1.0)[:, :, ::-1] * 255.0
        return result.astype(np.uint8), matrix

    def _resolve_named_model(self, filename: str) -> str:
        """
        Resolve a model filename against the same search path as the swapper.

        Args:
            filename: Weight filename

        Returns:
            Absolute path, which may not exist yet
        """
        if os.path.isdir('/workspace/models'):
            return os.path.join('/workspace/models', filename)

        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_dir, 'models', filename)

    def _resolve_model_path(self) -> str:
        """
        Resolve the model path, checking known locations in priority order.

        Priority:
        1. /workspace/models/ on the instance disk
        2. Relative to repo root (models/)
        3. Working directory fallback

        Returns:
            Full path to inswapper_128.onnx model
        """
        # /workspace/models (highest priority) — use if the directory exists,
        # even when the file hasn't been downloaded yet (pre_check will create it here)
        workspace_model = '/workspace/models/inswapper_128.onnx'
        if os.path.isdir('/workspace/models'):
            return workspace_model

        # Relative to repo root (pipeline package lives one level down)
        pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_model = os.path.join(pipeline_dir, 'models', 'inswapper_128.onnx')
        if os.path.exists(relative_model):
            return relative_model

        # Fall back to working directory
        return os.path.join(os.getcwd(), 'models', 'inswapper_128.onnx')

    def swap(self, source: Face, target: Face, frame: Frame) -> Frame:
        """
        Swap a face in a frame, using InsightFace's own compositing.

        Fallback path — prefer `swap_aligned` so compositing can be done
        properly. Kept for InsightFace builds that do not return the affine.

        Args:
            source: Source face to swap from
            target: Target face from the current frame
            frame: Frame to swap in

        Returns:
            Frame with swapped face

        Raises:
            FileNotFoundError: If model not found
            RuntimeError: If swap fails
        """
        model = self.model()
        if model.kind != 'inswapper':
            # These have no pasted form. `swap_aligned` is their only path, and
            # the caller treats None as "no swap this frame" rather than pasting
            # the raw frame — which on the live path would be the operator's own
            # face. See ProcessingPipeline._swap_face.
            emit_error(
                f'{model.name} has no pasted fallback; aligned swap is required.',
                scope='SWAPPER',
            )
            return frame

        try:
            swapper = self._get_swapper()
            return swapper.get(
                frame, target, self.pushed_source(source, target),
                paste_back=True,
            )
        except Exception as e:
            emit_error(f"Face swap failed: {e}", exception=e, scope='SWAPPER')
            return frame

    def swap_aligned(
        self,
        source: Face,
        target: Face,
        frame: Frame,
    ) -> Optional[Tuple[Frame, Any]]:
        """
        Swap a face and return the raw aligned crop instead of a pasted frame.

        `paste_back=False` hands back the generated crop together with the
        affine that produced it, which lets the caller own compositing —
        masking, colour, detail and grain all work far better in aligned
        space than they do after the model has already pasted.

        Args:
            source: Source face to swap from
            target: Target face (fresh detection — its `kps` drives the warp)
            frame: Frame to swap in

        Returns:
            (aligned_crop, matrix) where matrix is the 2x3 affine mapping
            frame space to the crop, or None if this InsightFace build does
            not support the unpasted form. Callers should fall back to
            `swap()` in that case.
        """
        model = self.model()

        # Non-inswapper families run on our own session: InsightFace's INSwapper
        # knows inswapper's emap projection and 128px crop specifically, so it
        # cannot host them.
        if model.kind != 'inswapper':
            return self._swap_session(model, source, target, frame)

        if self._aligned_unsupported:
            return None

        try:
            swapper = self._get_swapper()
            result = swapper.get(
                frame, target, self.pushed_source(source, target),
                paste_back=False,
            )
        except Exception as e:
            emit_error(f"Face swap failed: {e}", exception=e, scope='SWAPPER')
            return None

        # Guard the return shape rather than assuming it: older and patched
        # InsightFace builds have returned a bare frame here.
        crop, matrix = (result if isinstance(result, tuple) and len(result) == 2
                        else (None, None))

        if (
            crop is None
            or getattr(crop, 'ndim', 0) != 3
            or getattr(matrix, 'shape', None) != (2, 3)
        ):
            self._aligned_unsupported = True
            emit_status(
                'InsightFace did not return an affine for the unpasted swap — '
                'falling back to its built-in compositing. Masking, colour '
                'matching and grain will be unavailable.',
                scope='SWAPPER',
                level='warning',
            )
            return None

        return crop, matrix

    def pre_check(self) -> bool:
        """
        Check if model is available and prompt for download if needed.

        Returns:
            True if model is ready, False if user canceled or download failed
        """
        model_path = self._resolve_model_path()
        model_dir = os.path.dirname(model_path)

        if os.path.exists(model_path):
            emit_status(f'Model found: {os.path.basename(model_path)}', scope='SWAPPER')
            return True

        emit_status(f'Model not found: {os.path.basename(model_path)}', scope='SWAPPER')

        # Create models directory if needed
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Auto-download — no prompt; model is required for the app to function
        hf_url = (
            'https://huggingface.co/xingren23/comfyflow-models/resolve/'
            '976de8449674de379b02c144d0b3cfa2b61482f2/insightface/inswapper_128.onnx'
            '?download=true'
        )

        emit_status('Downloading inswapper_128.onnx from Hugging Face...', scope='SWAPPER')
        try:
            from pipeline.io.ffmpeg import conditional_download
            conditional_download(model_dir, [hf_url])
            if os.path.exists(model_path):
                emit_status('Model downloaded successfully.', scope='SWAPPER')
                return True
        except Exception as e:
            emit_error(f"Model download failed: {e}", exception=e, scope='SWAPPER')

        emit_status(
            'Auto-download failed. Download inswapper_128.onnx manually from: '
            'https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view '
            f'and place it at: {model_path}',
            scope='SWAPPER',
            level='warning',
        )
        return False

    def clear(self) -> None:
        """Clear the cached models (useful for memory cleanup)."""
        with self._lock:
            self._swapper = None
            self._session = None
            self._session_model = ''
            self._converter = None
            self._converter_model = ''
            self._converted.clear()
