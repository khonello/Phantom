"""
Face swapping service for the Phantom pipeline.

Two inference families, selected by `config.swapper_model` and described in
`pipeline/services/swapper_models.py`:

- **inswapper** — run through InsightFace's own `INSwapper`, which owns the
  alignment, the `emap` projection of the source embedding, and the crop. This
  is the incumbent path and is untouched.
- **hyperswap** — 256px native, run on a plain onnxruntime session here. Same
  ArcFace alignment template and the same *embedding* source contract, which is
  the whole reason it can be swapped in without the compositor, masker or guards
  changing.

Both return the aligned crop plus the affine that produced it, so compositing
stays where it belongs.
"""

import os
import threading
from typing import Any, Optional, Tuple

import cv2
import insightface
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face
from pipeline.services import swapper_models
from pipeline.logging import emit_status, emit_error, emit_warning

# ArcFace 5-point template, normalised to [0, 1]. Multiplied by the crop size at
# use, so one constant serves 128 and 256 alike.
#
# This is exactly InsightFace's `arcface_dst` shifted by +8px in x and divided by
# 128 — the transform its own `estimate_norm` applies for a 128px crop. Verified
# equal to facefusion's `arcface_128` to eight decimal places, which is why the
# two model families produce crops in the same space and the affine convention
# carries over unchanged.
_ARCFACE_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float64)


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
    # Non-inswapper families (hyperswap): plain onnxruntime
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
        if not model.url:
            return False

        megabytes = swapper_models.HYPERSWAP_SIZE_BYTES // (1024 * 1024)
        emit_status(
            f'Downloading {model.filename} (~{megabytes} MB)...',
            scope='SWAPPER',
        )
        try:
            from pipeline.io.ffmpeg import conditional_download

            os.makedirs(os.path.dirname(path), exist_ok=True)
            conditional_download(os.path.dirname(path), [model.url])
        except Exception as e:
            emit_error(
                f'Download failed for {model.filename}: {type(e).__name__}: {e}',
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

        embedding = getattr(source, 'normed_embedding', None)
        if embedding is None:
            emit_error(
                f'{model.name} needs a normalised source embedding and the '
                f'source face has none.',
                scope='SWAPPER',
            )
            return None

        kps = getattr(target, 'kps', None)
        if kps is None or len(kps) != len(_ARCFACE_TEMPLATE):
            return None

        # Umeyama rather than cv2.estimateAffinePartial2D, which is what
        # facefusion uses here. The OpenCV estimators are randomized, and
        # anything that varies frame to frame feeds straight into the shimmer
        # the compositor exists to remove — same reasoning as
        # compositor.estimate_similarity, and the same function.
        from pipeline.processing.compositor import estimate_similarity

        matrix = estimate_similarity(
            np.asarray(kps, dtype=np.float64),
            _ARCFACE_TEMPLATE * model.size,
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
            return swapper.get(frame, target, source, paste_back=True)
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
            result = swapper.get(frame, target, source, paste_back=False)
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
