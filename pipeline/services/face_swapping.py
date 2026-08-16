"""
Face swapping service for the Phantom pipeline.

Extracted from pipeline/processors/frame/face_swapper.py.
Provides ONNX-based face swapping without global state.
"""

import os
import threading
from typing import Any, Optional, Tuple

import insightface

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face
from pipeline.logging import emit_status, emit_error


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

                    import onnxruntime as ort
                    session_options = ort.SessionOptions()
                    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                    session_options.intra_op_num_threads = 4

                    self._swapper = insightface.model_zoo.get_model(
                        model_path,
                        providers=self.config.execution_providers,
                    )
        return self._swapper

    def _resolve_model_path(self) -> str:
        """
        Resolve the model path, checking known locations in priority order.

        Priority:
        1. RunPod Network Volume (/workspace/models/)
        2. Relative to repo root (models/)
        3. Working directory fallback

        Returns:
            Full path to inswapper_128.onnx model
        """
        # RunPod Network Volume (highest priority) — use if volume dir exists,
        # even when the file hasn't been downloaded yet (pre_check will create it here)
        runpod_model = '/workspace/models/inswapper_128.onnx'
        if os.path.isdir('/workspace/models'):
            return runpod_model

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
        """Clear the cached model (useful for memory cleanup)."""
        with self._lock:
            self._swapper = None
