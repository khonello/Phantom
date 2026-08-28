"""
Face mask construction for the Phantom pipeline.

Produces the soft alpha mask used to composite a swapped face back into the
frame. The mask is built in *aligned* face space (the same normalized crop the
swapper operates in), which is what makes it correct under head rotation.

Three terms are multiplied together:

1. **Landmark hull** — convex hull of the 106 landmarks InsightFace already
   computes, so this costs nothing extra. Follows the real jawline instead of
   assuming an ellipse.
2. **Valid region** — the part of the aligned crop that actually sampled real
   pixels. Faces near the frame edge sample outside it; without this term the
   black border bleeds into the composite.
3. **Occlusion** — optional DFL XSeg segmentation, so hands, microphones and
   hair crossing the face are not painted over with swapped skin.
"""

import os
import threading
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face, Mask, Matrix
from pipeline.services import guards
from pipeline.logging import emit_status, emit_warning

# DFL XSeg, distributed by the facefusion project. Segments the face area
# with obstructions removed; output is an *inclusion* mask (1 = swap here).
_OCCLUDER_MODEL_NAME = 'dfl_xseg.onnx'
_OCCLUDER_MODEL_URL = (
    'https://github.com/facefusion/facefusion-assets/releases/download/'
    'models-3.0.0/dfl_xseg.onnx'
)


class FaceMasker:
    """
    Builds soft compositing masks in aligned face space.

    The occluder model is optional: if it is missing and cannot be downloaded,
    masking degrades to hull + valid-region and the pipeline keeps running.

    Example:
        masker = FaceMasker(CONFIG)
        mask = masker.build(face, matrix, aligned_crop, frame.shape[:2])
    """

    # Radial expansion of the landmark hull. The 106 points stop at the
    # eyebrows and hug the jaw, so a bare hull clips the swap.
    _HULL_EXPAND = 0.10
    # Extra upward push, as a fraction of hull height, to recover forehead.
    # Kept moderate on purpose — masking below the hairline hides the
    # hairline seam, which is a far worse tell than a slightly short forehead.
    _FOREHEAD_PUSH = 0.16
    # Erosion of the valid-pixel region, in pixels at 256.
    _VALID_ERODE = 5
    # Feather width as a fraction of crop size.
    _FEATHER = 0.05

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize the masker.

        Args:
            config: Configuration object (execution_providers, occluder toggle)
        """
        self.config = config
        self._session: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._input_name: str = 'input'
        self._input_size: int = 256
        self._input_nchw: bool = False
        self._occluder_ready = False
        self._occluder_attempted = False
        self._lock = threading.Lock()
        self._white: Optional[Frame] = None
        self._white_shape: Tuple[int, int] = (0, 0)
        # Fraction of the landmark hull the occluder left in, from the most
        # recent `build()`. None when occlusion was not evaluated — the model is
        # missing, or the occluder is switched off. Read by the occlusion guard,
        # which cannot compute it itself without a second XSeg inference.
        self.last_coverage: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        face: Face,
        matrix: Matrix,
        aligned: Frame,
        frame_shape: Tuple[int, int],
    ) -> Mask:
        """
        Build the compositing mask for one face.

        Args:
            face: Detection whose landmarks define the face shape
            matrix: 2x3 affine mapping frame space -> aligned space
            aligned: The aligned crop sampled from the frame (HxWx3)
            frame_shape: (height, width) of the source frame

        Returns:
            float32 mask in [0, 1], same height/width as `aligned`
        """
        size = aligned.shape[0]

        hull = self._hull_mask(face, matrix, size)
        mask: Mask = hull * self._valid_mask(matrix, size, frame_shape)

        self.last_coverage = None
        if self.config.occluder:
            occlusion = self._occlusion_mask(aligned)
            if occlusion is not None:
                # Measured against the hull alone, not the hull times the valid
                # region: a face at the frame edge is cropped, not occluded, and
                # including that term would guard it for the wrong reason.
                self.last_coverage = guards.hull_coverage(hull, occlusion)
                mask *= occlusion

        # Feather. Blurring a hard-edged mask here plus a second, smaller blur
        # in frame space (applied by the compositor) is what keeps the seam
        # smooth regardless of how large the face is in the frame.
        ksize = int(size * self._FEATHER) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return np.clip(mask, 0.0, 1.0)

    def reset(self) -> None:
        """Release the occluder session (memory cleanup)."""
        with self._lock:
            self._session = None
            self._runner = None
            self._occluder_ready = False

    # ------------------------------------------------------------------
    # Mask terms
    # ------------------------------------------------------------------

    def _hull_mask(self, face: Face, matrix: Matrix, size: int) -> Mask:
        """
        Filled convex hull of the face landmarks, in aligned space.

        Falls back to a template ellipse when landmarks are unavailable. The
        fallback is safe because aligned space is normalized — unlike an
        ellipse in frame space, it is at least always centred on the face.
        """
        points = self._aligned_landmarks(face, matrix)
        if points is None:
            return self._template_ellipse(size)

        hull = cv2.convexHull(points.astype(np.float32))
        hull = self._expand_hull(hull)

        mask = np.zeros((size, size), dtype=np.float32)
        cv2.fillConvexPoly(mask, cv2.convexHull(hull).astype(np.int32), (1.0,))
        return mask

    @staticmethod
    def _aligned_landmarks(face: Face, matrix: Matrix) -> Optional[npt.NDArray[Any]]:
        """Transform the 106 landmarks into aligned space, or None if absent."""
        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is None or len(landmarks) < 3:
            return None

        points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.transform(points, matrix.astype(np.float32))
        return transformed.reshape(-1, 2)

    def _expand_hull(self, hull: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Grow the hull radially, then push its upper half further up.

        The upward weight ramps smoothly from the centroid to the topmost
        point so the polygon does not gain a kink at the midline.
        """
        points = hull.reshape(-1, 2).astype(np.float32)
        centre = points.mean(axis=0)

        points = centre + (points - centre) * (1.0 + self._HULL_EXPAND)

        top = float(points[:, 1].min())
        span = float(centre[1] - top)
        if span > 1e-3:
            weight = np.clip((centre[1] - points[:, 1]) / span, 0.0, 1.0)
            height = float(points[:, 1].max() - top)
            points[:, 1] -= weight * height * self._FOREHEAD_PUSH

        return points.reshape(-1, 1, 2)

    @staticmethod
    def _template_ellipse(size: int) -> Mask:
        """Fallback face-shaped region for aligned space (no landmarks)."""
        mask = np.zeros((size, size), dtype=np.float32)
        centre = (size // 2, int(size * 0.52))
        axes = (int(size * 0.38), int(size * 0.46))
        cv2.ellipse(mask, centre, axes, 0, 0, 360, (1.0,), -1)
        return mask

    def _valid_mask(
        self,
        matrix: Matrix,
        size: int,
        frame_shape: Tuple[int, int],
    ) -> Mask:
        """
        Region of the aligned crop that sampled real frame pixels.

        Warping a solid white frame through the same affine tells us exactly
        which aligned pixels came from inside the frame. Eroded slightly so
        interpolation at the boundary does not drag in the black border.
        """
        if self._white is None or self._white_shape != frame_shape:
            self._white = np.full(frame_shape, 255, dtype=np.uint8)
            self._white_shape = frame_shape

        valid = cv2.warpAffine(
            self._white,
            matrix,
            (size, size),
            flags=cv2.INTER_NEAREST,
        )

        erode = max(1, int(self._VALID_ERODE * size / 256))
        kernel = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
        valid = cv2.erode(valid, kernel, iterations=1)

        return (valid.astype(np.float32) / 255.0)

    def _occlusion_mask(self, aligned: Frame) -> Optional[Mask]:
        """
        Run DFL XSeg on the aligned crop.

        Returns:
            float32 inclusion mask matching `aligned`, or None if the model
            is unavailable or inference failed.
        """
        session = self._get_session()
        if session is None:
            return None

        size = aligned.shape[0]

        try:
            blob = cv2.resize(aligned, (self._input_size, self._input_size))
            blob = blob.astype(np.float32) / 255.0
            blob = np.expand_dims(blob, axis=0)
            if self._input_nchw:
                blob = blob.transpose(0, 3, 1, 2)

            runner = self._runner
            inputs = {self._input_name: blob}
            raw = (
                runner.run(inputs) if runner is not None
                else session.run(None, inputs)
            )[0]
            occlusion = np.squeeze(raw).astype(np.float32)

            if occlusion.ndim != 2:
                emit_warning(
                    f'Occluder returned unexpected shape {raw.shape} — skipping',
                    scope='MASKER',
                )
                return None

            occlusion = np.clip(occlusion, 0.0, 1.0)
            if occlusion.shape[0] != size:
                occlusion = cv2.resize(occlusion, (size, size))
            return occlusion

        except Exception as e:
            emit_warning(
                f'Occlusion mask failed: {type(e).__name__}: {e} — disabling occluder',
                scope='MASKER',
            )
            with self._lock:
                self._session = None
                self._occluder_ready = False
            return None

    # ------------------------------------------------------------------
    # Occluder model loading
    # ------------------------------------------------------------------

    def _get_session(self) -> Optional[Any]:
        """Lazily load the occluder ONNX session. Attempted at most once."""
        if self._occluder_ready:
            return self._session
        if self._occluder_attempted:
            return None

        with self._lock:
            if self._occluder_attempted:
                return self._session
            self._occluder_attempted = True
            self._session = self._load_occluder()
            self._occluder_ready = self._session is not None

        return self._session

    def _load_occluder(self) -> Optional[Any]:
        """
        Load (downloading if needed) the XSeg model.

        Returns None on any failure — masking then falls back to hull +
        valid-region, matching the graceful degradation the Enhancer uses.
        """
        model_path = self._resolve_model_path()

        if not os.path.isfile(model_path):
            if not self._download_occluder(model_path):
                return None

        try:
            from pipeline.services.onnx_session import BoundRunner, create_session

            # Static shapes: XSeg always sees `_input_size` square, whatever
            # the aligned crop was — the resize above guarantees it.
            session = create_session(
                self.config, model_path, 'xseg', static_shapes=True,
            )
            self._runner = BoundRunner(session, 'xseg')

            # Introspect rather than assume — input name and layout differ
            # between exports of this model.
            model_input = session.get_inputs()[0]
            self._input_name = model_input.name
            shape = model_input.shape
            self._input_nchw = len(shape) == 4 and shape[1] == 3
            spatial = shape[2] if self._input_nchw else shape[1]
            if isinstance(spatial, int) and spatial > 0:
                self._input_size = spatial

            emit_status(
                f'Occlusion masking available ({self._input_size}px, '
                f'{"NCHW" if self._input_nchw else "NHWC"})',
                scope='MASKER',
            )
            return session

        except Exception as e:
            emit_warning(
                f'Occluder failed to load: {type(e).__name__}: {e} — '
                'compositing without occlusion masking',
                scope='MASKER',
            )
            return None

    @staticmethod
    def _resolve_model_path() -> str:
        """
        Resolve the occluder model path.

        Checks the RunPod Network Volume first so the model survives pod
        restarts, then `pipeline/models/` — where the swapper and enhancer
        already keep their weights.
        """
        runpod_path = os.path.join('/workspace', 'models', _OCCLUDER_MODEL_NAME)
        if os.path.isdir('/workspace/models'):
            return runpod_path
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_dir, 'models', _OCCLUDER_MODEL_NAME)

    @staticmethod
    def _download_occluder(model_path: str) -> bool:
        """Download the occluder model. Returns True if it is now present."""
        model_dir = os.path.dirname(model_path)

        emit_status(f'Downloading {_OCCLUDER_MODEL_NAME} for occlusion masking...', scope='MASKER')
        try:
            from pipeline.io.ffmpeg import conditional_download

            conditional_download(model_dir, [_OCCLUDER_MODEL_URL])
            if os.path.isfile(model_path):
                emit_status('Occluder model downloaded', scope='MASKER')
                return True
        except Exception as e:
            emit_warning(f'Occluder download failed: {type(e).__name__}: {e}', scope='MASKER')

        emit_warning(
            f'Occlusion masking disabled. To enable it, download {_OCCLUDER_MODEL_URL} '
            f'and place it at: {model_path}',
            scope='MASKER',
        )
        return False
