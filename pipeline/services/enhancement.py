"""
Face restoration service for the Phantom pipeline.

Two backends, selected by `config.enhancer_model`:

- **codeformer** (default) — ONNX, runs on the existing onnxruntime CUDA
  session. Exposes a fidelity weight, which is the knob that matters here:
  it trades restoration strength against staying faithful to the input.
- **gfpgan** — the previous backend, kept so the two can be compared on real
  footage. Requires torch + the gfpgan package; degrades gracefully if absent.

Both expect an **FFHQ-framed 512x512 crop**, not an arcface crop. They were
trained on FFHQ alignment and have strong priors about where the eyes and
mouth sit in the frame; feeding them the swapper's tighter arcface crop
degrades restoration. The caller owns that warp — see
`pipeline/processing/compositor.py`.

Why CodeFormer is the default: GFPGAN v1.4 restores toward a beautified,
poreless look with no way to dial it back, and that plastic skin is the single
strongest "this is AI" signal on a video call. CodeFormer's fidelity weight
lets us ask for less.
"""

import os
import threading
from typing import Any, Optional

import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame
from pipeline.logging import emit_status, emit_warning

# Both backends are trained on FFHQ-aligned 512x512 crops.
CROP_SIZE = 512

_CODEFORMER_MODEL_NAME = 'codeformer.onnx'
_CODEFORMER_MODEL_URL = (
    'https://github.com/facefusion/facefusion-assets/releases/download/'
    'models-3.0.0/codeformer.onnx'
)
_GFPGAN_MODEL_NAME = 'GFPGANv1.4.pth'


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve a model path.

    Checks the RunPod Network Volume first so weights survive pod restarts,
    then `pipeline/models/` where the swapper keeps its own.

    Args:
        model_name: Bare filename of the model

    Returns:
        Full path to the model
    """
    if os.path.isdir('/workspace/models'):
        return os.path.join('/workspace', 'models', model_name)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_dir, 'models', model_name)


class _CodeFormerBackend:
    """
    CodeFormer restoration via ONNX Runtime.

    The model takes an FFHQ-aligned 512x512 crop plus a fidelity weight.
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        self.config = config
        self._session: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._image_input = 'input'
        self._weight_input: Optional[str] = None

    def load(self) -> bool:
        """Load the session, downloading the model if needed."""
        model_path = _resolve_model_path(_CODEFORMER_MODEL_NAME)

        if not os.path.isfile(model_path):
            if not self._download(model_path):
                return False

        try:
            from pipeline.services.onnx_session import BoundRunner, create_session

            # Static shapes: the crop is always CROP_SIZE square and the
            # fidelity weight is a scalar, so this model never sees a shape it
            # has not seen before. That is what makes CUDA graph capture legal
            # here and not on the detector.
            self._session = create_session(
                self.config, model_path, 'codeformer',
                static_shapes=True, bound=True,
            )
            self._runner = BoundRunner(self._session, 'codeformer')

            # Introspect rather than assume. The image input is whichever one
            # is not the scalar fidelity weight.
            for model_input in self._session.get_inputs():
                if model_input.name == 'weight':
                    self._weight_input = model_input.name
                else:
                    self._image_input = model_input.name

            emit_status(
                'CodeFormer restoration available'
                + ('' if self._weight_input else ' (fixed fidelity — no weight input)'),
                scope='ENHANCER',
            )
            return True

        except Exception as e:
            emit_warning(
                f'CodeFormer failed to load: {type(e).__name__}: {e}',
                scope='ENHANCER',
            )
            self._session = None
            self._runner = None
            return False

    @staticmethod
    def _download(model_path: str) -> bool:
        """Download the CodeFormer model. Returns True if it is now present."""
        emit_status(f'Downloading {_CODEFORMER_MODEL_NAME}...', scope='ENHANCER')
        try:
            from pipeline.io.ffmpeg import conditional_download

            conditional_download(os.path.dirname(model_path), [_CODEFORMER_MODEL_URL])
            if os.path.isfile(model_path):
                emit_status('CodeFormer model downloaded', scope='ENHANCER')
                return True
        except Exception as e:
            emit_warning(f'CodeFormer download failed: {type(e).__name__}: {e}', scope='ENHANCER')

        emit_warning(
            f'Restoration disabled. To enable it, download {_CODEFORMER_MODEL_URL} '
            f'and place it at: {model_path}',
            scope='ENHANCER',
        )
        return False

    def restore(self, crop: Frame, weight: float) -> Optional[Frame]:
        """
        Restore an FFHQ-aligned 512x512 BGR crop.

        Args:
            crop: Input crop (BGR, uint8, CROP_SIZE square)
            weight: Fidelity weight. 0.0 restores hardest and hallucinates
                    most; 1.0 stays closest to the input.

        Returns:
            Restored BGR crop, or None on failure.
        """
        if self._session is None:
            return None

        # BGR -> RGB, [0,1], then [-1,1], then NCHW.
        blob = crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5
        blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)

        inputs = {self._image_input: blob}
        if self._weight_input:
            inputs[self._weight_input] = np.array([weight]).astype(np.double)

        runner = self._runner
        raw = runner.run(inputs) if runner is not None else self._session.run(None, inputs)
        output = raw[0][0]

        # Inverse of the preparation above.
        output = np.clip(output, -1, 1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        restored: Frame = (output * 255.0).round().astype(np.uint8)[:, :, ::-1]
        return restored


class _GFPGANBackend:
    """
    GFPGAN restoration via the gfpgan package (torch).

    Kept so CodeFormer can be compared against the previous behaviour. Has no
    usable strength control of its own — `Enhancer` blends the result instead.
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        self.config = config
        self._enhancer: Optional[Any] = None

    def load(self) -> bool:
        """Load GFPGAN. Returns False if the model or package is missing."""
        model_path = _resolve_model_path(_GFPGAN_MODEL_NAME)

        if not os.path.exists(model_path):
            emit_warning(
                f'GFPGAN model not found: {model_path}',
                scope='ENHANCER',
            )
            return False

        try:
            # Shim for torchvision >= 0.18 which removed functional_tensor.
            # basicsr (gfpgan dependency) imports the removed module at load time.
            import sys
            if 'torchvision.transforms.functional_tensor' not in sys.modules:
                import types
                import torchvision.transforms.functional as _F
                _shim = types.ModuleType('torchvision.transforms.functional_tensor')
                _shim.rgb_to_grayscale = _F.rgb_to_grayscale  # type: ignore[attr-defined]
                sys.modules['torchvision.transforms.functional_tensor'] = _shim

            from gfpgan import GFPGANer

            self._enhancer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
            )
            emit_status('GFPGAN restoration available', scope='ENHANCER')
            return True

        except ImportError:
            emit_warning('gfpgan package not installed', scope='ENHANCER')
            return False
        except Exception as e:
            emit_warning(f'GFPGAN failed to load: {type(e).__name__}: {e}', scope='ENHANCER')
            return False

    def restore(self, crop: Frame, weight: float) -> Optional[Frame]:
        """
        Restore an FFHQ-aligned 512x512 BGR crop.

        `has_aligned=True` skips GFPGAN's internal face detection and its own
        crop/paste. That removes a redundant full detection from every frame,
        stops it enhancing bystanders, and leaves compositing to the caller.

        Args:
            crop: Input crop (BGR, uint8, CROP_SIZE square)
            weight: Ignored — GFPGAN's clean architecture has no weight input.

        Returns:
            Restored BGR crop, or None on failure.
        """
        if self._enhancer is None:
            return None

        _, restored_faces, _ = self._enhancer.enhance(
            crop,
            has_aligned=True,
            only_center_face=False,
            paste_back=False,
        )
        if not restored_faces:
            return None
        restored: Frame = restored_faces[0]
        return restored


class Enhancer:
    """
    Face restoration with a selectable backend.

    Thread-safe; the backend is loaded once on first use and reused. If the
    selected backend cannot load, restoration is disabled and the pipeline
    continues without it.

    Example:
        enhancer = Enhancer(CONFIG)
        restored = enhancer.restore(ffhq_crop)   # None if unavailable
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize the enhancer.

        Args:
            config: Configuration (enhancer_model, enhancer_weight,
                    execution_providers)
        """
        self.config = config
        self._backend: Optional[Any] = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True if a restoration backend is loaded and ready."""
        self._ensure_loaded()
        return self._backend is not None

    def load(self) -> None:
        """Eagerly load the backend (used by pipeline warm-up)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load the configured backend once."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            self._backend = self._build_backend()

    def _build_backend(self) -> Optional[Any]:
        """
        Construct and load the configured backend.

        Falls back to the other backend if the configured one is unavailable,
        so a missing model file degrades to "restoration still works" rather
        than "restoration is off".
        """
        requested = (self.config.enhancer_model or 'codeformer').lower()
        order = ['codeformer', 'gfpgan']
        if requested in order:
            order.remove(requested)
        order.insert(0, requested)

        for name in order:
            backend: Any
            if name == 'gfpgan':
                backend = _GFPGANBackend(self.config)
            elif name == 'codeformer':
                backend = _CodeFormerBackend(self.config)
            else:
                emit_warning(f"Unknown enhancer_model '{name}'", scope='ENHANCER')
                continue

            if backend.load():
                if name != requested:
                    emit_warning(
                        f"Falling back to '{name}' restoration "
                        f"('{requested}' unavailable)",
                        scope='ENHANCER',
                    )
                return backend

        emit_warning('No restoration backend available — enhancement disabled', scope='ENHANCER')
        return None

    def restore(self, crop: Frame) -> Optional[Frame]:
        """
        Restore an FFHQ-aligned crop.

        Args:
            crop: FFHQ-framed BGR crop, CROP_SIZE square

        Returns:
            Restored crop, or None if restoration is unavailable or failed.
            Callers decide how much of the result to blend in.
        """
        self._ensure_loaded()
        if self._backend is None:
            return None

        weight = float(np.clip(self.config.enhancer_weight, 0.0, 1.0))

        with self._lock:
            try:
                return self._backend.restore(crop, weight)
            except Exception as e:
                emit_warning(
                    f'Face restoration failed: {type(e).__name__}: {e}',
                    scope='ENHANCER',
                )
                return None

    def clear(self) -> None:
        """Release the backend (memory cleanup)."""
        with self._lock:
            self._backend = None
            self._loaded = False
