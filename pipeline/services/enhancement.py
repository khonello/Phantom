"""
Face restoration service for the Phantom pipeline.

Models come from `enhancer_models.py`; `config.enhancer_model` names one.
Two *backends* run them:

- **codeformer** — the ONNX path, on the onnxruntime session the swapper
  already requires. Despite the name it runs **any** single-input ONNX
  restorer, GPEN included, because it introspects the graph rather than
  assuming it: the fidelity weight is wired only if the model declares one,
  and the crop size is read from the declared input shape.
- **gfpgan** — the previous backend, kept so the two can be compared on real
  footage. Requires torch + the gfpgan package; degrades gracefully if absent.

The default is **gpen_bfr_256**, and the reason is measured. On an RTX 4090
CodeFormer costs 29.4ms against GPEN-256's 5.4ms, and on a 101px webcam face
that 29.4ms buys **+0.03** on the face/frame detail ratio — because restoration
runs on a fixed 512 crop that is warped straight back down into a 128-192
aligned space, discarding ~86% of what it produced. Note that `gpen_bfr_512` is
*slower* than CodeFormer, so the win is the crop size and not the architecture.

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
from pipeline.services import enhancer_models

# Both backends are trained on FFHQ-aligned 512x512 crops. This is the default
# and the size the FFHQ template is defined against; `config.restore_size` can
# ask for less, and `Enhancer.crop_size` decides what the loaded model will
# actually accept.
CROP_SIZE = 512

# Floor on the FFHQ crop. Below this the model has less to work with than the
# 128px swap it is restoring, so there is nothing left for it to do.
MIN_CROP_SIZE = 128


def _spatial_size(shape: Optional[Any]) -> Optional[int]:
    """
    The square input size an ONNX graph insists on, or None if it is dynamic.

    ONNX declares a symbolic dimension as a string (`'height'`) or None, and a
    fixed one as an int. Only a graph whose last two dims are equal ints is
    pinned to a size; anything else can be fed whatever the caller builds.

    Args:
        shape: The input's declared shape, e.g. `[1, 3, 512, 512]`

    Returns:
        Edge length if the graph is fixed and square, else None.
    """
    if not shape or len(shape) < 2:
        return CROP_SIZE
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int) and height == width:
        return height
    return None


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve a model path.

    Checks /workspace first so weights survive a container restart,
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

    def __init__(self, config: FaceSwapConfig, spec: Any = None) -> None:
        self.config = config
        # Which weights to load. Defaulted rather than required so the existing
        # construction sites keep working and a caller that does not care about
        # the registry still gets the configured model.
        self.spec = spec if spec is not None else enhancer_models.resolve(
            config.enhancer_model or enhancer_models.DEFAULT_ENHANCER_MODEL)
        self._session: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._image_input = 'input'
        self._weight_input: Optional[str] = None
        # Edge length the graph insists on, or None if it accepts any. Read
        # from the model rather than assumed: `restore_size` is worth nothing
        # against an export with fixed 512 spatial dims, and finding that out
        # by throwing once per frame on a paid pod is the wrong way to learn it.
        self.native_size: Optional[int] = CROP_SIZE

    def load(self) -> bool:
        """Load the session, downloading the model if needed."""
        model_path = _resolve_model_path(self.spec.filename)

        if not os.path.isfile(model_path):
            if not self._download(model_path, self.spec.url):
                return False

        try:
            from pipeline.services.onnx_session import BoundRunner, create_session

            # Static shapes: the crop is always CROP_SIZE square and the
            # fidelity weight is a scalar, so this model never sees a shape it
            # has not seen before. That is what makes CUDA graph capture legal
            # here and not on the detector.
            self._session = create_session(
                self.config, model_path, self.spec.name,
                static_shapes=True, bound=True,
            )
            self._runner = BoundRunner(self._session, self.spec.name)

            # Introspect rather than assume. The image input is whichever one
            # is not the scalar fidelity weight.
            for model_input in self._session.get_inputs():
                if model_input.name == 'weight':
                    self._weight_input = model_input.name
                else:
                    self._image_input = model_input.name
                    self.native_size = _spatial_size(getattr(model_input, 'shape', None))

            emit_status(
                '{} restoration available'.format(self.spec.name)
                + ('' if self._weight_input else ' (fixed fidelity — no weight input)')
                + (
                    ' — input is dynamic, restore_size applies'
                    if self.native_size is None
                    else f' — input fixed at {self.native_size}px'
                ),
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
    def _download(model_path: str, url: str) -> bool:
        """Download a restoration model. Returns True if it is now present."""
        name = os.path.basename(model_path)
        emit_status(f'Downloading {name}...', scope='ENHANCER')
        try:
            from pipeline.io.ffmpeg import conditional_download

            conditional_download(os.path.dirname(model_path), [url])
            if os.path.isfile(model_path):
                emit_status(f'{name} downloaded', scope='ENHANCER')
                return True
        except Exception as e:
            emit_warning(f'{name} download failed: {type(e).__name__}: {e}', scope='ENHANCER')

        emit_warning(
            f'Restoration disabled. To enable it, download {url} '
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

    def __init__(self, config: FaceSwapConfig, spec: Any = None) -> None:
        self.config = config
        self.spec = spec if spec is not None else enhancer_models.resolve('gfpgan')
        self._enhancer: Optional[Any] = None
        # `has_aligned=True` hands the crop straight to a network built for
        # FFHQ 512. There is no smaller path through it.
        self.native_size: Optional[int] = CROP_SIZE

    def load(self) -> bool:
        """Load GFPGAN. Returns False if the model or package is missing."""
        model_path = _resolve_model_path(self.spec.filename)

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
        self._size_warned: Optional[int] = None

    @property
    def crop_size(self) -> int:
        """
        Edge length the caller should build its FFHQ crop at.

        This is the largest single cost in the pipeline. Restoration runs on a
        512x512 crop regardless of how many frame pixels the face covers, so a
        101px face is upsampled ~20x in pixel count before the heaviest model
        in the chain sees it — roughly 4x the compute of restoring at 256 and
        16x of 128, spent reconstructing detail the source never held.

        `config.restore_size` asks for less. The **model decides**: an export
        with fixed spatial dims is honoured over the request, because feeding
        it another shape throws once per frame rather than running faster. The
        disagreement is said once, not per frame — this sits on the live path.

        Returns:
            Even edge length in [MIN_CROP_SIZE, CROP_SIZE].
        """
        requested = int(getattr(self.config, 'restore_size', CROP_SIZE) or CROP_SIZE)
        requested = max(MIN_CROP_SIZE, min(CROP_SIZE, requested))
        # Conv stacks halve the spatial dims repeatedly; an odd edge rounds
        # somewhere inside and comes back a different size than it went in.
        requested -= requested % 8

        self._ensure_loaded()
        native = getattr(self._backend, 'native_size', CROP_SIZE)
        if native is None or native == requested:
            return requested

        if self._size_warned != requested:
            self._size_warned = requested
            emit_warning(
                f'restore_size={requested} ignored — the loaded restoration model '
                f'has fixed {native}px inputs. Restoring at a smaller size needs a '
                f'model exported for it, not a config change.',
                scope='ENHANCER',
            )
        return int(native)

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
        requested = (self.config.enhancer_model
                     or enhancer_models.DEFAULT_ENHANCER_MODEL).lower()
        if requested not in enhancer_models.ENHANCER_MODELS:
            emit_warning(
                f"Unknown enhancer_model '{requested}' — using "
                f"'{enhancer_models.DEFAULT_ENHANCER_MODEL}'",
                scope='ENHANCER',
            )
            requested = enhancer_models.DEFAULT_ENHANCER_MODEL

        # Requested first, then the default, then anything else registered. A
        # missing weight file degrades to "restoration still works" rather than
        # "restoration is off", which is the same reason the two-backend
        # version fell back — only now the ladder is the registry rather than a
        # pair of names.
        order = [requested, enhancer_models.DEFAULT_ENHANCER_MODEL]
        order += [n for n in enhancer_models.names() if n not in order]

        for name in order:
            spec = enhancer_models.ENHANCER_MODELS[name]
            backend: Any
            if spec.backend == 'gfpgan':
                backend = _GFPGANBackend(self.config, spec)
            else:
                backend = _CodeFormerBackend(self.config, spec)

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
            crop: FFHQ-framed BGR crop, square, built at `crop_size`

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
            # The next backend may accept a size this one refused.
            self._size_warned = None
