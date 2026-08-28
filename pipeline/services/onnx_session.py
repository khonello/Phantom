"""
Central construction of ONNX Runtime sessions.

Every model that decides how the output looks is ONNX — the swapper, CodeFormer
and XSeg — and until now each service built its own session with its own idea of
what options to pass. That is how `face_swapping.py` came to construct a
`SessionOptions`, set two fields on it, and never hand it to the model: nothing
owned session construction, so nothing noticed.

It also does not scale. Four separate speed levers — session options, IOBinding,
CUDA graphs, fp16 weights and the TensorRT provider — all hook the same moment,
and bolting each onto three call sites independently is how a codebase acquires
three subtly different answers to the same question.

So this module owns that moment. Services say *which* model they want and
whether its shapes are static; everything about how the session is built is
decided here, once.

**Nothing here changes numerics by default.** `fp16`, `cuda_graphs` and `trt`
are all off unless asked for, so the default path produces bit-identical output
to the sessions this replaced.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pipeline.config import FaceSwapConfig
from pipeline.logging import emit_status, emit_warning

# A provider entry is either a bare name or a (name, options) pair.
Provider = Union[str, Tuple[str, Dict[str, Any]]]

_CUDA = 'CUDAExecutionProvider'
_TRT = 'TensorrtExecutionProvider'
_CPU = 'CPUExecutionProvider'

# Where engine and timing caches live. Follows the model weights: the network
# volume when it is mounted, so a build survives the pod it was made on.
_VOLUME_ROOT = '/workspace'
_CACHE_DIRNAME = 'trt-cache'

# Suffix for a converted half-precision copy, kept beside the fp32 original
# rather than replacing it — the two must stay comparable on real footage, and
# a conversion that turns out to hallucinate has to be revertible by config
# rather than by re-downloading 384 MB.
_FP16_SUFFIX = '-fp16.onnx'


def _cache_root() -> str:
    """
    Root directory for TensorRT engine and timing caches.

    Returns:
        `/workspace/trt-cache` when the network volume is mounted, else a
        directory beside the package. Created if missing.
    """
    if os.path.isdir(_VOLUME_ROOT):
        root = os.path.join(_VOLUME_ROOT, _CACHE_DIRNAME)
    else:
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(package_dir, _CACHE_DIRNAME)

    os.makedirs(root, exist_ok=True)
    return root


def gpu_identity() -> str:
    """
    A short, filesystem-safe name for the GPU this process can see.

    TensorRT engines are built for one architecture and are invalid on another,
    so this is the primary key of the engine cache. Read from torch when it is
    present (it is, on the GPU image) and from `nvidia-smi` otherwise.

    Returns:
        Something like `NVIDIA-GeForce-RTX-4090`, or `unknown-gpu` if nothing
        could be determined — which still caches, just under one shared name.
    """
    name = ''

    try:
        import torch
        if torch.cuda.is_available():
            name = str(torch.cuda.get_device_name(0))
    except Exception:
        name = ''

    if not name:
        try:
            import subprocess
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10,
            )
            name = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ''
        except Exception:
            name = ''

    if not name:
        return 'unknown-gpu'

    safe = ''.join(c if c.isalnum() else '-' for c in name)
    return '-'.join(part for part in safe.split('-') if part)


def _library_versions() -> str:
    """
    Version fingerprint of the libraries an engine is tied to.

    An engine built by one TensorRT is not loadable by another, and ORT's own
    version decides how the graph was partitioned before TensorRT ever saw it.
    Both belong in the cache key; getting this wrong surfaces as a corrupt-engine
    error minutes into a paid session.

    Returns:
        A short string like `ort1.17.1-trt8.6.1`
    """
    ort_version = 'ortX'
    try:
        import onnxruntime as ort
        ort_version = 'ort' + str(ort.__version__)
    except Exception:
        pass

    trt_version = 'trtX'
    try:
        import tensorrt  # type: ignore[import-not-found]
        trt_version = 'trt' + str(tensorrt.__version__)
    except Exception:
        pass

    return f'{ort_version}-{trt_version}'


def _model_fingerprint(model_path: str) -> str:
    """
    Short hash identifying the weights an engine was built from.

    Hashes size and mtime rather than content: these files are hundreds of
    megabytes, this runs on the startup path, and the question being asked is
    only "is this the same file as last time".

    Args:
        model_path: Path to the .onnx file

    Returns:
        Twelve hex characters
    """
    try:
        stat = os.stat(model_path)
        raw = f'{os.path.basename(model_path)}:{stat.st_size}:{int(stat.st_mtime)}'
    except OSError:
        raw = os.path.basename(model_path)

    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]


def trt_enabled_for_gpu(config: FaceSwapConfig) -> bool:
    """
    Whether this GPU is one we are willing to spend a build on.

    Engine building costs minutes, once per architecture — and it is spent from
    a paid hour with an operator waiting. On a fast card that is a good trade
    amortised over every later session; on a slow one it is minutes lost to make
    a card that was never going to hold the deadline slightly less slow.

    So TensorRT is opt-in per architecture rather than global. `trt_gpus` is a
    comma-separated list of substrings matched against the device name, which
    keeps the decision readable in `.env` and avoids duplicating the
    orchestrator's ranking table inside the pipeline — the two answer different
    questions and would drift.

    Args:
        config: Supplies `trt_gpus`

    Returns:
        True if this GPU's name matches an entry in the list
    """
    allowed = [
        entry.strip().lower()
        for entry in str(getattr(config, 'trt_gpus', '') or '').split(',')
        if entry.strip()
    ]
    if not allowed:
        return False

    device = gpu_identity().replace('-', ' ').lower()
    return any(entry.replace('-', ' ') in device for entry in allowed)


def engine_cache_dir(config: FaceSwapConfig, model_path: str) -> Optional[str]:
    """
    Directory holding this model's TensorRT engine on this GPU.

    Keyed by every property an engine is invalid across: the GPU architecture,
    the TensorRT and ONNX Runtime versions, the weights themselves, and the
    precision. A change to any of them lands in a different directory rather
    than loading an engine that will fail or, worse, silently misbehave.

    Caching per architecture rather than pinning to one is deliberate. The
    orchestrator picks across datacenters because availability is the binding
    constraint, and pinning would trade "sometimes a slower card" for "sometimes
    no pod at all". Each architecture pays its build once, ever.

    Args:
        config: Supplies `fp16`
        model_path: The .onnx file an engine would be built from

    Returns:
        An existing directory path, or None if a cache root is unavailable
    """
    try:
        precision = 'fp16' if getattr(config, 'fp16', False) else 'fp32'
        path = os.path.join(
            _cache_root(),
            gpu_identity(),
            _library_versions(),
            _model_fingerprint(model_path),
            precision,
        )
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as e:
        emit_warning(
            f'TensorRT engine cache unavailable: {type(e).__name__}: {e}',
            scope='ONNX',
        )
        return None


def resolve_weights(config: FaceSwapConfig, model_path: str) -> str:
    """
    The weights file to actually load, honouring the fp16 preference.

    A converted copy sits beside the original as `<name>-fp16.onnx`. If it has
    not been produced yet the fp32 original is used and that is said out loud —
    silently running fp32 while the config claims fp16 is the same class of
    problem as a provider silently falling back to CPU.

    Args:
        config: Supplies `fp16`
        model_path: Path to the fp32 weights

    Returns:
        Path to load
    """
    if not getattr(config, 'fp16', False):
        return model_path

    base, extension = os.path.splitext(model_path)
    if extension.lower() != '.onnx':
        return model_path

    half = base + _FP16_SUFFIX
    if os.path.isfile(half):
        return half

    emit_warning(
        f'fp16 requested but {os.path.basename(half)} is not present — loading '
        f'fp32 weights. Convert with: python tools/convert_fp16.py '
        f'{model_path}',
        scope='ONNX',
    )
    return model_path


def default_session_options(config: FaceSwapConfig) -> Any:
    """
    Session options shared by every model in the pipeline.

    Args:
        config: Supplies `execution_threads`

    Returns:
        A configured `ort.SessionOptions`, or None if onnxruntime is missing
    """
    try:
        import onnxruntime as ort
    except Exception:
        return None

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Sequential, not parallel. These are single-stream, batch-1 models run one
    # after another; ORT_PARALLEL spends threads looking for independent
    # subgraphs that a face model does not have, and the contention shows up as
    # jitter on the frame deadline rather than as throughput.
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    threads = int(getattr(config, 'execution_threads', 0) or 0)
    if threads > 0:
        options.intra_op_num_threads = threads

    return options


def build_providers(
    config: FaceSwapConfig,
    model_path: str,
    static_shapes: bool = False,
) -> List[Provider]:
    """
    The provider list for one model, with per-provider options attached.

    Args:
        config: Supplies `execution_providers`, `trt`, `fp16`, `cuda_graphs`
        model_path: Used to key the TensorRT engine cache
        static_shapes: True when every input to this model has a fixed shape.
                       CUDA graph capture requires it — a graph records fixed
                       buffer addresses, so a model whose input size changes
                       between calls would replay a graph describing the
                       previous shape

    Returns:
        Providers in priority order, ready to hand to `InferenceSession`
    """
    requested = list(config.execution_providers or [_CPU])
    providers: List[Provider] = []

    for name in requested:
        if name == _CUDA:
            options: Dict[str, Any] = {}

            # CUDA graphs collapse per-kernel launch overhead, which is a real
            # share of the cost for batch-1 models made of many small kernels.
            # They require IOBinding with stable buffer addresses, so this is
            # only offered where the caller has said shapes are static.
            if static_shapes and getattr(config, 'cuda_graphs', False):
                options['enable_cuda_graph'] = 1

            # Host<->device copies on their own stream, so a transfer can
            # overlap compute instead of serialising behind it. Worth having
            # here because this pipeline alternates device and host constantly
            # — the compositor is OpenCV on the CPU, so every model hands its
            # output back before the next one runs.
            #
            # Not combined with CUDA graphs: a captured graph replays a fixed
            # sequence against fixed buffers, and a copy running outside the
            # graph's stream is exactly the ordering assumption capture makes.
            # Where both are asked for, graphs win — they are the larger and
            # better-understood effect.
            if getattr(config, 'cuda_streams', False) and 'enable_cuda_graph' not in options:
                options['do_copy_in_default_stream'] = 0

            providers.append((name, options) if options else name)
            continue

        if name == _TRT:
            providers.append((_TRT, _trt_options(config, model_path)))
            continue

        providers.append(name)

    # TensorRT is requested through its own flag rather than by asking the
    # operator to spell the provider list correctly. It must come first: ORT
    # offers each subgraph to providers in order, and behind CUDA it would
    # never be asked.
    if getattr(config, 'trt', False) and _TRT not in requested:
        if trt_enabled_for_gpu(config):
            providers.insert(0, (_TRT, _trt_options(config, model_path)))
        else:
            emit_status(
                f'TensorRT skipped on {gpu_identity()} — not in trt_gpus. '
                f'Running on {requested[0] if requested else _CPU}.',
                scope='ONNX',
            )

    return providers


def _trt_options(config: FaceSwapConfig, model_path: str) -> Dict[str, Any]:
    """
    Provider options for the TensorRT execution provider.

    Args:
        config: Supplies `fp16`
        model_path: Keys the engine cache

    Returns:
        Option dict; empty if no cache directory could be made, which still
        works but rebuilds the engine every session
    """
    options: Dict[str, Any] = {}

    if getattr(config, 'fp16', False):
        options['trt_fp16_enable'] = True

    cache = engine_cache_dir(config, model_path)
    if cache:
        options['trt_engine_cache_enable'] = True
        options['trt_engine_cache_path'] = cache

        # The timing cache is not the engine. It records how fast each candidate
        # kernel measured on this device, so a *miss* on the engine cache still
        # builds substantially faster once one model has been through.
        options['trt_timing_cache_enable'] = True
        options['trt_timing_cache_path'] = cache

    return options


def create_session(
    config: FaceSwapConfig,
    model_path: str,
    label: str,
    static_shapes: bool = False,
) -> Any:
    """
    Build an `InferenceSession` for one model.

    Args:
        config: Supplies providers and the speed flags
        model_path: Path to the fp32 weights; an fp16 copy is preferred when
                    one exists and `fp16` is set
        label: Short name for log lines, e.g. `codeformer`
        static_shapes: Whether every input has a fixed shape — gates CUDA graphs

    Returns:
        A live session

    Raises:
        Whatever onnxruntime raises. Callers already handle load failure by
        degrading (restoration off, landmark-hull masking), and that decision
        belongs to them rather than here.
    """
    import onnxruntime as ort

    weights = resolve_weights(config, model_path)
    providers = build_providers(config, weights, static_shapes)
    options = default_session_options(config)

    session = ort.InferenceSession(
        weights, sess_options=options, providers=providers,
    )

    active = list(session.get_providers())
    detail = 'fp16' if weights.endswith(_FP16_SUFFIX) else 'fp32'
    if _TRT in active:
        detail += ', TensorRT'
    elif static_shapes and getattr(config, 'cuda_graphs', False) and _CUDA in active:
        detail += ', CUDA graph'

    emit_status(
        f'{label}: {active[0] if active else "no provider"} ({detail})',
        scope='ONNX',
    )
    return session


class BoundRunner:
    """
    Runs one fixed-shape session through IOBinding with reused device buffers.

    `session.run(None, {...})` hands ORT a numpy array and gets one back. Each
    call allocates a device buffer, copies from pageable host memory, runs,
    allocates a host array, and copies back. The copies are unavoidable here —
    the compositor is OpenCV on the CPU, so pixels have to come home between
    models — but the *allocation* and the pageable-memory penalty are not, and
    at four to six inferences a frame they are paid that many times.

    This also exists because CUDA graphs need it. A captured graph records fixed
    device addresses, so it can only be replayed against buffers that do not
    move. Binding once and reusing is what makes capture legal.

    Falls back to a plain `run` whenever anything is not static — a symbolic
    output dimension, a missing binding API, a shape that changed. The fallback
    is silent by design: it is a performance path, and a warning per frame would
    be worse than the cost it reports.
    """

    def __init__(self, session: Any, label: str) -> None:
        self.session = session
        self.label = label
        self._binding: Optional[Any] = None
        self._outputs: Optional[List[Tuple[str, Any]]] = None
        self._usable = True

    @staticmethod
    def _static_shape(shape: Sequence[Any]) -> Optional[Tuple[int, ...]]:
        """
        The shape as concrete integers, or None if any dimension is symbolic.

        Args:
            shape: A session input/output shape, which may hold strings or None
                   for dynamic dimensions

        Returns:
            A tuple of positive ints, or None
        """
        resolved: List[int] = []
        for dimension in shape:
            if not isinstance(dimension, int) or dimension <= 0:
                return None
            resolved.append(dimension)
        return tuple(resolved)

    def _prepare(self) -> bool:
        """
        Allocate the output buffers once. Returns False if binding is not usable.
        """
        if self._outputs is not None:
            return True
        if not self._usable:
            return False

        try:
            import numpy as np

            outputs: List[Tuple[str, Any]] = []
            for meta in self.session.get_outputs():
                shape = self._static_shape(meta.shape)
                if shape is None:
                    self._usable = False
                    return False
                outputs.append((meta.name, np.empty(shape, dtype=np.float32)))

            self._binding = self.session.io_binding()
            self._outputs = outputs
            return True
        except Exception:
            self._usable = False
            return False

    def run(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Run the session on `inputs`.

        Args:
            inputs: Input name -> numpy array

        Returns:
            Outputs in the session's declared order
        """
        if not self._prepare() or self._binding is None or self._outputs is None:
            result: List[Any] = self.session.run(None, inputs)
            return result

        try:
            binding = self._binding
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()

            for name, array in inputs.items():
                binding.bind_cpu_input(name, array)
            for name, buffer in self._outputs:
                binding.bind_output(name, 'cpu', 0, buffer.dtype, buffer.shape, buffer.ctypes.data)

            self.session.run_with_iobinding(binding)
            return [buffer for _, buffer in self._outputs]
        except Exception:
            # One failure is enough to stop trying: whatever made binding
            # invalid for this model will not fix itself next frame.
            self._usable = False
            self._binding = None
            self._outputs = None
            fallback: List[Any] = self.session.run(None, inputs)
            return fallback
