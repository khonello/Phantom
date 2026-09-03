"""
Execution provider verification for the Phantom pipeline.

ONNX Runtime does not fail when a provider cannot be initialised. Ask for
`CUDAExecutionProvider` on a machine missing `libcudnn.so.9` and you get a
session that runs happily on CPU, with at most a warning on a logger nobody is
reading. Every model that decides how the output looks is ONNX — the swapper,
CodeFormer and XSeg — so that fallback is not a degradation, it is seconds per
frame instead of a live call.

It has already happened once (see vast/TROUBLESHOOTING.md section 5b) and was
found by reading the Dockerfile, not by anything failing. Build-time checks now
exist on both deploy paths, but they only cover the ways it went wrong before.
This module asks the question the ways-not-yet-invented cannot dodge: once the
models are loaded, what is each session *actually* running on?

Same principle as the guard capability probe — a silent fallback is
indistinguishable from a slow GPU, so it has to be stated rather than assumed.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline.config import FaceSwapConfig
from pipeline.logging import emit_status, emit_error

# Providers that mean "not on the accelerator we asked for".
_CPU_PROVIDER = 'CPUExecutionProvider'
_CUDA_PROVIDER = 'CUDAExecutionProvider'
_TRT_PROVIDER = 'TensorrtExecutionProvider'


class ExecutionProviderError(RuntimeError):
    """
    The requested accelerator is not the one the models are running on.

    Raised rather than warned. A pod that stops here is obvious and costs a
    redeploy; one that carries on spends a paid GPU hour producing output at
    seconds per frame, which is the entire reason the pod was rented. Running on
    CPU is not a degraded version of the product — it is a bill with nothing
    usable attached.

    Asking for CPU explicitly (`--execution-provider cpu`) is the supported way
    to run without an accelerator, and does not raise.
    """


# Requested provider -> the library whose absence usually explains it silently
# failing, so the error can name the fix rather than the symptom.
_LIKELY_CAUSE = {
    'CUDAExecutionProvider': (
        'libcudnn.so.9 is missing or not on the loader path. The Docker image '
        'installs nvidia-cudnn-cu12 and registers it with ldconfig; the SSH path '
        'does the same in vast/startup.sh step 6b'
    ),
    'ROCMExecutionProvider': 'the ROCm runtime is not installed or not visible',
    'DmlExecutionProvider': 'the DirectML runtime is not available',
}


def available_providers() -> List[str]:
    """
    Providers this onnxruntime build can offer.

    Returns:
        Provider names, or an empty list if onnxruntime is unavailable
    """
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def session_providers(holder: Any) -> Optional[List[str]]:
    """
    Providers actually in use by whatever session `holder` owns.

    Services keep their sessions in different places — a raw InferenceSession, an
    InsightFace model wrapping one, or a FaceAnalysis holding several — so this
    digs rather than assuming a shape. Returns None when no session can be found,
    which means "not loaded", not "on CPU".

    Args:
        holder: A service, model, or session

    Returns:
        Provider names in priority order, or None
    """
    if holder is None:
        return None

    if hasattr(holder, 'get_providers'):
        try:
            return list(holder.get_providers())
        except Exception:
            return None

    # One level down: services store the session privately, InsightFace models
    # expose `.session`.
    for attribute in ('_session', 'session'):
        nested = getattr(holder, attribute, None)
        if nested is not None and hasattr(nested, 'get_providers'):
            try:
                return list(nested.get_providers())
            except Exception:
                return None

    # FaceAnalysis holds a dict of task models, each with its own session.
    models = getattr(holder, 'models', None)
    if isinstance(models, dict):
        for model in models.values():
            found = session_providers(model)
            if found:
                return found

    # Two levels down. Every service that matters keeps its session behind one
    # more object than the checks above reach, and each returned None — which
    # `verify` reads as "not loaded" rather than "on CPU", so three of the four
    # models were never actually checked. The one that passed, the occluder,
    # happens to hold its session directly.
    #
    #   Enhancer     -> _backend   -> _session
    #   FaceSwapper  -> _swapper   -> .session   (INSwapper)
    #   FaceDetector -> _analyser  -> .models    (FaceAnalysis)
    #
    # A silent CPU fallback is the exact failure this module exists to catch,
    # and it could not see most of the pipeline.
    for attribute in ('_backend', '_swapper', '_analyser', '_enhancer'):
        nested = getattr(holder, attribute, None)
        if nested is not None and nested is not holder:
            found = session_providers(nested)
            if found:
                return found

    return None


def verify(
    config: FaceSwapConfig,
    sessions: Sequence[Tuple[str, Any]],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Check that loaded models are running on the provider that was requested.

    Args:
        config: Supplies `execution_providers`, in priority order
        sessions: (label, holder) pairs to inspect, e.g. ('swap', swapper)
        strict: Raise on a fallback rather than only reporting it. The error is
                emitted either way, so it reaches the desktop through the
                server's ERROR forwarding before anything unwinds

    Returns:
        A summary: the requested provider, what is available, and per-model
        actuals — suitable for logging or a health response

    Raises:
        ExecutionProviderError: under `strict`, when an accelerator was
            requested and the models are not using it
    """
    requested = list(config.execution_providers or [])
    preferred = next((p for p in requested if p != _CPU_PROVIDER), None)
    available = available_providers()

    actual: Dict[str, Optional[List[str]]] = {}
    for label, holder in sessions:
        actual[label] = session_providers(holder)

    summary: Dict[str, Any] = {
        'requested': requested,
        'preferred': preferred,
        'available': available,
        'actual': actual,
    }

    # CPU was asked for. Nothing to warn about — it is a supported choice, and
    # the local development default.
    if preferred is None:
        emit_status('Execution provider: CPU (as configured)', scope='RUNTIME')
        summary['ok'] = True
        return summary

    if preferred not in available:
        message = (
            f'{preferred} was requested but this onnxruntime build does not '
            f'offer it. Available: {", ".join(available) or "none"}. '
            f'Every ONNX model — the swapper, CodeFormer and XSeg — would run on '
            f'CPU, which is seconds per frame rather than a live call. '
            f'Likely cause: {_LIKELY_CAUSE.get(preferred, "unknown")}.'
        )
        emit_error(message, scope='RUNTIME')
        summary['ok'] = False
        if strict:
            raise ExecutionProviderError(message)
        return summary

    # Available is not the same as used: a session can still fall back when the
    # provider fails to initialise for that particular model.
    degraded = [
        label for label, providers in actual.items()
        if providers is not None and preferred not in providers
    ]

    if degraded:
        message = (
            f'{preferred} is available but these models fell back to CPU: '
            f'{", ".join(sorted(degraded))}. That is seconds per frame, not a '
            f'live call. Likely cause: {_LIKELY_CAUSE.get(preferred, "unknown")}.'
        )
        emit_error(message, scope='RUNTIME')
        summary['ok'] = False
        if strict:
            raise ExecutionProviderError(message)
        return summary

    loaded = [label for label, providers in actual.items() if providers]
    emit_status(
        f'Execution provider: {preferred} confirmed on '
        f'{", ".join(sorted(loaded)) or "no loaded models"}',
        scope='RUNTIME',
    )
    summary['ok'] = True
    summary['tensorrt'] = _report_tensorrt(config, actual)
    return summary


def _report_tensorrt(
    config: FaceSwapConfig,
    actual: Dict[str, Optional[List[str]]],
) -> Dict[str, Any]:
    """
    Report which models TensorRT actually claimed, when it was asked for.

    Deliberately a **warning, not an error**, which is the one place this module
    departs from fail-closed — and the reason is the same reason CPU fallback
    raises. A model on CPU is a paid GPU hour producing nothing usable. A model
    that fell back from TensorRT to CUDA is still on the GPU and still holds a
    live call; it is merely not as fast as intended. Stopping the session over
    it would cost the operator more than the fallback does.

    It has to be *said*, though. TensorRT's whole failure mode is silence: the
    provider registers, declines the graph, and CUDA runs it — indistinguishable
    from a successful build except by the engine cache staying empty and the
    first frame arriving on time when it should have been late.

    Args:
        config: Supplies `trt`
        actual: Per-model provider lists, from `verify`

    Returns:
        A summary: whether it was requested, and which models it took
    """
    requested = bool(getattr(config, 'trt', False))
    report: Dict[str, Any] = {'requested': requested}
    if not requested:
        return report

    from pipeline.services.onnx_session import gpu_identity, trt_enabled_for_gpu

    eligible = trt_enabled_for_gpu(config)
    report['gpu'] = gpu_identity()
    report['eligible'] = eligible

    if not eligible:
        emit_status(
            f'TensorRT requested but {gpu_identity()} is not in trt_gpus — '
            f'running on {_CUDA_PROVIDER}. Add it to TRT_GPUS if a '
            f'multi-minute engine build is worth it on this card.',
            scope='RUNTIME',
        )
        return report

    claimed = sorted(
        label for label, providers in actual.items()
        if providers and _TRT_PROVIDER in providers
    )
    missing = sorted(
        label for label, providers in actual.items()
        if providers and _TRT_PROVIDER not in providers
    )
    report['claimed'] = claimed
    report['missing'] = missing

    if not claimed:
        emit_error(
            f'TensorRT was requested and {gpu_identity()} is eligible, but no '
            f'model is running on it — every one fell back to '
            f'{_CUDA_PROVIDER}. The session still works and still uses the GPU, '
            f'so this is not fatal, but nothing was gained. Usual cause: the '
            f'onnxruntime build has no TensorRT support, or the engine cache '
            f'directory is not writable.',
            scope='RUNTIME',
        )
    elif missing:
        emit_status(
            f'TensorRT claimed {", ".join(claimed)}; '
            f'{", ".join(missing)} fell back to {_CUDA_PROVIDER}.',
            scope='RUNTIME',
        )
    else:
        emit_status(
            f'TensorRT confirmed on {", ".join(claimed)}',
            scope='RUNTIME',
        )

    return report
