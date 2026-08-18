"""
Execution provider verification for the Phantom pipeline.

ONNX Runtime does not fail when a provider cannot be initialised. Ask for
`CUDAExecutionProvider` on a machine missing `libcudnn.so.9` and you get a
session that runs happily on CPU, with at most a warning on a logger nobody is
reading. Every model that decides how the output looks is ONNX — the swapper,
CodeFormer and XSeg — so that fallback is not a degradation, it is seconds per
frame instead of a live call.

It has already happened once (see runpod/TROUBLESHOOTING.md section 5b) and was
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
        'does the same in runpod/startup.sh step 6b'
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
    return summary
