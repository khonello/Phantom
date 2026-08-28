"""
Command handlers for the Phantom API.

Type-safe, side-effect-free command handlers that process API requests.
Each handler takes typed arguments and returns a ResponseMessage.

Handlers are called by the WebSocket server (api/server.py) when
commands are received from clients.

Extracted from pipeline/control.py:_dispatch() but with proper typing
and separation of concerns.

HandlerContext provides dependency injection — no module-level globals.
"""

import base64
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline.config import FaceSwapConfig
from pipeline.api.schema import (
    MAX_PHOTO_BYTES,
    MAX_PHOTO_TARGETS,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_SECONDS,
    VIDEO_CHUNK_BYTES,
    ResponseMessage,
)
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.services import guards
from pipeline.services import swapper_models
from pipeline.services.database import SourceReview
from pipeline.services.templates import TemplateLibrary
from pipeline.logging import emit_status, emit_error, emit_warning
from pipeline.io.ffmpeg import (
    is_image, is_video, is_video_name, normalize_output_path, probe_duration,
)


def _upload_dir() -> str:
    """
    Where uploaded media is written.

    Was hardcoded to `/tmp/phantom_uploads`, which on a pod is container disk:
    it does not survive a stop, so every restart cost a re-upload of a file
    that had already crossed the network once. `get_temp_root` already knew
    better — it prefers PHANTOM_TEMP_DIR, then the network volume, and the
    batch scratch has used it all along. This just stops the two disagreeing.

    Returns:
        Directory for uploads, created if missing
    """
    from pipeline.io.ffmpeg import get_temp_root

    path = os.path.join(get_temp_root(), 'uploads')
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class HandlerContext:
    """
    Dependency injection context for command handlers.

    Replaces module-level globals (_pipeline, _shutdown_event).
    Passed through dispatch_command() so handlers remain testable.

    Attributes:
        pipeline: ProcessingPipeline instance
        shutdown_event: threading.Event for shutdown signaling
    """

    pipeline: Optional[ProcessingPipeline]
    shutdown_event: Optional[threading.Event]
    reset_auto_stop: Optional[Callable[[], None]] = None


# ============================================================================
# Source/Target/Output Handlers
# ============================================================================

def handle_set_source(config: FaceSwapConfig, path: str) -> ResponseMessage:
    """
    Set source image path.

    Args:
        config: FaceSwapConfig
        path: Path to source image or embedding file

    Returns:
        ResponseMessage with success status
    """
    if not path:
        return ResponseMessage(
            type='set_source',
            data={'path': path},
            success=False,
            error='Source path cannot be empty',
        )

    if not os.path.exists(path):
        return ResponseMessage(
            type='set_source',
            data={'path': path},
            success=False,
            error=f'Source path does not exist: {path}',
        )

    # Accept .npy or image files
    if not (path.lower().endswith('.npy') or is_image(path)):
        return ResponseMessage(
            type='set_source',
            data={'path': path},
            success=False,
            error=f'Source must be an image or .npy file: {path}',
        )

    config.set('source_path', path)
    emit_status(f'Source set to: {path}', scope='API')

    return ResponseMessage(
        type='set_source',
        data={'path': path},
        success=True,
    )


def handle_set_source_paths(config: FaceSwapConfig, paths: List[str]) -> ResponseMessage:
    """
    Set multiple source paths (for averaging).

    Args:
        config: FaceSwapConfig
        paths: List of source image/embedding paths

    Returns:
        ResponseMessage with success status
    """
    if not paths:
        return ResponseMessage(
            type='set_source_paths',
            data={'paths': paths},
            success=False,
            error='Source paths cannot be empty',
        )

    # Validate all paths
    for path in paths:
        if not os.path.exists(path):
            return ResponseMessage(
                type='set_source_paths',
                data={'paths': paths},
                success=False,
                error=f'Source path does not exist: {path}',
            )

        if not (path.lower().endswith('.npy') or is_image(path)):
            return ResponseMessage(
                type='set_source_paths',
                data={'paths': paths},
                success=False,
                error=f'Source must be images or .npy files: {path}',
            )

    config.set('source_paths', paths)
    emit_status(f'Source paths set: {len(paths)} files', scope='API')

    return ResponseMessage(
        type='set_source_paths',
        data={'paths': paths, 'count': len(paths)},
        success=True,
    )


def handle_set_target(config: FaceSwapConfig, path: str) -> ResponseMessage:
    """
    Set target image/video path.

    Args:
        config: FaceSwapConfig
        path: Path to target image or video

    Returns:
        ResponseMessage with success status
    """
    if not path:
        return ResponseMessage(
            type='set_target',
            data={'path': path},
            success=False,
            error='Target path cannot be empty',
        )

    if not os.path.exists(path):
        return ResponseMessage(
            type='set_target',
            data={'path': path},
            success=False,
            error=f'Target path does not exist: {path}',
        )

    if not (is_image(path) or is_video(path)):
        return ResponseMessage(
            type='set_target',
            data={'path': path},
            success=False,
            error=f'Target must be an image or video: {path}',
        )

    config.set('target_path', path)
    # Clearing the photo targets is what keeps the two batch shapes exclusive:
    # a stale photo list would otherwise take precedence over the file just set.
    config.set('target_paths', [])
    _clear_template(config)
    emit_status(f'Target set to: {path}', scope='API')

    return ResponseMessage(
        type='set_target',
        data={'path': path},
        success=True,
    )


def handle_set_output(config: FaceSwapConfig, path: str) -> ResponseMessage:
    """
    Set output path.

    Args:
        config: FaceSwapConfig
        path: Output file or directory path

    Returns:
        ResponseMessage with success status
    """
    if not path:
        return ResponseMessage(
            type='set_output',
            data={'path': path},
            success=False,
            error='Output path cannot be empty',
        )

    # An output path is chosen on the desktop and applied here, and the two are
    # only the same filesystem when the pipeline runs locally. On a pod a
    # Windows path is not a path at all, and a Linux one names a directory that
    # very likely does not exist — either way the render fails at the last step,
    # after all the work. Keep the filename the operator's choice implies and
    # put it beside the uploaded target, which is somewhere this machine can
    # certainly write.
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        fallback_dir = (
            os.path.dirname(config.target_path)
            if config.target_path and os.path.isdir(os.path.dirname(config.target_path))
            else _upload_dir()
        )
        os.makedirs(fallback_dir, exist_ok=True)
        relocated = os.path.join(fallback_dir, os.path.basename(path))
        emit_status(
            f'Output directory {directory} is not on this machine — '
            f'writing to {relocated}',
            scope='API',
        )
        path = relocated

    # Normalize if directory
    if config.source_path and config.target_path:
        normalized = normalize_output_path(config.source_path, config.target_path, path)
    else:
        normalized = path

    config.set('output_path', normalized)
    emit_status(f'Output set to: {normalized}', scope='API')

    return ResponseMessage(
        type='set_output',
        data={'path': normalized},
        success=True,
    )


# ============================================================================
# Pipeline Control Handlers
# ============================================================================

def handle_start(config: FaceSwapConfig, pipeline: Optional[ProcessingPipeline]) -> ResponseMessage:
    """
    Start the processing pipeline (batch mode).

    Args:
        config: FaceSwapConfig
        pipeline: ProcessingPipeline instance (from server)

    Returns:
        ResponseMessage with success status
    """
    # A photo job carries its targets in `target_paths` and derives an output
    # per photo, so neither `target_path` nor `output_path` applies to it.
    photo_job = bool(config.target_paths)

    if not photo_job and not config.target_path:
        return ResponseMessage(
            type='start',
            data={},
            success=False,
            error='Target path not set',
        )

    if not config.source_path and not config.source_paths:
        return ResponseMessage(
            type='start',
            data={},
            success=False,
            error='Source path not set',
        )

    if not photo_job and not config.output_path:
        return ResponseMessage(
            type='start',
            data={},
            success=False,
            error='Output path not set',
        )

    if pipeline is None:
        return ResponseMessage(
            type='start',
            data={},
            success=False,
            error='Pipeline not initialized',
        )

    if pipeline.is_running():
        return ResponseMessage(
            type='start',
            data={},
            success=False,
            error='Pipeline already running',
        )

    # Start in background thread
    thread = threading.Thread(target=pipeline.run_batch, daemon=True)
    thread.start()
    emit_status('Batch processing started', scope='API')

    return ResponseMessage(
        type='start',
        data={},
        success=True,
    )


def handle_get_state(
    config: FaceSwapConfig,
    pipeline: Optional[ProcessingPipeline],
) -> ResponseMessage:
    """Return current pipeline state so a reconnecting client can sync its UI."""
    source_loaded = False
    if pipeline is not None and hasattr(pipeline, '_swapping_proc'):
        source_loaded = pipeline._swapping_proc.source_face is not None

    return ResponseMessage(
        type='get_state',
        data={
            'source_path': config.source_path,
            'source_paths': config.source_paths,
            'quality': config.quality,
            'enhance': config.enhance,
            'pipeline_running': pipeline.is_running() if pipeline else False,
            'source_loaded': source_loaded,
        },
        success=True,
    )


def handle_start_stream(config: FaceSwapConfig, pipeline: Optional[ProcessingPipeline]) -> ResponseMessage:
    """
    Start the streaming pipeline (webcam/realtime mode).

    Args:
        config: FaceSwapConfig
        pipeline: ProcessingPipeline instance (from server)

    Returns:
        ResponseMessage with success status
    """
    if not config.source_path and not config.source_paths:
        return ResponseMessage(
            type='start_stream',
            data={},
            success=False,
            error='Source path not set',
        )

    if pipeline is None:
        return ResponseMessage(
            type='start_stream',
            data={},
            success=False,
            error='Pipeline not initialized',
        )

    if pipeline.is_running():
        # Pipeline still running from a previous session (e.g. desktop
        # closed without pressing stop).  Let the new client join the
        # existing stream instead of rejecting it.
        return ResponseMessage(
            type='start_stream',
            data={'rejoined': True},
            success=True,
        )

    # Start in background thread
    thread = threading.Thread(target=pipeline.run_stream, daemon=True)
    thread.start()
    emit_status('Stream pipeline started', scope='API')

    return ResponseMessage(
        type='start_stream',
        data={},
        success=True,
    )


def handle_stop(pipeline: Optional[ProcessingPipeline]) -> ResponseMessage:
    """
    Stop the processing pipeline.

    Args:
        pipeline: ProcessingPipeline instance (from server)

    Returns:
        ResponseMessage with success status
    """
    if pipeline is None:
        return ResponseMessage(
            type='stop',
            data={},
            success=False,
            error='Pipeline not initialized',
        )

    if not pipeline.is_running():
        return ResponseMessage(
            type='stop',
            data={},
            success=True,
            error='Pipeline not running',
        )

    pipeline.stop()
    emit_status('Pipeline stop requested', scope='API')

    return ResponseMessage(
        type='stop',
        data={},
        success=True,
    )


# ============================================================================
# Configuration Handlers
# ============================================================================

def handle_set_quality(config: FaceSwapConfig, preset: str) -> ResponseMessage:
    """
    Set quality preset (fast/optimal/production).

    Args:
        config: FaceSwapConfig
        preset: Preset name

    Returns:
        ResponseMessage with success status
    """
    from pipeline.api.schema import PRESETS

    if preset not in PRESETS:
        return ResponseMessage(
            type='set_quality',
            data={'preset': preset},
            success=False,
            error=f'Unknown preset: {preset}. Available: {list(PRESETS.keys())}',
        )

    config.apply_preset(preset)
    emit_status(f'Quality preset set to: {preset}', scope='API')

    return ResponseMessage(
        type='set_quality',
        data={'preset': preset},
        success=True,
    )


def handle_set_blend(config: FaceSwapConfig, value: float) -> ResponseMessage:
    """
    Set blend factor (0.0-1.0).

    Args:
        config: FaceSwapConfig
        value: Blend factor

    Returns:
        ResponseMessage with success status
    """
    if not (0.0 <= value <= 1.0):
        return ResponseMessage(
            type='set_blend',
            data={'value': value},
            success=False,
            error='Blend must be between 0.0 and 1.0',
        )

    config.set('blend', value)
    emit_status(f'Blend set to: {value}', scope='API')

    return ResponseMessage(
        type='set_blend',
        data={'value': value},
        success=True,
    )


def handle_set_alpha(config: FaceSwapConfig, value: float) -> ResponseMessage:
    """
    Set alpha factor for keypoint smoothing (0.0-1.0).

    Args:
        config: FaceSwapConfig
        value: Alpha factor

    Returns:
        ResponseMessage with success status
    """
    if not (0.0 <= value <= 1.0):
        return ResponseMessage(
            type='set_alpha',
            data={'value': value},
            success=False,
            error='Alpha must be between 0.0 and 1.0',
        )

    config.set('alpha', value)
    emit_status(f'Alpha set to: {value}', scope='API')

    return ResponseMessage(
        type='set_alpha',
        data={'value': value},
        success=True,
    )


def handle_set_keep_fps(config: FaceSwapConfig, value: bool) -> ResponseMessage:
    """
    Preserve the target's own frame rate, or retime the output to 30fps.

    An output-format choice rather than a quality one, which is why it is
    settable at runtime while `many_faces` and `keep_frames` are not: it
    decides what file the operator gets, not how the face looks.

    It was declared and never implemented, so a desktop render inherited
    `False` and quietly retimed every clip to 30fps - duration preserved, but
    frames duplicated or dropped against the source cadence.

    Args:
        config: FaceSwapConfig
        value: True to keep the source rate, False to retime to 30fps

    Returns:
        ResponseMessage with success status
    """
    config.set('keep_fps', bool(value))
    emit_status(
        'Output frame rate: source' if value else 'Output frame rate: 30fps',
        scope='API',
    )
    return ResponseMessage(
        type='set_keep_fps',
        data={'value': bool(value)},
        success=True,
    )


def handle_set_keep_audio(config: FaceSwapConfig, value: bool) -> ResponseMessage:
    """
    Keep the target's audio track in the rendered output, or drop it.

    Args:
        config: FaceSwapConfig
        value: True to restore the source audio, False for a silent output

    Returns:
        ResponseMessage with success status
    """
    config.set('keep_audio', bool(value))
    emit_status(
        'Output audio: kept' if value else 'Output audio: dropped',
        scope='API',
    )
    return ResponseMessage(
        type='set_keep_audio',
        data={'value': bool(value)},
        success=True,
    )


def handle_set_enhance(config: FaceSwapConfig, value: bool) -> ResponseMessage:
    """
    Enable or disable GFPGAN face enhancement.

    Args:
        config: FaceSwapConfig
        value: True to enable, False to disable

    Returns:
        ResponseMessage with success status
    """
    config.set('enhance', bool(value))
    state = 'enabled' if value else 'disabled'
    emit_status(f'Enhancement {state}', scope='API')

    return ResponseMessage(
        type='set_enhance',
        data={'value': value},
        success=True,
    )


def handle_set_color_correction(config: FaceSwapConfig, value: bool) -> ResponseMessage:
    """
    Enable or disable color correction for cross-skin-tone swaps.

    Args:
        config: FaceSwapConfig
        value: True to enable, False to disable

    Returns:
        ResponseMessage with success status
    """
    config.set('color_correction', bool(value))
    state = 'enabled' if value else 'disabled'
    emit_status(f'Color correction {state}', scope='API')

    return ResponseMessage(
        type='set_color_correction',
        data={'value': value},
        success=True,
    )


def handle_set_preprocessing(config: FaceSwapConfig, value: bool) -> ResponseMessage:
    """
    Enable or disable frame preprocessing (CLAHE, white balance, denoise).

    Args:
        config: FaceSwapConfig
        value: True to enable, False to disable

    Returns:
        ResponseMessage with success status
    """
    config.set('preprocessing', bool(value))
    state = 'enabled' if value else 'disabled'
    emit_status(f'Preprocessing {state}', scope='API')

    return ResponseMessage(
        type='set_preprocessing',
        data={'value': value},
        success=True,
    )


# Realism tuning parameters settable at runtime, with their validators.
# Grouped into one command rather than a handler each: they are knobs that get
# adjusted together while comparing settings on real footage, and none of them
# has (or needs) its own control in the desktop header.
_REALISM_FIELDS: Dict[str, Any] = {
    # Switchable live, which is the point: comparing two swappers on the same
    # clip is the only way to know which is actually better. Changing it also
    # applies that model's realism profile, via the pipeline's config listener.
    'swapper_model': lambda v: str(v) if str(v) in swapper_models.names() else None,
    'enhancer_model': lambda v: str(v) if str(v) in ('codeformer', 'gfpgan') else None,
    'enhancer_weight': lambda v: min(1.0, max(0.0, float(v))),
    'enhance_strength': lambda v: min(1.0, max(0.0, float(v))),
    'aligned_size': lambda v: min(512, max(128, int(v))),
    'temporal_alpha': lambda v: min(1.0, max(0.0, float(v))),
    'color_strength': lambda v: min(1.0, max(0.0, float(v))),
    'grain': lambda v: bool(v),
    'occluder': lambda v: bool(v),
    # Restoration on/off. It has its own `set_enhance` command and is
    # deliberately absent from PRESETS — a preset must not silently undo
    # something the operator clicked. Neither reason applies to `set_realism`,
    # which is the live A/B mechanism, and restoration is 75% of the frame:
    # being unable to sweep the largest cost in the pipeline made the one
    # measurement that bounds every optimisation impossible to take.
    'enhance': lambda v: bool(v),
    # Inference speed levers. Live-switchable for the same reason the swapper
    # is: the only way to know what a lever is worth is to compare it against
    # the same clip, and a pod session is paid for by the hour. Restarting the
    # pipeline to change one would mean a cold start per measurement.
    #
    # The first four rebuild every ONNX session, which the pipeline's config
    # listener does. That costs model load — seconds, not the minutes a pod
    # restart costs — and TensorRT additionally builds or loads an engine.
    'fp16': lambda v: bool(v),
    'cuda_graphs': lambda v: bool(v),
    'cuda_streams': lambda v: bool(v),
    'trt': lambda v: bool(v),
    # Needs no rebuild: the server reads it per frame.
    'async_encode': lambda v: bool(v),
}


def handle_set_realism(config: FaceSwapConfig, values: Dict[str, Any]) -> ResponseMessage:
    """
    Set one or more realism, compositing or guard-threshold parameters.

    Guard thresholds share this command rather than getting one of their own,
    for the same reason the realism knobs share it: they are adjusted together
    while comparing behaviour on real footage. Guard values are *clamped* to
    their legal range rather than rejected, so an operator sweeping a threshold
    live gets the nearest legal value instead of an error.

    Unknown keys and values that fail validation are reported back rather than
    applied, so a typo during tuning does not silently do nothing.

    Args:
        config: FaceSwapConfig
        values: Mapping of field name to new value

    Returns:
        ResponseMessage listing what was applied and what was rejected
    """
    applied: Dict[str, Any] = {}
    rejected: Dict[str, str] = {}

    for field, raw in (values or {}).items():
        if field in guards.GUARD_FIELDS:
            accepted, value, error = guards.validate_guard_value(field, raw)
            if not accepted:
                rejected[field] = error
                continue
            config.set(field, value)
            applied[field] = value
            continue

        validator = _REALISM_FIELDS.get(field)
        if validator is None:
            rejected[field] = 'unknown field'
            continue
        try:
            value = validator(raw)
        except (TypeError, ValueError):
            value = None
        if value is None:
            rejected[field] = f'invalid value: {raw!r}'
            continue
        config.set(field, value)
        applied[field] = value

    if applied:
        summary = ', '.join(f'{k}={v}' for k, v in applied.items())
        emit_status(f'Realism settings updated: {summary}', scope='API')

    return ResponseMessage(
        type='set_realism',
        data={'applied': applied, 'rejected': rejected},
        success=not rejected,
        error=None if not rejected else f'Rejected: {rejected}',
    )


def handle_set_input_url(config: FaceSwapConfig, url: str) -> ResponseMessage:
    """
    Set network input stream URL.

    Args:
        config: FaceSwapConfig
        url: Stream URL (RTSP, RTMP, etc.)

    Returns:
        ResponseMessage with success status
    """
    config.set('input_url', url if url else None)
    emit_status(f'Input URL set to: {url}', scope='API')

    return ResponseMessage(
        type='set_input_url',
        data={'url': url},
        success=True,
    )


# ============================================================================
# Embedding/Session Handlers
# ============================================================================

def handle_upload_source(
    config: FaceSwapConfig,
    images: List[Dict[str, Any]],
    pipeline: Optional[ProcessingPipeline] = None,
) -> ResponseMessage:
    """
    Receive source image(s) as base64-encoded bytes, save to temp dir, and
    set source config. Replaces path-based set_source / create_embedding for
    remote deployments where the client cannot share a filesystem with the server.

    Images that fail the source guards are reported individually — which image
    and why — rather than as one opaque failure. This is an upload flow with a
    person present; they need to know which photo to replace.

    Args:
        config: FaceSwapConfig
        images: List of dicts with 'name' (filename) and 'data' (base64 string)
        pipeline: Pipeline whose source review should be reported, if available

    Returns:
        ResponseMessage with saved server-side paths on success
    """
    if not images:
        return ResponseMessage(
            type='upload_source',
            data={},
            success=False,
            error='No images provided',
        )

    saved: List[str] = []

    for img in images:
        name = os.path.basename(img.get('name', 'source.jpg'))
        b64 = img.get('data', '')
        if not b64:
            return ResponseMessage(
                type='upload_source',
                data={},
                success=False,
                error=f'No image data for: {name}',
            )
        try:
            image_bytes = base64.b64decode(b64)
        except Exception as e:
            emit_error(f'Base64 decode failed for {name}: {type(e).__name__}: {e}', scope='HANDLERS')
            return ResponseMessage(
                type='upload_source',
                data={},
                success=False,
                error=f'Invalid base64 data for: {name}',
            )

        path = os.path.join(_upload_dir(), name)
        with open(path, 'wb') as fh:
            fh.write(image_bytes)
        saved.append(path)

    # Setting these runs the source guards synchronously, via the pipeline's
    # config listener, so the review is available by the time this returns.
    config.set('source_paths', saved)
    config.set('source_path', saved[0])

    review = _source_review(pipeline)
    if review is None:
        emit_status(f'Source uploaded: {len(saved)} image(s)', scope='API')
        return ResponseMessage(
            type='upload_source',
            data={'paths': saved, 'count': len(saved)},
            success=True,
        )

    data: Dict[str, Any] = {
        'paths': review.accepted,
        'count': len(review.accepted),
        'uploaded': len(saved),
        **review.to_dict(),
    }

    if not review.usable:
        # Every image refused. Reported as a failure with a reason per image,
        # because there is a person on the other end of this who has to know
        # which photo to replace.
        reasons = '; '.join(
            f'{os.path.basename(p)}: {review.messages.get(p, r)}'
            for p, r in review.rejected
        )
        return ResponseMessage(
            type='upload_source',
            data=data,
            success=False,
            error=f'No usable source image — {reasons}',
        )

    if review.rejected:
        emit_status(
            f'Source uploaded: {len(review.accepted)} of {len(saved)} image(s) '
            f'accepted',
            scope='API',
        )
    else:
        emit_status(f'Source uploaded: {len(saved)} image(s)', scope='API')

    return ResponseMessage(
        type='upload_source',
        data=data,
        success=True,
    )


def _clear_template(config: FaceSwapConfig) -> None:
    """
    Forget the selected template.

    Its face point and foreground describe *that scene* and nothing else. Left
    in place they would silently steer face selection on an unrelated target,
    and quietly draw someone else's hair over it — both invisible until the
    output is wrong.
    """
    config.set('template_id', None)
    config.set('target_face_point', None)
    config.set('target_face_points', [])
    config.set('target_foreground', None)
    config.set('output_dir', None)


def handle_list_templates() -> ResponseMessage:
    """
    List the bundled target templates.

    Thumbnails travel inline because the library lives on the pipeline's
    filesystem, which on a pod is not the operator's machine — the same reason
    `get_photo_results` returns images rather than paths.

    Returns:
        ResponseMessage with one entry per template
    """
    library = TemplateLibrary()
    entries = []
    for template in library.all():
        entry = template.to_dict()
        thumbnail = template.thumbnail or template.image
        try:
            with open(thumbnail, 'rb') as fh:
                entry['thumbnail'] = base64.b64encode(fh.read()).decode('ascii')
        except OSError as e:
            # A template whose thumbnail cannot be read is still usable; it
            # just shows without a picture rather than vanishing.
            emit_warning(
                f'Template {template.id}: thumbnail unreadable ({e})',
                scope='API',
            )
        entries.append(entry)

    return ResponseMessage(
        type='list_templates',
        data={'templates': entries, 'count': len(entries)},
        success=True,
    )


def handle_set_template(config: FaceSwapConfig, template_id: str) -> ResponseMessage:
    """
    Choose a bundled template as the target.

    Sets the same `target_path` an uploaded photo would, plus the two things
    only a template knows: which face its author chose, and any layer that
    belongs in front of the swap.

    Args:
        config: FaceSwapConfig
        template_id: Id from `list_templates`

    Returns:
        ResponseMessage with the resolved template
    """
    if not template_id:
        return ResponseMessage(
            type='set_template',
            data={},
            success=False,
            error='Template id cannot be empty',
        )

    template = TemplateLibrary().get(template_id)
    if template is None:
        return ResponseMessage(
            type='set_template',
            data={'id': template_id},
            success=False,
            error=f'No such template: {template_id}',
        )

    # Run as a photo job of one. That is what it is — a single still, swapped
    # and handed back inline — and it means the result, the per-item event and
    # the image return path are the ones photo mode already proved rather than
    # a second set that would have to be kept in step.
    config.set('target_paths', [template.image])
    config.set('target_path', None)
    config.set('template_id', template.id)
    config.set('target_face_point', template.face_point)
    config.set('target_foreground', template.foreground)

    config.set('output_dir', tempfile.mkdtemp(prefix='template_', dir=_upload_dir()))

    emit_status(f'Template set: {template.name}', scope='API')
    return ResponseMessage(
        type='set_template',
        data={'template': template.to_dict()},
        success=True,
    )


# In-progress video uploads, keyed by the id handed back at begin.
#
# Server-side state, which nothing else in this module keeps. A video is too
# large for the one-message shape every other upload uses, so its transfer has a
# beginning and an end, and something has to remember the middle.
_VIDEO_UPLOADS: Dict[str, Dict[str, Any]] = {}


def _first_frame_jpeg(path: str, max_side: int = 640) -> Optional[str]:
    """
    Read frame zero of a video as a base64 JPEG, for a preview thumbnail.

    The render panes show what is going in and what came out, and neither file
    is reachable from the desktop: both live on the pipeline's filesystem, which
    on a pod is another machine. A path would be useless there, so the picture
    travels instead — the same reason `get_photo_results` returns images rather
    than paths.

    Downscaled because it is a thumbnail: a 1080p first frame is ~200 KB of JPEG
    to say something a 640px one says as well.

    Args:
        path: Video (or image) to read the first frame of
        max_side: Long side to fit the thumbnail within

    Returns:
        Base64 JPEG, or None if the file could not be read
    """
    try:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            ok, frame = capture.read()
        finally:
            capture.release()

        if not ok or frame is None:
            return None

        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > max_side:
            scale = max_side / float(longest)
            frame = cv2.resize(
                frame, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode('ascii')
    except Exception as e:
        emit_warning(
            f'Could not read a first frame from {path}: {type(e).__name__}: {e}',
            scope='HANDLERS',
        )
        return None


def handle_get_render_thumbnails(config: FaceSwapConfig) -> ResponseMessage:
    """
    First frames of the configured render target and its output.

    Deliberately takes no path. A command that thumbnailed any path the client
    named would read arbitrary files off the pod, so this answers only for what
    the config already points at — which is exactly what the render panes show.

    Either may be absent: the output does not exist until a render finishes, and
    asking before then is normal rather than an error.

    Args:
        config: FaceSwapConfig, supplying `target_path` and `output_path`

    Returns:
        ResponseMessage with `target` and `output` base64 JPEGs, each possibly
        None
    """
    target = config.target_path
    output = config.output_path

    return ResponseMessage(
        type='get_render_thumbnails',
        data={
            'target': _first_frame_jpeg(target) if target and os.path.isfile(target) else None,
            'output': _first_frame_jpeg(output) if output and os.path.isfile(output) else None,
            'target_path': target,
            'output_path': output,
        },
    )


def handle_get_output_info(config: FaceSwapConfig) -> ResponseMessage:
    """
    Size and name of the finished render, so the desktop can fetch it.

    A render writes on the *pipeline's* filesystem. When that is a pod it is
    another machine, so the operator has no way to reach the file they just paid
    to produce — the same gap uploading solved in the other direction. Photo
    mode already answers this by returning images inline; a video is too large
    for that, so it is read back in chunks.

    Args:
        config: FaceSwapConfig, supplying `output_path`

    Returns:
        ResponseMessage with `name` and `size`, or an error if nothing is there
    """
    path = config.output_path
    if not path or not os.path.isfile(path):
        return ResponseMessage(
            type='get_output_info', data={}, success=False,
            error='No output file to download',
        )
    return ResponseMessage(
        type='get_output_info',
        data={
            'name': os.path.basename(path),
            'size': os.path.getsize(path),
            'path': path,
        },
    )


def handle_get_output_chunk(
    config: FaceSwapConfig,
    offset: int,
    length: int,
) -> ResponseMessage:
    """
    Read a slice of the finished render.

    Takes no path, for the reason `get_render_thumbnails` does not: a command
    that read any path a client named would serve arbitrary files off the pod.
    It answers only for the output the config already points at.

    Args:
        config: FaceSwapConfig, supplying `output_path`
        offset: Byte offset to read from
        length: Bytes to read, clamped to the chunk size

    Returns:
        ResponseMessage with base64 `data` and whether the file ends here
    """
    path = config.output_path
    if not path or not os.path.isfile(path):
        return ResponseMessage(
            type='get_output_chunk', data={}, success=False,
            error='No output file to download',
        )

    size = os.path.getsize(path)
    if offset < 0 or offset > size:
        return ResponseMessage(
            type='get_output_chunk', data={}, success=False,
            error=f'Offset {offset} outside a {size} byte file',
        )

    length = max(1, min(int(length or VIDEO_CHUNK_BYTES), VIDEO_CHUNK_BYTES))
    with open(path, 'rb') as fh:
        fh.seek(offset)
        chunk = fh.read(length)

    return ResponseMessage(
        type='get_output_chunk',
        data={
            'offset': offset,
            'size': size,
            'data': base64.b64encode(chunk).decode('ascii'),
            'eof': offset + len(chunk) >= size,
        },
    )


def _remove_quietly(path: str) -> None:
    """Delete a refused upload; its own directory goes with it."""
    try:
        os.remove(path)
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass


def _discard_video_upload(upload_id: str) -> None:
    """Close and delete a partial upload, whatever state it is in."""
    entry = _VIDEO_UPLOADS.pop(upload_id, None)
    if not entry:
        return
    try:
        entry["handle"].close()
    except Exception:
        pass
    _remove_quietly(entry["path"])


def handle_upload_video_begin(
    config: FaceSwapConfig,
    name: str,
    size: int,
) -> ResponseMessage:
    """
    Open a chunked upload for a target video.

    Video needs a transfer path of its own because the photo one cannot stretch
    to it: `upload_target` carries an image base64 in a single message, and the
    server caps a message at 64 MB, which base64 turns into a 48 MB ceiling on
    the file itself. Chunking takes the message size out of the product limit.

    The declared size is refused here so an oversized file is stopped before any
    of it is sent rather than after a minutes-long transfer. It is checked again
    while chunks arrive, because a declaration is the client's claim and the
    bytes are the fact.

    Args:
        config: FaceSwapConfig
        name: Original filename, for the extension and the saved name
        size: Declared total size in bytes

    Returns:
        ResponseMessage carrying `upload_id` and the chunk size to use
    """
    name = os.path.basename(name or "") or "target.mp4"

    if not is_video_name(name):
        return ResponseMessage(
            type="upload_video_begin", data={}, success=False,
            error=f"{name} is not a video format this pipeline can read",
        )

    if size <= 0:
        return ResponseMessage(
            type="upload_video_begin", data={}, success=False,
            error="Declared size must be positive",
        )

    if size > MAX_VIDEO_BYTES:
        return ResponseMessage(
            type="upload_video_begin",
            data={"max_bytes": MAX_VIDEO_BYTES},
            success=False,
            error=(
                f"{size / (1024 * 1024):.0f} MB exceeds the "
                f"{MAX_VIDEO_BYTES // (1024 * 1024)} MB limit"
            ),
        )

    job_dir = tempfile.mkdtemp(prefix="video_", dir=_upload_dir())
    path = os.path.join(job_dir, name)

    upload_id = os.path.basename(job_dir)
    _VIDEO_UPLOADS[upload_id] = {
        "path": path,
        "handle": open(path, "wb"),
        "written": 0,
        "declared": size,
        "next_seq": 0,
    }

    emit_status(
        f"Receiving target video {name} ({size / (1024 * 1024):.0f} MB)...",
        scope="API",
    )
    return ResponseMessage(
        type="upload_video_begin",
        data={"upload_id": upload_id, "chunk_bytes": VIDEO_CHUNK_BYTES},
    )


def handle_upload_video_chunk(
    upload_id: str,
    seq: int,
    data: str,
) -> ResponseMessage:
    """
    Append one chunk to an open upload.

    Sequence numbers are checked rather than assumed. A single WebSocket
    connection delivers in order, so an out-of-order chunk means a client bug or
    a retry against a stale upload — and either way the assembled file would be
    silently corrupt, which surfaces as a render failing minutes later for no
    stated reason. Refusing here names the cause while it is still visible.

    Args:
        upload_id: From `upload_video_begin`
        seq: Zero-based chunk index
        data: Base64 chunk

    Returns:
        ResponseMessage with bytes received so far, for progress
    """
    entry = _VIDEO_UPLOADS.get(upload_id)
    if entry is None:
        return ResponseMessage(
            type="upload_video_chunk", data={}, success=False,
            error="Unknown or already-finished upload",
        )

    expected = entry["next_seq"]
    if seq != expected:
        _discard_video_upload(upload_id)
        return ResponseMessage(
            type="upload_video_chunk", data={}, success=False,
            error=f"Chunk out of order (expected {expected}, got {seq})",
        )

    try:
        chunk = base64.b64decode(data)
    except Exception as e:
        _discard_video_upload(upload_id)
        return ResponseMessage(
            type="upload_video_chunk", data={}, success=False,
            error=f"Invalid base64 in chunk {seq}: {type(e).__name__}",
        )

    if entry["written"] + len(chunk) > MAX_VIDEO_BYTES:
        _discard_video_upload(upload_id)
        return ResponseMessage(
            type="upload_video_chunk", data={}, success=False,
            error=(
                f"Upload exceeded the "
                f"{MAX_VIDEO_BYTES // (1024 * 1024)} MB limit"
            ),
        )

    entry["handle"].write(chunk)
    entry["written"] += len(chunk)
    entry["next_seq"] = seq + 1

    return ResponseMessage(
        type="upload_video_chunk",
        data={"received": entry["written"], "declared": entry["declared"]},
    )


def handle_upload_video_end(
    config: FaceSwapConfig,
    upload_id: str,
) -> ResponseMessage:
    """
    Close a completed upload, check its duration, and stage it as the target.

    Duration is probed here rather than trusted from the client, for the reason
    the byte count is re-checked: a limit only the desktop enforces is not a
    limit. Bytes and seconds refuse different things — bytes bound the transfer,
    seconds bound the render, and a well compressed ten minutes can be smaller
    than a badly compressed one while costing twenty times as much to process.

    A clip whose duration cannot be read is refused. An unknown duration against
    a limit is not a pass, and ffprobe failing here usually means the file is
    not the video it claimed to be.

    Args:
        config: FaceSwapConfig
        upload_id: From `upload_video_begin`

    Returns:
        ResponseMessage with the server-side path and the clip's duration
    """
    entry = _VIDEO_UPLOADS.get(upload_id)
    if entry is None:
        return ResponseMessage(
            type="upload_video_end", data={}, success=False,
            error="Unknown or already-finished upload",
        )

    path = entry["path"]
    try:
        entry["handle"].close()
    except Exception:
        pass
    _VIDEO_UPLOADS.pop(upload_id, None)

    written = entry["written"]
    if written == 0:
        _remove_quietly(path)
        return ResponseMessage(
            type="upload_video_end", data={}, success=False,
            error="No data received",
        )

    duration = probe_duration(path)
    if duration is None:
        _remove_quietly(path)
        return ResponseMessage(
            type="upload_video_end", data={}, success=False,
            error="Could not read the clip — it may be corrupt or not a video",
        )

    if duration > MAX_VIDEO_SECONDS:
        _remove_quietly(path)
        return ResponseMessage(
            type="upload_video_end",
            data={"max_seconds": MAX_VIDEO_SECONDS, "duration": duration},
            success=False,
            error=(
                f"{duration / 60:.1f} min exceeds the "
                f"{MAX_VIDEO_SECONDS // 60} min limit"
            ),
        )

    # Single-file target. Photo mode is signalled by which field holds the
    # targets, so the list has to be cleared or a stale batch would win.
    config.set("target_path", path)
    config.set("target_paths", [])
    config.set("target_face_points", [])
    _clear_template(config)
    config.set("output_dir", None)

    emit_status(
        f"Target video ready: {os.path.basename(path)} "
        f"({written / (1024 * 1024):.0f} MB, {duration:.0f}s)",
        scope="API",
    )
    return ResponseMessage(
        type="upload_video_end",
        data={
            "path": path,
            "bytes": written,
            "duration": duration,
            # Sent with the reply rather than fetched afterwards: the file is
            # already open here, and the pane should fill the moment the
            # transfer lands.
            "thumbnail": _first_frame_jpeg(path),
        },
    )


def handle_upload_video_cancel(upload_id: str) -> ResponseMessage:
    """
    Abandon an upload and delete what arrived.

    Without this a cancelled transfer leaves its partial file on the volume
    until the pod is discarded, and a 200 MB ceiling makes that worth cleaning
    up. Successful whether or not the id was still open, so a cancel racing a
    failure is not itself an error.

    Args:
        upload_id: From `upload_video_begin`

    Returns:
        ResponseMessage echoing the id
    """
    _discard_video_upload(upload_id)
    return ResponseMessage(
        type="upload_video_cancel", data={"upload_id": upload_id})


def handle_upload_target(
    config: FaceSwapConfig,
    images: List[Dict[str, Any]],
    pipeline: Optional[ProcessingPipeline] = None,
) -> ResponseMessage:
    """
    Receive up to four target photos as base64 bytes and stage them for a
    photo-mode job.

    Targets have never had a transfer path — `set_target` validates with
    `os.path.exists` against the *pipeline's* filesystem, so a file chosen on a
    desktop only worked when the pipeline ran on the same machine. Photos are
    small enough to carry inline, which is why this exists for images and not
    for video.

    A malformed or oversized image is refused individually and the rest are
    kept, matching how the job itself treats a photo it cannot swap.

    Each accepted photo is then *counted*: the response carries every face
    found, normalised, so the desktop can ask which one the operator meant
    before the job runs. Detection happens here rather than at swap time
    because here is where the person is — a photo refused mid-job for holding
    two faces tells them only that they already picked the wrong one.

    Args:
        config: FaceSwapConfig
        images: List of dicts with 'name' (filename) and 'data' (base64 string)
        pipeline: ProcessingPipeline, for the detector. Without one the photos
                  are still staged, just with no face counts

    Returns:
        ResponseMessage with the staged server-side paths and their faces
    """
    if not images:
        return ResponseMessage(
            type='upload_target',
            data={},
            success=False,
            error='No images provided',
        )

    if len(images) > MAX_PHOTO_TARGETS:
        return ResponseMessage(
            type='upload_target',
            data={'max': MAX_PHOTO_TARGETS},
            success=False,
            error=f'At most {MAX_PHOTO_TARGETS} target photos ({len(images)} given)',
        )

    # Its own directory per job. The upload dir is shared with sources and
    # persists across jobs, so without this a target named like a previously
    # uploaded file would silently overwrite it — and two photos picked from
    # different folders with the same camera filename would overwrite each
    # other.
    job_dir = tempfile.mkdtemp(prefix='targets_', dir=_upload_dir())

    saved: List[str] = []
    rejected: List[Dict[str, str]] = []

    for index, img in enumerate(images):
        name = os.path.basename(img.get('name', f'target_{index}.jpg')) or f'target_{index}.jpg'
        b64 = img.get('data', '')
        if not b64:
            rejected.append({'name': name, 'reason': 'no image data'})
            continue

        try:
            image_bytes = base64.b64decode(b64)
        except Exception as e:
            emit_error(
                f'Base64 decode failed for {name}: {type(e).__name__}: {e}',
                scope='HANDLERS',
            )
            rejected.append({'name': name, 'reason': 'invalid base64 data'})
            continue

        if len(image_bytes) > MAX_PHOTO_BYTES:
            rejected.append({
                'name': name,
                'reason': (
                    f'{len(image_bytes) / (1024 * 1024):.1f} MB exceeds the '
                    f'{MAX_PHOTO_BYTES // (1024 * 1024)} MB limit'
                ),
            })
            continue

        path = os.path.join(job_dir, name)
        with open(path, 'wb') as fh:
            fh.write(image_bytes)
        saved.append(path)

    if not saved:
        reasons = '; '.join(f"{r['name']}: {r['reason']}" for r in rejected)
        return ResponseMessage(
            type='upload_target',
            data={'paths': [], 'rejected': rejected},
            success=False,
            error=f'No usable target photo — {reasons}',
        )

    # Photo mode is signalled by which field holds the targets, so the
    # single-file path has to be cleared or a stale target would win.
    config.set('target_paths', saved)
    config.set('target_path', None)
    _clear_template(config)
    # A choice made about the previous batch says nothing about this one, and a
    # stale point would silently name a face in a photo nobody has looked at.
    config.set('target_face_points', [])
    # Uploaded photos are written beside themselves, inside the job directory.
    config.set('output_dir', None)

    emit_status(f'Target photos uploaded: {len(saved)}', scope='API')
    return ResponseMessage(
        type='upload_target',
        data={
            'paths': saved,
            'count': len(saved),
            'rejected': rejected,
            'faces': _count_target_faces(saved, pipeline),
        },
        success=True,
    )


def _count_target_faces(
    paths: List[str],
    pipeline: Optional[ProcessingPipeline],
) -> List[Dict[str, Any]]:
    """
    Every face in each staged photo, as normalised boxes.

    Normalised for the same reason `face_point` is: the desktop draws these
    over a scaled preview, and a pixel box would be wrong the moment the
    preview is not the photo's own size.

    Detection failing is not an upload failure. The photos are staged and the
    job will run; the operator is simply not offered a choice, and a multi-face
    photo is refused at swap time exactly as it was before.

    Args:
        paths: Staged photo paths
        pipeline: ProcessingPipeline, or None

    Returns:
        One entry per photo, in the same order
    """
    entries: List[Dict[str, Any]] = []

    for path in paths:
        boxes: List[Dict[str, float]] = []
        if pipeline is not None:
            try:
                boxes = pipeline.face_boxes(path)
            except Exception as e:
                emit_error(
                    f'Could not count faces in {os.path.basename(path)}: '
                    f'{type(e).__name__}: {e}',
                    scope='HANDLERS',
                )

        entries.append({
            'path': path,
            'name': os.path.basename(path),
            'boxes': boxes,
        })

    return entries


def handle_set_target_faces(
    config: FaceSwapConfig,
    points: List[Any],
) -> ResponseMessage:
    """
    Record which face the operator picked in each uploaded target photo.

    Aligned with `target_paths` by index, `None` for a photo they were not
    asked about. This is the operator's half of the seam a template's manifest
    fills from the other side: it names the face, so `select_by_point` uses it
    and the multi-face guard stands down for that photo alone.

    Args:
        config: FaceSwapConfig
        points: One [x, y] normalised pair, or null, per target photo

    Returns:
        ResponseMessage echoing the stored points
    """
    targets = config.target_paths
    if not targets:
        return ResponseMessage(
            type='set_target_faces',
            data={},
            success=False,
            error='No target photos to name faces in — upload them first',
        )

    if len(points) > len(targets):
        return ResponseMessage(
            type='set_target_faces',
            data={'targets': len(targets)},
            success=False,
            error=f'{len(points)} points for {len(targets)} target photos',
        )

    stored: List[Optional[Tuple[float, float]]] = []
    for index, point in enumerate(points):
        if point is None:
            stored.append(None)
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return ResponseMessage(
                type='set_target_faces',
                data={'index': index, 'point': point},
                success=False,
                error=f'Point {index} is not an [x, y] pair',
            )
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return ResponseMessage(
                type='set_target_faces',
                data={'index': index, 'point': [x, y]},
                success=False,
                error=f'Point {index} is outside the image: ({x:.3f}, {y:.3f})',
            )
        stored.append((x, y))

    config.set('target_face_points', stored)
    named = sum(1 for p in stored if p is not None)
    emit_status(
        f'Target face chosen in {named} of {len(targets)} photos', scope='API',
    )
    return ResponseMessage(
        type='set_target_faces',
        data={'points': [list(p) if p else None for p in stored]},
        success=True,
    )


def handle_get_photo_results(
    pipeline: Optional[ProcessingPipeline],
    include_images: bool = True,
) -> ResponseMessage:
    """
    Return the per-photo outcomes of the last photo job, with the swapped
    images inline.

    The outputs are written on the pipeline's filesystem, which on a pod is a
    machine the operator cannot see, so a path alone would be useless to them.

    Args:
        pipeline: Pipeline holding the results
        include_images: Attach base64 image bytes for the photos that swapped

    Returns:
        ResponseMessage with one entry per target
    """
    if pipeline is None:
        return ResponseMessage(
            type='get_photo_results',
            data={},
            success=False,
            error='Pipeline not initialized',
        )

    results = []
    for result in pipeline.photo_results:
        entry = result.to_dict()
        if include_images and result.ok and result.output_path:
            try:
                with open(result.output_path, 'rb') as fh:
                    entry['data'] = base64.b64encode(fh.read()).decode('ascii')
            except OSError as e:
                # The swap happened; only the read back failed. Reported as a
                # skip so the client never shows a result it cannot display.
                entry['ok'] = False
                entry['reason'] = f'output could not be read back: {e}'
        results.append(entry)

    swapped = sum(1 for r in results if r['ok'])
    return ResponseMessage(
        type='get_photo_results',
        data={
            'results': results,
            'total': len(results),
            'swapped': swapped,
            'skipped': len(results) - swapped,
        },
        success=True,
    )


def _source_review(pipeline: Optional[ProcessingPipeline]) -> Optional[SourceReview]:
    """
    The most recent source review, if the pipeline has run one.

    Returns None when there is no pipeline or it has not evaluated sources yet,
    in which case callers fall back to reporting a bare success — guards that
    never ran must not be reported as guards that passed.
    """
    processor = getattr(pipeline, '_swapping_proc', None)
    return getattr(processor, 'last_review', None) if processor else None


def handle_create_embedding(config: FaceSwapConfig, paths: List[str]) -> ResponseMessage:
    """
    Set source face paths for averaging (multi-image embedding).

    Validates all paths, sets source_paths on config (so the pipeline loads
    and averages them at stream start), then emits an 'Embedding ready' status
    so the desktop bridge can clear its pending indicator.

    Args:
        config: FaceSwapConfig
        paths: Source image paths

    Returns:
        ResponseMessage with success status
    """
    if not paths:
        return ResponseMessage(
            type='create_embedding',
            data={'paths': paths},
            success=False,
            error='No source paths provided',
        )

    for path in paths:
        if not os.path.exists(path):
            return ResponseMessage(
                type='create_embedding',
                data={'paths': paths},
                success=False,
                error=f'Source path does not exist: {path}',
            )
        if not is_image(path):
            return ResponseMessage(
                type='create_embedding',
                data={'paths': paths},
                success=False,
                error=f'Source must be an image file: {path}',
            )

    try:
        config.set('source_paths', paths)
        config.set('embedding_ready', True)
        # 'Embedding ready' matches bridge.py's status detection pattern
        emit_status('Embedding ready', scope='API')

        return ResponseMessage(
            type='create_embedding',
            data={'paths': paths, 'count': len(paths)},
            success=True,
        )
    except Exception as e:
        return ResponseMessage(
            type='create_embedding',
            data={'paths': paths},
            success=False,
            error=str(e),
        )


def handle_cleanup_session(config: FaceSwapConfig) -> ResponseMessage:
    """
    Clean up current session (clear source, temp files, etc.).

    Args:
        config: FaceSwapConfig

    Returns:
        ResponseMessage with success status
    """
    config.set('source_path', None)
    config.set('source_paths', [])
    config.set('embedding_ready', False)

    # Remove any uploaded temp files
    import shutil
    if os.path.isdir(_upload_dir()):
        shutil.rmtree(_upload_dir(), ignore_errors=True)

    emit_status('Session cleaned up', scope='API')

    return ResponseMessage(
        type='cleanup_session',
        data={},
        success=True,
    )


def handle_keep_alive(reset_fn: Optional[Callable[[], None]]) -> ResponseMessage:
    """
    Reset the auto-stop timer, extending the pod's uptime.

    Args:
        reset_fn: Callback to reset the auto-stop countdown

    Returns:
        ResponseMessage with success status
    """
    if reset_fn is None:
        return ResponseMessage(
            type='keep_alive',
            data={},
            success=False,
            error='Auto-stop timer not active',
        )

    reset_fn()
    emit_status('Auto-stop timer reset', scope='API')

    return ResponseMessage(
        type='keep_alive',
        data={},
        success=True,
    )


def handle_shutdown(shutdown_event: Optional[threading.Event]) -> ResponseMessage:
    """
    Shutdown the application.

    Args:
        shutdown_event: threading.Event to signal shutdown

    Returns:
        ResponseMessage with success status
    """
    if shutdown_event is None:
        return ResponseMessage(
            type='shutdown',
            data={},
            success=False,
            error='Shutdown event not initialized',
        )

    emit_status('Shutdown requested', scope='API')
    shutdown_event.set()

    return ResponseMessage(
        type='shutdown',
        data={},
        success=True,
    )


# ============================================================================
# Unified Handler Dispatcher
# ============================================================================

def dispatch_command(
    command_type: str,
    data: Dict[str, Any],
    config: FaceSwapConfig,
    ctx: HandlerContext,
) -> ResponseMessage:
    """
    Dispatch a command to the appropriate handler.

    Args:
        command_type: Type of command (e.g., 'set_source', 'start')
        data: Command data dictionary
        config: FaceSwapConfig
        ctx: HandlerContext with pipeline and shutdown_event references

    Returns:
        ResponseMessage with result
    """
    try:
        if command_type == 'set_source':
            return handle_set_source(config, data.get('path', ''))

        elif command_type == 'set_source_paths':
            return handle_set_source_paths(config, data.get('paths', []))

        elif command_type == 'set_target':
            return handle_set_target(config, data.get('path', ''))

        elif command_type == 'set_output':
            return handle_set_output(config, data.get('path', ''))

        elif command_type == 'start':
            return handle_start(config, ctx.pipeline)

        elif command_type == 'start_stream':
            return handle_start_stream(config, ctx.pipeline)

        elif command_type == 'stop':
            return handle_stop(ctx.pipeline)

        elif command_type == 'set_quality':
            return handle_set_quality(config, data.get('preset', 'optimal'))

        elif command_type == 'set_blend':
            return handle_set_blend(config, float(data.get('value', 0.65)))

        elif command_type == 'set_alpha':
            return handle_set_alpha(config, float(data.get('value', 0.6)))

        elif command_type == 'set_keep_fps':
            return handle_set_keep_fps(config, bool(data.get('value', True)))

        elif command_type == 'set_keep_audio':
            return handle_set_keep_audio(config, bool(data.get('value', True)))

        elif command_type == 'set_enhance':
            return handle_set_enhance(config, bool(data.get('value', True)))

        elif command_type == 'set_color_correction':
            return handle_set_color_correction(config, bool(data.get('value', True)))

        elif command_type == 'set_preprocessing':
            return handle_set_preprocessing(config, bool(data.get('value', True)))

        elif command_type == 'set_realism':
            return handle_set_realism(config, data.get('values', {}))

        elif command_type == 'set_input_url':
            return handle_set_input_url(config, data.get('url', ''))

        elif command_type == 'upload_source':
            return handle_upload_source(config, data.get('images', []), ctx.pipeline)

        elif command_type == 'list_templates':
            return handle_list_templates()

        elif command_type == 'set_template':
            return handle_set_template(config, str(data.get('id', '')))

        elif command_type == 'upload_target':
            return handle_upload_target(config, data.get('images', []), ctx.pipeline)

        elif command_type == 'upload_video_begin':
            return handle_upload_video_begin(
                config, str(data.get('name', '')), int(data.get('size', 0)))

        elif command_type == 'upload_video_chunk':
            return handle_upload_video_chunk(
                str(data.get('upload_id', '')),
                int(data.get('seq', -1)),
                str(data.get('data', '')),
            )

        elif command_type == 'upload_video_end':
            return handle_upload_video_end(
                config, str(data.get('upload_id', '')))

        elif command_type == 'get_render_thumbnails':
            return handle_get_render_thumbnails(config)

        elif command_type == 'get_output_info':
            return handle_get_output_info(config)

        elif command_type == 'get_output_chunk':
            return handle_get_output_chunk(
                config, int(data.get('offset', 0)), int(data.get('length', 0)))

        elif command_type == 'upload_video_cancel':
            return handle_upload_video_cancel(str(data.get('upload_id', '')))

        elif command_type == 'set_target_faces':
            return handle_set_target_faces(config, data.get('points', []))

        elif command_type == 'get_photo_results':
            return handle_get_photo_results(
                ctx.pipeline, bool(data.get('include_images', True))
            )
        elif command_type == 'create_embedding':
            return handle_create_embedding(config, data.get('paths', []))

        elif command_type == 'cleanup_session':
            return handle_cleanup_session(config)

        elif command_type == 'get_state':
            return handle_get_state(config, ctx.pipeline)

        elif command_type == 'keep_alive':
            return handle_keep_alive(ctx.reset_auto_stop)

        elif command_type == 'shutdown':
            return handle_shutdown(ctx.shutdown_event)

        else:
            return ResponseMessage(
                type=command_type,
                data=data,
                success=False,
                error=f'Unknown command: {command_type}',
            )

    except Exception as e:
        emit_error(f'Command handler error: {e}', exception=e, scope='API')
        return ResponseMessage(
            type=command_type,
            data=data,
            success=False,
            error=f'Handler error: {str(e)}',
        )
