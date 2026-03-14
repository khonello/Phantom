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

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pipeline.config import FaceSwapConfig
from pipeline.api.schema import ResponseMessage
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.logging import emit_status, emit_error
from pipeline.io.ffmpeg import is_image, is_video, normalize_output_path


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
    if not config.target_path:
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

    if not config.output_path:
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
        return ResponseMessage(
            type='start_stream',
            data={},
            success=False,
            error='Pipeline already running',
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

    emit_status('Session cleaned up', scope='API')

    return ResponseMessage(
        type='cleanup_session',
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

        elif command_type == 'set_input_url':
            return handle_set_input_url(config, data.get('url', ''))

        elif command_type == 'create_embedding':
            return handle_create_embedding(config, data.get('paths', []))

        elif command_type == 'cleanup_session':
            return handle_cleanup_session(config)

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
