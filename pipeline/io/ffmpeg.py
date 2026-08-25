"""
FFmpeg utilities for the Phantom pipeline.

Provides wrapper functions for FFmpeg operations:
- Video frame extraction
- Video creation from frames
- Audio restoration
- FPS detection

Extracted from pipeline/utilities.py. Uses config object instead of globals.
"""

import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from pipeline.config import FaceSwapConfig
from pipeline.logging import emit_status, emit_warning

# Monkey patch SSL for macOS
if platform.system().lower() == 'darwin':
    # The two factories accept different keyword sets, so this is not a clean
    # type substitution — but every call site here passes no arguments, which
    # both accept. Deliberate, and narrower than disabling the check.
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment] # noqa: E501

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'

# Frame filename width. Six digits rather than four because `get_temp_frame_paths`
# orders frames by sorting their names: at four digits ffmpeg rolls over to five
# after 9999 and '10000.png' sorts before '9999.png', silently reordering every
# frame past the five-minute mark of a 30fps clip.
FRAME_PATTERN = '%06d.png'

# Scratch space is keyed by session, not by target filename — two sessions
# handed the same filename must not share a directory, and the target may live
# somewhere unwritable (or, on a pod, inside the upload directory). Defaults to
# a per-process token, which is exactly one session while max_sessions is 1;
# the control plane calls set_temp_scope() with a real session id.
_TEMP_SCOPE: Optional[str] = None


def set_temp_scope(session_id: Optional[str]) -> None:
    """
    Bind batch scratch directories to a session.

    Args:
        session_id: Session identifier, or None to fall back to the default
                    per-process scope
    """
    global _TEMP_SCOPE
    _TEMP_SCOPE = session_id


def get_temp_scope() -> str:
    """
    Get the current scratch scope.

    Returns:
        The session id set by set_temp_scope(), else PHANTOM_SESSION_ID from the
        environment, else a per-process token
    """
    return _TEMP_SCOPE or os.environ.get('PHANTOM_SESSION_ID') or f'pid-{os.getpid()}'


def get_temp_root() -> str:
    """
    Get the root directory holding all scratch space.

    Batch video extracts every frame as lossless PNG, which is roughly 4 MB per
    1080p frame — about 36 GB for a five-minute clip at 30fps. On a pod the
    system temp lives on the *container* disk, which defaults to 20 GB and is
    shared with the OS, so a long job would fill it and fail partway through.
    The network volume is the disk the operator actually sized, so scratch goes
    there when one is mounted.

    Returns:
        PHANTOM_TEMP_DIR if set; else a directory on the network volume when one
        is mounted; else a 'phantom' directory under the system temp
    """
    override = os.environ.get('PHANTOM_TEMP_DIR')
    if override:
        return override

    if os.path.isdir('/workspace'):
        return '/workspace/tmp/phantom'

    return os.path.join(tempfile.gettempdir(), 'phantom')


def run_ffmpeg(config: FaceSwapConfig, args: List[str]) -> bool:
    """
    Run an FFmpeg command.

    Args:
        config: FaceSwapConfig for log_level setting
        args: FFmpeg arguments (without 'ffmpeg' command itself)

    Returns:
        True if successful, False otherwise

    Example:
        run_ffmpeg(CONFIG, ['-i', 'input.mp4', '-c:v', 'libx264', 'output.mp4'])
    """
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-loglevel', config.log_level]
    commands.extend(args)

    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except FileNotFoundError:
        emit_warning('FFmpeg not found in PATH', scope='FFMPEG')
        return False
    except subprocess.CalledProcessError as e:
        emit_warning(f'FFmpeg command failed (exit {e.returncode}): {e.output.decode(errors="replace").strip() if e.output else "no output"}', scope='FFMPEG')
        return False
    except Exception as e:
        emit_warning(f'FFmpeg unexpected error: {type(e).__name__}: {e}', scope='FFMPEG')
        return False


def detect_fps(video_path: str) -> float:
    """
    Detect FPS of a video file using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        FPS as float, or 30.0 if detection fails
    """
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path,
        ]
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL).decode().strip()

        if '/' in output:
            numerator, denominator = map(int, output.split('/'))
            return numerator / denominator
        else:
            return float(output)

    except Exception as e:
        emit_warning(f'FPS detection failed for {video_path}: {type(e).__name__}: {e} — defaulting to 30fps', scope='FFMPEG')
        return 30.0


def has_audio(video_path: str) -> bool:
    """
    Check whether a video carries at least one audio stream.

    Asked before restoring audio so a silent target takes the copy path
    deliberately, rather than by letting an FFmpeg stream-mapping error fail.

    Args:
        video_path: Path to video file

    Returns:
        True if an audio stream is present
    """
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path,
        ]
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL).decode().strip()
        return output == 'audio'
    except Exception:
        # Unknown is treated as absent: the copy path always produces a file,
        # where a bad stream map produces nothing.
        return False


def extract_frames(config: FaceSwapConfig, target_path: str) -> None:
    """
    Extract frames from a video file.

    Saves frames as PNG images to a temp directory.

    Args:
        config: FaceSwapConfig for log_level
        target_path: Path to video file
    """
    temp_directory_path = get_temp_directory_path(target_path)
    output_pattern = os.path.join(temp_directory_path, FRAME_PATTERN)

    emit_status(f'Extracting frames to {temp_directory_path}', scope='FFMPEG')
    run_ffmpeg(
        config,
        ['-i', target_path, '-pix_fmt', 'rgb24', output_pattern],
    )


def create_video(
    config: FaceSwapConfig,
    target_path: str,
    fps: float = 30.0,
    output_fps: Optional[float] = None,
) -> None:
    """
    Create a video file from extracted frames.

    Uses config.video_encoder and config.video_quality settings.

    `fps` is the rate the frames are *read* at, so it must be the source rate:
    every extracted frame is one source frame, and reading them at any other
    rate rescales the clip's duration, which desynchronises the audio restored
    afterwards. Retiming is `output_fps`, applied as a filter so frames are
    dropped or duplicated and the duration survives.

    Args:
        config: FaceSwapConfig with encoder and quality settings
        target_path: Original target path (for temp directory lookup)
        fps: Rate the frames were captured at — the source video's FPS
        output_fps: Re-time the result to this rate, preserving duration.
                    None encodes at the source rate
    """
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    input_pattern = os.path.join(temp_directory_path, FRAME_PATTERN)

    filters = ['colorspace=bt709:iall=bt601-6-625:fast=1']
    if output_fps and abs(output_fps - fps) > 0.01:
        filters.append(f'fps={output_fps}')

    emit_status(
        f'Creating video from frames (encoder: {config.video_encoder}, '
        f'quality: {config.video_quality}, {fps:.3f}fps'
        + (f' -> {output_fps:.3f}fps' if len(filters) > 1 else '') + ')',
        scope='FFMPEG',
    )

    run_ffmpeg(
        config,
        [
            '-r', str(fps),
            '-i', input_pattern,
            '-c:v', config.video_encoder,
            '-crf', str(config.video_quality),
            '-pix_fmt', 'yuv420p',
            '-vf', ','.join(filters),
            '-y',
            temp_output_path,
        ],
    )


def restore_audio(config: FaceSwapConfig, target_path: str, output_path: str) -> None:
    """
    Restore audio from original video to output video.

    Takes audio from target_path and combines with video from temp output.

    Args:
        config: FaceSwapConfig for log_level
        target_path: Original target path (for temp directory lookup)
        output_path: Final output path
    """
    temp_output_path = get_temp_output_path(target_path)

    emit_status('Restoring audio', scope='FFMPEG')

    done = run_ffmpeg(
        config,
        [
            '-i', temp_output_path,
            '-i', target_path,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-y',
            output_path,
        ],
    )

    if not done:
        # If audio restore fails, move temp output to final location
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    """
    Get list of extracted frame paths.

    Args:
        target_path: Original target path

    Returns:
        List of PNG frame paths in temp directory
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return sorted(glob.glob(os.path.join(glob.escape(temp_directory_path), '*.png')))


def get_temp_directory_path(target_path: str) -> str:
    """
    Get path to temp directory for a target, scoped to the current session.

    The target name is kept as the leaf so one session can process several
    targets without collision.

    Args:
        target_path: Original target path

    Returns:
        Temp directory path
    """
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(get_temp_root(), get_temp_scope(), TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    """
    Get path to temp output video file.

    Args:
        target_path: Original target path

    Returns:
        Temp output file path
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> str:
    """
    Normalize output path.

    If output_path is a directory, generates a name based on source and target.

    Args:
        source_path: Source image path
        target_path: Target image/video path
        output_path: Desired output path (file or directory)

    Returns:
        Normalized output file path
    """
    if source_path and target_path and os.path.isdir(output_path):
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        return os.path.join(output_path, f'{source_name}-{target_name}{target_extension}')
    return output_path


def create_temp(target_path: str) -> None:
    """
    Create temp directory for a target.

    Args:
        target_path: Original target path
    """
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def reset_temp(target_path: str) -> None:
    """
    Empty and recreate a target's temp directory.

    Unconditional, unlike clean_temp: frames left by an aborted run would
    otherwise be re-encoded as part of the next one, and `keep_frames` must not
    be able to turn that into silent corruption.

    Args:
        target_path: Original target path
    """
    temp_directory_path = get_temp_directory_path(target_path)
    if os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path, ignore_errors=True)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    """
    Move temp output to final location.

    Args:
        target_path: Original target path (for temp directory lookup)
        output_path: Final output path
    """
    temp_output_path = get_temp_output_path(target_path)

    if os.path.isfile(temp_output_path):
        # Remove existing output if present
        if os.path.isfile(output_path):
            os.remove(output_path)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        shutil.move(temp_output_path, output_path)


def clean_temp(config: FaceSwapConfig, target_path: str) -> None:
    """
    Clean up temp directory.

    Args:
        config: FaceSwapConfig for keep_frames setting
        target_path: Original target path
    """
    temp_directory_path = get_temp_directory_path(target_path)

    # Remove frame directory if keep_frames is False
    if not config.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path, ignore_errors=True)

    # Prune the now-empty scope directories back towards the root, so a session
    # that processed several targets does not leave a tree of empty folders.
    # Stops at the root itself, which is shared and not ours to remove.
    root = os.path.normpath(get_temp_root())
    directory = os.path.dirname(os.path.normpath(temp_directory_path))
    while directory.startswith(root) and directory != root:
        try:
            if os.listdir(directory):
                break
            os.rmdir(directory)
        except OSError:
            break
        directory = os.path.dirname(directory)


# ============================================================================
# File type utilities (migrated from pipeline/utilities.py)
# ============================================================================

# The image formats this application accepts, in one place because the file
# dialogs offer exactly this list and a check that disagreed with the offer
# would refuse a file the picker invited.
#
# It is a fixed tuple rather than a mimetype lookup. `mimetypes.guess_type`
# was the previous rule and it silently dropped **.webp**: the mapping only
# arrived in Python 3.11, and on Windows the module also consults
# HKEY_CLASSES_ROOT, so the same webp resolved on one machine and not on the
# next. A supported format failing by environment is worse than one failing
# outright, because nothing about it looks broken.
#
# `.gif` and `.heic` are left out on purpose, and not as an oversight to be
# corrected later: OpenCV has no decoder for either, so accepting one means
# uploading a file that is certain to be refused after the round trip. Better
# to say so while the file picker is still open.
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def is_image(image_path: str) -> bool:
    """
    Check if path points to an image file this application can read.

    Args:
        image_path: Path to check

    Returns:
        True if path is an existing file with a supported image extension
    """
    return bool(image_path) and os.path.isfile(image_path) and has_image_extension(image_path)


def is_video(video_path: str) -> bool:
    """
    Check if path points to a video file.

    Args:
        video_path: Path to check

    Deliberately still a mimetype lookup, unlike `is_image`. The image formats
    are a closed set we decode ourselves with OpenCV and offer in a file
    dialog, so the list has to be exact. Video is whatever FFmpeg can demux —
    open-ended, never enumerated in a picker, and handed straight to FFmpeg to
    accept or refuse. A fixed tuple there would reject working files.

    Returns:
        True if path is an existing video file
    """
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def has_image_extension(image_path: str) -> bool:
    """
    Check if path has a supported image extension, existing or not.

    The dot is part of the match. Without it `endswith('png')` also accepts a
    file called `diagram-png`, which is not one.

    Args:
        image_path: Path to check

    Returns:
        True if path has an image extension
    """
    return image_path.lower().endswith(IMAGE_EXTENSIONS)


def resolve_relative_path(path: str) -> str:
    """
    Resolve a path relative to the pipeline package directory.

    Args:
        path: Relative path

    Returns:
        Absolute path resolved from pipeline package root
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', path))


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    """
    Download files if they don't already exist.

    Args:
        download_directory_path: Directory to save downloaded files
        urls: List of URLs to download
    """
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)

    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(urllib.parse.urlparse(url).path))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(  # type: ignore[attr-defined]
                    url,
                    download_file_path,
                    reporthook=lambda count, block_size, total_size: progress.update(block_size),
                )
