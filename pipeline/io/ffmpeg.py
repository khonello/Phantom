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
import urllib.request
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from pipeline.config import FaceSwapConfig
from pipeline.logging import emit_status, emit_warning

# Monkey patch SSL for macOS
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'


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
    except subprocess.CalledProcessError:
        return False
    except Exception:
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

    except Exception:
        return 30.0


def extract_frames(config: FaceSwapConfig, target_path: str) -> None:
    """
    Extract frames from a video file.

    Saves frames as PNG images to a temp directory.

    Args:
        config: FaceSwapConfig for log_level
        target_path: Path to video file
    """
    temp_directory_path = get_temp_directory_path(target_path)
    output_pattern = os.path.join(temp_directory_path, '%04d.png')

    emit_status(f'Extracting frames to {temp_directory_path}', scope='FFMPEG')
    run_ffmpeg(
        config,
        ['-i', target_path, '-pix_fmt', 'rgb24', output_pattern],
    )


def create_video(config: FaceSwapConfig, target_path: str, fps: float = 30.0) -> None:
    """
    Create a video file from extracted frames.

    Uses config.video_encoder and config.video_quality settings.

    Args:
        config: FaceSwapConfig with encoder and quality settings
        target_path: Original target path (for temp directory lookup)
        fps: Output video FPS
    """
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    input_pattern = os.path.join(temp_directory_path, '%04d.png')

    emit_status(
        f'Creating video from frames (encoder: {config.video_encoder}, quality: {config.video_quality})',
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
            '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
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
    Get path to temp directory for a target.

    Args:
        target_path: Original target path

    Returns:
        Temp directory path
    """
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


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
    parent_directory_path = os.path.dirname(temp_directory_path)

    # Remove frame directory if keep_frames is False
    if not config.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)

    # Remove parent if empty
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


# ============================================================================
# File type utilities (migrated from pipeline/utilities.py)
# ============================================================================

def is_image(image_path: str) -> bool:
    """
    Check if path points to an image file.

    Args:
        image_path: Path to check

    Returns:
        True if path is an existing image file
    """
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    """
    Check if path points to a video file.

    Args:
        video_path: Path to check

    Returns:
        True if path is an existing video file
    """
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def has_image_extension(image_path: str) -> bool:
    """
    Check if path has a common image extension.

    Args:
        image_path: Path to check

    Returns:
        True if path has an image extension
    """
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


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
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(  # type: ignore[attr-defined]
                    url,
                    download_file_path,
                    reporthook=lambda count, block_size, total_size: progress.update(block_size),
                )
