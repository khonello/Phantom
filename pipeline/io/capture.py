"""
Input source abstraction for the Phantom pipeline.

Provides InputSource ABC and implementations for various input types:
- Webcam (default camera)
- Network stream (RTSP, RTMP, etc.)
- Video file
- Image sequence

Replaces hardcoded cv2.VideoCapture usage in stream.py.
"""

import glob
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2

from pipeline.types import Frame, VideoProperties
from pipeline.logging import emit_status, emit_warning


class InputSource(ABC):
    """
    Abstract base for video input sources.

    Implementations handle different input types (webcam, network, file, etc.)
    with a unified interface.
    """

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """
        Read the next frame.

        Returns:
            Frame as numpy array, or None if end of stream or error
        """
        pass

    @abstractmethod
    def properties(self) -> VideoProperties:
        """
        Get video properties (width, height, fps).

        Returns:
            VideoProperties object
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        pass


class WebcamInput(InputSource):
    """
    Webcam input (default camera device).

    Example:
        source = WebcamInput(device_id=0)
        frame = source.read()
        props = source.properties()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 960,
        height: int = 540,
        fps: float = 30.0,
    ) -> None:
        """
        Initialize webcam source.

        Args:
            device_id: Camera device ID (0 = default)
            width: Requested frame width
            height: Requested frame height
            fps: Requested FPS
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            emit_warning(f"Failed to open webcam device {device_id}", scope='INPUT')
            self._cap = None
            return

        # Set resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        emit_status(f'Webcam initialized (device {device_id})', scope='INPUT')

    def read(self) -> Optional[Frame]:
        """Read next frame from webcam."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        return frame if ret else None

    def properties(self) -> VideoProperties:
        """Get webcam properties."""
        if self._cap is None:
            return VideoProperties(width=0, height=0, fps=0.0)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.fps

        return VideoProperties(width=actual_width, height=actual_height, fps=actual_fps)

    def release(self) -> None:
        """Release webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class NetworkInput(InputSource):
    """
    Network stream input (RTSP, RTMP, HTTP, etc.).

    Example:
        source = NetworkInput(url="rtsp://example.com/stream")
        frame = source.read()
    """

    def __init__(self, url: str) -> None:
        """
        Initialize network stream source.

        Args:
            url: Stream URL (RTSP, RTMP, HTTP, etc.)
        """
        self.url = url
        self._cap = cv2.VideoCapture(url)

        if not self._cap.isOpened():
            emit_warning(f"Failed to open network stream: {url}", scope='INPUT')
            self._cap = None
            return

        emit_status(f'Network stream initialized: {url}', scope='INPUT')

    def read(self) -> Optional[Frame]:
        """Read next frame from stream."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        return frame if ret else None

    def properties(self) -> VideoProperties:
        """Get stream properties."""
        if self._cap is None:
            return VideoProperties(width=0, height=0, fps=0.0)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        return VideoProperties(width=width, height=height, fps=fps)

    def release(self) -> None:
        """Release stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FileInput(InputSource):
    """
    Video file input.

    Example:
        source = FileInput("video.mp4")
        frame = source.read()
        props = source.properties()
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize video file source.

        Args:
            file_path: Path to video file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file can't be opened
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self.file_path = file_path
        self._cap = cv2.VideoCapture(file_path)

        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {file_path}")

        emit_status(f'Video file initialized: {file_path}', scope='INPUT')

    def read(self) -> Optional[Frame]:
        """Read next frame from video."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        return frame if ret else None

    def properties(self) -> VideoProperties:
        """Get video properties."""
        if self._cap is None:
            return VideoProperties(width=0, height=0, fps=0.0)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        return VideoProperties(width=width, height=height, fps=fps)

    def release(self) -> None:
        """Release video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSequenceInput(InputSource):
    """
    Image sequence input (directory of images).

    Reads images in sorted order from a directory.

    Example:
        source = ImageSequenceInput("frame_dir/*.png")
        frame = source.read()
    """

    def __init__(self, glob_pattern: str) -> None:
        """
        Initialize image sequence source.

        Args:
            glob_pattern: Glob pattern matching image files (e.g., "frames/*.png")

        Raises:
            ValueError: If no images found
        """
        self.glob_pattern = glob_pattern
        self._image_paths = sorted(glob.glob(glob_pattern))

        if not self._image_paths:
            raise ValueError(f"No images found matching: {glob_pattern}")

        self._index = 0
        self._width = 0
        self._height = 0

        # Probe first image to get dimensions
        import cv2 as cv2_module
        first_frame = cv2_module.imread(self._image_paths[0])
        if first_frame is not None:
            self._height, self._width = first_frame.shape[:2]

        emit_status(
            f'Image sequence initialized: {len(self._image_paths)} images',
            scope='INPUT',
        )

    def read(self) -> Optional[Frame]:
        """Read next image from sequence."""
        if self._index >= len(self._image_paths):
            return None

        path = self._image_paths[self._index]
        self._index += 1

        import cv2 as cv2_module
        frame = cv2_module.imread(path)
        if frame is None:
            emit_warning(f"Failed to read image: {path}", scope='INPUT')
            return self.read()  # Skip to next

        return frame

    def properties(self) -> VideoProperties:
        """Get image sequence properties."""
        return VideoProperties(width=self._width, height=self._height, fps=30.0)

    def release(self) -> None:
        """No-op for image sequence."""
        pass

    def reset(self) -> None:
        """Reset to first image."""
        self._index = 0
