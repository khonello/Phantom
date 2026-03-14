"""
Output sink abstraction for the Phantom pipeline.

Provides OutputSink ABC and implementations for various output types:
- File (MP4, AVI, etc.)
- HTTP frame serving (for desktop GUI)
- WebSocket broadcast (for remote clients)

Allows flexible routing of processed frames.
"""

import io
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

import cv2

from pipeline.types import Frame
from pipeline.logging import emit_status, emit_warning


class OutputSink(ABC):
    """
    Abstract base for frame output sinks.

    Implementations handle different output destinations with a unified interface.
    """

    @abstractmethod
    def write(self, frame: Frame) -> bool:
        """
        Write a frame to the sink.

        Args:
            frame: Frame to write

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close and finalize the sink."""
        pass


class FileOutput(OutputSink):
    """
    File output sink (MP4, AVI, etc.).

    Uses OpenCV VideoWriter to encode and write frames to a file.

    Example:
        sink = FileOutput("output.mp4", width=1920, height=1080, fps=30.0)
        sink.write(frame)
        sink.close()
    """

    def __init__(
        self,
        file_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = 'mp4v',
    ) -> None:
        """
        Initialize file output sink.

        Args:
            file_path: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (e.g., 'mp4v', 'MJPG', 'XVID')
        """
        self.file_path = file_path
        self.width = width
        self.height = height
        self.fps = fps

        # Create output directory if needed
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        # Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            emit_warning(f"Failed to open video file for writing: {file_path}", scope='OUTPUT')
            self._writer = None
            return

        emit_status(f'File output initialized: {file_path}', scope='OUTPUT')

    def write(self, frame: Frame) -> bool:
        """Write frame to video file."""
        if self._writer is None or not self._writer.isOpened():
            return False

        # Ensure frame is correct size
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            self._writer.write(frame)
            return True
        except Exception as e:
            emit_warning(f"Failed to write frame: {e}", scope='OUTPUT')
            return False

    def close(self) -> None:
        """Finalize and close video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            emit_status(f'File output closed: {self.file_path}', scope='OUTPUT')


class HTTPFrameOutput(OutputSink):
    """
    HTTP frame server output sink.

    Serves the latest frame via HTTP endpoint (used by desktop GUI for preview).
    This is a simplified implementation; actual HTTP server is in api/server.py.

    Example:
        sink = HTTPFrameOutput()
        sink.write(frame)  # frame available via HTTP
    """

    def __init__(self) -> None:
        """Initialize HTTP frame buffer."""
        self._latest_frame: Optional[Frame] = None
        self._lock = threading.Lock()

    def write(self, frame: Frame) -> bool:
        """Buffer latest frame for HTTP serving."""
        with self._lock:
            self._latest_frame = frame.copy()
        return True

    def get_latest_frame(self) -> Optional[Frame]:
        """
        Get the latest buffered frame (for HTTP handler to serve).

        Returns:
            Latest frame or None if no frame written yet
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """
        Get latest frame as JPEG bytes (ready to serve).

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG bytes or None
        """
        frame = self.get_latest_frame()
        if frame is None:
            return None

        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return bytes(buffer)
        except Exception as e:
            import sys
            print(f'[HTTPOutput] JPEG encode error: {type(e).__name__}: {e}', file=sys.stderr)
            return None

    def close(self) -> None:
        """Close HTTP frame buffer."""
        with self._lock:
            self._latest_frame = None


class WebSocketOutput(OutputSink):
    """
    WebSocket broadcast output sink.

    Broadcasts frames (as JPEG) to connected WebSocket clients.
    Frame subscription is handled by api/server.py; this just buffers.

    Example:
        sink = WebSocketOutput()
        sink.write(frame)
        # api/server broadcasts sink.get_latest_frame_jpeg() to clients
    """

    def __init__(self) -> None:
        """Initialize WebSocket frame buffer."""
        self._latest_frame: Optional[Frame] = None
        self._lock = threading.Lock()

    def write(self, frame: Frame) -> bool:
        """Buffer latest frame for WebSocket broadcast."""
        with self._lock:
            self._latest_frame = frame.copy()
        return True

    def get_latest_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """
        Get latest frame as JPEG bytes for WebSocket broadcast.

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG bytes or None
        """
        with self._lock:
            if self._latest_frame is None:
                return None

            try:
                _, buffer = cv2.imencode('.jpg', self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                return bytes(buffer)
            except Exception as e:
                import sys
                print(f'[WebSocketOutput] JPEG encode error: {type(e).__name__}: {e}', file=sys.stderr)
                return None

    def close(self) -> None:
        """Close WebSocket frame buffer."""
        with self._lock:
            self._latest_frame = None


class RTMPOutput(OutputSink):
    """
    RTMP stream output sink (placeholder).

    Would broadcast frames to RTMP server (e.g., YouTube, Twitch).
    Not fully implemented; requires ffmpeg streaming setup.

    Example:
        sink = RTMPOutput(rtmp_url="rtmp://example.com/stream")
        sink.write(frame)
    """

    def __init__(self, rtmp_url: str) -> None:
        """
        Initialize RTMP output (placeholder).

        Args:
            rtmp_url: RTMP server URL
        """
        self.rtmp_url = rtmp_url
        emit_status(f'RTMP output placeholder: {rtmp_url}', scope='OUTPUT')

    def write(self, frame: Frame) -> bool:
        """No-op for placeholder implementation."""
        return True

    def close(self) -> None:
        """Close RTMP connection (placeholder)."""
        pass
