"""
Enhanced type definitions for the Phantom pipeline.

Provides dataclasses and type aliases for:
- Face detection results (Bbox, Detection)
- Video properties (VideoProperties)
- Processing results (SwapResult)
- Frame buffers

Extends the basic types from pipeline/typing.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from insightface.app.common import Face

# Re-export basic types from typing module for convenience
Face = Face
Frame = np.ndarray


@dataclass
class Bbox:
    """
    Bounding box representation with common operations.

    Stores box as (x, y, width, height) and provides conversion utilities.
    """

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_insightface(cls, bbox: np.ndarray) -> 'Bbox':
        """
        Convert InsightFace bbox format (x1, y1, x2, y2) to our format.

        Args:
            bbox: numpy array [x1, y1, x2, y2] from InsightFace detection

        Returns:
            Bbox object with (x, y, w, h) format
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        return cls(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def to_insightface(self) -> np.ndarray:
        """Convert back to InsightFace format (x1, y1, x2, y2)."""
        return np.array([self.x, self.y, self.x + self.w, self.y + self.h], dtype=np.float32)

    def in_frame(self, shape: Tuple[int, int]) -> bool:
        """
        Check if bounding box is fully contained within frame.

        Args:
            shape: Frame shape (height, width)

        Returns:
            True if bbox is completely within bounds
        """
        h, w = shape
        return (
            self.x >= 0 and
            self.y >= 0 and
            self.x + self.w <= w and
            self.y + self.h <= h
        )

    def clip_to_frame(self, shape: Tuple[int, int]) -> 'Bbox':
        """
        Clip bbox to frame boundaries.

        Args:
            shape: Frame shape (height, width)

        Returns:
            New Bbox clipped to valid frame region
        """
        h, w = shape
        x = max(0, min(self.x, w))
        y = max(0, min(self.y, h))
        new_w = min(self.w, w - x)
        new_h = min(self.h, h - y)
        return Bbox(x=x, y=y, w=new_w, h=new_h)


@dataclass
class Detection:
    """
    Represents a detected face in a frame.

    Combines the face model (for swapping) with its spatial location
    and confidence score. This is the single canonical face type.
    """

    face: Face
    bbox: Bbox
    kps: np.ndarray  # keypoints array
    confidence: float

    @classmethod
    def from_insightface(cls, face: 'Face') -> 'Detection':
        """
        Create a Detection from a raw InsightFace Face object.

        Args:
            face: Raw InsightFace Face object

        Returns:
            Detection wrapping the face with parsed bbox and kps
        """
        bbox = Bbox.from_insightface(face.bbox)
        kps = face.kps if hasattr(face, 'kps') and face.kps is not None else np.array([])
        score = getattr(face, 'det_score', None) or getattr(face, 'score', None)
        confidence = float(score) if score is not None else 0.0
        return cls(face=face, bbox=bbox, kps=kps, confidence=confidence)

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging/debugging."""
        return {
            'bbox': {'x': self.bbox.x, 'y': self.bbox.y, 'w': self.bbox.w, 'h': self.bbox.h},
            'confidence': float(self.confidence),
            'kps_shape': list(self.kps.shape) if self.kps is not None else None,
        }


@dataclass
class VideoProperties:
    """Metadata about a video source."""

    width: int
    height: int
    fps: float

    @property
    def frame_size(self) -> tuple:
        """Return frame dimensions as (height, width) tuple."""
        return (self.height, self.width)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
        }


@dataclass
class SwapResult:
    """Result of a face swap operation."""

    frame: Frame
    source_used: bool  # Was a source face found and used?
    detection: Optional[Detection]  # Detection info if available

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'source_used': self.source_used,
            'detection': self.detection.to_dict() if self.detection else None,
            'frame_shape': self.frame.shape if self.frame is not None else None,
        }
