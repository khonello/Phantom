"""
Enhanced type definitions for the Phantom pipeline.

Provides dataclasses and type aliases for:
- Face detection results (Bbox, Detection)
- Video properties (VideoProperties)
- Processing results (SwapResult)
- Frame buffers

Extends the basic types from pipeline/typing.py.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple
import numpy as np
import numpy.typing as npt
from insightface.app.common import Face

# Re-export basic types from typing module for convenience
Face = Face

# Array aliases. These are all the same underlying type — they exist to say
# what a given array *means* at each call site. Spelled with explicit type
# arguments so they satisfy mypy's `disallow_any_generics`; a bare
# `np.ndarray` is a generic without arguments and is rejected under strict
# mode (and is not usable as an annotation at all).
Frame = npt.NDArray[Any]   # image, HxWx3 (BGR) or HxW
Mask = npt.NDArray[Any]    # single-channel float mask in [0, 1]
Matrix = npt.NDArray[Any]  # 2x3 affine transform
Points = npt.NDArray[Any]  # Nx2 point set (landmarks, keypoints)


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
    def from_insightface(cls, bbox: npt.NDArray[Any]) -> 'Bbox':
        """
        Convert InsightFace bbox format (x1, y1, x2, y2) to our format.

        Args:
            bbox: numpy array [x1, y1, x2, y2] from InsightFace detection

        Returns:
            Bbox object with (x, y, w, h) format
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        return cls(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def to_insightface(self) -> npt.NDArray[Any]:
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
    kps: npt.NDArray[Any]  # keypoints array
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/debugging."""
        return {
            'bbox': {'x': int(self.bbox.x), 'y': int(self.bbox.y), 'w': int(self.bbox.w), 'h': int(self.bbox.h)},
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
    def frame_size(self) -> Tuple[int, int]:
        """Return frame dimensions as (height, width) tuple."""
        return (self.height, self.width)

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'source_used': self.source_used,
            'detection': self.detection.to_dict() if self.detection else None,
            'frame_shape': self.frame.shape if self.frame is not None else None,
        }


class FrameSwap(NamedTuple):
    """
    What one batch frame came back as.

    The frame alone is enough for video, which passes an unswapped frame
    through and carries on, and the human-readable reason is enough for a
    still, which writes nothing and reports it. Neither is enough to decide
    whether to *abort*: that needs the guard's reason code, because only
    `multiple_faces` says something about the target rather than about one
    frame in it. So the code travels with the other two rather than being
    reconstructed by matching on message text.
    """

    frame: Frame
    reason: str = ''    # human-readable; empty when every detected face swapped
    code: str = ''      # guards.* reason code, empty unless a guard failed
    faces: int = 0      # faces swapped


@dataclass
class PhotoResult:
    """
    Outcome of swapping one target photo.

    Photo mode processes each target independently and skips the ones that
    fail, so a job returns a list of these rather than a single verdict. The
    reason matters as much as the verdict: there is a person choosing the next
    photo to try, and "no face detected" and "two faces in frame" call for
    different fixes.
    """

    target_path: str
    ok: bool
    reason: str = ''                    # empty when ok
    output_path: Optional[str] = None   # written only when ok
    faces: int = 0                      # faces swapped

    @classmethod
    def swapped(cls, target_path: str, output_path: str, faces: int) -> 'PhotoResult':
        """A photo that produced a swap."""
        return cls(target_path=target_path, ok=True, output_path=output_path, faces=faces)

    @classmethod
    def skipped(cls, target_path: str, reason: str) -> 'PhotoResult':
        """A photo that did not, and why. No output file is written."""
        return cls(target_path=target_path, ok=False, reason=reason)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for the API."""
        return {
            'target': os.path.basename(self.target_path),
            'target_path': self.target_path,
            'ok': self.ok,
            'reason': self.reason,
            'output_path': self.output_path,
            'faces': self.faces,
        }
