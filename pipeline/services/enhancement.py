"""
Face enhancement service for the Phantom pipeline.

Extracted from pipeline/processors/frame/face_enhancer.py and stream.py.
Provides GFPGAN-based face enhancement with graceful fallback if unavailable.
"""

import os
import threading
from typing import Any, Optional

from pipeline.types import Frame
from pipeline.logging import emit_status


class Enhancer:
    """
    Face enhancement using GFPGAN (if available).

    This service gracefully handles missing GFPGAN installation by
    returning frames unchanged. Thread-safe with internal model caching.

    Example:
        enhancer = Enhancer(model_path)
        if enhancer.available:
            enhanced = enhancer.enhance(frame)
        else:
            enhanced = frame  # fallback
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the enhancer.

        Args:
            model_path: Path to GFPGANv1.4.pth model file.
                       If None, uses default location relative to pipeline/
        """
        self.model_path = model_path or self._resolve_model_path()
        self._enhancer: Optional[Any] = None
        self._lock = threading.Lock()
        self._available = False

        # Try to load GFPGAN on init
        self._enhancer = self._try_load_gfpgan()
        self._available = self._enhancer is not None

    def _resolve_model_path(self) -> str:
        """
        Resolve the GFPGAN model path (relative to pipeline/).

        Returns:
            Full path to GFPGANv1.4.pth
        """
        pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(pipeline_dir, 'models', 'GFPGANv1.4.pth')

    def _try_load_gfpgan(self) -> Optional[Any]:
        """
        Attempt to load GFPGAN model.

        Returns None if model missing, gfpgan not installed, or load fails.
        """
        if not os.path.exists(self.model_path):
            emit_status(
                f'GFPGAN model not found: {self.model_path} — enhancement disabled',
                scope='ENHANCER',
                level='warning',
            )
            return None

        try:
            # Lazy import to avoid requiring gfpgan for other features
            from gfpgan import GFPGANer

            enhancer = GFPGANer(
                model_path=self.model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
            )
            emit_status('GFPGAN enhancement available', scope='ENHANCER')
            return enhancer

        except ImportError:
            emit_status(
                'gfpgan package not installed — enhancement disabled',
                scope='ENHANCER',
                level='warning',
            )
            return None
        except Exception as e:
            emit_status(
                f'GFPGAN failed to load: {e} — enhancement disabled',
                scope='ENHANCER',
                level='warning',
            )
            return None

    @property
    def available(self) -> bool:
        """
        Check if GFPGAN is available and ready to use.

        Returns:
            True if enhancement is active, False otherwise
        """
        return self._available

    def enhance(self, frame: Frame) -> Frame:
        """
        Enhance a frame using GFPGAN.

        Returns frame unchanged if GFPGAN is unavailable.

        Args:
            frame: Input frame as numpy array

        Returns:
            Enhanced frame (or original if enhancement unavailable)
        """
        if self._enhancer is None:
            return frame

        with self._lock:
            try:
                _, _, restored = self._enhancer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                return restored if restored is not None else frame
            except Exception as e:
                # If enhancement fails, return original frame
                emit_status(
                    f'Face enhancement failed: {e}',
                    scope='ENHANCER',
                    level='warning',
                )
                return frame

    def clear(self) -> None:
        """Clear the cached model (useful for memory cleanup)."""
        with self._lock:
            self._enhancer = None
            self._available = False
