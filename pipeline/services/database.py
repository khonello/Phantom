"""
Face database service for the Phantom pipeline.

Handles caching of face embeddings, averaging multiple faces,
and loading pre-saved embeddings. Extracted from face_analyser.py.
"""

import os
import types
from typing import List, Optional

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Face
from pipeline.services.face_detection import FaceDetector


class FaceDatabase:
    """
    In-memory cache of face embeddings.

    Handles:
    - Loading and caching embeddings from image files
    - Loading pre-computed .npy embeddings
    - Averaging multiple faces into a single embedding
    - Clearing cache on demand

    Example:
        db = FaceDatabase(detector)
        source_face = db.get_source_face(['face1.jpg', 'face2.jpg'])
        db.clear()  # cleanup
    """

    def __init__(self, detector: FaceDetector) -> None:
        """
        Initialize face database.

        Args:
            detector: FaceDetector instance for face extraction
        """
        self.detector = detector
        self._cache: dict = {}  # path -> Face

    def get_source_face(self, paths: List[str]) -> Optional[Face]:
        """
        Get a source face from one or more paths.

        If multiple paths, returns averaged face embedding.
        Handles both image files (.jpg, .png) and embeddings (.npy).

        Args:
            paths: List of image or .npy file paths

        Returns:
            Face object with averaged embedding, or None if no valid faces
        """
        if not paths:
            return None

        faces = []

        for path in paths:
            if path.lower().endswith('.npy'):
                face = self._load_embedding(path)
            else:
                face = self._extract_from_image(path)

            if face is not None:
                faces.append(face)

        if not faces:
            return None

        # Return single face or average of multiple
        if len(faces) == 1:
            return faces[0]

        return self._average_faces(faces)

    def _load_embedding(self, npy_path: str) -> Optional[Face]:
        """
        Load a pre-computed face embedding from .npy file.

        Args:
            npy_path: Path to .npy file containing embedding vector

        Returns:
            Face object with embedding, or None if file not found
        """
        if not os.path.exists(npy_path):
            return None

        try:
            embedding = np.load(npy_path)
            # Create a Face-like object with just the embedding
            return types.SimpleNamespace(normed_embedding=embedding)
        except Exception:
            return None

    def _extract_from_image(self, image_path: str) -> Optional[Face]:
        """
        Extract face from an image file.

        Uses the FaceDetector to find and extract a face.
        Caches result for repeated access.

        Args:
            image_path: Path to image file

        Returns:
            Face object, or None if file not found or no face detected
        """
        # Check cache first
        if image_path in self._cache:
            return self._cache[image_path]

        if not os.path.exists(image_path):
            return None

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return None

            detection = self.detector.detect_one(frame)
            if detection is None:
                return None

            # Cache it
            face = detection.face
            self._cache[image_path] = face
            return face
        except Exception:
            return None

    def _average_faces(self, faces: List[Face]) -> Optional[Face]:
        """
        Average embeddings from multiple faces.

        Args:
            faces: List of Face objects with normed_embedding attribute

        Returns:
            New Face-like object with averaged embedding, or None if empty
        """
        if not faces:
            return None

        # Extract embeddings
        embeddings = []
        for face in faces:
            if hasattr(face, 'normed_embedding'):
                embeddings.append(face.normed_embedding)

        if not embeddings:
            return None

        # Average
        embeddings_array = np.array(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)

        # Normalize to unit vector
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        # Return Face-like object
        return types.SimpleNamespace(normed_embedding=avg_embedding)

    def save_embedding(self, face: Face, path: str) -> None:
        """
        Save a face embedding to a .npy file.

        Args:
            face: Face object with normed_embedding
            path: Output path for .npy file
        """
        if not hasattr(face, 'normed_embedding'):
            return

        # Create directory if needed (dirname is '' for bare filenames)
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            np.save(path, face.normed_embedding)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
