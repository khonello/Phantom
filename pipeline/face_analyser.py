import os
import threading
import types
from typing import Any, List, Optional

import cv2
import numpy as np
import insightface

import pipeline.globals
from pipeline.typing import Frame

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=pipeline.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None


def get_averaged_face(source_paths: List[str]) -> Optional[Any]:
    if not source_paths:
        return None

    faces = []

    for path in source_paths:
        if path.lower().endswith('.npy'):
            if not os.path.exists(path):
                continue
            embedding = np.load(path)
            faces.append(types.SimpleNamespace(normed_embedding=embedding))
        else:
            frame = cv2.imread(path)
            if frame is None:
                continue
            face = get_one_face(frame)
            if face is not None:
                faces.append(face)

    if not faces:
        return None

    if len(faces) == 1:
        return faces[0]

    embeddings = np.array([f.normed_embedding for f in faces])
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    return types.SimpleNamespace(normed_embedding=avg_embedding)


def save_face_embedding(face: Any, path: str) -> None:
    np.save(path, face.normed_embedding)
