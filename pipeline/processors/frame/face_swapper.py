from typing import Any, List, Callable
import cv2
import insightface
import threading
import os

import pipeline.globals
import pipeline.processors.frame.core
from pipeline.core import update_status
from pipeline.face_analyser import get_one_face, get_many_faces, get_averaged_face
from pipeline.typing import Face, Frame
from pipeline.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
SOURCE_FACE = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=pipeline.globals.execution_providers)
    return FACE_SWAPPER


def get_source_face() -> Any:
    global SOURCE_FACE

    if SOURCE_FACE is None:
        source_paths = pipeline.globals.source_paths or ([pipeline.globals.source_path] if pipeline.globals.source_path else [])
        SOURCE_FACE = get_averaged_face(source_paths)
    return SOURCE_FACE


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('models')
    model_path = os.path.join(download_directory_path, 'inswapper_128.onnx')

    if os.path.exists(model_path):
        update_status('Model found: inswapper_128.onnx', NAME)
        return True

    update_status('Model not found: inswapper_128.onnx', NAME)

    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)

    hf_url = 'https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/insightface/inswapper_128.onnx?download=true'
    answer = input('Would you like to download the model from Hugging Face? (y/n): ').strip().lower()
    if answer == 'y':
        try:
            conditional_download(download_directory_path, [hf_url])
            if os.path.exists(model_path):
                update_status('Model downloaded successfully.', NAME)
                return True
        except Exception:
            pass

    update_status(
        'Please download inswapper_128.onnx manually from: '
        'https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view '
        'and place it in: pipeline/models/inswapper_128.onnx',
        NAME
    )
    return False


def pre_start() -> bool:
    source_paths = pipeline.globals.source_paths or ([pipeline.globals.source_path] if pipeline.globals.source_path else [])

    if not source_paths:
        update_status('Select an image for source path.', NAME)
        return False

    for path in source_paths:
        if path.lower().endswith('.npy'):
            if not os.path.exists(path):
                update_status(f'Embedding file not found: {path}', NAME)
                return False
        elif is_video(path):
            update_status(f'Video files are not supported as source: {path}. Extract a frame first: ffmpeg -i "{path}" -ss 00:00:05 -frames:v 1 source_face.jpg', NAME)
            return False
        elif not is_image(path):
            update_status(f'Invalid source image: {path}', NAME)
            return False

    source_face = get_source_face()
    if source_face is None:
        update_status('No face in source path detected.', NAME)
        return False

    if len(source_paths) > 1:
        update_status(f'Averaged face from {len(source_paths)} sources.', NAME)

    if not is_image(pipeline.globals.target_path) and not is_video(pipeline.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    get_face_swapper()
    return True


def post_process() -> None:
    global FACE_SWAPPER
    global SOURCE_FACE

    FACE_SWAPPER = None
    SOURCE_FACE = None


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if pipeline.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_source_face()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_source_face()
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    pipeline.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
