#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import threading
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import pipeline.globals
import pipeline.metadata
from pipeline.predicter import predict_image, predict_video
from pipeline.processors.frame.core import get_frame_processors_modules
from pipeline.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in pipeline.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

from pipeline.api.schema import PRESETS


def _apply_preset(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.quality, PRESETS['optimal'])
    # Fill each unset (None) arg from the selected preset
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select source image(s) or embedding (.npy)', dest='source_path', nargs='+')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--save-embedding', help='save averaged face embedding to .npy file', dest='save_embedding_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('--quality', help='stream quality preset', dest='quality', default='optimal', choices=['fast', 'optimal', 'production'])
    program.add_argument('--tracker', help='face tracker for stream mode', dest='tracker', default=None, choices=['csrt', 'kcf', 'mosse'])
    program.add_argument('--alpha', help='EMA smoothing factor for landmarks (0.0=max smooth, 1.0=no smooth)', dest='alpha', type=float, default=None)
    program.add_argument('--blend', help='swap blend ratio (0.0=original, 1.0=full swap)', dest='blend', type=float, default=None)
    program.add_argument('--luminance-blend', help='enable luminance-adaptive blend', dest='luminance_blend', action='store_true', default=None)
    program.add_argument('--enhance-interval', help='run GFPGAN every N frames (0=off)', dest='enhance_interval', type=int, default=None)
    program.add_argument('--buffer-size', help='frame buffer depth', dest='buffer_size', type=int, default=None)
    program.add_argument('--redetect-interval', help='frames between forced face re-detections', dest='redetect_interval', type=int, default=None)
    program.add_argument('--warmup-frames', help='frames to display raw before swap starts', dest='warmup_frames', type=int, default=None)
    program.add_argument('--input-url', help='network stream URL as webcam source (RTSP/RTMP/HTTP); omit to use local webcam', dest='input_url', default=None)
    program.add_argument('--stream-url', help='RTMP URL for primary stream output (e.g. rtmp://live.twitch.tv/live/KEY)', dest='stream_url', default=None)
    program.add_argument('--preview-url', help='RTMP URL for preview stream (monitoring/desktop app)', dest='preview_url', default=None)
    program.add_argument('--virtual-cam', help='write swapped output to OBS virtual camera (requires pyvirtualcam)', dest='virtual_cam', action='store_true', default=False)
    program.add_argument('--control-port', help='start HTTP control server on this port (for desktop app communication)', dest='control_port', type=int, default=9000)
    program.add_argument('-v', '--version', action='version', version=f'{pipeline.metadata.name} {pipeline.metadata.version}')

    args = program.parse_args()
    _apply_preset(args)

    pipeline.globals.source_paths = args.source_path or []
    pipeline.globals.source_path = args.source_path[0] if args.source_path else None
    pipeline.globals.target_path = args.target_path
    pipeline.globals.output_path = normalize_output_path(pipeline.globals.source_path, pipeline.globals.target_path, args.output_path)
    pipeline.globals.save_embedding_path = args.save_embedding_path
    pipeline.globals.frame_processors = args.frame_processor
    pipeline.globals.headless = pipeline.globals.source_path or args.target_path or args.output_path
    pipeline.globals.keep_fps = args.keep_fps
    pipeline.globals.keep_audio = args.keep_audio
    pipeline.globals.keep_frames = args.keep_frames
    pipeline.globals.many_faces = args.many_faces
    pipeline.globals.video_encoder = args.video_encoder
    pipeline.globals.video_quality = args.video_quality
    pipeline.globals.max_memory = args.max_memory
    pipeline.globals.execution_providers = decode_execution_providers(args.execution_provider)
    pipeline.globals.execution_threads = args.execution_threads
    pipeline.globals.quality = args.quality
    pipeline.globals.tracker = args.tracker
    pipeline.globals.alpha = args.alpha
    pipeline.globals.blend = args.blend
    pipeline.globals.luminance_blend = args.luminance_blend
    pipeline.globals.enhance_interval = args.enhance_interval
    pipeline.globals.buffer_size = args.buffer_size
    pipeline.globals.redetect_interval = args.redetect_interval
    pipeline.globals.warmup_frames = args.warmup_frames
    pipeline.globals.input_url = args.input_url
    pipeline.globals.stream_url = args.stream_url
    pipeline.globals.preview_url = args.preview_url
    pipeline.globals.virtual_cam = args.virtual_cam
    pipeline.globals.control_port = args.control_port


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in pipeline.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in pipeline.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if pipeline.globals.max_memory:
        memory = pipeline.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = pipeline.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in pipeline.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    pipeline.globals.status_message = message


def start() -> None:
    for frame_processor in get_frame_processors_modules(pipeline.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(pipeline.globals.target_path):
        shutil.copy2(pipeline.globals.target_path, pipeline.globals.output_path)
        for frame_processor in get_frame_processors_modules(pipeline.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(pipeline.globals.source_path, pipeline.globals.output_path, pipeline.globals.output_path)
            frame_processor.post_process()
            release_resources()
        if is_image(pipeline.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    update_status('Creating temp resources...')
    create_temp(pipeline.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(pipeline.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(pipeline.globals.target_path)
    for frame_processor in get_frame_processors_modules(pipeline.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(pipeline.globals.source_path, temp_frame_paths)
        frame_processor.post_process()
        release_resources()
    # handles fps
    if pipeline.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(pipeline.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(pipeline.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(pipeline.globals.target_path)
    # handle audio
    if pipeline.globals.keep_audio:
        if pipeline.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(pipeline.globals.target_path, pipeline.globals.output_path)
    else:
        move_temp(pipeline.globals.target_path, pipeline.globals.output_path)
    # clean and validate
    clean_temp(pipeline.globals.target_path)
    if is_video(pipeline.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if pipeline.globals.target_path:
        clean_temp(pipeline.globals.target_path)
    from pipeline.control import _cleanup_temp_files
    _cleanup_temp_files()
    quit()


def run_headless() -> None:
    parse_args()
    if not pre_check():
        return
    if pipeline.globals.save_embedding_path:
        save_embedding()
        return
    for frame_processor in get_frame_processors_modules(pipeline.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    from pipeline.control import start_control_server
    start_control_server(pipeline.globals.control_port)
    if pipeline.globals.headless:
        start()
    else:
        pipeline.globals.shutdown_event.wait()
        destroy()


def save_embedding() -> None:
    from pipeline.face_analyser import get_averaged_face, save_face_embedding

    if not pipeline.globals.source_paths:
        update_status('No source images provided for embedding.')
        return
    update_status(f'Analyzing {len(pipeline.globals.source_paths)} source(s)...')
    face = get_averaged_face(pipeline.globals.source_paths)
    if face is None:
        update_status('No face detected in source images.')
        return
    save_face_embedding(face, pipeline.globals.save_embedding_path)
    update_status(f'Embedding saved to: {pipeline.globals.save_embedding_path}')
