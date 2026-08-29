#!/usr/bin/env python3

import os
import sys

# Load .env before any config initialization
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List, Optional
import platform
import signal
import argparse
import torch
import onnxruntime

import pipeline.metadata
from pipeline.config import CONFIG
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.events import BUS
from pipeline.logging import emit_status
from pipeline.api.schema import PRESETS
from pipeline.services import swapper_models

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def _env_float(name: str) -> Optional[float]:
    """Read a float from the environment, or None if unset/unparseable."""
    raw = os.environ.get(name)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _env_int(name: str) -> Optional[int]:
    """Read an int from the environment, or None if unset/unparseable."""
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _env_bool(name: str) -> Optional[bool]:
    """Read a bool from the environment, or None if unset/unrecognised."""
    raw = (os.environ.get(name) or '').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    return None


def parse_args() -> None:
    """Parse command-line arguments and update CONFIG."""
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())

    program = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100)
    )
    program.add_argument('-s', '--source', help='select source image(s) or embedding (.npy)', dest='source_path', nargs='+')
    program.add_argument('-t', '--target', help='select a target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    # These two decide what file the operator gets back, and both were unable
    # to say "no". `store_true` gave `--keep-fps` a default of False, so every
    # render retimed to 30fps unless asked otherwise - duplicating frames on a
    # 24fps source and discarding motion on a 60fps one, for a tool whose job
    # is swapping faces rather than transcoding. Worse, that is the path the
    # audio-desync bug lived on, so the default routed everyone through the
    # more fragile branch.
    #
    # `--keep-audio` was `store_true` with `default=True`, which can only ever
    # produce True: there was no way to drop audio at all.
    #
    # BooleanOptionalAction gives both a real off switch (`--no-keep-fps`,
    # `--no-keep-audio`) and lets the default be the conservative answer:
    # hand back what was handed in.
    program.add_argument('--keep-fps', help='keep the source frame rate', dest='keep_fps',
                         action=argparse.BooleanOptionalAction, default=True)
    program.add_argument('--keep-audio', help='keep the source audio', dest='keep_audio',
                         action=argparse.BooleanOptionalAction, default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder',
                        default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality',
                        type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory',
                        type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='available execution provider', dest='execution_provider',
                        default=suggest_default_execution_providers(), choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads',
                        type=int, default=suggest_execution_threads())
    program.add_argument('--quality', help='stream quality preset', dest='quality',
                        default='optimal', choices=['fast', 'optimal', 'production'])
    program.add_argument('--tracker', help='face tracker for stream mode', dest='tracker',
                        default=None, choices=['csrt', 'kcf', 'mosse'])
    program.add_argument('--alpha', help='EMA smoothing factor (0.0=max smooth, 1.0=no smooth)',
                        dest='alpha', type=float, default=None)
    program.add_argument('--blend', help='swap blend ratio (0.0=original, 1.0=full swap)',
                        dest='blend', type=float, default=None)
    program.add_argument('--luminance-blend', help='enable luminance-adaptive blend',
                        dest='luminance_blend', action='store_true', default=None)
    # Realism knobs. All default to None so the quality preset applies unless
    # a flag or environment variable explicitly overrides it. Environment
    # defaults exist because the RunPod pod is configured through .env rather
    # than a command line.
    program.add_argument('--swapper-model',
                        help='face swap model (also selects its realism profile)',
                        dest='swapper_model',
                        choices=list(swapper_models.names()),
                        default=os.environ.get('SWAPPER_MODEL'))
    program.add_argument('--enhancer-model', help='face restoration backend',
                        dest='enhancer_model', choices=['codeformer', 'gfpgan'],
                        default=os.environ.get('ENHANCER_MODEL'))
    program.add_argument('--enhancer-weight', help='CodeFormer fidelity (0=most restoration, 1=closest to input)',
                        dest='enhancer_weight', type=float, default=_env_float('ENHANCER_WEIGHT'))
    program.add_argument('--enhance-strength', help='how much of the restored face to keep (0.0-1.0)',
                        dest='enhance_strength', type=float, default=_env_float('ENHANCE_STRENGTH'))
    program.add_argument('--aligned-size', help='ceiling on compositing resolution (128-512); actual size follows face size',
                        dest='aligned_size', type=int, default=_env_int('ALIGNED_SIZE'))
    program.add_argument('--restore-size', help='FFHQ crop edge fed to the restorer (128-512); '
                                                'ignored if the model has fixed inputs',
                        dest='restore_size', type=int, default=_env_int('RESTORE_SIZE'))
    program.add_argument('--restore-min-face', help='skip restoration below this face size in px '
                                                    '(shorter side); 0 never skips',
                        dest='restore_min_face', type=int, default=_env_int('RESTORE_MIN_FACE'))
    program.add_argument('--temporal-alpha', help='EMA on composited pixels (1.0=off)',
                        dest='temporal_alpha', type=float, default=_env_float('TEMPORAL_ALPHA'))
    program.add_argument('--color-strength', help='scales the LAB colour transfer (0.0-1.0)',
                        dest='color_strength', type=float, default=_env_float('COLOR_STRENGTH'))
    program.add_argument('--no-enhance', help='disable face restoration',
                        dest='enhance', action='store_false', default=_env_bool('ENHANCE'))
    program.add_argument('--no-grain', help='disable sensor-noise matching',
                        dest='grain', action='store_false', default=_env_bool('GRAIN'))
    program.add_argument('--no-occluder', help='disable occlusion masking',
                        dest='occluder', action='store_false', default=_env_bool('OCCLUDER'))
    program.add_argument('--fp16', help='load half-precision weights where a -fp16.onnx copy exists',
                        dest='fp16', action='store_true', default=_env_bool('FP16'))
    program.add_argument('--cuda-graphs', help='capture and replay the CUDA kernel launch sequence',
                        dest='cuda_graphs', action='store_true', default=_env_bool('CUDA_GRAPHS'))
    program.add_argument('--cuda-streams', help='copies on a separate CUDA stream, overlapping transfer with compute',
                        dest='cuda_streams', action='store_true', default=_env_bool('CUDA_STREAMS'))
    program.add_argument('--async-encode', help='JPEG-encode on the sender thread, not the pipeline thread',
                        dest='async_encode', action='store_true', default=_env_bool('ASYNC_ENCODE'))
    program.add_argument('--trt', help='route inference through the TensorRT provider',
                        dest='trt', action='store_true', default=_env_bool('TRT'))
    program.add_argument('--trt-gpus', help='comma-separated GPU names worth building TensorRT engines for',
                        dest='trt_gpus', default=os.environ.get('TRT_GPUS'))
    program.add_argument('--debug-frames', help='write (input, output) frame pairs to this directory',
                        dest='debug_frames_dir', default=os.environ.get('DEBUG_FRAMES_DIR'))
    program.add_argument('--debug-frames-stride', help='keep every Nth frame pair (default 1)',
                        dest='debug_frames_stride', type=int, default=_env_int('DEBUG_FRAMES_STRIDE'))
    program.add_argument('--debug-frames-limit', help='stop after N frame pairs (0 = unlimited)',
                        dest='debug_frames_limit', type=int, default=_env_int('DEBUG_FRAMES_LIMIT'))
    program.add_argument('--guard-observe',
                        help='evaluate and record input guards without acting on them '
                             '(calibration: a session that enforces cannot measure)',
                        dest='guard_observe', action='store_true',
                        default=_env_bool('GUARD_OBSERVE') or None)
    program.add_argument('--guard-report',
                        help='write guard calibration telemetry as JSON to this path',
                        dest='guard_report', default=os.environ.get('GUARD_REPORT'))
    program.add_argument('--no-guards', help='disable runtime input guards entirely',
                        dest='guards', action='store_false', default=_env_bool('GUARDS'))
    program.add_argument('--input-url', help='network stream URL (RTSP/RTMP/HTTP)',
                        dest='input_url', default=None)
    program.add_argument('--control-port', help='API server port',
                        dest='control_port', type=int,
                        default=int(os.environ.get('API_PORT', '9000')))
    program.add_argument('--stream', help='start in stream mode (realtime webcam/network)',
                        dest='stream_mode', action='store_true', default=False)
    program.add_argument('--log-level', help='logging level (debug/info/warning/error)',
                        dest='log_level', default=os.environ.get('LOG_LEVEL', 'info'),
                        choices=['debug', 'info', 'warning', 'error'])
    program.add_argument('-v', '--version', action='version',
                        version=f'{pipeline.metadata.name} {pipeline.metadata.version}')

    args = program.parse_args()

    # Update CONFIG
    CONFIG.set('source_paths', args.source_path or [])
    CONFIG.set('source_path', args.source_path[0] if args.source_path else None)
    CONFIG.set('target_path', args.target_path)
    CONFIG.set('keep_fps', args.keep_fps)
    CONFIG.set('keep_audio', args.keep_audio)
    CONFIG.set('keep_frames', args.keep_frames)
    CONFIG.set('many_faces', args.many_faces)
    CONFIG.set('video_encoder', args.video_encoder)
    CONFIG.set('video_quality', args.video_quality)
    CONFIG.set('max_memory', args.max_memory)
    CONFIG.set('execution_providers', decode_execution_providers(args.execution_provider))
    CONFIG.set('execution_threads', args.execution_threads)
    CONFIG.set('quality', args.quality)

    # Quality preset first; explicit flags and environment variables then
    # override it. Applying to CONFIG rather than patching `args` means new
    # preset keys take effect without also having to be argparse destinations.
    if args.quality in PRESETS:
        CONFIG.apply_preset(args.quality)

    # Then the model's realism profile. The preset owns compute; the model owns
    # appearance. The explicit flags below override either.
    CONFIG.apply_model_profile(args.swapper_model or CONFIG.swapper_model)

    for field, value in (
        ('tracker', args.tracker),
        ('alpha', args.alpha),
        ('blend', args.blend),
        ('luminance_blend', args.luminance_blend),
        ('enhancer_model', args.enhancer_model),
        ('enhancer_weight', args.enhancer_weight),
        ('enhance_strength', args.enhance_strength),
        ('aligned_size', args.aligned_size),
        ('restore_size', args.restore_size),
        ('restore_min_face', args.restore_min_face),
        ('temporal_alpha', args.temporal_alpha),
        ('color_strength', args.color_strength),
        ('enhance', args.enhance),
        ('grain', args.grain),
        ('occluder', args.occluder),
        ('fp16', args.fp16),
        ('cuda_graphs', args.cuda_graphs),
        ('cuda_streams', args.cuda_streams),
        ('async_encode', args.async_encode),
        ('trt', args.trt),
        ('trt_gpus', args.trt_gpus),
        ('debug_frames_dir', args.debug_frames_dir),
        ('debug_frames_stride', args.debug_frames_stride),
        ('debug_frames_limit', args.debug_frames_limit),
        ('guards', args.guards),
        ('guard_observe', args.guard_observe),
        ('guard_report', args.guard_report),
    ):
        if value is not None:
            CONFIG.set(field, value)

    if args.input_url:
        CONFIG.set('input_url', args.input_url)
    CONFIG.set('control_port', args.control_port)
    CONFIG.set('log_level', args.log_level)
    CONFIG.set('stream_mode', args.stream_mode)

    # Set output path (normalize if directory)
    from pipeline.io.ffmpeg import normalize_output_path
    CONFIG.set('output_path', normalize_output_path(CONFIG.source_path or '', CONFIG.target_path or '', args.output_path or ''))

    # Determine if headless mode
    CONFIG.set('headless', bool(CONFIG.source_path and CONFIG.target_path and CONFIG.output_path))

    # Log CUDA availability
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            emit_status(f'GPU available: {gpu_name}', scope='CORE')
        else:
            emit_status('No GPU detected, using CPU', scope='CORE')
    except Exception as e:
        emit_status(f'GPU detection error: {type(e).__name__}: {e}', scope='CORE', level='warning')


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    decoded = [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
               if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    # Always include CPUExecutionProvider as fallback — InsightFace and ONNX
    # silently fall back to CPU if CUDA init fails without it in the list.
    if 'CPUExecutionProvider' not in decoded:
        decoded.append('CPUExecutionProvider')
    return decoded


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_default_execution_providers() -> List[str]:
    available = encode_execution_providers(onnxruntime.get_available_providers())
    return ['cuda'] if 'cuda' in available else ['cpu']


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in CONFIG.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in CONFIG.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak (lazy import — TF init can hang on some pods)
    try:
        import tensorflow
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
                tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
            ])
    except Exception as e:
        print(f"[CORE] WARNING: TensorFlow GPU memory limit skipped: {e}")
    # limit memory usage
    if CONFIG.max_memory:
        memory = CONFIG.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = CONFIG.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            # Windows-only, and this branch only runs there. mypy checks against
            # the current platform's stubs, so on Linux — where CI runs — it
            # cannot see this attribute. The mirror of the `resource` ignore
            # below, which is invisible on Windows for the same reason.
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            # POSIX only, which is why it is imported inside the non-Windows
            # branch. mypy checks against the current platform's stubs, so on
            # Windows it cannot see either name.
            import resource
            resource.setrlimit(  # type: ignore[attr-defined]
                resource.RLIMIT_DATA, (memory, memory),  # type: ignore[attr-defined]
            )


def release_resources() -> None:
    if 'CUDAExecutionProvider' in CONFIG.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    import shutil
    if sys.version_info < (3, 9):
        emit_status('Python version is not supported - please upgrade to 3.9 or higher.', scope='CORE')
        return False
    if not shutil.which('ffmpeg'):
        emit_status('ffmpeg is not installed.', scope='CORE')
        return False
    from pipeline.services.face_swapping import FaceSwapper
    if not FaceSwapper(CONFIG).pre_check():
        return False
    return True


def run_headless() -> None:
    """Run headless (batch or stream) pipeline with specified source/target/output."""
    parse_args()
    if not pre_check():
        return

    limit_resources()

    # Start WebSocket API server
    from pipeline.api.server import WebSocketAPIServer
    server = WebSocketAPIServer(CONFIG, None, CONFIG.control_port)
    server.start()

    # Stream mode: start realtime pipeline
    if getattr(CONFIG, 'stream_mode', False):
        # Reuse the server's pipeline — creating a second instance would cause
        # two capture loops fighting over the same webcam/URL, and WebSocket
        # commands (start/stop) would target a different pipeline than the one
        # actually running.
        emit_status('Starting in stream mode', scope='CORE')
        import threading
        t = threading.Thread(target=server.pipeline.run_stream, daemon=True)
        t.start()
        CONFIG.shutdown_event.wait()
    elif CONFIG.headless:
        # Batch mode: run and exit
        pipeline = ProcessingPipeline(CONFIG, BUS)
        pipeline.run_batch()
    else:
        # Server-only mode: wait for commands via WebSocket
        CONFIG.shutdown_event.wait()

    server.stop()
    destroy()


def destroy() -> None:
    """Clean up resources on shutdown."""
    from pipeline.io.ffmpeg import clean_temp
    if CONFIG.target_path:
        clean_temp(CONFIG, CONFIG.target_path)
    quit()
