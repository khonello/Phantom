import queue
import threading
import time
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

import pipeline.globals
import pipeline.ws_server
from pipeline.face_analyser import get_one_face

NAME = 'ROOP.STREAM'

_stop_event: threading.Event = threading.Event()
_latest_frame: Optional[np.ndarray] = None
_running: bool = False

# Async enhancement queues (maxsize=1 — always latest frame only)
_enhance_input: queue.Queue = queue.Queue(maxsize=1)
_enhance_output: queue.Queue = queue.Queue(maxsize=1)


def _make_tracker(tracker_name: str) -> Optional[cv2.Tracker]:
    name = tracker_name.lower()
    try:
        if name == 'kcf':
            creator = getattr(cv2, 'TrackerKCF_create', None) or getattr(cv2.legacy, 'TrackerKCF_create', None)
        elif name == 'mosse':
            creator = getattr(cv2.legacy, 'TrackerMOSSE_create', None)
        else:
            creator = getattr(cv2, 'TrackerCSRT_create', None) or getattr(cv2.legacy, 'TrackerCSRT_create', None)
        return creator() if creator else None
    except Exception:
        return None


def _bbox_insightface_to_cv2(bbox: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return x1, y1, x2 - x1, y2 - y1


def _bbox_in_frame(bbox_cv2: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> bool:
    x, y, w, h = bbox_cv2
    fh, fw = frame_shape[:2]
    return x >= 0 and y >= 0 and x + w <= fw and y + h <= fh and w > 0 and h > 0


def _ema(current: np.ndarray, previous: Optional[np.ndarray], alpha: float) -> np.ndarray:
    if previous is None:
        return current.copy()
    return alpha * current + (1.0 - alpha) * previous


def _luminance_adaptive_blend(
    swapped: np.ndarray,
    original: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    base_blend: float,
) -> np.ndarray:
    x, y, w, h = face_bbox
    fh, fw = original.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        return cv2.addWeighted(swapped, base_blend, original, 1.0 - base_blend, 0)

    orig_lum = float(np.mean(cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0]))
    swap_lum = float(np.mean(cv2.cvtColor(swapped[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0]))
    lum_delta = abs(swap_lum - orig_lum)

    if lum_delta < 10.0:
        return cv2.addWeighted(swapped, base_blend, original, 1.0 - base_blend, 0)

    adaptive_blend = base_blend * max(0.5, 1.0 - lum_delta / 255.0)
    return cv2.addWeighted(swapped, adaptive_blend, original, 1.0 - adaptive_blend, 0)


def _load_gfpgan() -> Any:
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'GFPGANv1.4.pth')
    if not os.path.exists(model_path):
        print(f'[{NAME}] GFPGANv1.4.pth not found in models/ — enhancement disabled')
        return None
    try:
        from gfpgan import GFPGANer
        return GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
    except ImportError:
        print(f'[{NAME}] gfpgan not installed — enhancement disabled')
        return None
    except Exception as e:
        print(f'[{NAME}] GFPGAN failed to load: {e}')
        return None


def _enhancement_worker(stop_event: threading.Event) -> None:
    gfpganer = _load_gfpgan()
    if gfpganer is None:
        while not stop_event.is_set():
            try:
                seq, frame = _enhance_input.get(timeout=0.1)
                if _enhance_output.full():
                    try:
                        _enhance_output.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    _enhance_output.put_nowait((seq, frame))
                except queue.Full:
                    pass
            except queue.Empty:
                continue
        return

    print(f'[{NAME}] GFPGAN enhancement active (interval: {pipeline.globals.enhance_interval} frames)')
    while not stop_event.is_set():
        try:
            seq, frame = _enhance_input.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            _, _, restored = gfpganer.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            enhanced = restored if restored is not None else frame
        except Exception:
            enhanced = frame

        if _enhance_output.full():
            try:
                _enhance_output.get_nowait()
            except queue.Empty:
                pass
        try:
            _enhance_output.put_nowait((seq, enhanced))
        except queue.Full:
            pass


def _pipeline_loop(on_stop: Optional[Callable[[], None]] = None) -> None:
    import pipeline.processors.frame.face_swapper as fs_module
    from pipeline.processors.frame.face_swapper import get_face_swapper, get_source_face, swap_face

    get_face_swapper()
    source_face = get_source_face()

    width, height, fps = 960, 540, 30

    input_source: Any = pipeline.globals.input_url if pipeline.globals.input_url else 0
    cap = cv2.VideoCapture(input_source)
    if not pipeline.globals.input_url:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    enhance_interval = pipeline.globals.enhance_interval
    enhancement_thread: Optional[threading.Thread] = None
    if enhance_interval > 0:
        enhancement_thread = threading.Thread(
            target=_enhancement_worker,
            args=(_stop_event,),
            daemon=True,
        )
        enhancement_thread.start()

    drop_count: int = 0
    drop_window_start: float = time.time()

    tracker: Optional[cv2.Tracker] = None
    tracker_initialized: bool = False
    tracker_bbox: Optional[Tuple[int, int, int, int]] = None

    cached_face: Optional[Any] = None
    prev_kps: Optional[np.ndarray] = None

    frame_count: int = 0
    seq: int = 0
    active_source_path: Optional[str] = pipeline.globals.source_path

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            alpha = pipeline.globals.alpha
            blend = pipeline.globals.blend

            if pipeline.globals.source_path != active_source_path:
                active_source_path = pipeline.globals.source_path
                fs_module.SOURCE_FACE = None
                source_face = get_source_face()
                tracker = None
                tracker_initialized = False
                tracker_bbox = None
                cached_face = None
                prev_kps = None
                frame_count = 0

            need_detect = (
                not tracker_initialized
                or frame_count % pipeline.globals.redetect_interval == 0
            )

            if need_detect:
                detected = get_one_face(frame)
                if detected is not None and detected.kps is not None:
                    bbox_cv2 = _bbox_insightface_to_cv2(detected.bbox)
                    if _bbox_in_frame(bbox_cv2, frame.shape):
                        smoothed_kps = _ema(detected.kps, prev_kps, alpha)
                        detected.kps = smoothed_kps
                        prev_kps = smoothed_kps
                        cached_face = detected

                        tracker = _make_tracker(pipeline.globals.tracker)
                        if tracker is not None:
                            tracker.init(frame, bbox_cv2)
                            tracker_initialized = True
                        else:
                            tracker_initialized = False
                        tracker_bbox = bbox_cv2
                    else:
                        tracker_initialized = False
                        tracker_bbox = None
                        cached_face = None
                        prev_kps = None
                else:
                    tracker_initialized = False
                    tracker_bbox = None
                    cached_face = None
                    prev_kps = None

            elif tracker_initialized and tracker is not None:
                ok, bbox_raw = tracker.update(frame)
                if ok:
                    new_bbox = (
                        int(bbox_raw[0]), int(bbox_raw[1]),
                        int(bbox_raw[2]), int(bbox_raw[3]),
                    )
                    if _bbox_in_frame(new_bbox, frame.shape):
                        if tracker_bbox is not None and prev_kps is not None:
                            dx = float(new_bbox[0] - tracker_bbox[0])
                            dy = float(new_bbox[1] - tracker_bbox[1])
                            estimated_kps = prev_kps + np.array([[dx, dy]], dtype=np.float32)
                            smoothed_kps = _ema(estimated_kps, prev_kps, alpha)
                            prev_kps = smoothed_kps
                            if cached_face is not None:
                                cached_face.kps = smoothed_kps
                        tracker_bbox = new_bbox
                    else:
                        tracker_initialized = False
                        tracker_bbox = None
                        cached_face = None
                        prev_kps = None
                else:
                    tracker_initialized = False
                    tracker_bbox = None
                    cached_face = None
                    prev_kps = None

            in_warmup = frame_count < pipeline.globals.warmup_frames and not tracker_initialized

            if in_warmup or source_face is None or cached_face is None:
                display_frame = frame.copy()
            else:
                try:
                    original = frame.copy()
                    swapped = swap_face(source_face, cached_face, frame)
                    if pipeline.globals.luminance_blend and tracker_bbox is not None:
                        display_frame = _luminance_adaptive_blend(swapped, original, tracker_bbox, blend)
                    else:
                        display_frame = cv2.addWeighted(swapped, blend, original, 1.0 - blend, 0)
                except Exception:
                    display_frame = frame.copy()

            seq += 1
            current_seq = seq

            if enhance_interval > 0:
                try:
                    eseq, enhanced_frame = _enhance_output.get_nowait()
                    if current_seq - eseq <= enhance_interval:
                        display_frame = enhanced_frame
                except queue.Empty:
                    pass
                if frame_count % enhance_interval == 0:
                    if _enhance_input.full():
                        try:
                            _enhance_input.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        _enhance_input.put_nowait((current_seq, display_frame))
                    except queue.Full:
                        pass

            # Expose latest frame for HTTP /frame endpoint
            global _latest_frame
            _latest_frame = display_frame.copy()

            # Broadcast to WebSocket clients
            enc_ok, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if enc_ok:
                pipeline.ws_server.broadcast(buf.tobytes())

            now = time.time()
            if now - drop_window_start >= 1.0:
                if drop_count > 0:
                    print(f'[{NAME}] Dropped {drop_count} frames/sec')
                drop_count = 0
                drop_window_start = now

            frame_count += 1

    finally:
        cap.release()
        if enhancement_thread is not None:
            enhancement_thread.join(timeout=2)
        _stop_event.set()
        if on_stop is not None:
            on_stop()


def _reset_queues() -> None:
    global _stop_event, _enhance_input, _enhance_output
    _stop_event = threading.Event()
    _enhance_input = queue.Queue(maxsize=1)
    _enhance_output = queue.Queue(maxsize=1)


def start_pipeline() -> None:
    global _running
    if _running:
        return
    _running = True
    _reset_queues()
    print(f'[{NAME}] Pipeline started.')
    try:
        _pipeline_loop()
    finally:
        _running = False


def stop_pipeline() -> None:
    _stop_event.set()
