import json
import subprocess
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

NAME = 'DESKTOP.CONTROLLER'
UDP_INGEST_PORT: int = 5000


class PipelineClient:
    """HTTP client for communicating with a running roop-cam pipeline."""

    def __init__(self, host: str = 'localhost', port: int = 9000) -> None:
        self.host = host
        self.port = port
        self.base_url = f'http://{host}:{port}'

    def _post(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        payload = json.dumps({'action': action, **kwargs}).encode()
        req = urllib.request.Request(
            f'{self.base_url}/control',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return {'error': json.loads(e.read()).get('error', str(e))}
        except Exception as e:
            return {'error': str(e)}

    def status(self) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(f'{self.base_url}/status', timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {'error': str(e)}

    def get_frame(self) -> Optional[bytes]:
        try:
            with urllib.request.urlopen(f'{self.base_url}/frame', timeout=2) as resp:
                return resp.read()
        except Exception:
            return None

    # Source / target / output
    def set_source(self, path: str) -> Dict[str, Any]:
        return self._post('set_source', path=path)

    def set_target(self, path: str) -> Dict[str, Any]:
        return self._post('set_target', path=path)

    def set_output(self, path: str) -> Dict[str, Any]:
        return self._post('set_output', path=path)

    # Processing settings
    def set_keep_fps(self, value: bool) -> Dict[str, Any]:
        return self._post('set_keep_fps', value=value)

    def set_keep_frames(self, value: bool) -> Dict[str, Any]:
        return self._post('set_keep_frames', value=value)

    def set_keep_audio(self, value: bool) -> Dict[str, Any]:
        return self._post('set_keep_audio', value=value)

    def set_many_faces(self, value: bool) -> Dict[str, Any]:
        return self._post('set_many_faces', value=value)

    # Source embedding
    def create_embedding(self, paths: List[str]) -> Dict[str, Any]:
        return self._post('create_embedding', paths=paths)

    # Stream routing
    def set_input_url(self, url: str) -> Dict[str, Any]:
        return self._post('set_input_url', url=url)

    def set_stream_url(self, url: str) -> Dict[str, Any]:
        return self._post('set_stream_url', url=url)

    # Stream tuning
    def set_quality(self, preset: str) -> Dict[str, Any]:
        return self._post('set_quality', preset=preset)

    def set_blend(self, value: float) -> Dict[str, Any]:
        return self._post('set_blend', value=value)

    def set_alpha(self, value: float) -> Dict[str, Any]:
        return self._post('set_alpha', value=value)

    # Pipeline control
    def start(self) -> Dict[str, Any]:
        return self._post('start')

    def start_stream(self) -> Dict[str, Any]:
        return self._post('start_stream')

    def stop(self) -> Dict[str, Any]:
        return self._post('stop')

    def stop_stream(self) -> Dict[str, Any]:
        return self._post('stop_stream')

    def cleanup_session(self) -> Dict[str, Any]:
        return self._post('cleanup_session')

    def shutdown(self) -> Dict[str, Any]:
        return self._post('shutdown')


def _run_webcam_broadcast(
    webcam_index: int,
    server_host: str,
    udp_port: int,
    stop_event: threading.Event,
) -> None:
    cap = cv2.VideoCapture(webcam_index)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    cmd: List[str] = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'mpegts',
        f'udp://{server_host}:{udp_port}',
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f'[{NAME}] FFmpeg webcam broadcast failed to start: {e}')
        cap.release()
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                assert proc.stdin is not None
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                break
    finally:
        cap.release()
        try:
            assert proc.stdin is not None
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def start_webcam_broadcast(
    webcam_index: int,
    server_host: str,
    udp_port: int = UDP_INGEST_PORT,
) -> Tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_run_webcam_broadcast,
        args=(webcam_index, server_host, udp_port, stop_event),
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def stop_webcam_broadcast(thread: threading.Thread, stop_event: threading.Event) -> None:
    stop_event.set()
    thread.join(timeout=5)
