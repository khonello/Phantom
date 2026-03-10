import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import pipeline.globals
from pipeline.api.schema import COMMANDS, PRESETS

NAME = 'ROOP.CONTROL'


def _create_embedding_bg(paths: List[str]) -> None:
    from pipeline.face_analyser import get_averaged_face, save_face_embedding
    pipeline.globals.embedding_ready = False
    pipeline.globals.status_message = f'creating embedding from {len(paths)} image(s)...'
    face = get_averaged_face(paths)
    if face is None:
        pipeline.globals.status_message = 'no face detected in source images'
        return
    tmp = os.path.join(tempfile.gettempdir(), 'roop_source_embedding.npy')
    save_face_embedding(face, tmp)
    pipeline.globals.source_path = tmp
    pipeline.globals.embedding_ready = True
    pipeline.globals.status_message = 'embedding ready'


def _cleanup_temp_files() -> None:
    tmp = os.path.join(tempfile.gettempdir(), 'roop_source_embedding.npy')
    if os.path.exists(tmp):
        os.remove(tmp)
        print(f'[{NAME}] Removed temp embedding: {tmp}')
    pipeline.globals.source_path = None
    pipeline.globals.source_paths = []
    pipeline.globals.embedding_ready = False
    pipeline.globals.status_message = 'session cleared'


def _dispatch(action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if action == 'set_source':
        pipeline.globals.source_path = data['path']
        return {'status': 'ok', 'source_path': pipeline.globals.source_path}

    if action == 'set_target':
        pipeline.globals.target_path = data['path']
        return {'status': 'ok', 'target_path': pipeline.globals.target_path}

    if action == 'set_output':
        pipeline.globals.output_path = data['path']
        return {'status': 'ok', 'output_path': pipeline.globals.output_path}

    if action == 'set_keep_fps':
        pipeline.globals.keep_fps = bool(data['value'])
        return {'status': 'ok', 'keep_fps': pipeline.globals.keep_fps}

    if action == 'set_keep_frames':
        pipeline.globals.keep_frames = bool(data['value'])
        return {'status': 'ok', 'keep_frames': pipeline.globals.keep_frames}

    if action == 'set_keep_audio':
        pipeline.globals.keep_audio = bool(data['value'])
        return {'status': 'ok', 'keep_audio': pipeline.globals.keep_audio}

    if action == 'set_many_faces':
        pipeline.globals.many_faces = bool(data['value'])
        return {'status': 'ok', 'many_faces': pipeline.globals.many_faces}

    if action == 'set_quality':
        preset = data['preset']
        if preset not in PRESETS:
            return {'error': f'unknown preset: {preset}. choices: fast, optimal, production'}
        for key, value in PRESETS[preset].items():
            setattr(pipeline.globals, key, value)
        pipeline.globals.quality = preset
        return {'status': 'ok', 'preset': preset}

    if action == 'create_embedding':
        paths: List[str] = data.get('paths', [])
        pipeline.globals.source_paths = paths
        thread = threading.Thread(target=_create_embedding_bg, args=(paths,), daemon=True)
        thread.start()
        return {'status': 'ok', 'message': f'creating embedding from {len(paths)} image(s)'}

    if action == 'set_input_url':
        pipeline.globals.input_url = data.get('url') or None
        return {'status': 'ok', 'input_url': pipeline.globals.input_url}

    if action == 'set_blend':
        pipeline.globals.blend = float(data['value'])
        return {'status': 'ok', 'blend': pipeline.globals.blend}

    if action == 'set_alpha':
        pipeline.globals.alpha = float(data['value'])
        return {'status': 'ok', 'alpha': pipeline.globals.alpha}

    if action == 'start':
        from pipeline.core import start
        thread = threading.Thread(target=start, daemon=True)
        thread.start()
        return {'status': 'ok', 'started': True}

    if action == 'start_stream':
        from pipeline.stream import start_pipeline
        thread = threading.Thread(target=start_pipeline, daemon=True)
        thread.start()
        return {'status': 'ok', 'started': True}

    if action == 'stop':
        from pipeline.stream import stop_pipeline
        stop_pipeline()
        return {'status': 'ok', 'stopped': True}

    if action == 'stop_stream':
        from pipeline.stream import stop_pipeline
        stop_pipeline()
        return {'status': 'ok', 'stopped': True}

    if action == 'cleanup_session':
        _cleanup_temp_files()
        return {'status': 'ok'}

    if action == 'shutdown':
        _cleanup_temp_files()
        threading.Timer(0.2, pipeline.globals.shutdown_event.set).start()
        return {'status': 'ok', 'shutting_down': True}

    return {'error': f'unknown action: {action}'}


def _get_status() -> Dict[str, Any]:
    return {
        'status_message': pipeline.globals.status_message,
        'source_path': pipeline.globals.source_path,
        'target_path': pipeline.globals.target_path,
        'output_path': pipeline.globals.output_path,
        'keep_fps': pipeline.globals.keep_fps,
        'keep_frames': pipeline.globals.keep_frames,
        'keep_audio': pipeline.globals.keep_audio,
        'many_faces': pipeline.globals.many_faces,
        'quality': pipeline.globals.quality,
        'tracker': pipeline.globals.tracker,
        'alpha': pipeline.globals.alpha,
        'blend': pipeline.globals.blend,
        'luminance_blend': pipeline.globals.luminance_blend,
        'enhance_interval': pipeline.globals.enhance_interval,
        'redetect_interval': pipeline.globals.redetect_interval,
        'buffer_size': pipeline.globals.buffer_size,
        'embedding_ready': pipeline.globals.embedding_ready,
        'ws_port': pipeline.globals.control_port + 1,
    }


class _ControlHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == '/status':
            self._respond_json(200, _get_status())
        elif self.path == '/frame':
            self._respond_frame()
        else:
            self._respond_json(404, {'error': 'not found'})

    def do_POST(self) -> None:
        if self.path != '/control':
            self._respond_json(404, {'error': 'not found'})
            return
        length = int(self.headers.get('Content-Length', 0))
        try:
            data = json.loads(self.rfile.read(length))
            action = data.get('action', '')
            if action not in COMMANDS:
                self._respond_json(400, {'error': f'unknown action: {action}', 'valid': list(COMMANDS.keys())})
                return
            result = _dispatch(action, data)
            self._respond_json(200, result)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._respond_json(400, {'error': str(e)})
        except Exception as e:
            self._respond_json(500, {'error': str(e)})

    def _respond_frame(self) -> None:
        from pipeline.stream import _latest_frame
        if _latest_frame is None:
            self._respond_json(404, {'error': 'no frame available'})
            return
        try:
            import cv2
            ok, buf = cv2.imencode('.jpg', _latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self._respond_json(500, {'error': 'encode failed'})
                return
            payload = buf.tobytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self._respond_json(500, {'error': str(e)})

    def _respond_json(self, code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        print(f'[{NAME}] ' + format % args)


def start_control_server(port: int) -> Optional[threading.Thread]:
    try:
        import pipeline.ws_server
        pipeline.ws_server.start(port + 1)

        server = HTTPServer(('0.0.0.0', port), _ControlHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f'[{NAME}] Listening on port {port}')
        print(f'[{NAME}] GET  http://localhost:{port}/status')
        print(f'[{NAME}] GET  http://localhost:{port}/frame')
        print(f'[{NAME}] POST http://localhost:{port}/control')
        return thread
    except Exception as e:
        print(f'[{NAME}] Failed to start: {e}')
        return None
