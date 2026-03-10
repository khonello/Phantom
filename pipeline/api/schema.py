from typing import Any, Dict

# Quality presets — single source of truth used by core.py, control.py
PRESETS: Dict[str, Dict[str, Any]] = {
    'fast': {
        'tracker': 'kcf',
        'alpha': 0.7,
        'blend': 0.65,
        'luminance_blend': False,
        'enhance_interval': 10,
        'buffer_size': 3,
        'redetect_interval': 30,
        'warmup_frames': 3,
    },
    'optimal': {
        'tracker': 'csrt',
        'alpha': 0.6,
        'blend': 0.65,
        'luminance_blend': True,
        'enhance_interval': 5,
        'buffer_size': 4,
        'redetect_interval': 30,
        'warmup_frames': 5,
    },
    'production': {
        'tracker': 'csrt',
        'alpha': 0.5,
        'blend': 0.65,
        'luminance_blend': True,
        'enhance_interval': 1,
        'buffer_size': 5,
        'redetect_interval': 20,
        'warmup_frames': 5,
    },
}

# Commands accepted by the control server (POST /control)
COMMANDS: Dict[str, Dict[str, Any]] = {
    # Source / target / output
    'set_source':      {'path': str},
    'set_target':      {'path': str},
    'set_output':      {'path': str},
    # Processing settings
    'set_keep_fps':    {'value': bool},
    'set_keep_frames': {'value': bool},
    'set_keep_audio':  {'value': bool},
    'set_many_faces':  {'value': bool},
    # Stream tuning
    'set_quality':     {'preset': str},
    'set_blend':       {'value': float},
    'set_alpha':       {'value': float},
    # Stream routing
    'set_input_url':   {'url': str},
    # Source embedding
    'create_embedding': {'paths': list},
    # Pipeline control
    'start':           {},
    'start_stream':    {},
    'stop':            {},
    'stop_stream':     {},
    'cleanup_session': {},
    'shutdown':        {},
}

# Events emitted by the pipeline (GET /status, future WebSocket push)
EVENTS: Dict[str, Dict[str, Any]] = {
    'status':    {'message': str},
    'started':   {},
    'stopped':   {},
    'face_lost': {},
    'drop_rate': {'count': int},
}
