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


# ============================================================================
# WebSocket Message Types (New in Phase 0)
# ============================================================================
# Typed message envelopes for future WebSocket API server (Phase 4)

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIMessage:
    """Base WebSocket message envelope."""
    type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON transmission."""
        return {
            'type': self.type,
            'data': self.data,
        }


@dataclass
class CommandMessage(APIMessage):
    """Command message sent to the server."""
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.request_id:
            result['request_id'] = self.request_id
        return result


@dataclass
class EventMessage(APIMessage):
    """Event message broadcast from the server."""
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.timestamp:
            result['timestamp'] = self.timestamp
        return result


@dataclass
class ResponseMessage(APIMessage):
    """Response to a command."""
    request_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result['success'] = self.success
        if self.request_id:
            result['request_id'] = self.request_id
        if self.error:
            result['error'] = self.error
        return result


# Command type constants
CMD_SET_SOURCE = 'set_source'
CMD_SET_TARGET = 'set_target'
CMD_SET_OUTPUT = 'set_output'
CMD_START = 'start'
CMD_START_STREAM = 'start_stream'
CMD_STOP = 'stop'
CMD_STOP_STREAM = 'stop_stream'
CMD_SET_QUALITY = 'set_quality'
CMD_SET_BLEND = 'set_blend'
CMD_SET_ALPHA = 'set_alpha'
CMD_SET_INPUT_URL = 'set_input_url'
CMD_CREATE_EMBEDDING = 'create_embedding'
CMD_CLEANUP_SESSION = 'cleanup_session'
CMD_SHUTDOWN = 'shutdown'

# Event type constants
EVT_FRAME_READY = 'frame_ready'
EVT_DETECTION = 'detection'
EVT_FACE_LOST = 'face_lost'
EVT_STATUS = 'status'
EVT_DROP_RATE = 'drop_rate'
EVT_EMBEDDING_READY = 'embedding_ready'
EVT_PIPELINE_STARTED = 'pipeline_started'
EVT_PIPELINE_STOPPED = 'pipeline_stopped'
