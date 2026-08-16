from typing import Any, Dict

# Quality presets — single source of truth, applied via FaceSwapConfig.apply_preset().
#
# A preset picks how much compute to spend. It does **not** change how the face
# looks.
#
# That distinction is the whole design. `enhancer_weight` and `enhance_strength`
# decide whether the output reads as a real video call or as AI, and neither one
# costs anything to compute — the weight is a scalar model input, the strength is
# one addWeighted. Varying them per preset meant "production" restored hardest
# and so looked the *most* synthetic, while presenting itself as the best
# option. They are now identical across all three presets, so an operator picking
# a preset for their GPU cannot accidentally change the look.
#
# What varies is only what actually costs: capture resolution and frame rate,
# detector input size, compositing working resolution, and whether the occluder
# runs. Landmark and pixel EMA vary with frame rate, since smoothing across
# frames means smoothing across time — the same factor at 15fps reaches twice as
# far back as it does at 30fps.
#
# Deliberately absent: `enhance` and `color_correction`. Both have explicit
# toggles in the desktop header, and a preset must not silently undo something
# the operator just clicked.
#
# Also absent: `tracker`, `blend`, `luminance_blend`, `redetect_interval`.
# Face tracking was replaced by per-frame detection plus landmark EMA, and
# blending is handled by the compositor's mask. The fields still exist on
# FaceSwapConfig so the `set_blend` / `set_alpha` API commands keep working,
# but nothing reads them.

# Look parameters — identical in every preset. Change these to change the look
# globally; override per-run with --enhancer-weight / --enhance-strength or the
# set_realism command.
_LOOK: Dict[str, Any] = {
    'enhancer_weight': 0.7,   # CodeFormer fidelity: 0=most restoration, 1=closest to input
    'enhance_strength': 0.7,  # how much of the restored face to keep
    'grain': True,            # cheap, and the biggest believability win per ms
}

PRESETS: Dict[str, Dict[str, Any]] = {
    'fast': {
        **_LOOK,
        # Capture
        'capture_width': 480,
        'capture_height': 270,
        'capture_fps': 15,
        'jpeg_quality': 60,
        # Compute
        'det_size': 320,          # detector input; runs every frame, so this
                                  # is the single largest cost in the loop
        'aligned_size': 192,      # cheaper compositing
        'occluder': False,        # skips an ONNX pass per frame
        # Smoothing, scaled to frame rate
        'alpha': 0.7,
        'temporal_alpha': 0.7,
        'buffer_size': 3,
        'warmup_frames': 3,
    },
    'optimal': {
        **_LOOK,
        'capture_width': 640,
        'capture_height': 360,
        'capture_fps': 20,
        'jpeg_quality': 70,
        'det_size': 448,
        'aligned_size': 256,
        'occluder': True,
        'alpha': 0.6,
        'temporal_alpha': 0.6,
        'buffer_size': 4,
        'warmup_frames': 5,
    },
    'production': {
        **_LOOK,
        'capture_width': 960,
        'capture_height': 540,
        'capture_fps': 30,
        'jpeg_quality': 85,
        'det_size': 640,
        'aligned_size': 320,
        'occluder': True,
        'alpha': 0.5,
        'temporal_alpha': 0.5,
        'buffer_size': 5,
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
    'set_enhance':     {'value': bool},
    'set_realism':     {'values': dict},
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
    'keep_alive':      {},
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

from dataclasses import dataclass
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
        # Include 'action' so _dispatch_message() can route by key without
        # relying on 'type' as a fallback. Both fields carry the command name.
        result['action'] = self.type
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
CMD_SET_ENHANCE = 'set_enhance'
CMD_SET_REALISM = 'set_realism'
CMD_SET_INPUT_URL = 'set_input_url'
CMD_CREATE_EMBEDDING = 'create_embedding'
CMD_CLEANUP_SESSION = 'cleanup_session'
CMD_KEEP_ALIVE = 'keep_alive'
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
EVT_AUTO_STOP_WARNING = 'auto_stop_warning'
