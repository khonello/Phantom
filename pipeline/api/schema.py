from typing import Any, Dict, Optional

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
# What every preset shares. `enhancer_weight` and `enhance_strength` used to live
# here, on the correct reasoning that a preset must not change how the face
# looks — but that put them in the wrong place rather than the wrong preset. How
# much restoration a face needs depends on **what generated it**, not on the
# frame rate, so they now belong to the model profile in
# pipeline/services/swapper_models.py. Applied after the preset, which is why
# nothing here can contradict them.
_LOOK: Dict[str, Any] = {
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
        # OFF, and put back after being switched on. `fast` is the gear an
        # operator drops to when the link is failing, and the one configuration
        # measured to hold on theirs - so it stays byte-identical to what was
        # tested rather than carrying an untested change, however cheap that
        # change looked. Occlusion costs nothing on the uplink and the pipeline
        # has the headroom for it; that is an argument for revisiting this on a
        # good link, not for altering the fallback gear.
        'occluder': False,        # skips an ONNX pass per frame
        # Smoothing, scaled to frame rate
        'alpha': 0.7,
        'temporal_alpha': 0.7,
        'buffer_size': 3,
        'warmup_frames': 3,
    },
    # The default, and it is chosen by UPLINK rather than by compute.
    #
    # Measured 2026-09-05 from West Africa to a Denmark RTX 4090: the pipeline
    # held its 50ms deadline comfortably at p50 38.8ms with 7ms of headroom and
    # guarded nothing across 1105 frames. The GPU was never the problem. What
    # the operator saw as sluggish was the uplink - 640x360 q70 at 20fps is
    # 3.96 Mbps, which delivered only 84% of frames, while 480x270 q60 at 15fps
    # is 1.58 Mbps and delivered 91%. Switching by hand, they described the
    # lower rate as "much smoother" immediately.
    #
    # So this preset keeps everything that decides how the output LOOKS -
    # 640x360, det_size 448, occlusion masking, aligned 256 - and pays for it
    # in frame rate and JPEG quality, which are the two axes that cost uplink
    # without costing detail:
    #
    #     640x360 q70 @20fps   3.96 Mbps    was optimal, measured not smooth
    #     640x360 q60 @15fps   2.45 Mbps    this
    #     480x270 q60 @15fps   1.58 Mbps    fast, measured smooth
    #
    # 15fps rather than 12, and that floor is deliberate. Speech runs at 4-8
    # syllables a second, so 12fps gives one to three frames per syllable and
    # lip movement stutters - on a product whose entire subject is a talking
    # face. The virtual camera also ticks at 30/s and repeats to fill, so at
    # 12fps three frames in five that reach the call are repeats. 15 is the
    # rate `fast` already runs and the one the operator judged acceptable.
    'optimal': {
        **_LOOK,
        'capture_width': 640,
        'capture_height': 360,
        'capture_fps': 15,
        'jpeg_quality': 60,
        'det_size': 448,
        'aligned_size': 256,
        'occluder': True,
        # Matches `fast`, because both run at 15fps. Smoothing across frames is
        # smoothing across time, so this factor belongs to the rate and not to
        # the quality tier.
        'alpha': 0.7,
        'temporal_alpha': 0.7,
        'buffer_size': 4,
        'warmup_frames': 5,
    },
    # What `optimal` used to be. Same picture, 20fps and q70 instead of 15fps
    # and q60 - so it costs 3.96 Mbps up, which is more than a home connection
    # in West Africa carried on the day this was measured.
    #
    # The old `production` - 960x540 at 30fps, det_size 640, aligned 320 - is
    # gone rather than renamed. docs/PERFORMANCE_AUDIT.md had it at 39ms of a
    # 33ms deadline before detection, swap, restoration or encode, so it missed
    # its frame budget on the compositor alone. It was never a usable live
    # preset, and at 85 quality and 30fps it asked ~11 Mbps of the one leg that
    # is asymmetric.
    'production': {
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
}

# Photo mode limits. Defined here so the desktop and the pipeline read the same
# numbers, the way both read PRESETS for capture settings — a client-side limit
# the server does not also enforce is not a limit.
#
# The byte ceiling is per image and applies to what arrives over the socket. It
# is deliberately generous: a photo that already fits is forwarded untouched, so
# the cap only ever bites on camera originals, and re-encoding is a last resort
# rather than routine. Photos only — video targets need a real transfer path,
# not a bigger message.
MAX_PHOTO_TARGETS = 4
MAX_PHOTO_BYTES = 6 * 1024 * 1024

# Video target limits, and the chunk the file is carried in.
#
# Video gets a transfer path of its own because a photo's cannot stretch to it:
# `upload_target` carries an image base64 in one message, and the server caps a
# message at 64 MB (`server.py`), which base64's 4/3 inflation turns into a 48 MB
# ceiling on the file. Chunking means the message size stops being the product
# limit — MAX_VIDEO_BYTES can move without the transport being rewritten.
#
# Two limits, because they refuse different things. Bytes bound the transfer;
# seconds bound the *work*, and the two do not track each other — a well
# compressed ten minutes can be smaller than a poorly compressed one, while
# costing twenty times as much to render at ~130ms a frame.
#
# Duration is checked after assembly rather than trusted from the client: the
# desktop reports what it probed, and a limit the server does not enforce is
# not a limit.
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 180

# Comfortably inside the 64 MB message cap once base64 has inflated it by 4/3,
# and large enough that a 200 MB file is ~50 messages rather than thousands.
VIDEO_CHUNK_BYTES = 4 * 1024 * 1024

# Every command the server accepts, and only those.
#
# This list is checked against `handlers.dispatch_command` by
# `tests/test_wiring.py`, in both directions. It used to be neither checked nor
# read by anything, and had drifted both ways: five entries no handler
# answered - so a client method could be written against a declaration and get
# `Unknown command` - and five working commands missing entirely.
#
# Two settings are deliberately **absent** rather than pending:
#
#   `many_faces`  swaps every face in frame, which bypasses every runtime
#                 guard and both temporal EMAs. A live switch for it works
#                 against the guards; it stays CLI/env (`--many-faces`).
#   `keep_frames` retains the extracted PNG scratch - a debugging aid, and on
#                 a pod a disk filler at roughly 4 MB per 1080p frame. CLI
#                 only (`--keep-frames`).
#
# `keep_fps` and `keep_audio` are here because they are output-format choices,
# not quality knobs: they decide what file the operator gets, not how the face
# looks.
# Restoration strength, as named steps rather than a raw 0-1.
#
# The header once had an ENHANCE *toggle*, and it was removed because off was
# never "less plastic" — it was no restoration at all, a 128-native swap
# dropped into a sharp frame. A switch across an axis that is not binary.
#
# A list fixes that rather than working around it: **`off` as the bottom of a
# scale is a different object from `off` as a switch.** It reads as one end of
# a range, which is what it actually is.
#
# `auto` is the default and means "whatever this swap model's profile asks
# for". It exists so the operator's choice and the model's profile cannot fight:
# `apply_model_profile` sets `enhance_strength` per model — 0.7 for
# inswapper_128, 0.5 for hyperswap — and without `auto` a model change would
# silently revert a choice the operator had made. That is precisely the
# `set_enhance` mistake already recorded in CLAUDE.md, where `startPipeline`
# reverted a pipeline launched with `--no-enhance`.
#
# The numbers are starting points chosen on the design target rather than
# measured: full-strength restoration is what makes a swap read as AI, so
# `full` is named honestly rather than as the best option. Retune them from
# footage; that is what a named step is for.
RESTORATION_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    'auto': None,
    'off': {'enhance': False},
    'subtle': {'enhance': True, 'enhance_strength': 0.35},
    'balanced': {'enhance': True, 'enhance_strength': 0.7},
    'full': {'enhance': True, 'enhance_strength': 1.0},
}

DEFAULT_RESTORATION_PRESET = 'auto'


COMMANDS: Dict[str, Dict[str, Any]] = {
    # Source / target / output
    'set_source':      {'path': str},
    'set_source_paths': {'paths': list},
    'upload_source':   {'images': list},
    'set_target':      {'path': str},
    'upload_target':   {'images': list},
    'upload_video_begin': {'name': str, 'size': int},
    'upload_video_chunk': {'upload_id': str, 'seq': int, 'data': str},
    'upload_video_end':   {'upload_id': str},
    'upload_video_cancel': {'upload_id': str},
    'get_render_thumbnails': {},
    'get_output_info':  {},
    'get_output_chunk': {'offset': int, 'length': int},
    'set_target_faces': {'points': list},
    'list_templates':  {},
    'set_template':    {'id': str},
    'get_photo_results': {},
    'set_output':      {'path': str},
    # Output format
    'set_keep_fps':    {'value': bool},
    'set_keep_audio':  {'value': bool},
    # Stream tuning
    'set_quality':     {'preset': str},
    'set_blend':       {'value': float},
    'set_alpha':       {'value': float},
    'set_enhance':     {'value': bool},
    'set_color_correction': {'value': bool},
    'set_preprocessing': {'value': bool},
    'set_realism':     {'values': dict},
    'set_restoration': {'preset': str},
    # Stream routing
    'set_input_url':   {'url': str},
    # Source embedding
    'create_embedding': {'paths': list},
    # Pipeline control
    'start':           {},
    'start_stream':    {},
    'stop':            {},
    'cleanup_session': {},
    'get_state':       {},
    'get_stats':       {},
    'keep_alive':      {},
    'shutdown':        {},
}

# Answered by `WebSocketAPIServer` itself, before dispatch ever runs, so it is
# not in COMMANDS and has no handler.
SERVER_COMMANDS = ('health',)

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
CMD_UPLOAD_TARGET = 'upload_target'
CMD_UPLOAD_VIDEO_BEGIN = 'upload_video_begin'
CMD_UPLOAD_VIDEO_CHUNK = 'upload_video_chunk'
CMD_UPLOAD_VIDEO_END = 'upload_video_end'
CMD_UPLOAD_VIDEO_CANCEL = 'upload_video_cancel'
CMD_GET_RENDER_THUMBNAILS = 'get_render_thumbnails'
CMD_GET_OUTPUT_INFO = 'get_output_info'
CMD_GET_OUTPUT_CHUNK = 'get_output_chunk'
CMD_SET_TARGET_FACES = 'set_target_faces'
CMD_LIST_TEMPLATES = 'list_templates'
CMD_SET_TEMPLATE = 'set_template'
CMD_GET_PHOTO_RESULTS = 'get_photo_results'
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
