import threading
from typing import List, Optional

source_path: Optional[str] = None
source_paths: List[str] = []
target_path: Optional[str] = None
output_path: Optional[str] = None
save_embedding_path: Optional[str] = None
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_audio: Optional[bool] = None
keep_frames: Optional[bool] = None
many_faces: Optional[bool] = None
video_encoder: Optional[str] = None
video_quality: Optional[int] = None
max_memory: Optional[int] = None
execution_providers: List[str] = []
execution_threads: Optional[int] = None
headless: Optional[bool] = None
log_level: str = 'error'

# Stream pipeline (Phase 1)
quality: str = 'optimal'
tracker: str = 'csrt'
buffer_size: int = 3
redetect_interval: int = 30
warmup_frames: int = 5

# Stream pipeline (Phase 2)
alpha: float = 0.6
blend: float = 0.65

# Stream pipeline (Phase 3)
luminance_blend: bool = False
enhance_interval: int = 5

# Architecture
input_url: Optional[str] = None    # network stream source (replaces local webcam)
control_port: int = 9000            # HTTP control server port

# Status
status_message: str = ''
embedding_ready: bool = False

# Lifecycle
shutdown_event: threading.Event = threading.Event()
