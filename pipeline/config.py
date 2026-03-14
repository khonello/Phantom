"""
Configuration management for the Phantom face-swapping pipeline.

Replaces pipeline.globals with a typed, observable configuration object.
Supports change notifications via callbacks for reactive updates.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict


@dataclass
class FaceSwapConfig:
    """
    Centralized runtime configuration for the Phantom pipeline.

    All fields are typed and have sensible defaults. Change listeners can be
    registered via on_change() to react to configuration updates.
    """

    # Input/output paths
    source_path: Optional[str] = None
    source_paths: List[str] = field(default_factory=list)
    target_path: Optional[str] = None
    output_path: Optional[str] = None
    save_embedding_path: Optional[str] = None

    # Processing pipeline
    frame_processors: List[str] = field(default_factory=lambda: ['face_swapper'])
    keep_fps: bool = False
    keep_audio: bool = True
    keep_frames: bool = False
    many_faces: bool = False
    video_encoder: str = 'libx264'
    video_quality: int = 18
    max_memory: int = 16
    execution_providers: List[str] = field(default_factory=lambda: ['CPUExecutionProvider'])
    execution_threads: int = 8

    # Stream pipeline - quality presets
    quality: str = 'optimal'
    tracker: str = 'csrt'
    alpha: float = 0.6
    blend: float = 0.65
    luminance_blend: bool = True
    enhance_interval: int = 5
    buffer_size: int = 4
    redetect_interval: int = 30
    warmup_frames: int = 5

    # I/O configuration
    input_url: Optional[str] = None
    stream_url: Optional[str] = None
    preview_url: Optional[str] = None
    virtual_cam: bool = False
    control_port: int = 9000

    # Logging & status
    log_level: str = 'error'
    status_message: str = ''
    embedding_ready: bool = False
    headless: bool = False
    stream_mode: bool = False

    # Internal state
    shutdown_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    _listeners: List[Callable[[str, Any], None]] = field(default_factory=list, repr=False, compare=False)

    def on_change(self, cb: Callable[[str, Any], None]) -> None:
        """
        Register a callback to be invoked when configuration changes.

        Callback signature: cb(field_name: str, new_value: Any)
        """
        self._listeners.append(cb)

    def off_change(self, cb: Callable[[str, Any], None]) -> None:
        """Unregister a previously registered change listener."""
        if cb in self._listeners:
            self._listeners.remove(cb)

    def set(self, field: str, value: Any) -> None:
        """
        Set a configuration field and notify all listeners.

        Args:
            field: Name of the configuration field (must exist)
            value: New value for the field

        Raises:
            AttributeError: If field doesn't exist
            TypeError: If value type is incompatible
        """
        if not hasattr(self, field):
            raise AttributeError(f"FaceSwapConfig has no field '{field}'")

        setattr(self, field, value)

        # Notify all listeners
        for cb in self._listeners:
            try:
                cb(field, value)
            except Exception as e:
                # Log but don't crash if callback fails
                import sys
                print(f"Warning: config change listener failed: {e}", file=sys.stderr)

    def apply_preset(self, preset_name: str) -> None:
        """
        Apply a named quality preset to the configuration.

        Available presets: 'fast', 'optimal', 'production'

        Args:
            preset_name: Name of the preset to apply

        Raises:
            ValueError: If preset name is not recognized
        """
        from pipeline.api.schema import PRESETS

        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")

        for key, value in PRESETS[preset_name].items():
            self.set(key, value)

    def get_preset_config(self) -> Dict[str, Any]:
        """
        Export current configuration as a dictionary.
        Useful for serialization or logging.
        """
        return {
            'source_path': self.source_path,
            'source_paths': self.source_paths,
            'target_path': self.target_path,
            'output_path': self.output_path,
            'frame_processors': self.frame_processors,
            'keep_fps': self.keep_fps,
            'keep_audio': self.keep_audio,
            'keep_frames': self.keep_frames,
            'many_faces': self.many_faces,
            'video_encoder': self.video_encoder,
            'video_quality': self.video_quality,
            'max_memory': self.max_memory,
            'execution_providers': self.execution_providers,
            'execution_threads': self.execution_threads,
            'quality': self.quality,
            'tracker': self.tracker,
            'alpha': self.alpha,
            'blend': self.blend,
            'luminance_blend': self.luminance_blend,
            'enhance_interval': self.enhance_interval,
            'buffer_size': self.buffer_size,
            'redetect_interval': self.redetect_interval,
            'warmup_frames': self.warmup_frames,
            'input_url': self.input_url,
            'control_port': self.control_port,
            'log_level': self.log_level,
            'headless': self.headless,
        }


# Global configuration singleton
CONFIG: FaceSwapConfig = FaceSwapConfig()
