"""
Per-frame latency budgeting for the Phantom pipeline.

Stage 1 of the implementation plan asks whether a single session holds its
latency budget at each preset, and notes that one session missing frame
deadlines is a quality problem regardless of how many others exist.

The per-stage timings already existed behind `--log-level debug`, but as
individual lines every thirtieth frame. That answers "how long did frame 240
take"; it does not answer "does this preset hold", which needs a distribution
against the deadline the preset itself sets.

The deadline is not a target to average against. Frames arrive on a clock: at
20fps a frame is due every 50ms, and one that takes 70ms does not borrow time
back from a fast neighbour — it pushes every later frame along behind it. So the
number that matters is the fraction over deadline and the p95, not the mean.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.config import FaceSwapConfig

# Fraction of frames allowed over deadline before a preset is judged not to
# hold. Some overshoot is normal and invisible — a dropped frame here and there
# reads as bandwidth. Sustained overshoot is what desynchronises from audio.
_TOLERANCE_PCT = 5.0


@dataclass
class LatencyBudget:
    """
    Records per-stage frame timings and judges them against the preset deadline.

    Cheap enough to leave on: three floats appended per frame, and the
    percentiles are computed once at the end.
    """

    limit: int = 50000

    stages: Dict[str, List[float]] = field(default_factory=dict)
    frames: int = 0

    def record(self, detect_ms: float, swap_ms: float, total_ms: float) -> None:
        """
        Record one frame's stage timings, in milliseconds.

        Args:
            detect_ms: Preprocessing and detection
            swap_ms: Swap and compositing
            total_ms: Whole frame, capture to emit
        """
        self.frames += 1
        for name, value in (
            ('detect', detect_ms),
            ('swap+composite', swap_ms),
            ('total', total_ms),
        ):
            bucket = self.stages.setdefault(name, [])
            if len(bucket) < self.limit:
                bucket.append(value)

    @staticmethod
    def deadline_ms(config: FaceSwapConfig) -> float:
        """
        Milliseconds available per frame at the configured capture rate.

        Args:
            config: Supplies `capture_fps`

        Returns:
            Frame period in milliseconds
        """
        fps = float(getattr(config, 'capture_fps', 0) or 0)
        return (1000.0 / fps) if fps > 0 else 0.0

    def report(self, config: FaceSwapConfig) -> Dict[str, Any]:
        """
        Summarise the session against this preset's deadline.

        Args:
            config: Supplies the preset name and capture rate

        Returns:
            A JSON-serialisable report, including a pass/fail verdict
        """
        deadline = self.deadline_ms(config)
        stages: Dict[str, Any] = {}
        over_pct: Optional[float] = None

        for name, values in self.stages.items():
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            entry = {
                'count': int(array.size),
                'p50': round(float(np.percentile(array, 50)), 2),
                'p95': round(float(np.percentile(array, 95)), 2),
                'p99': round(float(np.percentile(array, 99)), 2),
                'max': round(float(array.max()), 2),
            }
            if name == 'total' and deadline > 0:
                over = float((array > deadline).mean() * 100.0)
                entry['over_deadline_pct'] = round(over, 2)
                entry['headroom_ms'] = round(deadline - entry['p95'], 2)
                over_pct = over
            stages[name] = entry

        holds = deadline > 0 and over_pct is not None and over_pct <= _TOLERANCE_PCT

        return {
            'preset': getattr(config, 'quality', 'unknown'),
            'capture_fps': getattr(config, 'capture_fps', 0),
            'deadline_ms': round(deadline, 2),
            'tolerance_pct': _TOLERANCE_PCT,
            'frames': self.frames,
            'stages': stages,
            'holds': holds,
        }

    def format_report(self, config: FaceSwapConfig) -> str:
        """
        The report as text, for the log.

        Args:
            config: Thresholds to judge against

        Returns:
            A multi-line summary
        """
        data = self.report(config)
        if not data['frames']:
            return 'Latency budget: no frames recorded'

        verdict = 'HOLDS' if data['holds'] else 'MISSES'
        lines = [
            'Latency budget [{}] — preset {} at {}fps, deadline {}ms, '
            '{} frames'.format(
                verdict, data['preset'], data['capture_fps'],
                data['deadline_ms'], data['frames'],
            ),
        ]

        for name, entry in data['stages'].items():
            line = '  {:<16} p50={:>7.1f}ms  p95={:>7.1f}ms  p99={:>7.1f}ms'.format(
                name, entry['p50'], entry['p95'], entry['p99'],
            )
            if 'over_deadline_pct' in entry:
                line += '  -> {}% over deadline, {}ms headroom at p95'.format(
                    entry['over_deadline_pct'], entry['headroom_ms'],
                )
            lines.append(line)

        if not data['holds']:
            lines.append(
                '  A frame over deadline does not borrow time back from a fast '
                'neighbour — it pushes every later frame along behind it.'
            )

        return '\n'.join(lines)
