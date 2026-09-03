"""
Per-frame realism readings, reported as distributions when a stream stops.

`LatencyBudget` answers "does this preset hold". This answers the questions the
realism work keeps needing and keeps not having: whether a limit is binding,
whether a stage had anything to do. Those are questions about a *distribution*
over a run, not about one frame, and they cannot be answered by a log line
sampled one frame in thirty.

Two are recorded today and both have a decision waiting on them:

**`detail_ratio`** — the correction `_match_detail` wanted before its clamp. The
stage scales the swap's high-frequency band toward the target's and bounds the
result to `_DETAIL_RATIO`, and on the one clip measured the face still came out
at 0.584 of the frame's detail *after* it ran. Either the clamp is binding, in
which case part of that gap is simply the stage not being allowed to correct far
enough and raising a constant is the cheapest lever this project has; or it is
not, in which case the band genuinely holds nothing to amplify and the case for
extracting real detail is made outright. One run of a stream decides which, and
the number to read is `share_at_limit`.

**`texture_headroom`** — how much high-frequency deviation the texture layer was
allowed. Routinely zero means detail matching has already taken the face to the
target's texture level and the layer has nothing to add, which would be worth
knowing before tuning its strength.

Cheap enough to leave on: one float appended per frame per reading, percentiles
computed once at the end. Same reasoning `LatencyBudget` records unconditionally
rather than at debug level.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Readings:
    """
    Named scalar readings accumulated over a stream.

    Example:
        readings = Readings()
        readings.record('detail_ratio', 1.6, limit=1.6)
        print(readings.format_report())
    """

    # Same ceiling `LatencyBudget` uses, and for the same reason: an unbounded
    # list over a long session is a memory leak wearing a diagnostic's clothes.
    limit: int = 50000

    values: Dict[str, List[float]] = field(default_factory=dict)
    limits: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, value: float, limit: Optional[float] = None) -> None:
        """
        Add one reading.

        Args:
            name: What was measured
            value: The reading
            limit: The bound this reading is pressing against, if any. Recorded
                so the report can say what share of frames reached it — which is
                the whole question for a clamped quantity, and is not visible in
                percentiles of the clamped value because those cannot exceed it.
        """
        series = self.values.setdefault(name, [])
        if len(series) < self.limit:
            series.append(float(value))
        if limit is not None:
            self.limits[name] = float(limit)

    def reset(self) -> None:
        """Drop everything. Called when a stream starts, never mid-run."""
        self.values.clear()
        self.limits.clear()

    def report(self) -> Dict[str, Any]:
        """
        Percentiles per reading, plus the share sitting at any recorded limit.

        Returns:
            {name: {n, p50, p95, max, mean, share_at_limit?}}
        """
        summary: Dict[str, Any] = {}

        for name, series in self.values.items():
            if not series:
                continue
            array = np.asarray(series, dtype=np.float64)
            entry: Dict[str, Any] = {
                'n': int(array.size),
                'p50': round(float(np.percentile(array, 50)), 4),
                'p95': round(float(np.percentile(array, 95)), 4),
                'max': round(float(array.max()), 4),
                'mean': round(float(array.mean()), 4),
            }

            bound = self.limits.get(name)
            if bound is not None:
                # Within a whisker of the bound rather than equal to it: the
                # readings are floats that have been through a clip, and asking
                # for exact equality would report zero on a quantity that is
                # pinned every frame.
                at_limit = int(np.count_nonzero(array >= bound - 1e-6))
                entry['limit'] = round(bound, 4)
                entry['share_at_limit'] = round(at_limit / array.size, 4)

            summary[name] = entry

        return summary

    def format_report(self) -> str:
        """
        The report as text, for the log.

        Returns:
            A multi-line summary, or a single line when nothing was recorded.
        """
        data = self.report()
        if not data:
            return 'Realism readings: none recorded'

        lines = ['Realism readings']
        for name, entry in data.items():
            line = '  {:<18} p50={:>7.3f}  p95={:>7.3f}  max={:>7.3f}  n={}'.format(
                name, entry['p50'], entry['p95'], entry['max'], entry['n'],
            )
            if 'share_at_limit' in entry:
                line += '  at limit {:.0f}% of frames (limit {:.2f})'.format(
                    entry['share_at_limit'] * 100.0, entry['limit'],
                )
            lines.append(line)

        lines.extend(self._verdicts(data))
        return '\n'.join(lines)

    @staticmethod
    def _verdicts(data: Dict[str, Any]) -> List[str]:
        """
        Say what the numbers mean, for the two that have a decision waiting.

        A percentile nobody can interpret is a number, not a finding. These are
        deliberately phrased as what to do next rather than as a grade.
        """
        notes: List[str] = []

        detail = data.get('detail_ratio')
        if detail is not None and 'share_at_limit' in detail:
            share = detail['share_at_limit']
            if share > 0.5:
                notes.append(
                    '  -> detail matching is CLAMPED on {:.0f}% of frames '
                    '(wanted p95 {:.2f}). Part of the face/frame detail gap is '
                    'the clamp, not the swap: raise _DETAIL_RATIO and '
                    're-measure before building anything else.'.format(
                        share * 100.0, detail['p95']))
            elif share > 0.05:
                notes.append(
                    '  -> detail matching reaches its clamp on {:.0f}% of '
                    'frames. Worth a sweep, but it is not the main term.'.format(
                        share * 100.0))
            else:
                notes.append(
                    '  -> detail matching is not clamp-bound ({:.0f}% of '
                    'frames). The high band holds nothing more to amplify, '
                    'which is the case for adding real detail.'.format(
                        share * 100.0))

        headroom = data.get('texture_headroom')
        if headroom is not None:
            if headroom['p95'] < 0.25:
                notes.append(
                    '  -> texture had no headroom on this clip. The swap is '
                    'already at the real face\'s texture level, so '
                    'texture_strength has nothing to spend.')
            else:
                notes.append(
                    '  -> texture headroom p50 {:.2f} units, so '
                    'texture_strength=1.0 would add that much and land at '
                    'parity.'.format(headroom['p50']))

        return notes
