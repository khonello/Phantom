"""
Per-frame realism readings, reported as distributions when a stream stops or a
batch job finishes.

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
knowing before tuning its strength. It was, and it did: the measured p50 was
**0.78** of an 8-bit unit against a face carrying several, which is why the layer
was invisible at every strength.

**`shape_mismatch` / `outline_shift`** — whether the output took the source's
head shape or kept the target's. Alone among the readings here, the first is a
property of the *pairing* rather than of a setting: no knob moves it, and it is
the quantity behind "swaps read better when the two heads are similar". It is
reported beside the `id_*` cosines and is deliberately not folded into them —
ArcFace is trained to be invariant to most of the geometry it describes, so the
two can disagree completely and a good cosine does not cover head shape. See
pipeline/services/shape.py.

**`detail_reserve`** — the share of that budget `_match_detail` now holds back so
the texture layer has something to fill. This is the reading that says whether
the fix is engaged on a given clip, and it is the first thing to check when
texture looks weak: a reserve of zero while `texture_strength` is set means the
layer is declining for a reason that has nothing to do with the strength.

Cheap enough to leave on: one float appended per frame per reading, percentiles
computed once at the end. Same reasoning `LatencyBudget` records unconditionally
rather than at debug level.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    def _identity_verdicts(data: Dict[str, Any]) -> List[str]:
        """
        Say where the likeness went, when identity was measured.

        The readings are cosines and a cosine on its own tells almost nobody
        anything. What is actionable is which *step* lost the most, because each
        step has a different lever behind it — and that only exists as a
        difference between two of these numbers.

        Deliberately reported as attribution rather than as a grade. There is no
        score at which a swap is "good"; there is only a stage that cost more
        than the others, and a knob attached to it.
        """
        notes: List[str] = []

        out = data.get('id_out')
        if out is None:
            return notes

        swap = data.get('id_swap')
        restore = data.get('id_restore')
        final = data.get('id_final')
        target = data.get('id_target')

        notes.append(
            '  -> identity: output {:.3f} against the source (p50). '
            'Read the drops below, not this number.'.format(out['p50']))

        # Attribution, stage by stage, but only for stages that ran. Each is a
        # different knob, and naming the knob is the point.
        if swap is not None:
            steps = []
            if restore is not None:
                steps.append((
                    'restoration', swap['p50'] - restore['p50'],
                    'enhance_strength, or restoration_preset'))
                previous = restore['p50']
            else:
                previous = swap['p50']

            if final is not None:
                steps.append((
                    'colour/detail/texture', previous - final['p50'],
                    'color_strength — skin tone is an identity cue'))
                previous = final['p50']

            steps.append((
                'mask and paste', previous - out['p50'],
                'mask_erode, mask_feather, mask_shape_growth'))

            notes.append(
                '     the swapper produced {:.3f}; what happened after:'.format(
                    swap['p50']))
            for name, cost, lever in steps:
                notes.append(
                    '       {:<22} {:+.3f}   ({})'.format(name, -cost, lever))

            worst = max(steps, key=lambda item: item[1])
            if worst[1] > 0.02:
                notes.append(
                    '     -> {} is the largest single loss. Sweep {} before '
                    'changing the swap model.'.format(worst[0], worst[2]))
            elif swap['p50'] < 0.45:
                notes.append(
                    '     -> the compositor costs almost nothing here; the '
                    'swap itself is the ceiling. This is the case for a '
                    'different swapper, or identity_push.')

        if target is not None:
            if target['p50'] >= out['p50']:
                notes.append(
                    '     -> WARNING: the output resembles the TARGET more '
                    'than the source ({:.3f} vs {:.3f}). The swap is not '
                    'taking — check the source loaded, and that the mask is '
                    'not handing most of the face back to the frame.'.format(
                        target['p50'], out['p50']))
            else:
                notes.append(
                    '     target similarity {:.3f}, source {:.3f}. The gap is '
                    'the swap actually working.'.format(
                        target['p50'], out['p50']))

        return notes

    @staticmethod
    def _shape_verdicts(data: Dict[str, Any]) -> List[str]:
        """
        Say whether the output took the source's head shape or the target's.

        Reported apart from the identity verdicts on purpose. A cosine and a
        shape residual can disagree completely — ArcFace is trained to be
        invariant to much of the geometry this measures — and folding them into
        one paragraph would invite reading a good cosine as covering both.

        The number to act on is `outline_shift`, because the silhouette is the
        channel that carries identity and the channel the mask clips.
        """
        notes: List[str] = []

        mismatch = data.get('shape_mismatch')
        if mismatch is None:
            return notes

        shift = data.get('outline_shift') or data.get('shape_shift')
        outline = data.get('outline_mismatch') or mismatch

        # How different the two heads are in the first place. This is the
        # quantity behind "swaps read better when the shapes are close", and it
        # is a property of the pairing rather than of any setting.
        if mismatch['p50'] < 0.02:
            # And stop here. Everything below recommends a lever for recovering
            # the source's contour, which is advice for a problem this pairing
            # does not have — there is almost no contour difference to recover,
            # so a shift near zero is the correct outcome rather than a finding.
            notes.append(
                '  -> source and target head shapes are close (mismatch '
                '{:.3f}). There is little geometric conflict to resolve here, '
                'so shape is not what is costing this swap, and a shift near '
                'zero is expected rather than a fault.'.format(
                    mismatch['p50']))
            return notes
        else:
            notes.append(
                '  -> source and target head shapes differ by {:.3f} of face '
                'size ({:.3f} at the outline). This is the geometric conflict '
                'the swap has to resolve, and the larger it is the more the '
                'output leans on the target\'s silhouette.'.format(
                    mismatch['p50'], outline['p50']))

        if shift is None:
            return notes

        moved = shift['p50']
        notes.append(
            '     the output carries {:+.1%} of the source\'s head shape '
            '(0 = kept the target\'s, 1 = took the source\'s).'.format(moved))

        notes.extend(Readings._shape_attribution(data, moved))
        return notes

    @staticmethod
    def _shape_attribution(data: Dict[str, Any], survived: float) -> List[str]:
        """
        Say *where* the source's head shape was lost, not merely that it was.

        The whole reason the intermediate readings exist. A final `outline_shift`
        near zero has two causes with opposite remedies — the generator moved
        the contour and something downstream clipped it, or the generator never
        moved it at all — and they are indistinguishable from the finished frame.

        Args:
            data: The full reading summary
            survived: `outline_shift` p50 on the finished frame

        Returns:
            Attribution lines, or a fallback recommendation when the
            intermediate readings are absent
        """
        notes: List[str] = []

        made = data.get('outline_swap')
        if made is None:
            # No attribution available, so recommend on the final number alone.
            if survived < 0.05:
                notes.append(
                    '     -> the silhouette is entirely the target\'s. The '
                    'levers are mask_shape_growth and a swap model that moves '
                    'the contour at all — hififace_unofficial_256 is the only '
                    'registered one.')
            return notes

        produced = made['p50']
        notes.append(
            '     the generator produced {:+.3f} and {:+.3f} survived:'.format(
                produced, survived))

        # Each step, with the lever attached to it. Same shape as the identity
        # attribution above, and for the same reason: a loss nobody can act on
        # is a number rather than a finding.
        steps: List[Tuple[str, float, str]] = []
        final = data.get('outline_final')
        if final is not None:
            steps.append((
                'restoration etc', produced - final['p50'],
                'enhance_strength — a restorer regresses a face toward its '
                'training manifold, which is a plausible way to lose a jaw'))
            steps.append((
                'mask and paste', final['p50'] - survived, 'mask_shape_growth'))
        else:
            steps.append((
                'everything after', produced - survived,
                'mask_shape_growth first, then enhance_strength'))

        for name, cost, _lever in steps:
            notes.append('       {:<22} {:+.3f}'.format(name, -cost))

        # The finding, and it is the one that decides which lever to reach for.
        if produced < 0.05:
            notes.append(
                '     -> the GENERATOR never moved the contour, so there is '
                'nothing for the mask to clip and mask_shape_growth is not '
                'your lever. Expected for inswapper and alphaface. Changing '
                'the swap model is the only thing that moves this.')
            return notes

        if survived >= produced * 0.5:
            notes.append(
                '     -> most of what the generator produced is reaching the '
                'output, so the ceiling is the model rather than anything '
                'downstream of it.')
            return notes

        # Something took it back. Name the stage that took the most, rather
        # than assuming the mask — restoration can eat a contour too, and
        # recommending the wrong knob is worse than recommending none.
        worst = max(steps, key=lambda item: item[1])
        notes.append(
            '     -> the generator moved the contour and {:.0f}% of it was '
            'taken back downstream, most of it by {}. Sweep {}.'.format(
                (1.0 - survived / produced) * 100.0, worst[0], worst[2]))

        return notes

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

        # The budget against the spend. Read together or neither means much:
        # headroom alone says the layer had room, and only this pair says
        # whether it used it.
        budget = data.get('texture_headroom')
        spend = data.get('texture_delivered')
        if budget is not None and spend is not None and budget['p50'] > 1e-6:
            share = spend['p50'] / budget['p50']
            if share >= 0.6:
                notes.append(
                    '  -> texture delivered {:.2f} of a {:.2f} budget ({:.0%}), '
                    'so the reservation is being filled.'.format(
                        spend['p50'], budget['p50'], share))
            else:
                notes.append(
                    '  -> texture delivered {:.2f} of a {:.2f} budget ({:.0%}), '
                    'so the face is SOFTER than with texture off — detail '
                    'matching stood down by {:.2f} and this did not fill it. '
                    'Do not raise texture_strength; it scales both sides.'
                    .format(spend['p50'], budget['p50'], share,
                            (data.get('detail_reserve') or {}).get('p50', 0.0)))

                # Which of the two causes it is, and they want opposite fixes.
                # A field that is unit-deviation over a fraction `c` of the
                # region it is measured across reads `sqrt(c)`, so coverage
                # predicts the shortfall exactly when area is the whole story.
                cover = data.get('texture_coverage')
                if cover is not None and cover['p50'] > 1e-6:
                    predicted = math.sqrt(cover['p50'])
                    if abs(predicted - share) < 0.08:
                        notes.append(
                            '     the map covers {:.0%} of the mask, which '
                            'predicts a {:.0%} spend on area alone — so it is '
                            'arriving at full amplitude over less of the face '
                            'than the reserve assumed. The reserve is uniform '
                            'and the fill is skin-only; that mismatch is the '
                            'defect, not the map.'.format(
                                cover['p50'], predicted))
                    else:
                        notes.append(
                            '     the map covers {:.0%} of the mask, which '
                            'would predict {:.0%} on area alone against the '
                            '{:.0%} measured — so amplitude is being lost too. '
                            'Look at the warp and the normalisation.'.format(
                                cover['p50'], predicted, share))

        reserve = data.get('detail_reserve')
        if reserve is not None:
            if reserve['p95'] <= 1e-6:
                notes.append(
                    '  -> detail matching reserved nothing for texture on any '
                    'frame. Either the layer is off, or it is declining — no '
                    'source photograph, or every frame too far from its pose. '
                    'Read texture_confidence before touching a strength.')
            else:
                notes.append(
                    '  -> detail matching held back {:.0%} of the target band '
                    'for real skin detail (p50). Without that reservation this '
                    'stage reaches parity on its own and leaves the texture '
                    'layer a rounding error to spend.'.format(reserve['p50']))

        kept = data.get('complexion_kept')
        if kept is not None:
            if kept['p95'] <= 1e-6:
                notes.append(
                    '  -> complexion_keep had nothing to spend: the swap and '
                    'the target were already the same skin tone on every '
                    'frame. The knob is not weak, the two faces simply match.')
            elif kept.get('share_at_limit', 0.0) > 0.5:
                notes.append(
                    '  -> complexion_keep is capped on {:.0f}% of frames at '
                    '{:.1f} LAB units. The two complexions are far enough '
                    'apart that the bound, not the knob, is deciding — raising '
                    'it will not keep more skin tone, it will only start '
                    'showing a colour step at the jaw.'.format(
                        kept['share_at_limit'] * 100.0, kept['limit']))
            else:
                notes.append(
                    '  -> complexion_keep left {:.1f} LAB units of the '
                    'source\'s own skin tone on the face (p50), against a '
                    '{:.1f} cap. Judge it on footage: ArcFace barely sees skin '
                    'tone, so id_out will not move much either way.'.format(
                        kept['p50'], kept['limit']))

        notes.extend(Readings._identity_verdicts(data))
        notes.extend(Readings._shape_verdicts(data))

        headroom = data.get('texture_headroom')
        if headroom is not None:
            if headroom['p95'] < 0.25:
                notes.append(
                    '  -> texture had no headroom on this clip. The swap is '
                    'already at the real face\'s texture level, so '
                    'texture_strength has nothing to spend. If detail_reserve '
                    'is also zero, that is the cause and not a property of the '
                    'footage.')
            else:
                notes.append(
                    '  -> texture headroom p50 {:.2f} units of deviation. That '
                    'is what the layer may add at strength 1.0, and it lands at '
                    'parity with the real face rather than past it. Judge it '
                    'against the band the face carries, not against zero: a '
                    'headroom under ~1 unit is invisible whatever the strength '
                    'says.'.format(headroom['p50']))

        return notes
