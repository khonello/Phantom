#!/usr/bin/env python3
"""
How much of the source's identity reaches the output, and where the rest went.

The complaint this exists to answer is "it looks good, but not quite like me",
and the reason it needed a tool is that the question was unanswerable: nothing
in this pipeline measured identity, so every stage between the swapper and the
screen was above suspicion by default. Four of them are not.

This runs one source and one target image through the **real** compositing path
— the same `FaceSwapper`, `FaceCompositor`, `Enhancer` and `FaceMasker` a live
frame goes through — and reports ArcFace cosine similarity at four points:

    id_swap      the swapper's raw crop, before restoration
    id_restore   after restoration
    id_final     after scatter, colour, detail and texture
    id_out       re-detected from the finished frame   <- the one that counts
    id_target    the finished frame against the TARGET's identity

Then it sweeps whatever you ask it to and prints the same five numbers per
configuration, so a lever is judged against one still rather than argued about.

**Read the differences, not the absolutes.** A swap is not a photograph of the
source and will not score like one. What is actionable is which step costs the
most, because each step has a different knob behind it.

Why a still rather than a clip: a still needs no stream, no warm-up and no pod
session, and identity is a per-frame property. Temporal behaviour is a separate
question and this deliberately does not touch it.

Usage:
    python tools/identity_probe.py --source me.jpg --target scene.jpg
    python tools/identity_probe.py -s me.jpg -t scene.jpg \\
        --sweep enhance_strength=0,0.35,0.7 --sweep identity_push=0,0.2,0.4
    python tools/identity_probe.py -s me.jpg -t scene.jpg \\
        --sweep swapper_model=inswapper_128,hififace_unofficial_256 \\
        --sweep mask_shape_growth=0,0.06 --json report.json
    python tools/identity_probe.py -s me.jpg -t scene.jpg --save-frames out/
"""

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FaceSwapConfig                        # noqa: E402
from pipeline.processing.compositor import FaceCompositor         # noqa: E402
from pipeline.services.database import FaceDatabase               # noqa: E402
from pipeline.services.enhancement import Enhancer                # noqa: E402
from pipeline.services.face_detection import FaceDetector         # noqa: E402
from pipeline.services.face_swapping import FaceSwapper           # noqa: E402
from pipeline.services.identity import IdentityProbe              # noqa: E402
from pipeline.services.masking import FaceMasker                  # noqa: E402

# Reported in this order, because it is the order the losses happen in.
_STAGES = ('id_swap', 'id_restore', 'id_final', 'id_out', 'id_target')

# What each step between two stages has a knob for. Printed with the attribution
# so a reading arrives with its remedy attached.
_LEVERS = (
    ('id_swap', 'id_restore', 'restoration',
     'enhance_strength / enhancer_model / restore_min_face'),
    ('id_restore', 'id_final', 'colour, detail, texture',
     'color_strength — the target\'s complexion is an identity cue'),
    ('id_final', 'id_out', 'mask and paste',
     'mask_erode / mask_feather / mask_shape_growth'),
)


def _parse_sweep(entries: List[str]) -> List[Tuple[str, List[Any]]]:
    """
    Turn `--sweep field=a,b,c` into (field, [values]).

    Values are typed from the config's own default, so `enhance_strength=0`
    becomes 0.0 and `identity_probe=5` becomes 5 without anyone saying so. An
    unknown field is refused here rather than silently ignored, which is the
    difference between a sweep that measured nothing and a sweep that said so.

    Args:
        entries: Raw `field=v1,v2` strings

    Returns:
        One (field, values) pair per entry, in the order given
    """
    defaults = FaceSwapConfig()
    parsed: List[Tuple[str, List[Any]]] = []

    for entry in entries:
        if '=' not in entry:
            raise SystemExit(f'--sweep needs field=values, got: {entry}')

        field, raw = entry.split('=', 1)
        field = field.strip()
        if not hasattr(defaults, field):
            raise SystemExit(f'unknown config field: {field}')

        current = getattr(defaults, field)
        values: List[Any] = []
        for piece in raw.split(','):
            piece = piece.strip()
            if isinstance(current, bool):
                values.append(piece.lower() in ('1', 'true', 'yes', 'on'))
            elif isinstance(current, int):
                values.append(int(piece))
            elif isinstance(current, float):
                values.append(float(piece))
            else:
                values.append(piece)

        parsed.append((field, values))

    return parsed


class Rig:
    """
    The real compositing path, held open across a sweep.

    Services are built once and the config is mutated between runs, which is
    what makes a sweep affordable: the detector, the swapper and the restorer
    each cost seconds to load and none of them needs reloading to change a
    strength. The two that *do* — a different swap or restoration model — are
    handled by the services themselves, which key their cached sessions on the
    model name.
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Args:
            config: The config every service will read, and that a sweep mutates
        """
        self.config = config
        self.detector = FaceDetector(config)
        self.database = FaceDatabase(self.detector, config)
        self.swapper = FaceSwapper(config)
        self.masker = FaceMasker(config, self.detector)
        self.compositor = FaceCompositor(config, Enhancer(config), self.masker)
        self.compositor.identity = IdentityProbe(self.detector)
        self.probe = IdentityProbe(self.detector)

    def load_source(self, paths: List[str]) -> Any:
        """
        Build the source identity exactly as the pipeline does.

        Through `review_sources` and `get_source_face` rather than a bare
        detection, so the guards, the outlier check and the pose-weighted
        average are all in the loop — a sweep that skipped them would be
        measuring a different identity from the one a real job uses.

        Args:
            paths: Source image paths

        Returns:
            The averaged source face

        Raises:
            SystemExit: If no usable source survives the guards
        """
        review = self.database.review_sources(paths)
        for path, reason in review.rejected:
            print('  refused {}: {} — {}'.format(
                os.path.basename(path), reason, review.messages.get(path, '')))

        if not review.usable:
            raise SystemExit('no usable source image')

        face = self.database.get_source_face(review.accepted)
        if face is None:
            raise SystemExit('source images produced no embedding')

        self.compositor.source_identity = getattr(
            face, 'normed_embedding', None)
        return face

    def run(self, source: Any, frame: np.ndarray) -> Tuple[
        Optional[np.ndarray], Dict[str, float]
    ]:
        """
        Swap one frame and return it with its identity readings.

        Args:
            source: The source face
            frame: The target image, BGR

        Returns:
            (output frame or None, {stage: cosine}). None means the swap was
            refused — the same fail-closed path a live frame takes
        """
        self.compositor.reset()
        self.compositor.clear_readings()

        detection = self.detector.detect_one(frame)
        if detection is None:
            return None, {}

        swapped = self.swapper.swap_aligned(source, detection.face, frame)
        if swapped is None:
            return None, {}

        crop, matrix = swapped
        output = self.compositor.composite(
            frame.copy(), detection.face, crop, matrix)

        return output, dict(self.compositor.last_identity)


def _label(settings: Dict[str, Any]) -> str:
    """One line naming a configuration, or 'baseline' when nothing was set."""
    if not settings:
        return 'baseline'
    return ' '.join('{}={}'.format(k, v) for k, v in settings.items())


def _print_row(label: str, readings: Dict[str, float], width: int) -> None:
    """Print one configuration's readings as a fixed-width row."""
    cells = []
    for stage in _STAGES:
        value = readings.get(stage)
        cells.append('   —  ' if value is None else '{:.3f}'.format(value))
    print('  {:<{}}  {}'.format(label, width, '  '.join(cells)))


def _attribute(readings: Dict[str, float]) -> List[str]:
    """
    Say which step cost the most, and what to turn.

    Only steps that actually ran are reported: a configuration with restoration
    off has no restoration loss, and printing a zero there would be a claim
    about a stage that did not happen.
    """
    lines: List[str] = []
    steps: List[Tuple[str, float, str]] = []

    for before, after, name, lever in _LEVERS:
        start, end = readings.get(before), readings.get(after)
        if start is None or end is None:
            continue
        steps.append((name, start - end, lever))

    if not steps:
        return lines

    for name, cost, _lever in steps:
        lines.append('    {:<24} {:+.3f}'.format(name, -cost))

    worst = max(steps, key=lambda item: item[1])
    if worst[1] > 0.02:
        lines.append('    -> largest loss is {}. Sweep: {}'.format(
            worst[0], worst[2]))
    else:
        lines.append('    -> the compositor costs almost nothing here. The '
                     'swapper is the ceiling: try another model, or '
                     'identity_push.')

    return lines


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Measure source-to-output identity through the real '
                    'compositing path.')
    parser.add_argument('-s', '--source', nargs='+', required=True,
                        help='source image(s) — the face to wear')
    parser.add_argument('-t', '--target', required=True,
                        help='target image — the picture to put it in')
    parser.add_argument('--sweep', action='append', default=[],
                        metavar='FIELD=V1,V2',
                        help='config field to vary; repeatable, and repeats '
                             'are combined as a full product')
    parser.add_argument('--save-frames', metavar='DIR',
                        help='write each configuration\'s output for looking '
                             'at, which is the half of this a number cannot do')
    parser.add_argument('--json', metavar='PATH', help='write the readings')
    parser.add_argument('--execution-provider', default=None,
                        help='override the execution provider (e.g. cpu)')
    args = parser.parse_args()

    frame = cv2.imread(args.target)
    if frame is None:
        raise SystemExit(f'could not read target: {args.target}')

    config = FaceSwapConfig()
    if args.execution_provider:
        config.set('execution_providers', [args.execution_provider])
    # Nothing reads these during a direct run — the compositor is called here,
    # not by the pipeline — but the probe interval gates the measurement, so it
    # has to be on for any of this to report anything at all.
    config.set('identity_probe', 1)

    print('Source: {}'.format(', '.join(
        os.path.basename(p) for p in args.source)))
    print('Target: {}\n'.format(os.path.basename(args.target)))

    rig = Rig(config)
    source = rig.load_source(list(args.source))

    sweeps = _parse_sweep(args.sweep)
    if sweeps:
        fields = [field for field, _ in sweeps]
        combinations = [
            dict(zip(fields, values))
            for values in itertools.product(*[v for _, v in sweeps])
        ]
    else:
        combinations = [{}]

    labels = [_label(settings) for settings in combinations]
    width = max(len(label) for label in labels)

    print('  {:<{}}  {}'.format('configuration', width, '  '.join(
        '{:<5}'.format(stage.replace('id_', '')) for stage in _STAGES)))
    print('  ' + '-' * (width + 2 + len(_STAGES) * 7))

    results: List[Dict[str, Any]] = []
    baseline: Optional[Dict[str, float]] = None

    for settings, label in zip(combinations, labels):
        for field, value in settings.items():
            config.set(field, value)

        # A model change reaches the profile through the same call the API
        # uses, so a sweep sees what an operator would rather than a config
        # with a new model name and the old model's restoration profile.
        if 'swapper_model' in settings:
            config.apply_model_profile(settings['swapper_model'])
            for field, value in settings.items():
                config.set(field, value)

        output, readings = rig.run(source, frame)
        _print_row(label, readings, width)

        if baseline is None:
            baseline = readings

        results.append({
            'settings': settings,
            'readings': readings,
            'swapped': output is not None,
        })

        if output is not None and args.save_frames:
            os.makedirs(args.save_frames, exist_ok=True)
            name = label.replace(' ', '_').replace('=', '-').replace('/', '-')
            cv2.imwrite(os.path.join(
                args.save_frames, '{}.png'.format(name)), output)

    if baseline:
        print('\n  where it goes, for `{}`:'.format(labels[0]))
        for line in _attribute(baseline):
            print(line)

    if len(results) > 1:
        best = max(
            (r for r in results if 'id_out' in r['readings']),
            key=lambda r: r['readings']['id_out'], default=None)
        if best is not None:
            print('\n  best id_out: {:.3f} at `{}`'.format(
                best['readings']['id_out'], _label(best['settings'])))
            print('  Now look at the frames. A higher cosine that reads as '
                  'plastic or seamed is not a better swap — this measures one '
                  'axis of three.')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump({
                'source': list(args.source),
                'target': args.target,
                'swapper_model': config.swapper_model,
                'results': results,
            }, handle, indent=2)
        print('\n  wrote {}'.format(args.json))

    return 0


if __name__ == '__main__':
    sys.exit(main())
