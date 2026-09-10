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

and two **shape** readings the cosines cannot give, because ArcFace is trained
to be invariant to most of the geometry they measure:

    shape        how far the output's head shape moved off the target's toward
                 the source's. 0 kept the target's, 1 took the source's
    outln        the same, measured only at the silhouette — the channel a
                 viewer reads first and the one the mask clips

Then it sweeps whatever you ask it to and prints the same numbers per
configuration, so a lever is judged against one still rather than argued about.

**Read the differences, not the absolutes.** A swap is not a photograph of the
source and will not score like one. What is actionable is which step costs the
most, because each step has a different knob behind it.

**The two axes disagree, and that is the reason to have both.** A model can
score well on identity while leaving the silhouette entirely the target's —
which is what `inswapper` and `alphaface` do by construction, and what
`mask_shape_growth` exists to recover.

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
import types
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FaceSwapConfig                        # noqa: E402
from pipeline.core import (                                        # noqa: E402
    decode_execution_providers,
    suggest_default_execution_providers,
)
from pipeline.processing import texture                           # noqa: E402
from pipeline.processing.compositor import FaceCompositor         # noqa: E402
from pipeline.services.database import (                          # noqa: E402
    FaceDatabase,
    off_axis,
)
from pipeline.services.enhancement import Enhancer                # noqa: E402
from pipeline.services.face_detection import FaceDetector         # noqa: E402
from pipeline.services.face_swapping import FaceSwapper           # noqa: E402
from pipeline.services.identity import IdentityProbe              # noqa: E402
from pipeline.services.masking import FaceMasker                  # noqa: E402
from pipeline.services.shape import ShapeProbe                    # noqa: E402

# Reported in this order, because it is the order the losses happen in.
_STAGES = ('id_swap', 'id_restore', 'id_final', 'id_out', 'id_target')

# Shape, reported beside identity because the two measure different axes and
# routinely disagree — ArcFace is trained to be invariant to much of the
# geometry the shape metric exists to see. `shape_mismatch` is deliberately not
# a column: it is a property of the source/target pairing rather than of any
# configuration, so it cannot vary down a sweep and is printed once above the
# table instead.
_SHAPE = ('shape_shift', 'interior_shift', 'outline_shift')

# Whether the texture layer had anything to spend, and whether pose let it spend
# it. Reported in their own block rather than as more columns, and only when the
# layer actually ran — a `texture_strength` sweep that comes back flat is
# ambiguous without them (too weak, or declining for a reason unrelated to
# strength), which is the exact question `detail_reserve` was added to answer.
_TEXTURE = ('texture_headroom', 'texture_delivered', 'texture_coverage',
            'detail_reserve', 'texture_confidence', 'detail_ratio')

# What each step between two stages has a knob for. Printed with the attribution
# so a reading arrives with its remedy attached.
_LEVERS = (
    ('id_swap', 'id_restore', 'restoration',
     'enhance_strength / enhancer_model / restore_min_face'),
    ('id_restore', 'id_final', 'colour, detail, texture',
     'complexion_keep first, then color_strength — the colour match moves the '
     'face onto the target\'s skin tone, and tone is an identity cue'),
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
        self.compositor.shape = ShapeProbe(self.detector)
        self.probe = IdentityProbe(self.detector)
        self._announced = False

    def load_source(self, paths: List[str], holdout: bool = False) -> Any:
        """
        Build the source identity exactly as the pipeline does.

        Through `review_sources` and `get_source_face` rather than a bare
        detection, so the guards, the outlier check and the blending strategy
        are all in the loop — a sweep that skipped them would be measuring a
        different identity from the one a real job uses.

        **`holdout` is what makes `source_blend` measurable at all.** The
        readings score the output against `compositor.source_identity`, and if
        that is the identity the strategy under test just built, the strategy
        is grading its own homework: every one of them wins, because every one
        of them is closest to itself. Holding one photograph out and scoring
        against *that* gives an independent yardstick — the same leave-one-out
        structure `_review_identity` already uses to spot an intruder.

        Args:
            paths: Source image paths
            holdout: Build the identity from all but the last photograph, and
                score against the one left out. Needs at least two

        Returns:
            The source face the swapper will be conditioned on

        Raises:
            SystemExit: If no usable source survives the guards
        """
        # A sweep over `source_blend` calls this once per configuration, and
        # the refusals and the holdout line are properties of the *photographs*
        # rather than of the configuration — so they are said once.
        announce = not self._announced
        self._announced = True

        review = self.database.review_sources(paths)
        for path, reason in review.rejected:
            if announce:
                print('  refused {}: {} — {}'.format(
                    os.path.basename(path), reason,
                    review.messages.get(path, '')))

        if not review.usable:
            raise SystemExit('no usable source image')

        accepted = list(review.accepted)
        reference = None

        if holdout:
            if len(accepted) < 2:
                raise SystemExit(
                    '--holdout needs at least two accepted source images')
            held = accepted.pop()
            reference = self.database.get_source_face([held])
            if reference is None:
                raise SystemExit('the held-out image produced no embedding')
            if announce:
                print('  holding out {} as the yardstick; building the '
                      'identity from the other {}\n'.format(
                          os.path.basename(held), len(accepted)))

        face = self.database.get_source_face(accepted)
        if face is None:
            raise SystemExit('source images produced no embedding')

        self.compositor.source_identity = getattr(
            reference if reference is not None else face,
            'normed_embedding', None)

        # Shape comes from one photograph, never the average � the same pick
        # the texture layer uses, and for the same reason: landmarks taken at
        # different angles average into a face nobody has. Unaffected by
        # `--holdout`, which is about which identity vector grades the output
        # and has nothing to say about geometry.
        # Skin texture, from the *texture* pick. Without this the whole layer is
        # inert here — `_add_texture` needs a `source_texture` and the pipeline
        # sets one in `SwappingProcessor._load_texture`, which this rig does not
        # go through. A `texture_strength` sweep would then return identical
        # rows and read as "the layer does nothing", which is worse than no
        # measurement: it is a confident wrong answer about a layer that was
        # never switched on.
        donor = self.database.select_texture_source(accepted)
        self.compositor.source_texture = (
            None if donor is None else texture.extract(donor[0], donor[1]))

        if announce and self.compositor.source_texture is not None:
            pores, marks = self.compositor.source_texture.octaves
            print('  texture source: {} ({}px face, pores {:.2f}, '
                  'marks {:.2f}){}'.format(
                      os.path.basename(donor[0]),
                      self.compositor.source_texture.native_px, pores, marks,
                      ' - upsampled, so the band is thinner than it looks'
                      if self.compositor.source_texture.upsampled else ''))
        elif announce and donor is not None:
            print('  texture source: extraction failed; texture_strength will '
                  'do nothing')

        best = self.database.select_shape_source(accepted)
        self.compositor.source_shape = (
            None if best is None
            else getattr(best[1], 'landmark_2d_106', None))

        if announce:
            if self.compositor.source_shape is None:
                print('  no 106-point landmarks on the source; the shape '
                      'readings will be absent\n')
            elif len(accepted) > 1:
                # Which photograph became the shape reference is not a detail
                # when several were given: `shape_mismatch` is inflated by
                # out-of-plane pose, so a reader seeing an implausibly large
                # one needs to know how frontal the reference was. The identity
                # is still the average of all of them, and the texture donor is
                # still chosen separately on sharpness.
                angle = off_axis(types.SimpleNamespace(
                    face=best[1], kps=getattr(best[1], 'kps', None)))
                off = ('' if angle is None
                       else '  ({:.0f} degrees off axis)'.format(angle))
                print('  shape reference: {} of {} accepted{}\n'.format(
                    os.path.basename(best[0]), len(accepted), off))

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

        readings = dict(self.compositor.last_identity)
        readings.update(self.compositor.last_shape)

        for name, value in (
                ('texture_headroom', self.compositor.last_texture_headroom),
                ('texture_delivered', self.compositor.last_texture_delivered),
                ('texture_coverage', self.compositor.last_texture_coverage),
                ('texture_confidence', self.compositor.last_texture_confidence),
                ('detail_reserve', self.compositor.last_detail_reserve),
                ('detail_ratio', self.compositor.last_detail_ratio)):
            if value is not None:
                readings[name] = float(value)

        return output, readings


def _label(settings: Dict[str, Any]) -> str:
    """One line naming a configuration, or 'baseline' when nothing was set."""
    if not settings:
        return 'baseline'
    return ' '.join('{}={}'.format(k, v) for k, v in settings.items())


def _print_row(label: str, readings: Dict[str, float], width: int) -> None:
    """Print one configuration's readings as a fixed-width row."""
    cells = []
    for stage in _STAGES + _SHAPE:
        value = readings.get(stage)
        if value is None:
            cells.append('   —  ')
        elif stage in _SHAPE:
            # Signed, because a negative shift is a real and different finding:
            # the output moved *away* from the source's shape.
            cells.append('{:+.3f}'.format(value))
        else:
            cells.append('{:.3f}'.format(value))
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


def _shape_note(readings: Dict[str, float]) -> List[str]:
    """
    Say how far apart the two head shapes are, and how much of that was closed.

    `shape_mismatch` is the quantity behind the observation that swaps read
    better when the source and target heads are similar. It is a property of the
    *pairing* — nothing in the config moves it — so it is stated as context for
    the sweep rather than as a result of one.
    """
    mismatch = readings.get('shape_mismatch')
    if mismatch is None:
        return ['    (no shape reading — the pack has no 106-point landmark '
                'model, or the source carried none)']

    outline = readings.get('outline_mismatch', mismatch)
    lines = [
        '',
        '    head-shape mismatch  {:.3f} of face size, {:.3f} at the outline'
        .format(mismatch, outline),
    ]

    if mismatch < 0.02:
        # And say nothing further. What follows recommends levers for
        # recovering the source's contour, which is advice for a problem this
        # pairing does not have.
        lines.append('    -> these two heads are geometrically close. Shape is '
                     'not what is costing this pairing, and a shift near zero '
                     'is expected rather than a fault.')
        return lines

    shift = readings.get('outline_shift')
    if shift is None:
        return lines

    lines.append(
        '    outline shift        {:+.3f}   (0 = kept the target\'s head '
        'shape, 1 = took the source\'s)'.format(shift))

    produced = readings.get('outline_swap')
    if produced is None:
        if shift < 0.05:
            lines.append('    -> the silhouette is entirely the target\'s. '
                         'Sweep mask_shape_growth, and try '
                         'hififace_unofficial_256 — the only registered model '
                         'that moves the contour.')
        return lines

    # Where it went. The two causes of a low shift have opposite remedies and
    # are indistinguishable from the finished frame alone.
    lines.append('    the generator produced {:+.3f}, and {:+.3f} survived'
                 .format(produced, shift))

    steps: List[Tuple[str, float, str]] = []
    final = readings.get('outline_final')
    if final is not None:
        steps.append(('restoration etc', produced - final, 'enhance_strength'))
        steps.append(('mask and paste', final - shift, 'mask_shape_growth'))
    else:
        steps.append(('everything after', produced - shift,
                      'mask_shape_growth, then enhance_strength'))

    for name, cost, _lever in steps:
        lines.append('      {:<20} {:+.3f}'.format(name, -cost))

    if produced < 0.05:
        lines.append('    -> the GENERATOR never moved the contour. Nothing '
                     'for the mask to clip, so mask_shape_growth is not the '
                     'lever here — only a different swap model is.')
    elif shift >= produced * 0.5:
        lines.append('    -> most of what the generator produced is reaching '
                     'the output; the model is the ceiling, not anything '
                     'downstream of it.')
    else:
        # Name the stage that took the most rather than assuming the mask.
        # Restoration can eat a contour too, and pointing at the wrong knob is
        # worse than pointing at none.
        worst = max(steps, key=lambda item: item[1])
        lines.append('    -> generated and then taken back ({:.0f}% lost), '
                     'most of it by {}. Sweep {}.'.format(
                         (1.0 - shift / produced) * 100.0, worst[0], worst[2]))

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
    parser.add_argument('--holdout', action='store_true',
                        help='build the identity from all but the last source '
                             'image and score against the one left out. '
                             'Required for judging --sweep source_blend, which '
                             'otherwise grades its own homework')
    parser.add_argument('--json', metavar='PATH', help='write the readings')
    parser.add_argument('--execution-provider', default=None,
                        help='override the execution provider (e.g. cpu)')
    args = parser.parse_args()

    frame = cv2.imread(args.target)
    if frame is None:
        raise SystemExit(f'could not read target: {args.target}')

    config = FaceSwapConfig()

    # `FaceSwapConfig` defaults to CPU, and `core.py` is what normally replaces
    # that with the best available provider. This tool never went through
    # `core.py`, so it ran every model on CPU — on a rented GPU, silently, which
    # is the exact failure `execution.verify` exists to halt and which this
    # bypasses by building its services directly. The readings themselves are
    # provider-independent (fp32 either way), so nothing measured before this
    # was wrong; it was just paying GPU rates for CPU inference.
    #
    # Resolved through `core.py`'s own functions rather than a second copy of
    # the logic, for the reason `onnx_session.py` gives about four levers and
    # three call sites: a duplicate would drift.
    requested = ([args.execution_provider] if args.execution_provider
                 else suggest_default_execution_providers())
    resolved = decode_execution_providers(requested)
    config.set('execution_providers', resolved)
    # Nothing reads these during a direct run — the compositor is called here,
    # not by the pipeline — but the probe interval gates the measurement, so it
    # has to be on for any of this to report anything at all.
    config.set('identity_probe', 1)

    print('Source: {}'.format(', '.join(
        os.path.basename(p) for p in args.source)))
    print('Target: {}'.format(os.path.basename(args.target)))
    # Said rather than assumed. A run that quietly used CPU is not wrong — the
    # readings are provider-independent — but it is slow enough to matter on a
    # paid pod, and the reader should not have to infer it from onnxruntime's
    # own debug lines.
    print('Providers: {}\n'.format(', '.join(resolved)))

    rig = Rig(config)
    source = rig.load_source(list(args.source), holdout=args.holdout)

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

    columns = [stage.replace('id_', '') for stage in _STAGES]
    # `gen` is what the swap model produced, `kept` is what survived the mask
    # and the paste. Reading them left to right is the attribution.
    columns += ['shape', 'inner', 'outln']
    print('  {:<{}}  {}'.format('configuration', width, '  '.join(
        '{:<5}'.format(name[:5]) for name in columns)))
    print('  ' + '-' * (width + 2 + len(columns) * 7))

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

        # The identity is built when photographs are accepted, not per frame,
        # so a sweep over how they are combined has to rebuild it. Cheap: the
        # detections are cached by path, and only the averaging is redone.
        if 'source_blend' in settings:
            source = rig.load_source(list(args.source), holdout=args.holdout)

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

    # Whether the texture layer had anything to spend. Printed only when it
    # ran, and as its own block rather than as four more columns — the main
    # table is already eight wide.
    if any(any(k in r['readings'] for k in _TEXTURE) for r in results):
        print('\n  texture readings')
        print('  {:<{}}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}'.format(
            '', width, 'headroom', 'delivered', 'coverage', 'reserve', 'pose',
            'detail'))
        for result, label in zip(results, labels):
            cells = []
            for name in _TEXTURE:
                value = result['readings'].get(name)
                cells.append('       —' if value is None
                             else '{:9.3f}'.format(value))
            print('  {:<{}}  {}'.format(label, width, ' '.join(cells)))
        print('  headroom is the budget and delivered is the spend. Delivered '
              'well under headroom means the map is not')
        print('  reaching the face at the amplitude the reserve already stood '
              'detail matching down for.')

    if baseline:
        print('\n  where it goes, for `{}`:'.format(labels[0]))
        for line in _attribute(baseline):
            print(line)
        for line in _shape_note(baseline):
            print(line)

    if len(results) > 1:
        best = max(
            (r for r in results if 'id_out' in r['readings']),
            key=lambda r: r['readings']['id_out'], default=None)
        if best is not None:
            print('\n  best id_out: {:.3f} at `{}`'.format(
                best['readings']['id_out'], _label(best['settings'])))

        shaped = [r for r in results if 'outline_shift' in r['readings']]
        if shaped:
            top = max(shaped, key=lambda r: r['readings']['outline_shift'])
            print('  best outline_shift: {:+.3f} at `{}`'.format(
                top['readings']['outline_shift'], _label(top['settings'])))
            if top['settings'] != (best or {}).get('settings'):
                print('  -> the two disagree, which is the point of measuring '
                      'both: a cosine barely sees head shape, and head shape '
                      'is what a viewer reads first.')

        if best is not None or shaped:
            print('  Now look at the frames. A higher number that reads as '
                  'plastic or seamed is not a better swap — these measure two '
                  'axes of several.')

    if args.json:
        # `--save-frames` makes its directory and this did not, so a run that
        # had already done all its compute died at the last line with
        # FileNotFoundError and took every reading with it.
        parent = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(parent, exist_ok=True)
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
