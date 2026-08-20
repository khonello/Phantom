"""
Check every bundled template against the guards that will judge it at runtime.

A template is a target *we* chose, so a user must never meet a refusal caused by
one. This runs the real `guards.check_frame` over the library and reports, per
template, whether the scene would swap — before it ships, rather than in front
of somebody.

It also verifies the thing a manifest can most easily get wrong: that
`face_point` lands on a face. A point that misses resolves to the nearest face
instead, which is not an error and not a crash — it is simply the wrong person,
silently. That is exactly the failure this file exists to catch.

Usage:
    python tools/validate_templates.py [--dir DIR] [--json OUT]

Exit code is non-zero if any template would fail, so it can gate a build.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from pipeline.config import FaceSwapConfig  # noqa: E402
from pipeline.services import guards, templates  # noqa: E402
from pipeline.services.face_detection import FaceDetector  # noqa: E402


def check_template(
    template: templates.Template,
    detector: FaceDetector,
    config: FaceSwapConfig,
) -> Dict[str, Any]:
    """
    Run one template through detection and the runtime guards.

    Args:
        template: Template to check
        detector: Detector to find faces with
        config: Thresholds, with the template's own face point applied

    Returns:
        A record of what was found and whether it would swap
    """
    record: Dict[str, Any] = {
        'id': template.id,
        'name': template.name,
        'ok': False,
        'faces': 0,
        'problems': [],
    }

    frame = cv2.imread(template.image)
    if frame is None:
        record['problems'].append('image could not be read')
        return record

    detections = detector.detect(frame)
    record['faces'] = len(detections)

    if not detections:
        record['problems'].append('no face detected')
        return record

    # The template's own choice is what the pipeline will apply, so evaluate
    # under it rather than under the defaults.
    config.target_face_point = template.face_point

    if template.face_point is None and len(detections) > 1:
        record['problems'].append(
            f'{len(detections)} faces and no face_point — the largest would be '
            f'picked, and the multi-face guard would refuse the scene'
        )

    if template.face_point is not None:
        chosen = templates.select_by_point(
            detections, template.face_point,
            (int(frame.shape[0]), int(frame.shape[1])),
        )
        height, width = frame.shape[:2]
        x = template.face_point[0] * width
        y = template.face_point[1] * height
        inside = chosen is not None and (
            chosen.bbox.x <= x <= chosen.bbox.x + chosen.bbox.w
            and chosen.bbox.y <= y <= chosen.bbox.y + chosen.bbox.h
        )
        if not inside:
            # Resolvable, but by proximity — which means the manifest is
            # asserting something it does not actually point at.
            record['problems'].append(
                'face_point does not land on any detected face; it resolved by '
                'proximity, which may not be the intended person'
            )
        if chosen is not None:
            record['chosen_bbox'] = [
                int(chosen.bbox.x), int(chosen.bbox.y),
                int(chosen.bbox.w), int(chosen.bbox.h),
            ]
            record['face_px'] = int(min(chosen.bbox.w, chosen.bbox.h))

    verdict = guards.check_frame(config, detections)
    if not verdict.ok:
        record['problems'].append(verdict.message or verdict.reason)

    if template.foreground:
        layer = cv2.imread(template.foreground, cv2.IMREAD_UNCHANGED)
        if layer is None:
            record['problems'].append('foreground layer could not be read')
        elif layer.ndim != 3 or layer.shape[2] != 4:
            record['problems'].append('foreground layer has no alpha channel')
        elif layer.shape[:2] != frame.shape[:2]:
            # Not fatal — it is resized — but a mismatch usually means the
            # layer was exported from a different comp than the scene.
            record['problems'].append(
                f'foreground is {layer.shape[1]}x{layer.shape[0]}, scene is '
                f'{frame.shape[1]}x{frame.shape[0]} — it will be resized'
            )

    record['ok'] = not record['problems']
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', default=None, help='Templates directory')
    parser.add_argument('--json', default=None, help='Write the report here')
    args = parser.parse_args()

    library = templates.TemplateLibrary(args.dir)
    entries = library.all()

    print(f'Templates directory: {library.directory}')
    if not entries:
        print('No templates found.')
        return 0

    config = FaceSwapConfig()
    detector = FaceDetector(config)
    records: List[Dict[str, Any]] = []

    for template in entries:
        record = check_template(template, detector, config)
        records.append(record)
        mark = 'OK  ' if record['ok'] else 'FAIL'
        print(f'  [{mark}] {record["id"]} — {record["faces"]} face(s)')
        for problem in record['problems']:
            print(f'         {problem}')

    failed = [r for r in records if not r['ok']]
    print(f'\n{len(records) - len(failed)} of {len(records)} templates would swap.')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump({'templates': records}, fh, indent=2)
        print(f'Report written to {args.json}')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
