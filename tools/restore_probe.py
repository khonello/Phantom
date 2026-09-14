#!/usr/bin/env python3
"""
Score restored stills against the source — RESEMBLANCE.md Route B's instrument.

    python tools/restore_probe.py -s source/one --frames gpen.png refstar.png

For each frame: ArcFace cosine against the source identity (the same averaged
embedding the pipeline conditions on, built through `review_sources` so the
guards are in the loop), the face's high-frequency deviation on a cheek window
(the texture the restorer put there), and the face's size. Two frames of the
same target through two restorers are the comparison the route hangs on:

    id higher AND texture higher    the reference restorer carries the source
    id higher, texture lower        it carries identity but smooths — GPEN
                                    territory with a better prior
    id lower                        it hallucinated someone else's detail

Also `--reference`: the photograph handed to the restorer as its reference,
so the reading can say how much of the output is that photograph's texture
rather than the swap's (a restorer that copies the reference verbatim scores
perfectly on identity and is a different picture).

Needs a detector and the recognition model, so it runs where the pipeline
runs — the pod — not on the development machine.
"""

import argparse
import os
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FaceSwapConfig                        # noqa: E402
from pipeline.core import (                                        # noqa: E402
    decode_execution_providers,
    suggest_default_execution_providers,
)
from pipeline.services import identity                            # noqa: E402
from pipeline.services.database import FaceDatabase               # noqa: E402
from pipeline.services.face_detection import FaceDetector         # noqa: E402
from pipeline.services.identity import IdentityProbe              # noqa: E402
from pipeline.processing.geometry import DETAIL_SIGMA            # noqa: E402


def _cheek_window(frame: np.ndarray, bbox: Any, side: float = 0.28) -> Optional[np.ndarray]:
    """A window on the cheek, below the eye line, as a fraction of the face box."""
    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
    cx = x + int(w * 0.30)
    cy = y + int(h * 0.55)
    half = max(4, int(min(w, h) * side / 2))
    patch = frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    return patch if patch.size and min(patch.shape[:2]) >= 8 else None


def _high_frequency(patch: np.ndarray) -> float:
    """Deviation of the detail band the compositor scales, on grey."""
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(grey, (0, 0), DETAIL_SIGMA)
    return float((grey - low).std())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-s', '--source', nargs='+', required=True,
                        help='source photograph(s) or a directory of them')
    parser.add_argument('--frames', nargs='+', required=True,
                        help='restored stills to score, in the order to print them')
    parser.add_argument('--reference', help='the photograph the restorer was given')
    parser.add_argument('--execution-provider', nargs='+',
                        default=suggest_default_execution_providers())
    args = parser.parse_args()

    config = FaceSwapConfig()
    config.execution_providers = decode_execution_providers(args.execution_provider)

    detector = FaceDetector(config)
    database = FaceDatabase(detector, config)
    probe = IdentityProbe(detector)

    paths: List[str] = []
    for entry in args.source:
        if os.path.isdir(entry):
            paths += sorted(os.path.join(entry, n) for n in os.listdir(entry)
                            if n.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        else:
            paths.append(entry)
    review = database.review_sources(paths)
    source = database.get_source_face(list(review.accepted))
    if source is None:
        raise SystemExit('no usable source identity')
    source_embedding = getattr(source, 'normed_embedding', None)
    print('source identity from {} of {} photograph(s)'.format(
        len(review.accepted), len(paths)))

    reference_embedding = None
    if args.reference:
        image = cv2.imread(args.reference)
        detection = detector.detect_one(image) if image is not None else None
        if detection is not None:
            reference_embedding = probe.embed_frame(image, getattr(detection.face, 'kps', None))
            print('reference photograph: {}'.format(os.path.basename(args.reference)))

    print()
    print('  {:<28} {:>8} {:>8} {:>9} {:>8}'.format(
        'frame', 'id_src', 'id_ref', 'cheek_hf', 'face_px'))
    print('  ' + '-' * 66)

    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []
    for path in args.frames:
        frame = cv2.imread(path)
        label = os.path.basename(path)[:28]
        if frame is None:
            print('  {:<28} unreadable'.format(label))
            continue
        detection = detector.detect_one(frame)
        if detection is None:
            print('  {:<28} no face'.format(label))
            continue
        embedding = probe.embed_frame(frame, getattr(detection.face, 'kps', None))
        id_src = identity.cosine(source_embedding, embedding)
        id_ref = identity.cosine(reference_embedding, embedding)
        window = _cheek_window(frame, detection.bbox)
        hf = _high_frequency(window) if window is not None else None
        rows.append((label, id_src, id_ref, hf))
        print('  {:<28} {:>8} {:>8} {:>9} {:>8}'.format(
            label,
            '—' if id_src is None else '{:.3f}'.format(id_src),
            '—' if id_ref is None else '{:.3f}'.format(id_ref),
            '—' if hf is None else '{:.2f}'.format(hf),
            int(min(detection.bbox.w, detection.bbox.h))))

    scored = [r for r in rows if r[1] is not None]
    if len(scored) >= 2:
        first, best = scored[0], max(scored, key=lambda r: r[1] or 0.0)
        print()
        if best is not first:
            delta = (best[1] or 0.0) - (first[1] or 0.0)
            print('  {} carries {:+.3f} more of the source than {}'.format(
                best[0], delta, first[0]))
            if best[3] is not None and first[3] is not None:
                print('  and {} cheek detail ({:.2f} vs {:.2f})'.format(
                    'more' if best[3] > first[3] else 'LESS', best[3], first[3]))
        else:
            print('  nothing beat {} on identity'.format(first[0]))
        if reference_embedding is not None:
            for label, id_src, id_ref, _ in scored:
                if id_ref is not None and id_src is not None and id_ref > id_src + 0.08:
                    print('  -> {} resembles the REFERENCE PHOTOGRAPH more than the '
                          'source identity ({:.3f} vs {:.3f}): check it is not '
                          'copying the reference rather than restoring the face'.format(
                              label, id_ref, id_src))
    return 0


if __name__ == '__main__':
    sys.exit(main())
