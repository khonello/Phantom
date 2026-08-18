#!/usr/bin/env python3
"""
Measure a debug-frame capture for the tells that give a swap away.

`--debug-frames DIR` writes lossless `NNNNNN_in.png` / `NNNNNN_out.png` pairs
from a live session. This turns those pairs into numbers, so a realism change
can be judged against a fixed clip instead of re-recorded and argued about.

It measures the three failure modes this project is built around:

  1. **Too clean** — the swapped face sharper or less noisy than the frame it
     sits in. Noise sigma and high-frequency energy, inside the face against
     outside it. A ratio near 1.0 means the face matches its surroundings;
     below 1.0 means it is smoother than the camera that supposedly shot it.
  2. **A visible seam** — a gradient step across the mask boundary that does not
     exist elsewhere. Measured as the boundary gradient over the surrounding
     region, so near 1.0 means the edge is invisible and a large value means
     there is a line where the swap ends.
  3. **Wrong motion** — during head movement the real frame smears and a
     generated face does not, so the swap ends up *sharper than what it
     replaced*. Blur anisotropy inside the face against outside, on frames where
     motion actually occurred.

Plus colour: LAB mean and spread inside the face against outside, which is what
drifts when lighting changes and the transfer under- or over-corrects.

Every ratio is inside-over-outside, both taken from the **output** frame. That
is the comparison that matters: not whether the swap resembles the original
face, but whether it belongs in the picture it is now part of.

The face region comes from the difference between input and output, so this
needs no detector, no models and no GPU — the swap marks its own territory.

Usage:
    python tools/compare_frames.py <dir> [--json report.json] [--limit N]
    python tools/compare_frames.py <dir_a> --against <dir_b>
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# A pixel counts as "changed by the swap" above this mean absolute difference.
# Low, because the composite fades out at the mask edge and that faint outer
# ring is exactly the part a seam check needs to see.
_CHANGE_THRESHOLD = 6.0

# Gaussian sigma splitting texture from shape, specified at 256px of face and
# scaled with the face's real size. Matches the compositor's own detail-matching
# split, so the two are talking about the same band.
_DETAIL_SIGMA = 1.5
_DETAIL_REFERENCE = 256.0

# Laplacian (4-neighbour) amplifies noise variance by 4*(1^2) + (-4)^2 = 20.
_LAPLACIAN_GAIN = float(np.sqrt(20.0))

# Width in pixels of the band either side of the mask edge used for the seam
# measurement, and of the ring beyond it used as the comparison.
_SEAM_BAND = 3
_SEAM_CONTEXT = 12

# Mean absolute frame-to-frame difference above which a frame counts as motion.
_MOTION_THRESHOLD = 2.0


def load_pairs(directory: str, limit: int = 0) -> List[Tuple[str, str, str]]:
    """
    Find (seq, input, output) triples in a debug-frame directory.

    Args:
        directory: Directory written by --debug-frames
        limit: Stop after this many pairs; 0 for all

    Returns:
        Triples sorted by sequence number
    """
    pairs = []
    for in_path in sorted(glob.glob(os.path.join(glob.escape(directory), '*_in.png'))):
        seq = os.path.basename(in_path)[:-len('_in.png')]
        out_path = os.path.join(directory, '{}_out.png'.format(seq))
        if os.path.isfile(out_path):
            pairs.append((seq, in_path, out_path))
        if limit and len(pairs) >= limit:
            break
    return pairs


def face_mask(source: np.ndarray, output: np.ndarray) -> Optional[np.ndarray]:
    """
    Where the swap changed the frame.

    Derived rather than detected: anything the compositor touched differs from
    the input and nothing else does, so the composite marks its own extent. A
    detector here would be a second opinion about where the face is, when what
    is wanted is where the swap actually landed.

    Args:
        source: Input frame
        output: Composited frame

    Returns:
        uint8 mask (255 = changed), or None if nothing changed
    """
    if source.shape != output.shape:
        return None

    difference = cv2.absdiff(source, output).mean(axis=2)
    mask = (difference > _CHANGE_THRESHOLD).astype(np.uint8) * 255

    # Close pinholes: flat skin is sometimes reproduced almost exactly, and
    # those pixels are inside the face however little they moved.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if cv2.countNonZero(mask) < 256:
        return None

    # Largest region only — stray compression differences elsewhere in the
    # frame are not the face.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8) * 255

    return mask


def noise_sigma(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Robust noise estimate over a masked region, via the MAD of its Laplacian.

    Median-based so facial detail and edges do not inflate it. Deliberately the
    same estimator the compositor uses to match grain, so a disagreement between
    them means a real mismatch rather than two different definitions of noise.

    Args:
        gray: Single-channel image
        mask: uint8 region selector

    Returns:
        Estimated sigma in intensity units
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    values = laplacian[mask > 0]
    if values.size < 64:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad / _LAPLACIAN_GAIN


def high_frequency_energy(gray: np.ndarray, mask: np.ndarray, face_px: float) -> float:
    """
    Standard deviation of the high-frequency band inside a region.

    The band split scales with the face size in frame, so "texture" means the
    same physical detail whether the subject is close to the camera or far.

    Args:
        gray: Single-channel image
        mask: uint8 region selector
        face_px: Face extent in pixels, used to scale the split

    Returns:
        High-band standard deviation
    """
    sigma = _DETAIL_SIGMA * max(face_px, 1.0) / _DETAIL_REFERENCE
    low = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(sigma, 0.4))
    high = gray.astype(np.float32) - low
    values = high[mask > 0]
    return float(values.std()) if values.size >= 64 else 0.0


def blur_anisotropy(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Directional smear, as the ratio of gradient energy across two axes.

    Motion blur suppresses detail along the direction of travel and leaves it
    across, so a smeared region reads well above 1.0. A generated face that did
    not smear reads near 1.0 while the frame around it does not — which is
    exactly the "sharper than what it replaced" tell.

    Args:
        gray: Single-channel image
        mask: uint8 region selector

    Returns:
        max/min axis gradient energy ratio, 1.0 when isotropic
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    inside = mask > 0
    if int(inside.sum()) < 64:
        return 1.0
    ex = float(np.mean(np.square(gx[inside])))
    ey = float(np.mean(np.square(gy[inside])))
    low, high = min(ex, ey), max(ex, ey)
    return (high / low) if low > 1e-6 else 1.0


def seam_ratio(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Gradient across the mask boundary, relative to the region just outside it.

    A composite that hands back to the frame cleanly has no more gradient at its
    edge than the surrounding skin does. Near 1.0 means the edge is invisible; a
    large value means there is a line where the swap ends.

    Args:
        gray: Single-channel image
        mask: uint8 region selector

    Returns:
        Boundary gradient over context gradient
    """
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    band_k = np.ones((_SEAM_BAND * 2 + 1,) * 2, np.uint8)
    context_k = np.ones((_SEAM_CONTEXT * 2 + 1,) * 2, np.uint8)

    boundary = cv2.subtract(cv2.dilate(mask, band_k), cv2.erode(mask, band_k))
    context = cv2.subtract(cv2.dilate(mask, context_k), cv2.dilate(mask, band_k))

    edge = gradient[boundary > 0]
    around = gradient[context > 0]
    if edge.size < 64 or around.size < 64:
        return 1.0

    outer = float(np.mean(around))
    return (float(np.mean(edge)) / outer) if outer > 1e-6 else 1.0


def lab_stats(image: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """
    LAB mean and spread over a region.

    Args:
        image: BGR image
        mask: uint8 region selector

    Returns:
        Per-channel mean and standard deviation, or None if the region is tiny
    """
    if cv2.countNonZero(mask) < 64:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean, dev = cv2.meanStdDev(lab, mask=mask)
    return {
        'L_mean': float(mean[0][0]), 'L_std': float(dev[0][0]),
        'a_mean': float(mean[1][0]), 'a_std': float(dev[1][0]),
        'b_mean': float(mean[2][0]), 'b_std': float(dev[2][0]),
    }


def measure_pair(
    source: np.ndarray,
    output: np.ndarray,
    previous: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """
    Measure one (input, output) pair.

    Args:
        source: Input frame
        output: Composited frame
        previous: Previous input frame, for motion detection

    Returns:
        Measurements, or None if no swap was found in this frame
    """
    mask = face_mask(source, output)
    if mask is None:
        return None

    # Dilated before inverting, so the "outside" sample starts clear of the
    # feathered edge rather than straddling it.
    outside = cv2.bitwise_not(cv2.dilate(mask, np.ones((21, 21), np.uint8)))
    if cv2.countNonZero(outside) < 256:
        return None

    out_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    face_px = float(np.sqrt(cv2.countNonZero(mask)))

    inside_sigma = noise_sigma(out_gray, mask)
    outside_sigma = noise_sigma(out_gray, outside)
    inside_hf = high_frequency_energy(out_gray, mask, face_px)
    outside_hf = high_frequency_energy(out_gray, outside, face_px)

    motion = 0.0
    if previous is not None and previous.shape == source.shape:
        motion = float(cv2.absdiff(source, previous).mean())

    record: Dict[str, Any] = {
        'face_px': round(face_px, 1),
        'coverage_pct': round(cv2.countNonZero(mask) / mask.size * 100.0, 2),
        'noise_ratio': (
            round(inside_sigma / outside_sigma, 4) if outside_sigma > 1e-6 else None
        ),
        'hf_ratio': round(inside_hf / outside_hf, 4) if outside_hf > 1e-6 else None,
        'seam_ratio': round(seam_ratio(out_gray, mask), 4),
        'anisotropy_in': round(blur_anisotropy(out_gray, mask), 4),
        'anisotropy_out': round(blur_anisotropy(out_gray, outside), 4),
        'motion': round(motion, 3),
    }

    inside_lab = lab_stats(output, mask)
    outside_lab = lab_stats(output, outside)
    if inside_lab and outside_lab:
        record['lab_delta'] = {
            key: round(inside_lab[key] - outside_lab[key], 2) for key in inside_lab
        }

    return record


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reduce per-frame measurements to a comparable summary.

    Motion frames are summarised separately for anisotropy, because that measure
    exists for what happens *during* movement — averaging it over a mostly-still
    clip hides the effect entirely.

    Args:
        records: Per-frame measurements

    Returns:
        Summary suitable for printing or diffing against another run
    """
    def stat(key: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            'p50': round(float(np.percentile(array, 50)), 4),
            'p95': round(float(np.percentile(array, 95)), 4),
            'mean': round(float(array.mean()), 4),
        }

    moving = [r for r in records if r.get('motion', 0.0) >= _MOTION_THRESHOLD]

    summary: Dict[str, Any] = {
        'frames': len(records),
        'moving_frames': len(moving),
        'noise_ratio': stat('noise_ratio', records),
        'hf_ratio': stat('hf_ratio', records),
        'seam_ratio': stat('seam_ratio', records),
        'coverage_pct': stat('coverage_pct', records),
        'anisotropy_in_motion': stat('anisotropy_in', moving),
        'anisotropy_out_motion': stat('anisotropy_out', moving),
    }

    deltas = [r['lab_delta'] for r in records if 'lab_delta' in r]
    if deltas:
        summary['lab_delta'] = {
            key: round(float(np.mean([d[key] for d in deltas])), 2)
            for key in deltas[0]
        }

    return summary


def verdict(summary: Dict[str, Any]) -> List[str]:
    """
    Plain-language reading of the numbers, with the reasoning attached.

    Args:
        summary: Output of `summarise`

    Returns:
        One line per finding
    """
    notes = []

    noise = (summary.get('noise_ratio') or {}).get('p50')
    if noise is not None:
        if noise < 0.7:
            notes.append(
                'TOO CLEAN: the face carries {:.0%} of the sensor noise the rest '
                'of the frame has. Poreless skin on a noisy webcam is the '
                'strongest "this is AI" signal there is.'.format(noise))
        elif noise > 1.4:
            notes.append(
                'NOISIER THAN THE FRAME ({:.2f}x) — grain matching is '
                'overshooting.'.format(noise))
        else:
            notes.append('Noise matches the frame ({:.2f}x).'.format(noise))

    hf = (summary.get('hf_ratio') or {}).get('p50')
    if hf is not None:
        if hf > 1.3:
            notes.append(
                'SHARPER THAN THE FRAME ({:.2f}x) — restoration is winning over '
                'detail matching.'.format(hf))
        elif hf < 0.7:
            notes.append('Softer than the frame ({:.2f}x).'.format(hf))
        else:
            notes.append('Detail matches the frame ({:.2f}x).'.format(hf))

    seam = (summary.get('seam_ratio') or {}).get('p95')
    if seam is not None:
        if seam > 1.5:
            notes.append(
                'VISIBLE SEAM: p95 boundary gradient is {:.2f}x the surrounding '
                'region.'.format(seam))
        else:
            notes.append('No seam detected (p95 {:.2f}x).'.format(seam))

    inside = (summary.get('anisotropy_in_motion') or {}).get('p50')
    outside = (summary.get('anisotropy_out_motion') or {}).get('p50')
    if summary['moving_frames'] < 10:
        notes.append(
            'Not enough motion in this clip to judge blur matching — record one '
            'with head movement.')
    elif inside is not None and outside is not None:
        if outside - inside > 0.25:
            notes.append(
                'MOTION MISMATCH: during movement the frame smears ({:.2f}) '
                'while the face does not ({:.2f}). This is what motion-blur '
                'matching would fix.'.format(outside, inside))
        else:
            notes.append(
                'Motion blur consistent ({:.2f} face vs {:.2f} frame).'.format(
                    inside, outside))

    return notes


def analyse(directory: str, limit: int = 0) -> Dict[str, Any]:
    """
    Measure every pair in a directory.

    Args:
        directory: Directory written by --debug-frames
        limit: Stop after this many pairs; 0 for all

    Returns:
        Summary with per-frame records attached
    """
    pairs = load_pairs(directory, limit)
    if not pairs:
        raise SystemExit('No *_in.png / *_out.png pairs found in {}'.format(directory))

    records: List[Dict[str, Any]] = []
    previous: Optional[np.ndarray] = None
    skipped = 0

    for seq, in_path, out_path in pairs:
        source = cv2.imread(in_path)
        output = cv2.imread(out_path)
        if source is None or output is None:
            skipped += 1
            continue

        record = measure_pair(source, output, previous)
        previous = source
        if record is None:
            skipped += 1
            continue
        record['seq'] = seq
        records.append(record)

    if not records:
        raise SystemExit(
            'No frames with a detectable swap in {} — every pair was identical. '
            'Was a source face set?'.format(directory))

    summary = summarise(records)
    summary['directory'] = directory
    summary['pairs_found'] = len(pairs)
    summary['pairs_without_swap'] = skipped
    summary['records'] = records
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print a measured summary.

    Args:
        summary: Output of `analyse`
    """
    print('=' * 68)
    print('{}  -  {} frames measured, {} without a swap'.format(
        summary['directory'], summary['frames'], summary['pairs_without_swap']))
    print('=' * 68)

    rows = [
        ('noise_ratio', 'sensor noise, face / frame', '1.00'),
        ('hf_ratio', 'high-freq detail, face / frame', '1.00'),
        ('seam_ratio', 'gradient at mask edge / around', '1.00'),
        ('coverage_pct', 'face as % of frame', '-'),
        ('anisotropy_in_motion', 'blur anisotropy, face (moving)', 'match'),
        ('anisotropy_out_motion', 'blur anisotropy, frame (moving)', 'match'),
    ]
    print('{:<34} {:>7} {:>7} {:>7}   ideal'.format('', 'p50', 'p95', 'mean'))
    for key, label, ideal in rows:
        entry = summary.get(key)
        if not entry:
            continue
        print('{:<34} {:>7.3f} {:>7.3f} {:>7.3f}   {}'.format(
            label, entry['p50'], entry['p95'], entry['mean'], ideal))

    if summary.get('lab_delta'):
        print('\nLAB difference, face minus frame (0 = identical):')
        print('  ' + '  '.join(
            '{}={:+.1f}'.format(k, v) for k, v in summary['lab_delta'].items()))

    print('\nReading:')
    for note in verdict(summary):
        print('  - ' + note)


def main() -> int:
    """
    Entry point.

    Returns:
        Process exit code
    """
    parser = argparse.ArgumentParser(
        description='Measure realism tells in a --debug-frames capture.')
    parser.add_argument('directory', help='directory written by --debug-frames')
    parser.add_argument('--against', help='second directory to compare against')
    parser.add_argument('--json', help='write the full report here')
    parser.add_argument('--limit', type=int, default=0,
                        help='measure at most N pairs (0 = all)')
    args = parser.parse_args()

    summary = analyse(args.directory, args.limit)
    print_summary(summary)

    other: Optional[Dict[str, Any]] = None
    if args.against:
        other = analyse(args.against, args.limit)
        print()
        print_summary(other)
        print('\n' + '=' * 68)
        print('Change: {} -> {}'.format(args.directory, args.against))
        print('=' * 68)
        for key in ('noise_ratio', 'hf_ratio', 'seam_ratio'):
            first, second = summary.get(key), other.get(key)
            if first and second:
                print('{:<24} {:+.3f}  ({:.3f} -> {:.3f})'.format(
                    key, second['p50'] - first['p50'], first['p50'], second['p50']))

    if args.json:
        payload: Dict[str, Any] = {'primary': summary}
        if other is not None:
            payload['against'] = other
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        print('\nWritten to {}'.format(args.json))

    return 0


if __name__ == '__main__':
    sys.exit(main())
