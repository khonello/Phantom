"""
Route A — the source's complexion on every visible skin pixel, before the swap.

RESEMBLANCE.md §4, Route A. `_match_color` moves the swapped face onto the
*target's* skin tone, because a face that does not match the neck it sits on
is a seam. That is correct, and it is also most of what "it doesn't quite look
like me" is in colour terms. A face-only transfer cannot fix it — it moves the
seam to the jaw. So this stage changes what the colour match matches *to*:
every skin pixel of the **target frame** — face, neck, ears, chest, hands —
is graded toward the source's complexion *before* the swap, and the existing
colour match then pulls the swapped face toward surroundings that already
carry the source's tone. The seam logic is untouched; the face and the neck
agree because one stage graded both.

    frame in  ->  detect  ->  GRADE SKIN  ->  swap  ->  composite  ->  out
                                 ^ here, on the target frame

Three properties carry it:

- **Chroma is shifted, lightness is scaled.** Pigment lives in a/b and is
  moved by an offset. Lightness is shading and light direction — the
  target's to keep — so it is moved by a *gain*, which preserves every
  shading ratio: a neck at 0.75 of the forehead's L stays at 0.75. An offset
  on L would flatten the face; a gain on chroma would desaturate shadows.
- **Bounded and ramped, like `_match_color`.** Under `CLOSE` units of gap the
  stage does nothing, since the two complexions already agree to within what
  a 4:2:0 JPEG preserves. The chroma shift is capped at `_MAX_SHIFT` and the
  gain at `_MAX_GAIN`, so a pairing far apart lands on a bounded correction
  and says so in the readings rather than producing an implausible colour.
- **Smoothed on the parameters, not the pixels.** One offset and one gain per
  frame, EMA'd, so the grade cannot flicker with the face's own exposure
  wobble. Reset with the compositor's temporal state.

The reference is `complexion.resolve_reference`: the operator's baseline
(`complexion_base`, a Monk Skin Tone step) with the photographs' undertone
inside a bound, or the photographs alone under `auto`. See there for why a
prior rather than a destination.

Hands are in scope by default. The seeded segmenter knows colour and not
anatomy, so "hands" here means body-skin components outside a corridor below
the face — the neck and chest are inside it, arms and hands are not. Turning
them off leaves the neck graded and the hands the target's, which is a
visible mismatch at the wrists in some shots and the safer choice in others
(a cross-tone hand recolour is where the eye catches it). That is a footage
question, which is why it is a switch.

What it does not do: hair and beard. Excluded by the skin masks, and a
separate identity cue with a separate decision behind it.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.services import complexion
from pipeline.services.skin import SkinSegmenter
from pipeline.types import Face, Frame, Mask


# Largest chroma offset applied, in 8-bit LAB a/b units. Past this the two
# complexions are far enough apart that a flat shift stops reading as skin;
# the reading reports the cap binding so it is a number, not a mystery.
_MAX_SHIFT = 28.0

# Bounds on the lightness gain. Narrow on purpose, after the first footage run
# (2026-09-14): the face measured 40 L units darker than the source
# photographs — lighting, not complexion — and a cap of 2.0 let the stage try
# to grade the lighting away, doubling every exposure wobble on the way. A
# real complexion difference under the SAME light is a modest ratio; anything
# past this band is the room, and the room is the target's to keep.
_MIN_GAIN = 0.80
_MAX_GAIN = 1.25

# EMA on the per-frame parameters. About a second at 15fps: slow enough that
# a hunting auto-exposure averages out instead of being followed, which is
# what read as pulsing on the first footage run (`complexion_shift` swung
# from 6 to 19 units between frames). A step change in the source is a new
# source and comes with a reset.
_PARAM_ALPHA = 0.08

# The corridor below the face inside which body skin counts as neck and
# chest rather than hands, as multiples of the face's width either side of
# its centre. Only consulted when hands are switched off.
_NECK_CORRIDOR = 1.25

# Harmonisation — the second pass, body toward face. After the grade, the
# face's skin and the body's skin are measured on the GRADED frame and the
# body is corrected toward the face: chroma all the way, lightness toward a
# plausible neck-to-face ratio. This is what makes `complexion_seam` go to
# zero whatever caused it, which is the number the operator's complaint maps
# to — "the neck and shoulders do not meet the face". Measured 2026-09-14 on
# a fair-on-dark pairing after a declared-tone gain: still 6 units apart.
#
# A neck is darker than the forehead on everyone — it sits in the jaw's
# shadow — so lightness is not matched, it is brought toward a floor ratio.
# Under ordinary light the ratio is already ~0.85 and this does little; under
# a torch at the face it is ~0.3, and lifting it is correcting a light that
# was never the person's. Bounded, so a body already close is left alone.
_HARMONISE_L_RATIO = 0.82
_HARMONISE_MAX_GAIN = 2.0
_HARMONISE_MAX_SHIFT = 20.0


class ComplexionStage:
    """
    Grade the target frame's skin toward the source's complexion.

    Example:
        stage = ComplexionStage(config, segmenter)
        graded = stage.apply(frame, face, reference)   # or `frame` itself
    """

    def __init__(self, config: FaceSwapConfig, segmenter: SkinSegmenter) -> None:
        self.config = config
        self.segmenter = segmenter
        self.last_ms: float = 0.0
        # What the last frame applied, for the readings: chroma shift in a/b
        # units and the lightness gain. None when the stage did not run.
        self.last_shift: Optional[float] = None
        self.last_gain: Optional[float] = None
        # Body skin found, as a share of the face's area — whether the neck
        # was there to grade at all.
        self.last_coverage: Optional[float] = None
        self._shift: Optional[np.ndarray] = None
        self._gain: Optional[float] = None
        # The harmoniser's smoothed correction and what it applied.
        self._h_shift: Optional[np.ndarray] = None
        self._h_gain: Optional[float] = None
        self.last_harmonise: Optional[float] = None

    def reset(self) -> None:
        """Drop the smoothed parameters. Face lost, source changed."""
        self._shift = None
        self._gain = None
        self._h_shift = None
        self._h_gain = None

    def clear_readings(self) -> None:
        """Drop the last frame's readings, per frame."""
        self.last_shift = None
        self.last_gain = None
        self.last_coverage = None
        self.last_harmonise = None

    def enabled(self) -> bool:
        """Whether `skin_complexion` asks for anything at all."""
        return float(getattr(self.config, 'skin_complexion', 0.0) or 0.0) > 0.0

    def apply(
        self, frame: Frame, face: Face, reference: Optional[np.ndarray],
    ) -> Frame:
        """
        Grade one frame. Returns the frame itself, untouched, when there is
        nothing to do — no reference, strength zero, no skin found, or the two
        complexions already agree.

        Args:
            frame: The target frame, BGR
            face: The detection made on it
            reference: The complexion to grade toward, (L, a, b) 8-bit LAB

        Returns:
            A new graded frame, or `frame` unchanged
        """
        strength = float(np.clip(
            float(getattr(self.config, 'skin_complexion', 0.0) or 0.0), 0.0, 1.0))
        if strength <= 0.0 or reference is None:
            return frame

        started = time.perf_counter()
        graded = self._grade(frame, face, reference, strength)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return graded

    # ------------------------------------------------------------------

    def _grade(
        self, frame: Frame, face: Face, reference: np.ndarray, strength: float,
    ) -> Frame:
        masks = self.segmenter.segment(frame, face)
        if masks is None:
            return frame

        # The target's complexion is the sample the segmenter fitted its
        # model to, already smoothed — one measurement per frame, not two,
        # and the two cannot disagree about what the face's colour is.
        target = self.segmenter.sample_lab
        if target is None:
            return frame

        face_area = float(np.count_nonzero(masks.face > 0.5))
        if face_area > 0.0:
            self.last_coverage = float(np.count_nonzero(masks.body > 0.5)) / face_area

        # This frame's correction, then the smoothed one actually applied.
        shift = (reference[1:] - target[1:]) * strength
        magnitude = float(np.hypot(shift[0], shift[1]))
        if magnitude > _MAX_SHIFT:
            shift = shift * (_MAX_SHIFT / magnitude)

        # Lightness. Two regimes, and which one applies is declared, not
        # guessed. With BOTH tone classes named, the ratio of their swatches
        # is complexion with the room cancelled out, and it may move L as far
        # as the classes are apart. With either side `auto`, the only L
        # signal is photographs-against-frame — two rooms — and it stays
        # inside the narrow band that cannot grade the lighting away. This
        # was measured: on a fair-on-dark pairing the narrow band pinned at
        # 1.25 on every frame while the seam stayed at 6 units.
        declared = complexion.lightness_ratio(
            getattr(self.config, 'complexion_base', None),
            getattr(self.config, 'complexion_target_base', None))
        if declared is not None:
            gain = float(np.clip(
                declared ** strength,
                1.0 / complexion.CLASS_GAIN_MAX, complexion.CLASS_GAIN_MAX))
        else:
            ratio = float(reference[0]) / max(float(target[0]), 1.0)
            gain = float(np.clip(ratio ** strength, _MIN_GAIN, _MAX_GAIN))

        if self._shift is None or self._gain is None:
            self._shift, self._gain = shift.astype(np.float32), gain
        else:
            a = _PARAM_ALPHA
            self._shift = (a * shift + (1.0 - a) * self._shift).astype(np.float32)
            self._gain = a * gain + (1.0 - a) * self._gain

        self.last_shift = float(np.hypot(self._shift[0], self._shift[1]))
        self.last_gain = self._gain

        # Nothing worth a full-frame colour conversion: the complexions agree.
        if self.last_shift < complexion.CLOSE and abs(self._gain - 1.0) < 0.02:
            return frame

        body = masks.body
        if not bool(getattr(self.config, 'skin_complexion_hands', True)):
            body = _neck_only(body, face)
        alpha = np.maximum(masks.face, body)

        graded = _apply(frame, alpha, self._shift, self._gain)
        return self._harmonise(graded, masks.face, body)

    def _harmonise(self, graded: Frame, face_mask: Mask, body: Mask) -> Frame:
        """
        Second pass: correct the body toward the graded face.

        Args:
            graded: The frame after the global grade
            face_mask: The face skin mask
            body: The body skin mask (hands already removed if switched off)

        Returns:
            The frame with the body harmonised, or `graded` unchanged when the
            knob is off or either region cannot be measured
        """
        strength = float(np.clip(
            float(getattr(self.config, 'skin_harmonise', 0.0) or 0.0), 0.0, 1.0))
        if strength <= 0.0:
            return graded

        face_lab = complexion.masked_lab(graded, face_mask)
        body_lab = complexion.masked_lab(graded, body)
        if face_lab is None or body_lab is None:
            return graded

        shift = (face_lab[1:] - body_lab[1:]) * strength
        magnitude = float(np.hypot(shift[0], shift[1]))
        if magnitude > _HARMONISE_MAX_SHIFT:
            shift = shift * (_HARMONISE_MAX_SHIFT / magnitude)

        ratio = float(body_lab[0]) / max(float(face_lab[0]), 1.0)
        # Lift only toward the floor ratio, never past it, never down.
        wanted = max(ratio, _HARMONISE_L_RATIO)
        gain = float(np.clip((wanted / max(ratio, 1e-3)) ** strength, 1.0, _HARMONISE_MAX_GAIN))

        if self._h_shift is None or self._h_gain is None:
            self._h_shift, self._h_gain = shift.astype(np.float32), gain
        else:
            a = _PARAM_ALPHA
            self._h_shift = (a * shift + (1.0 - a) * self._h_shift).astype(np.float32)
            self._h_gain = a * gain + (1.0 - a) * self._h_gain

        self.last_harmonise = float(np.hypot(self._h_shift[0], self._h_shift[1]))
        if self.last_harmonise < complexion.CLOSE and abs(self._h_gain - 1.0) < 0.02:
            return graded
        return _apply(graded, body, self._h_shift, self._h_gain)


def _apply(frame: Frame, alpha: Mask, shift: np.ndarray, gain: float) -> Frame:
    """
    Shift chroma and scale lightness under a soft mask, inside its bounding box.

    Args:
        frame: BGR uint8
        alpha: Float mask in [0, 1], frame-sized
        shift: (da, db) in 8-bit LAB units
        gain: Multiplier on L

    Returns:
        A new frame; pixels with zero alpha are byte-identical to the input
    """
    rows = np.flatnonzero(alpha.max(axis=1) > 0.0)
    cols = np.flatnonzero(alpha.max(axis=0) > 0.0)
    if rows.size == 0 or cols.size == 0:
        return frame

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    roi = frame[y0:y1, x0:x1]
    weight = alpha[y0:y1, x0:x1][:, :, None]

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    graded = lab.copy()
    graded[:, :, 0] *= gain
    graded[:, :, 1] += float(shift[0])
    graded[:, :, 2] += float(shift[1])
    np.clip(graded, 0.0, 255.0, out=graded)

    blended = lab + (graded - lab) * weight
    np.rint(blended, out=blended)
    out_roi = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Only the masked pixels are written back, so everything outside the mask
    # never went through a colour round trip — byte-identical by construction
    # rather than by luck.
    result = frame.copy()
    touched = weight[:, :, 0] > 0.0
    region = result[y0:y1, x0:x1]
    region[touched] = out_roi[touched]
    return result


def _neck_only(body: Mask, face: Face) -> Mask:
    """
    Keep body-skin components inside the corridor below the face.

    The seeded segmenter cannot tell a neck from a hand; geometry can, well
    enough for a switch: the neck and chest sit under the face, arms and hands
    do not.
    """
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return body
    box = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if box.size < 4:
        return body
    centre_x = (box[0] + box[2]) / 2.0
    width = max(box[2] - box[0], 1.0)
    left = centre_x - width * _NECK_CORRIDOR
    right = centre_x + width * _NECK_CORRIDOR

    binary = (body > 0.5).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for index in range(1, count):
        cx = float(centroids[index][0])
        if left <= cx <= right:
            keep[labels == index] = 1
    return np.asarray(body * keep, dtype=np.float32)


def corridor(face: Face) -> Optional[Tuple[float, float]]:
    """The neck corridor's horizontal extent, for tests and diagnostics."""
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return None
    box = np.asarray(bbox, dtype=np.float64).reshape(-1)
    centre_x = (box[0] + box[2]) / 2.0
    width = max(box[2] - box[0], 1.0)
    return centre_x - width * _NECK_CORRIDOR, centre_x + width * _NECK_CORRIDOR
