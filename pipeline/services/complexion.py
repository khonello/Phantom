"""
Complexion — the identity cue the cosine cannot see and the colour match spends.

`_match_color` moves the swapped face onto the *target's* skin tone, because a
face whose tone does not match the neck it sits on is failure mode 2. It is
correct, and it is also most of what "it doesn't quite look like me" is in
colour terms. Nothing measured that cost: ArcFace is nearly blind to skin tone,
so `id_out` barely moves whether the complexion transferred or not — and where
it does move, it cannot say whether it moved for likeness or for tone.

This measures it directly, in the same place and on the same interval as the
identity and shape probes, so a configuration is scored on all three axes at
once. Four readings:

    complexion_gap     source skin against target skin. A property of the
                       PAIRING — no setting moves it — and the denominator for
                       everything below. Under a few units the two complexions
                       already agree and nothing here has anything to do
    complexion_face    output face skin against the SOURCE's. The number to
                       drive toward zero
    complexion_target  output face skin against the target's ORIGINAL skin.
                       The leakage direction: this rising is the swap taking
    complexion_lum     output lightness less the source's, signed. Reported
                       apart from the chroma distances because it is mostly
                       lighting, and lighting is the target's to keep
    complexion_neck    the target's OTHER skin — neck, ears, chest, hands —
                       against the source's. Whether the rest of the person
                       agrees with the face. Needs the skin segmenter
    complexion_seam    the output's face skin against its own neck skin. The
                       seam a face-only complexion transfer creates, and the
                       number Route A exists to hold at zero

Chroma only for the three distances — the a/b plane of OpenCV's 8-bit LAB, the
same units `_COMPLEXION_RESIDUAL` and `complexion_kept` are in. Pigment lives
in chroma; luminance is shading and light direction, which a correct swap
takes from the target, so folding it into the distance would report a face
lit from the window as the wrong colour.

Two things this deliberately does not do:

- **It does not read the whole face.** Skin is the landmark hull with eyes,
  nostrils and mouth cut out, through the same `skin_mask` the texture layer
  extracts under, so the two layers agree on what skin is. Lips are red on
  everyone. Medians rather than means, since a freckle field and a specular
  highlight are not complexion.
- **It does not trust one photograph.** The source reference is the median
  across every accepted photograph's skin — each carries its own white balance,
  and the median across several is the closest thing to the person's colour
  that a set of uncontrolled uploads can give. `SourceComplexion.spread` says
  how much they disagreed, which is the honest error bar on `complexion_face`.

The neck comes from `pipeline/services/skin.py` — the body mask, which is
classified skin outside the grown face hull — and the two readings that need
it are simply absent when no segmenter is attached or no body skin is visible.
A face-only complexion transfer creates the seam at the jaw that the colour
match exists to prevent, and `complexion_seam` is what prices it.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.processing.geometry import canonical_from_frame
from pipeline.processing.texture import skin_mask
from pipeline.types import Face, Frame


# Canonical edge the skin is measured at. A median over a few thousand pixels
# does not need the restorer's 512, and this runs on the live path every Nth
# frame — the same argument `_texture_headroom` makes for its 160px window.
MEASURE_SIZE = 256

# Fewer skin pixels than this and the median is a guess: a face at the edge of
# the frame, or a mask that mostly landed on the feature exclusions.
_MIN_PIXELS = 256

# Lightness outside this band is shadow or specular, and neither carries
# pigment. Bounds are in OpenCV's 8-bit L, so 255 is white.
_L_FLOOR = 8.0
_L_CEILING = 247.0

# Below this chroma gap the two complexions agree to within what a JPEG at
# 4:2:0 preserves, and the readings describe noise rather than a transfer.
CLOSE = 3.0


@dataclass(frozen=True)
class SourceComplexion:
    """
    The source person's skin colour, as one LAB triple.

    Attributes:
        lab: Median (L, a, b) over the skin of every usable photograph, in
            OpenCV's 8-bit LAB convention
        photographs: How many photographs contributed
        spread: Median chroma distance of each photograph from the consensus.
            The disagreement between uploads — their white balance, mostly —
            and therefore the resolution `complexion_face` can be read at
    """

    lab: np.ndarray
    photographs: int
    spread: float


@dataclass(frozen=True)
class ComplexionReading:
    """One measured frame's complexion, in the units described above."""

    gap: float
    face: float
    target: float
    lightness: float
    neck: Optional[float] = None
    seam: Optional[float] = None

    def as_readings(self) -> Dict[str, float]:
        """The reading under the names the REALISM block reports."""
        out = {
            'complexion_gap': self.gap,
            'complexion_face': self.face,
            'complexion_target': self.target,
            'complexion_lum': self.lightness,
        }
        if self.neck is not None:
            out['complexion_neck'] = self.neck
        if self.seam is not None:
            out['complexion_seam'] = self.seam
        return out


# The Monk Skin Tone scale — ten steps, published as sRGB by Google's Skin
# Tone Research (skintone.google), CC-BY. Chosen over Fitzpatrick because it
# was designed for camera and image work and resolves dark tones where
# Fitzpatrick collapses above type IV. Stored as hex and converted once into
# the 8-bit LAB convention every other number here uses.
MST_SRGB: Dict[str, str] = {
    'mst01': 'f6ede4', 'mst02': 'f3e7db', 'mst03': 'f7ead0', 'mst04': 'eadaba',
    'mst05': 'd7bd96', 'mst06': 'a07e56', 'mst07': '825c43', 'mst08': '604134',
    'mst09': '3a312a', 'mst10': '292420',
}

BASE_AUTO = 'auto'
BASES: Tuple[str, ...] = (BASE_AUTO,) + tuple(MST_SRGB)

# How far the photographs may pull the baseline toward their own undertone, in
# a/b units. Inside it the photographs are trusted for the fine tone; past it
# the two references disagree about the person, and the dropdown wins.
UNDERTONE_BOUND = 6.0


def base_lab(base: str) -> Optional[np.ndarray]:
    """The LAB triple for a baseline name, or None for `auto` and unknowns."""
    hexcode = MST_SRGB.get((base or '').strip().lower())
    if hexcode is None:
        return None
    r, g, b = (int(hexcode[i:i + 2], 16) for i in (0, 2, 4))
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)


@dataclass(frozen=True)
class Reference:
    """
    The complexion a frame is graded toward, and where it came from.

    Attributes:
        lab: The effective (L, a, b)
        base: The baseline name, `auto` when the photographs alone decided
        disagreement: Chroma distance between the baseline and the
            photographs, when both exist. Past `UNDERTONE_BOUND` the dropdown
            and the uploads disagree about the person — one of them is wrong,
            and the operator should hear so before a session
    """

    lab: np.ndarray
    base: str
    disagreement: Optional[float]

    @property
    def disagrees(self) -> bool:
        """Whether the baseline and the photographs conflict."""
        return self.disagreement is not None and self.disagreement > UNDERTONE_BOUND


def resolve_reference(
    source: Optional[SourceComplexion], base: Optional[str],
) -> Optional[Reference]:
    """
    Combine the operator's baseline with the photographs' measured tone.

        dropdown  ->  baseline L, a, b and a bound      coarse, stable
        photos    ->  undertone offset within the bound  fine, measured
        past the bound  ->  the dropdown wins, and says so

    A prior rather than a destination: undertone — warm, cool, olive — lives
    inside a scale step, and snapping to the step would erase the thing that
    makes a complexion someone's. `auto` is the photographs alone, which is
    the behaviour that existed before the baseline did.

    Args:
        source: The measured reference, if any photographs yielded skin
        base: `auto`, or an `MST_SRGB` key

    Returns:
        The effective reference, or None when neither exists
    """
    anchor = base_lab(base or BASE_AUTO)
    if anchor is None:
        if source is None:
            return None
        return Reference(lab=source.lab.copy(), base=BASE_AUTO, disagreement=None)

    name = (base or '').strip().lower()
    if source is None:
        return Reference(lab=anchor, base=name, disagreement=None)

    offset = source.lab[1:] - anchor[1:]
    distance = float(np.hypot(offset[0], offset[1]))
    if distance > UNDERTONE_BOUND:
        offset = offset * (UNDERTONE_BOUND / distance)
    effective = anchor.copy()
    effective[1:] += offset
    return Reference(lab=effective, base=name, disagreement=distance)


def skin_lab(image: Frame, face: Face, size: int = MEASURE_SIZE) -> Optional[np.ndarray]:
    """
    Median LAB of the skin in one image, for one detected face.

    Args:
        image: BGR frame or photograph
        face: The detection made on it, for its keypoints and landmarks
        size: Canonical edge to measure at

    Returns:
        (L, a, b) as float64 in OpenCV's 8-bit convention, or None if the
        geometry cannot be fitted or too little skin is visible
    """
    matrix = canonical_from_frame(face, size)
    if matrix is None:
        return None

    crop = cv2.warpAffine(
        image, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE,
    )
    inside = skin_mask(face, matrix, size) > 0.5
    if int(inside.sum()) < _MIN_PIXELS:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    pixels = lab[inside].astype(np.float64)

    lit = (pixels[:, 0] > _L_FLOOR) & (pixels[:, 0] < _L_CEILING)
    if int(lit.sum()) >= _MIN_PIXELS:
        pixels = pixels[lit]

    return np.median(pixels, axis=0)


def masked_lab(image: Frame, mask: 'npt.NDArray[np.float32]') -> Optional[np.ndarray]:
    """
    Median LAB under a frame-space mask — the body-skin reading.

    Args:
        image: BGR frame
        mask: Float mask in [0, 1], frame-sized; pixels above 0.5 count

    Returns:
        (L, a, b) as float64, or None when too little is masked
    """
    inside = mask > 0.5
    if int(inside.sum()) < _MIN_PIXELS:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[inside].astype(np.float64)
    lit = (pixels[:, 0] > _L_FLOOR) & (pixels[:, 0] < _L_CEILING)
    if int(lit.sum()) >= _MIN_PIXELS:
        pixels = pixels[lit]
    return np.median(pixels, axis=0)


def chroma_distance(a: 'npt.ArrayLike', b: 'npt.ArrayLike') -> float:
    """Euclidean distance in the a/b plane — the pigment, not the light."""
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    return float(np.hypot(first[1] - second[1], first[2] - second[2]))


def measure_source(
    photographs: Iterable[Tuple[Union[str, Frame], Face]],
) -> Optional[SourceComplexion]:
    """
    Build the source reference from every usable photograph.

    Args:
        photographs: (path or frame, face) pairs — the accepted source set with
            the detections the review already made

    Returns:
        The consensus complexion, or None when no photograph yielded skin
    """
    samples: List[np.ndarray] = []
    for image, face in photographs:
        frame = cv2.imread(image) if isinstance(image, str) else image
        if frame is None:
            continue
        sample = skin_lab(frame, face)
        if sample is not None:
            samples.append(sample)

    if not samples:
        return None

    stack = np.stack(samples)
    consensus = np.median(stack, axis=0)
    spread = float(np.median([chroma_distance(s, consensus) for s in samples]))
    return SourceComplexion(lab=consensus, photographs=len(samples), spread=spread)


def measure_output(
    frame: Frame,
    pasted: Frame,
    face: Face,
    source: SourceComplexion,
    body: Optional['npt.NDArray[np.float32]'] = None,
) -> Optional[ComplexionReading]:
    """
    Score the finished frame's face skin against the source and the target.

    Args:
        frame: The target frame before the swap
        pasted: The finished frame
        face: The target detection, whose landmarks bound the skin in both
        source: The reference built by `measure_source`
        body: The skin segmenter's body mask for this frame, if one ran —
            classified skin outside the face. Adds the neck and seam readings

    Returns:
        The reading, or None when face skin could not be measured in either
        frame. The neck and seam are None when no body skin was visible
    """
    before = skin_lab(frame, face)
    after = skin_lab(pasted, face)
    if before is None or after is None:
        return None

    neck = seam = None
    if body is not None:
        rest = masked_lab(pasted, body)
        if rest is not None:
            neck = chroma_distance(rest, source.lab)
            seam = chroma_distance(after, rest)

    return ComplexionReading(
        gap=chroma_distance(source.lab, before),
        face=chroma_distance(after, source.lab),
        target=chroma_distance(after, before),
        lightness=float(after[0]) - float(source.lab[0]),
        neck=neck,
        seam=seam,
    )
