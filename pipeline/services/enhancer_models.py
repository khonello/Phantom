"""
Face-restoration model registry.

The sibling of `swapper_models.py`, and it exists for the same reason: the
model decides things about itself that nothing else can know, and hard-coding
one model's answers into the backend is how a second model becomes impossible
to add.

Two things live together here:

**Model spec** — facts about the weights. Backend family, native crop size,
whether it accepts a fidelity weight, where to download it. Not tunable.

**Look profile** — the appearance knobs whose correct value depends on *this*
model. `enhancer_weight` only means something to CodeFormer; GPEN has no such
input, so a value carried over from CodeFormer would silently mean nothing.

**Why `crop` is the field that matters.** Restoration is the most expensive
stage in the pipeline and the only one whose cost ignores how big the face
actually is — it runs on a fixed crop either way. Measured on an RTX 4090,
inference alone:

    codeformer     512px   29.4ms   377 MB
    gpen_bfr_512   512px   37.5ms   284 MB
    gpen_bfr_256   256px    5.4ms    76 MB

Note which way `gpen_bfr_512` falls. It is **slower** than CodeFormer at the
same resolution, so the saving is entirely resolution and not architecture.
GPEN is not a lighter model; 256 is a quarter of the pixels.

The pipeline does not have to be told any of this. `Enhancer.crop_size` reads
the declared input shape from the ONNX graph itself and the compositor builds
its FFHQ crop at whatever size comes back, so a 256 model needs no change in
`compositor.py`. `crop` here is documentation and a pre-flight expectation, not
the thing that drives the geometry.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# Same host and release the swapper weights come from. Note the tag differs:
# the restoration models live under `models-3.0.0`, while hyperswap needs
# `models-3.3.0` — 3.3.0 carries no restoration assets at all. Verified
# 2026-08-30 by listing both releases.
_ASSETS = 'https://github.com/facefusion/facefusion-assets/releases/download'
_RESTORE_TAG = 'models-3.0.0'


@dataclass(frozen=True)
class EnhancerModel:
    """
    One face-restoration model: how to load it, and how to finish its output.

    Attributes:
        name: Registry key, and the config value that selects it
        backend: Inference family — `codeformer` for the ONNX path (which also
                 runs GPEN, since that path introspects its inputs rather than
                 assuming them), `gfpgan` for the torch package
        crop: Native FFHQ crop edge the weights were trained on. Authoritative
              only as an expectation — `Enhancer.crop_size` reads the real
              value out of the graph
        fidelity: Whether the model accepts a fidelity weight. False means
                  `enhancer_weight` has nothing to act on, and the only
                  remaining control is the compositor-side `enhance_strength`
        filename: Local weight filename
        url: Download URL, or empty when the weights are fetched another way
        enhance_strength: How much of the restored face to blend back in
        notes: Why this profile differs from the others
    """

    name: str
    backend: str
    crop: int
    fidelity: bool
    filename: str
    url: str

    enhance_strength: float

    notes: str = ''

    def look(self) -> Dict[str, float]:
        """
        The appearance knobs this model wants, as a config overlay.

        `enhancer_weight` is deliberately absent: it is CodeFormer's fidelity
        input, and setting it for a model without one would look like a
        configured value while doing nothing at all.

        Returns:
            Field name -> value, applied after the quality preset
        """
        return {'enhance_strength': self.enhance_strength}


ENHANCER_MODELS: Dict[str, EnhancerModel] = {

    # ── The small one, and the reason this registry exists ────────────────────
    'gpen_bfr_256': EnhancerModel(
        name='gpen_bfr_256',
        backend='codeformer',
        crop=256,
        fidelity=False,
        filename='gpen_bfr_256.onnx',
        url='{}/{}/gpen_bfr_256.onnx'.format(_ASSETS, _RESTORE_TAG),
        enhance_strength=0.7,
        notes=(
            '5.4ms against CodeFormer 29.4ms on a 4090, and the reason is the '
            'crop rather than the architecture. Restoring at 512 and warping '
            'down into a 128-192 aligned space discards ~86% of what the model '
            'produced, one warp after it was made; at 256 the supersampling '
            'margin over the aligned size survives and so does all the '
            'low-frequency work — tone, structure, artefact cleanup — that a '
            'downsample does not destroy. What is given up is the 512-to-256 '
            'octave, which the final resize into a ~101px face deletes anyway. '
            'Has no fidelity input, so enhancer_weight stops meaning anything '
            'and enhance_strength is the only remaining control. '
            'ADOPTED ON SPEED EVIDENCE ONLY — never yet judged on footage.'
        ),
    ),

    # ── The incumbent ────────────────────────────────────────────────────────
    'codeformer': EnhancerModel(
        name='codeformer',
        backend='codeformer',
        crop=512,
        fidelity=True,
        filename='codeformer.onnx',
        url='{}/{}/codeformer.onnx'.format(_ASSETS, _RESTORE_TAG),
        enhance_strength=0.7,
        notes=(
            'The only registered model with a fidelity weight, which is what '
            'made it the default over GFPGAN: 0 restores hardest and '
            'hallucinates most, 1 stays closest to the input, and that is the '
            'axis believability actually lives on. Its graph is static '
            '[1, 3, 512, 512], so it cannot be asked to restore smaller — that '
            'needs a different model, not a config change. Measured worth on a '
            '101px face: +0.03 on the face/frame detail ratio, for 29.4ms.'
        ),
    ),

    # ── Kept for comparison ──────────────────────────────────────────────────
    'gpen_bfr_512': EnhancerModel(
        name='gpen_bfr_512',
        backend='codeformer',
        crop=512,
        fidelity=False,
        filename='gpen_bfr_512.onnx',
        url='{}/{}/gpen_bfr_512.onnx'.format(_ASSETS, _RESTORE_TAG),
        enhance_strength=0.7,
        notes=(
            'Registered mainly so the 256 result cannot be misread as "GPEN is '
            'faster". At 512 it is 37.5ms — slower than CodeFormer, and '
            'without a fidelity weight. The comparison that isolates '
            'resolution from architecture.'
        ),
    ),

    'gfpgan': EnhancerModel(
        name='gfpgan',
        backend='gfpgan',
        crop=512,
        fidelity=False,
        filename='GFPGANv1.4.pth',
        url='',
        enhance_strength=0.7,
        notes=(
            'The previous default, kept so the two can be compared on real '
            'footage. Needs torch plus the gfpgan package. Restores toward a '
            'beautified, poreless look with no way to dial it back, and that '
            'plastic skin is the strongest "this is AI" signal on a call.'
        ),
    ),
}

# The measured choice. CodeFormer buys +0.03 of detail ratio for 29.4ms on a
# 101px face, which is the normal operating case at 640x360 — so the default is
# the model that keeps the part of restoration that survives compositing and
# stops paying for the part that does not.
DEFAULT_ENHANCER_MODEL = 'gpen_bfr_256'


def resolve(name: str) -> EnhancerModel:
    """
    Look up a model by name.

    Args:
        name: Registry key

    Returns:
        The model spec, falling back to the default for an unknown name — a
        typo in `.env` degrades to the default rather than taking down a pod
        that has already been paid for.
    """
    return ENHANCER_MODELS.get(name, ENHANCER_MODELS[DEFAULT_ENHANCER_MODEL])


def names() -> Tuple[str, ...]:
    """
    Every registered model name.

    Returns:
        Names in registry order, for CLI choices and the API schema
    """
    return tuple(ENHANCER_MODELS)
