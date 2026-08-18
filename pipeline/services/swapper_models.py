"""
Face-swap model registry, with the realism profile each model needs.

Two things live together here, and the distinction matters:

**Model spec** — facts about the weights. Type, alignment template, native
output size, input normalisation, where to download it. Not tunable; getting
any of them wrong produces garbage, not a different look.

**Look profile** — the realism knobs whose *correct value depends on the model*.
These were previously in `PRESETS._LOOK`, identical across every preset, because
a quality preset decides how much compute to spend and deliberately does not
change how the face looks. That reasoning holds — but it puts them in the wrong
place. How much restoration a face needs is a property of **what generated it**,
not of the frame rate.

Which matters because the current values were tuned around a 128px swap. Feed
them a native 256 face and they are wrong in a specific, predictable direction:
CodeFormer over-restores something that no longer needs rescuing. Switching
model without switching profile would make a better model look worse, and invite
exactly the wrong conclusion.

Ownership, so the three layers never fight:

    quality preset  ->  compute      (capture, det_size, aligned ceiling, EMA)
    model profile   ->  appearance   (restoration burden, aligned floor)
    CLI / env       ->  explicit operator override of either
    set_realism     ->  live A/B on top of all of it
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# facefusion publishes its ONNX weights as GitHub release assets. Same host the
# occluder already downloads from (see services/masking.py).
#
# Verified reachable on 2026-08-18: all three hyperswap URLs below serve a range
# request (HTTP 206) and the payload is a real ONNX protobuf — header
# `pytorch`, ir_version 8. Each is **384 MB**, and all three are
# byte-identical in size, which is expected for siblings sharing an architecture
# and differing only in weights.
#
# The release tag is load-bearing. `models-3.3.0` serves them; `models-3.0.0`
# and `models-3.4.0` both 404. Bumping the tag without re-checking would break
# the download at pod-provision time, which is the most expensive place to find
# out — the same class of failure as the dead `runtime` image tag.
#
# 384 MB is also a cold-start cost: three variants pulled for a comparison run
# is ~1.2 GB before the pipeline serves a frame. Pull the one being measured,
# not all three, unless the volume is being pre-seeded.
_ASSETS = 'https://github.com/facefusion/facefusion-assets/releases/download'

# Bytes, for pre-seed budgeting and for spotting a truncated download.
HYPERSWAP_SIZE_BYTES = 402742682


@dataclass(frozen=True)
class SwapperModel:
    """
    One face-swap model: how to run it, and how to finish its output.

    Attributes:
        name: Registry key, and the config value that selects it
        kind: Inference family — decides how the source is prepared. `inswapper`
              and `hyperswap` both take an ArcFace *embedding*; `blendswap` and
              `uniface` take a source *image* and are deliberately absent, since
              an image source would break multi-photo embedding averaging,
              `.npy` embeddings and the identity-outlier guard
        template: Alignment template for the target crop. Both registered models
                  use `arcface_128`, which is why swapping between them needs no
                  change to the compositor, masker or guards
        size: Native output edge in pixels. The single most consequential number
              here — everything downstream is either using this detail or
              inventing what is missing
        mean: Per-channel input mean for normalisation
        standard_deviation: Per-channel input standard deviation
        filename: Local weight filename
        url: Download URL, or empty when the model is fetched by InsightFace
        enhancer_weight: CodeFormer fidelity. 0 restores hardest and hallucinates
                         most; 1 stays closest to the input
        enhance_strength: How much of the restored face to blend back in
        aligned_min: Floor on compositing resolution. Compositing below a model's
                     native size throws away output it already generated
        notes: Why this profile differs from the others
    """

    name: str
    kind: str
    template: str
    size: int
    mean: Tuple[float, float, float]
    standard_deviation: Tuple[float, float, float]
    filename: str
    url: str

    enhancer_weight: float
    enhance_strength: float
    aligned_min: int

    notes: str = ''

    def look(self) -> Dict[str, float]:
        """
        The appearance knobs this model wants, as a config overlay.

        Returns:
            Field name -> value, applied after the quality preset
        """
        return {
            'enhancer_weight': self.enhancer_weight,
            'enhance_strength': self.enhance_strength,
            'aligned_min': float(self.aligned_min),
        }


SWAPPER_MODELS: Dict[str, SwapperModel] = {

    # ── The incumbent ────────────────────────────────────────────────────────
    'inswapper_128': SwapperModel(
        name='inswapper_128',
        kind='inswapper',
        template='arcface_128',
        size=128,
        mean=(0.0, 0.0, 0.0),
        standard_deviation=(1.0, 1.0, 1.0),
        filename='inswapper_128.onnx',
        url='',  # resolved locally / by InsightFace, see FaceSwapper
        # Tuned around a 128px swap upsampled to 192-320 for compositing. The
        # restorer is doing heavy lifting here: it is inventing most of the
        # detail the output appears to have, and `enhance_strength` at 0.7 is
        # what keeps that invention from reading as AI.
        enhancer_weight=0.7,
        enhance_strength=0.7,
        aligned_min=128,
        notes='128px native. Restoration supplies most apparent detail.',
    ),

    # ── 256px, same source contract and same alignment ───────────────────────
    # facefusion's current default, which is a meaningful signal about where
    # that project landed after comparing all of them.
    #
    # The three variants differ in training, not interface, so registering all
    # three costs nothing beyond three lines and lets one session compare them.
    'hyperswap_1a_256': SwapperModel(
        name='hyperswap_1a_256',
        kind='hyperswap',
        template='arcface_128',
        size=256,
        mean=(0.5, 0.5, 0.5),
        standard_deviation=(0.5, 0.5, 0.5),
        filename='hyperswap_1a_256.onnx',
        url='{}/models-3.3.0/hyperswap_1a_256.onnx'.format(_ASSETS),
        # Both values move for one mechanical reason: at 256 native there is
        # real detail where there used to be upsampled guesswork, so the
        # restorer has less to fix and should be trusted less.
        #
        #   enhancer_weight 0.7 -> 0.8   stay closer to an input that is better
        #   enhance_strength 0.7 -> 0.5  blend in less of the restored face
        #
        # Direction is mechanical; the magnitudes are starting points, not
        # measured truth. Sweep them with `set_realism` against a fixed clip.
        enhancer_weight=0.8,
        enhance_strength=0.5,
        # Compositing below 256 would discard half of what the model produced.
        aligned_min=256,
        notes='256px native. Less restoration needed; floor raised to native size.',
    ),

    'hyperswap_1b_256': SwapperModel(
        name='hyperswap_1b_256',
        kind='hyperswap',
        template='arcface_128',
        size=256,
        mean=(0.5, 0.5, 0.5),
        standard_deviation=(0.5, 0.5, 0.5),
        filename='hyperswap_1b_256.onnx',
        url='{}/models-3.3.0/hyperswap_1b_256.onnx'.format(_ASSETS),
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='Sibling of 1a; differs in training, not interface.',
    ),

    'hyperswap_1c_256': SwapperModel(
        name='hyperswap_1c_256',
        kind='hyperswap',
        template='arcface_128',
        size=256,
        mean=(0.5, 0.5, 0.5),
        standard_deviation=(0.5, 0.5, 0.5),
        filename='hyperswap_1c_256.onnx',
        url='{}/models-3.3.0/hyperswap_1c_256.onnx'.format(_ASSETS),
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='Sibling of 1a; differs in training, not interface.',
    ),
}

DEFAULT_SWAPPER_MODEL = 'inswapper_128'


def resolve(name: str) -> SwapperModel:
    """
    Look up a model by name.

    Args:
        name: Registry key

    Returns:
        The model spec, falling back to the default for an unknown name

    Raises:
        KeyError: never — an unknown name falls back rather than failing, so a
                  typo in `.env` degrades to the incumbent instead of taking a
                  pod down after it has already been paid for
    """
    return SWAPPER_MODELS.get(name, SWAPPER_MODELS[DEFAULT_SWAPPER_MODEL])


def names() -> Tuple[str, ...]:
    """
    Every registered model name.

    Returns:
        Names in registry order, for CLI choices and the API schema
    """
    return tuple(SWAPPER_MODELS)
