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
# The release tag is load-bearing, and it differs per model: `models-3.0.0`,
# `models-3.1.0`, `models-3.4.0` and `models-3.9.0` are all in use below, and an
# asset served under one 404s under another. Bumping a tag without re-checking
# breaks the download at pod-provision time, which is the most expensive place
# to find out — the same class of failure as the dead `runtime` image tag.
#
# Removed 2026-09-09: **hyperswap_1a/1b/1c_256**. facefusion's own default, and
# wrong for this product on the one axis that decides it. It is tuned to blend
# well and respect the target, which is the opposite of low target leakage; it
# measured 62.2ms/frame against inswapper's 58.9 on a 4090, so it was slower
# too; and judged by eye on real footage it was the worst of the three tried.
# Kept in git, not in the registry — three 384 MB entries nobody should reach
# for is a trap, not an option.
_ASSETS = 'https://github.com/facefusion/facefusion-assets/releases/download'

# Bytes, for pre-seed budgeting and for spotting a truncated download. Verified
# by range request against the release assets on 2026-09-08.
#
# The whole registry is ~4.6 GB if every model is pulled. That is a `VAST_DISK`
# cost on a rented instance and a cold-start cost on an empty volume, so pull
# the one being measured rather than the set.
HIFIFACE_SIZE_BYTES = 203784742
ALPHAFACE_SIZE_BYTES = 555624110
GHOST_1_SIZE_BYTES = 514902950
SIMSWAP_256_SIZE_BYTES = 220368724
SIMSWAP_512_SIZE_BYTES = 239249034
# blendswap is by far the largest weight here at 1.6 GB — worth knowing
# before pre-seeding a volume, and worth not pulling speculatively.
BLENDSWAP_SIZE_BYTES = 1661432957
UNIFACE_SIZE_BYTES = 406964143

# The `crossface_*` converters are one architecture with per-model weights, so
# every file is the same size. Do not collapse them to one download.
CROSSFACE_SIZE_BYTES = 22083800


@dataclass(frozen=True)
class SwapperModel:
    """
    One face-swap model: how to run it, and how to finish its output.

    Attributes:
        name: Registry key, and the config value that selects it
        kind: Inference family — decides how the source is prepared. Most take
              an ArcFace *embedding*; `blendswap` and `uniface` take a source
              *image*. See `source_kind`, which is the field that actually
              carries that distinction
        template: Alignment template for the **target** crop. Five are in use
                  across the registry; everything downstream works off the
                  returned affine, so a new template needs no change to the
                  compositor, masker or guards
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

    # Bytes, for the download message and for spotting a truncated file. Zero
    # when unknown, which is the inswapper case — InsightFace fetches it.
    size_bytes: int = 0

    # Some models were not trained against ArcFace's embedding space and need a
    # small learned map into their own. It runs **once per source**, not per
    # frame, so it is free on the live path — but it is a second weight file,
    # and a model that silently ran without it would produce a face that is
    # confidently the wrong person, which is the exact failure the source guards
    # exist to prevent. Empty means the model takes the ArcFace vector directly.
    converter_filename: str = ''
    converter_url: str = ''
    converter_size_bytes: int = 0

    # What this model is conditioned on: `'embedding'` or `'image'`.
    #
    # This registry used to assume the first, and excluded `blendswap` and
    # `uniface` on the grounds that an image source "would break multi-photo
    # averaging, .npy embeddings and the identity-outlier guard". That was an
    # architecture deciding which models were allowed to exist. It is the wrong
    # way round: the source contract is a fact about the weights, and the
    # pipeline should meet each model where it is.
    #
    # None of the three worries survives contact. The guards run at **upload**,
    # over every photograph, and are untouched by what the swapper is later
    # handed. Averaging is not broken, it is inapplicable — an image model gets
    # `select_texture_source`'s single best photograph, chosen on sharpness,
    # size, frontality and clipping, which is the same picker the texture layer
    # and the studio backends already use and a better answer than "the first
    # path". And an all-`.npy` source set genuinely cannot feed an image model,
    # which is said once rather than discovered as a bad swap.
    source_kind: str = 'embedding'

    # For `source_kind == 'image'`: the framing and edge length the **source**
    # crop must be in. Named separately from `template`/`size` because a model
    # can want different spaces for its two inputs — blendswap reads its target
    # in FFHQ framing and its source in arcface_112_v2.
    source_template: str = ''
    source_size: int = 0

    # Which form of the ArcFace vector this model — or its converter — was
    # fitted on. `'raw'` is the unnormalised vector of norm ~22, `'normed'` the
    # unit one. It matters because a converter is a non-linear map: feeding it a
    # unit vector when it was fitted on a raw one is not a scaling difference,
    # it is a different input. It also separates the two converter-less
    # families, which disagree — inswapper is conditioned on the unit vector
    # and alphaface on the raw one.
    source_form: str = 'normed'

    # Whether to L2-normalise the vector finally handed to the session.
    #
    # False for exactly two reasons and they are different. `alphaface` is
    # conditioned on the raw embedding and its magnitude is part of the signal.
    # `ghost` reads the **unnormalised** output of its converter — facefusion
    # computes both forms and hands ghost the un-normalised one while every
    # other converter model gets the normalised one, which is the kind of
    # detail that produces a washed-out identity rather than an error.
    normalise_source: bool = True

    # Whether the model's output needs the input normalisation undone.
    #
    # A property of the export, not of the numbers: `simswap` is fed ImageNet
    # mean and deviation but emits [0, 1] directly, so applying the inverse
    # would tint and stretch a correct image. Mirrors the model-type list in
    # facefusion's `normalize_crop_frame`.
    denormalize_output: bool = True

    # Which tier this belongs to, derived from two declared facts rather than
    # written down — see `pipeline/services/tiers.py`. Every model in this
    # registry is general-purpose and holds a frame deadline, so both are
    # constant here; they are fields rather than a hard-coded tier so that a
    # future entry which is neither cannot quietly inherit the wrong one, and
    # so all three registries answer `require_live` the same way.
    live_capable: bool = True
    needs_training: bool = False

    @property
    def tier(self) -> str:
        """
        The tier this model belongs to.

        Returns:
            A key from `tiers.TIERS`, derived so it cannot disagree with the
            two facts above
        """
        from pipeline.services import tiers
        return tiers.classify(self.live_capable, self.needs_training)

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

    # ── 3D-shape supervised, and the first model here that is not arcface ────
    #
    # HiFiFace (Wang et al., IJCAI 2021) is the reason this entry exists: it is
    # trained with a 3DMM in the loop, recombining the *source's* identity
    # coefficients with the *target's* expression and pose, so the generator
    # learns to move the face **contour** toward the source rather than only
    # repainting the interior. Face outline is one of the strongest identity
    # cues a viewer has, and it is the one thing every other model here leaves
    # at the target's.
    #
    # Three things about that claim are worth stating precisely, because two of
    # them are easy to over-read:
    #
    # 1. **The 3D is training-time.** This export takes an embedding and a crop,
    #    nothing else — there is no 3DMM fit at inference and no per-frame
    #    reconstruction cost. What ships is a generator that learned shape
    #    awareness, not one that computes it.
    # 2. **The compositor has to allow it.** The mask here is the convex hull of
    #    the *target's* landmarks, so a contour this model widens is clipped
    #    straight back off. See `mask_shape_growth` — without it, roughly half
    #    of what this entry is for never reaches the screen.
    # 3. **`mtcnn_512`, not `arcface_128`.** The first model registered here
    #    that needs its own template. Everything downstream works off the
    #    returned affine, so nothing else changes — but assuming the arcface
    #    framing would feed it a crop ~6% off in y, which degrades quietly
    #    rather than failing.
    #
    # Two weight files. The converter maps ArcFace's embedding space into the
    # recognition space this model was trained against, and runs once per
    # source. Note the model itself is tagged `models-3.1.0`, not the
    # `models-3.0.0` most of this registry comes from — the asset does not
    # exist under other tags, and bumping it would 404 at pod-provision time,
    # which is the most expensive place to find out.
    #
    # The **converter** is `crossface_hififace` from `models-3.4.0`, which is
    # what facefusion moved to; it replaced the `arcface_converter_hififace`
    # this entry originally carried. Same interface, same one-shot cost, and it
    # is the maintained one — but it is a different map, so a hififace result
    # measured before this change is not comparable with one measured after.
    'hififace_unofficial_256': SwapperModel(
        name='hififace_unofficial_256',
        kind='hififace',
        template='mtcnn_512',
        size=256,
        mean=(0.5, 0.5, 0.5),
        standard_deviation=(0.5, 0.5, 0.5),
        filename='hififace_unofficial_256.onnx',
        url='{}/models-3.1.0/hififace_unofficial_256.onnx'.format(_ASSETS),
        size_bytes=HIFIFACE_SIZE_BYTES,
        converter_filename='crossface_hififace.onnx',
        converter_url='{}/models-3.4.0/crossface_hififace.onnx'.format(_ASSETS),
        converter_size_bytes=CROSSFACE_SIZE_BYTES,
        source_form='raw',
        # 256 native, so there is real detail where inswapper had upsampled
        # guesswork and the restorer should be trusted less. Starting points,
        # not measured: sweep them against a fixed clip with `identity_probe`
        # on, which is the whole reason that reading exists.
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='256px native, 3D-shape supervised. Needs mask_shape_growth to '
              'deliver the contour it generates.',
    ),

    # ── Identity injected at every scale, not only the bottleneck ────────────
    #
    # AlphaFace (arXiv 2601.16429, 2026). The reason it is registered is its
    # conditioning: *Cross-Adaptive Identity Injection* applies the source code
    # at every encoder stage rather than once at the bottleneck, which is the
    # mechanism that suppresses target leakage rather than a claim about it.
    # Reported 98.77 ID retrieval on FF++ at 24.1ms, and — the number that
    # matters more here — 0.471 CSIM on the pose-hard MPIE set against
    # FaceDancer 0.401, BlendFace 0.392, SimSwap 0.180, HifiFace 0.092.
    #
    # Two conventions differ from every other entry and both are silent
    # failures rather than errors:
    #
    # 1. It is conditioned on the **raw** ArcFace embedding, unnormalised. A
    #    unit vector produces a weaker, blander identity — exactly the
    #    degradation that gets blamed on the model.
    # 2. Its output is already in [0, 1], so the input normalisation is **not**
    #    undone. Harmless here since mean is 0 and deviation 1, but stated
    #    rather than relied upon.
    #
    # Licence is non-commercial, which is the same constraint inswapper already
    # carries — see the note on ghost, which is the only permissive entry.
    'alphaface_256': SwapperModel(
        name='alphaface_256',
        kind='alphaface',
        template='arcface_128',
        size=256,
        mean=(0.0, 0.0, 0.0),
        standard_deviation=(1.0, 1.0, 1.0),
        filename='alphaface_256.onnx',
        url='{}/models-3.9.0/alphaface_256.onnx'.format(_ASSETS),
        size_bytes=ALPHAFACE_SIZE_BYTES,
        source_form='raw',
        normalise_source=False,
        denormalize_output=False,
        # 256 native, so the restorer has less to repair and should be trusted
        # less. Starting points, not measured.
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='256px native. Identity injected at every encoder stage, which '
              'is the reason to try it. Raw embedding, no output denorm.',
    ),

    # ── Apache-2.0, and the only entry that is ───────────────────────────────
    #
    # GHOST (ai-forever). Registered as much for its licence as its numbers:
    # every other model here is non-commercial, ResearchRAIL or unknown, and
    # this product ships to paying customers. Identity is middling — the 2026
    # survey does not rank it near inswapper — so treat it as the answer to
    # "what could we actually ship", not to "what looks most like the source".
    #
    # The trap is its converter output. facefusion computes both the raw and
    # the normalised form and hands **ghost the un-normalised one**, while
    # hififace and simswap get the normalised one. Same converter architecture,
    # opposite convention, no error either way.
    #
    # Only variant 1 is registered. Upstream ships three (491/704/816 MB)
    # differing in training, not interface, and 2 and 3 were dropped for two
    # reasons: nothing here rates ghost on identity, so three untested
    # siblings of a low-expectation model is 1.5 GB of catalogue inviting a
    # wasted pod session; and `ghost_2_256` read confusingly beside the
    # STUDIO backend `ghost_2`, which is GHOST 2.0 and a different thing
    # entirely. Re-add them from git if ghost ever earns a measurement.
    'ghost_1_256': SwapperModel(
        name='ghost_1_256',
        kind='ghost',
        template='arcface_112_v1',
        size=256,
        mean=(0.5, 0.5, 0.5),
        standard_deviation=(0.5, 0.5, 0.5),
        filename='ghost_1_256.onnx',
        url='{}/models-3.0.0/ghost_1_256.onnx'.format(_ASSETS),
        size_bytes=GHOST_1_SIZE_BYTES,
        converter_filename='crossface_ghost.onnx',
        converter_url='{}/models-3.4.0/crossface_ghost.onnx'.format(_ASSETS),
        converter_size_bytes=CROSSFACE_SIZE_BYTES,
        source_form='raw',
        normalise_source=False,
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='256px native, Apache-2.0 — the ONLY permissively licensed '
              'model here, which is why it survives a prune it does not '
              'earn on identity. Converter output is used un-normalised.',
    ),

    # ── Registered to be falsified, and for the 512 ──────────────────────────
    #
    # SimSwap is the weakest bet in this registry on the one axis that matters
    # here: the 2026 survey puts it at 0.61 ID similarity against inswapper's
    # 0.73, and its documented failure is precisely target leakage — it matches
    # attributes well and does **not** carry the source's face shape. It is
    # registered anyway because it is ~20 lines and because
    # `simswap_unofficial_512` is the only 512-native swapper available, which
    # is a different axis (detail) from the one it is expected to lose on.
    #
    # Note the two variants disagree about normalisation: 256 is fed ImageNet
    # mean and deviation, 512 is fed [0, 1]. Neither undoes it on the way out —
    # applying the inverse to the 256 would tint and stretch a correct image.
    'simswap_256': SwapperModel(
        name='simswap_256',
        kind='simswap',
        template='arcface_112_v1',
        size=256,
        mean=(0.485, 0.456, 0.406),
        standard_deviation=(0.229, 0.224, 0.225),
        filename='simswap_256.onnx',
        url='{}/models-3.0.0/simswap_256.onnx'.format(_ASSETS),
        size_bytes=SIMSWAP_256_SIZE_BYTES,
        converter_filename='crossface_simswap.onnx',
        converter_url='{}/models-3.4.0/crossface_simswap.onnx'.format(_ASSETS),
        converter_size_bytes=CROSSFACE_SIZE_BYTES,
        source_form='raw',
        denormalize_output=False,
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='256px native, ImageNet input normalisation, [0, 1] output.',
    ),

    # ── Conditioned on a picture, not a vector ───────────────────────────────
    #
    # These take the source **image**. Registered because excluding them was a
    # decision made by this pipeline's averaging rather than by their quality —
    # see `source_kind`. Whether either is worth using is now a question that
    # can be answered on footage instead of one closed by architecture.
    #
    # What actually changes for them: the identity comes from one photograph,
    # so it is sharper and less "typical" than an average, and it carries
    # whatever that photograph's pose and lighting carry. That is the trade,
    # and it is the same trade the texture layer already makes deliberately.
    #
    # Note each wants a *different* space for its source crop from the one it
    # wants for its target, which is the detail that would silently produce a
    # weak identity if assumed.
    'blendswap_256': SwapperModel(
        name='blendswap_256',
        kind='blendswap',
        template='ffhq_512',
        size=256,
        mean=(0.0, 0.0, 0.0),
        standard_deviation=(1.0, 1.0, 1.0),
        filename='blendswap_256.onnx',
        url='{}/models-3.0.0/blendswap_256.onnx'.format(_ASSETS),
        size_bytes=BLENDSWAP_SIZE_BYTES,
        source_kind='image',
        source_template='arcface_112_v2',
        source_size=112,
        denormalize_output=False,
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='256px native, source is a 112px arcface_112_v2 crop.',
    ),

    'uniface_256': SwapperModel(
        name='uniface_256',
        kind='uniface',
        template='ffhq_512',
        size=256,
        mean=(0.5, 0.5, 0.5),
        standard_deviation=(0.5, 0.5, 0.5),
        filename='uniface_256.onnx',
        url='{}/models-3.0.0/uniface_256.onnx'.format(_ASSETS),
        size_bytes=UNIFACE_SIZE_BYTES,
        source_kind='image',
        source_template='ffhq_512',
        source_size=256,
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=256,
        notes='256px native, source is a 256px FFHQ crop — a different space '
              'from blendswap, and from its own target crop size.',
    ),

    'simswap_unofficial_512': SwapperModel(
        name='simswap_unofficial_512',
        kind='simswap',
        template='arcface_112_v1',
        size=512,
        mean=(0.0, 0.0, 0.0),
        standard_deviation=(1.0, 1.0, 1.0),
        filename='simswap_unofficial_512.onnx',
        url='{}/models-3.0.0/simswap_unofficial_512.onnx'.format(_ASSETS),
        size_bytes=SIMSWAP_512_SIZE_BYTES,
        converter_filename='crossface_simswap.onnx',
        converter_url='{}/models-3.4.0/crossface_simswap.onnx'.format(_ASSETS),
        converter_size_bytes=CROSSFACE_SIZE_BYTES,
        source_form='raw',
        denormalize_output=False,
        # The only 512-native entry, so the floor is 512 — and note the floor
        # beats the quality preset's ceiling in `_aligned_size`, deliberately.
        # Every compositing stage therefore runs at 512 on every frame, which
        # is the most expensive configuration this registry can ask for. It is
        # a measurement config, not a live one.
        enhancer_weight=0.8,
        enhance_strength=0.5,
        aligned_min=512,
        notes='512px native — the only one. Forces 512 compositing regardless '
              'of preset, so it is expensive by construction.',
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
