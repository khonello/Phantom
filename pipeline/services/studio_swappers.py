"""
Registry of studio face-swap backends — the ones that cannot run on a call.

The sibling of `swapper_models.py`, and deliberately *not* an extension of it.
Everything in that registry is a single ONNX graph taking an identity vector and
an aligned crop, returning a crop for `FaceCompositor` to finish. Nothing here
is. These are whole pipelines that do their own detection, alignment, masking
and blending, and they return a finished picture.

That difference is the reason for a second registry rather than a `kind` on the
first one. Three consequences follow from it and all three are load-bearing:

- **They bypass the compositor entirely.** Running `_match_color`,
  `_match_detail` and the landmark-hull mask over a result that has already been
  blended would re-introduce exactly the target information these models exist
  to remove, and would clip a head swap back to a face swap. So the studio path
  is a *replacement* for compositing, not a stage in front of it.
- **They cannot hold a frame deadline.** The fastest of them is ~0.6s per
  image, against a 50ms live budget. `is_live_safe()` is False for every entry
  and `ProcessingPipeline` refuses them on the stream path — see the gate there.
  This is not a performance note, it is the only thing standing between a
  selected studio backend and a call that emits nothing.
- **They are separate processes, not imports.** Each ships as a repository to
  clone rather than a package to install, each vendors its own copy of a
  diffusion stack, and their requirements conflict with each other and with
  ours — REFace carries an old latent-diffusion tree, DreamID-V needs torch
  >= 2.4 with a forked Wan. One environment cannot satisfy all three, so each
  backend names its own interpreter and its own checkout, and an unconfigured
  one reports itself unavailable rather than failing at use.

Nothing here is bundled. Every path comes from the environment, because these
are tens of gigabytes of weights that must not land in an image build or a
clone — the same reasoning that keeps the template library and the ONNX weights
out of the repository.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class StudioSwapperModel:
    """
    One studio backend: what it consumes, what it needs, and what it costs.

    Attributes:
        name: Registry key, and the value `config.studio_swapper` selects with
        media: `'image'` or `'video'` — what the *target* must be. A backend is
               refused for the other kind rather than silently producing
               nothing, since these disagree: two of them swap a still and the
               third swaps a clip
        repo_env: Environment variable holding the repository checkout
        python_env: Environment variable holding the interpreter for that
                    checkout's own virtualenv. Falls back to the running
                    interpreter, which is almost certainly wrong and is
                    reported as such rather than assumed
        entrypoint: Script within the checkout, as it appears in its own docs
        resolution: Native output edge, for the log line
        seconds_per_item: Order-of-magnitude cost, for the log line and for the
                          refusal message on the live path. Not a measurement
        licence: Named because this ships to paying customers and two of the
                 three are research-only
        swaps_head: Whether it replaces the whole head — hair, outline,
                    accessories — rather than the face interior. The only
                    property here that raises the identity ceiling rather than
                    moving along it
        notes: Why this entry exists
    """

    name: str
    media: str
    repo_env: str
    python_env: str
    entrypoint: str
    resolution: int
    seconds_per_item: float
    licence: str
    swaps_head: bool
    notes: str = ''

    # The same two facts every registry declares — see
    # `pipeline/services/tiers.py`. False and False puts all of these in the
    # STUDIO tier: too slow for a call, but general rather than trained per
    # person, so anyone's photographs work with them.
    live_capable: bool = False
    needs_training: bool = False

    @property
    def tier(self) -> str:
        """
        The tier this backend belongs to.

        Returns:
            A key from `tiers.TIERS`
        """
        from pipeline.services import tiers
        return tiers.classify(self.live_capable, self.needs_training)

    def is_live_safe(self) -> bool:
        """
        Whether this backend may be used on a stream.

        Retained as the name the stream gate already reads. It now answers from
        the declared fact rather than from a constant, so a backend cannot be
        cleared for a call by editing this method.

        Returns:
            `live_capable`, which is False for every registered backend
        """
        return self.live_capable


STUDIO_SWAPPERS: Dict[str, StudioSwapperModel] = {

    # ── Best identity of the three, and the only one that is image-native ────
    #
    # REFace (WACV 2025). Re-frames swapping as self-supervised inpainting with
    # multi-step DDIM sampling at *training* time to enforce identity, and CLIP
    # feature disentanglement to take pose, expression and lighting from the
    # target. Reported 98.8% ID retrieval top-1 on CelebA and 95.4% on FFHQ at
    # FID 6.09 — the strongest identity figure in the open literature, against
    # inswapper's 0.73 ID similarity from a different protocol.
    #
    # The reason it is first: its mask-shuffling training gives it a **head**
    # swap, hair and accessories included. Every ONNX model in
    # `swapper_models.py` is bounded by the target's silhouette because our mask
    # is a hull of the target's landmarks; this one is not bounded at all. That
    # is the only route past the interior-only ceiling that does not require
    # training a model per person.
    #
    # 50 DDIM steps is the published figure; the authors report usable output at
    # 5, which is the first lever to pull if 4.7s per photo is too slow.
    'reface': StudioSwapperModel(
        name='reface',
        media='image',
        repo_env='REFACE_REPO',
        python_env='REFACE_PYTHON',
        entrypoint='scripts/one_inference.py',
        resolution=512,
        seconds_per_item=4.7,
        licence='research (see repository)',
        swaps_head=True,
        notes='Highest reported identity retrieval, and swaps the head rather '
              'than the face interior.',
    ),

    # ── Head transfer, and the one with a permissive-looking pedigree ────────
    #
    # GHOST 2.0 (ai-forever, 2025). Two modules: an Aligner that reenacts the
    # head while preserving identity at multiple scales, and a Blender that
    # integrates it into the target background, transferring skin colour and
    # inpainting the gap the old head left behind.
    #
    # That inpainting step is the point. A head swap leaves a hole wherever the
    # source's head is smaller than the target's, which is why every other head
    # method here either avoids the problem or fails at the hairline — and it is
    # why this backend pulls a LaMa inpainter and a matting model alongside its
    # own two checkpoints. Expect the heaviest checkout of the three.
    'ghost_2': StudioSwapperModel(
        name='ghost_2',
        media='image',
        repo_env='GHOST2_REPO',
        python_env='GHOST2_PYTHON',
        entrypoint='inference.py',
        resolution=512,
        seconds_per_item=6.0,
        licence='Apache-2.0 (repository); check bundled weights separately',
        swaps_head=True,
        notes='Head transfer with explicit background inpainting, so a smaller '
              'source head does not leave a hole.',
    ),

    # ── Video-native, and the only one that is ───────────────────────────────
    #
    # DreamID-V (ByteDance, ECCV 2026 Oral). A diffusion transformer on a Wan
    # 2.1 backbone, so it is the only entry that reasons about a clip rather
    # than a sequence of stills — which is the whole argument for it here.
    # Frame-independent swapping of a video is what produces shimmer, and
    # shimmer is failure mode 3.
    #
    # The cost is proportional. It wants the Wan 2.1 VAE and text encoder
    # *alongside* its own checkpoint, 16 GB of VRAM at minimum, and 16-20
    # sampling steps per clip. On a rented pod it competes with the live
    # pipeline for the same card, so treat a render as exclusive rather than
    # concurrent.
    #
    # `seconds_per_item` is per *clip* here, not per frame, and is a
    # placeholder — nothing has been timed.
    'dreamid_v': StudioSwapperModel(
        name='dreamid_v',
        media='video',
        repo_env='DREAMIDV_REPO',
        python_env='DREAMIDV_PYTHON',
        entrypoint='generate_dreamidv_faster.py',
        resolution=480,
        seconds_per_item=300.0,
        licence='Apache-2.0 (repository); research-use notice in README',
        swaps_head=False,
        notes='The only video-native backend. Temporally coherent by '
              'construction rather than by smoothing after the fact.',
    ),
}


def resolve(name: str) -> StudioSwapperModel:
    """
    Look up a studio backend by name.

    Unlike `swapper_models.resolve`, an unknown name **raises**. That registry
    falls back to the incumbent because a typo in `.env` must not take down a
    pod that has already been paid for, and any swap model produces a swap.
    Here there is no incumbent: falling back would silently run a different
    model — a different architecture, a different licence, and for
    `dreamid_v` a different media type entirely.

    Args:
        name: Registry key

    Returns:
        The backend spec

    Raises:
        KeyError: If the name is not registered
    """
    try:
        return STUDIO_SWAPPERS[name]
    except KeyError:
        raise KeyError(
            '{!r} is not a studio swapper. Registered: {}'.format(
                name, ', '.join(names())))


def names() -> Tuple[str, ...]:
    """
    Every registered studio backend name.

    Returns:
        Names in registry order, for CLI choices and the API schema
    """
    return tuple(STUDIO_SWAPPERS)
