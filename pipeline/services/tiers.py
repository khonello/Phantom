"""
The three tiers a face model can belong to, and the rules that separate them.

This exists because "which models can be used on a call" had been three
different implicit answers in three places — a speed comment in one registry, a
refusal in the stream loop, and a paragraph in CLAUDE.md — and an implicit
answer is one that drifts. A tier is now declared by the model, derived from two
facts about it, and **enforced** at the two places a model is put to work.

Two orthogonal facts, deliberately not one enum:

    live_capable    can it hold a frame deadline?
    needs_training  does it require a per-identity artifact before it runs?

They are orthogonal because the interesting tier is the one where they
disagree. A DeepFaceLab model is *fast* — it is a small GAN and it runs at frame
rate — and it is also unusable until someone has spent hours training it on one
person. Collapsing that into a single speed ladder would file it next to a
diffusion model it has nothing in common with, and would hide the only question
that matters about it, which is not "how fast" but "trained on whom".

    tier      live  training   what it is
    LIVE      yes   no         general models, usable on a call today
    STUDIO    no    no         whole pipelines; RENDER and photo jobs only
    TRAINED   yes   yes        one model per person, trained elsewhere first

One registry per tier, so the distinction is structural rather than a field to
remember to check:

    swapper_models.py    LIVE
    studio_swappers.py   STUDIO
    identity_models.py   TRAINED

**What "forced" means here.** `require_live` is called on the stream path before
any model is warmed, and every registry entry answers it. A model cannot reach a
call by being added to the wrong list, by having its speed comment edited, or by
being selected through an environment variable nobody checked — the refusal
reads the model's own declaration, and there is no default that lets an
undeclared model through.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Tier keys. Strings rather than an Enum so they cross the WebSocket API and
# land in a QML model without a conversion nobody would remember to write.
LIVE = 'live'
STUDIO = 'studio'
TRAINED = 'trained'

TIERS: Tuple[str, ...] = (LIVE, STUDIO, TRAINED)


@dataclass(frozen=True)
class TierInfo:
    """
    What a tier is, in the words the operator should see.

    Attributes:
        key: One of `TIERS`
        label: Short name for a control
        summary: One line, stating the constraint rather than the technology —
                 an operator choosing a model cares that it cannot be used on a
                 call, not that it is a diffusion transformer
        live_capable: Whether models here may run on a stream
        needs_training: Whether they require a per-identity artifact first
    """

    key: str
    label: str
    summary: str
    live_capable: bool
    needs_training: bool


TIER_INFO: Dict[str, TierInfo] = {
    LIVE: TierInfo(
        key=LIVE,
        label='Live',
        summary='Usable on a call. Runs inside the frame deadline and needs '
                'nothing but the source photographs.',
        live_capable=True,
        needs_training=False,
    ),
    STUDIO: TierInfo(
        key=STUDIO,
        label='Studio',
        summary='Not for a call — seconds per image, not milliseconds. For '
                'rendered video and photos, where the extra fidelity is '
                'worth the wait.',
        live_capable=False,
        needs_training=False,
    ),
    TRAINED: TierInfo(
        key=TRAINED,
        label='Trained',
        summary='One model per person, trained beforehand on hours of their '
                'footage. Fast enough for a call once it exists, and unusable '
                'until it does.',
        live_capable=True,
        needs_training=True,
    ),
}


def classify(live_capable: bool, needs_training: bool) -> str:
    """
    The tier a model with these two properties belongs to.

    Args:
        live_capable: Whether it holds a frame deadline
        needs_training: Whether it needs a per-identity artifact

    Returns:
        A key from `TIERS`

    Raises:
        ValueError: For the one combination that has no meaning — a model that
                    is too slow for a call *and* has to be trained per person
                    would be the worst of both, and nothing here is that. It
                    raises rather than inventing a fourth tier, so a registry
                    entry that declares it is caught at import
    """
    if needs_training and not live_capable:
        raise ValueError(
            'a model that is both too slow for a call and trained per person '
            'has no tier here; if one appears, it needs a name of its own '
            'rather than being filed under an existing one')
    if needs_training:
        return TRAINED
    return LIVE if live_capable else STUDIO


def describe(tier: str) -> TierInfo:
    """
    The tier's own description.

    Args:
        tier: A key from `TIERS`

    Returns:
        Its `TierInfo`

    Raises:
        KeyError: For an unknown tier. Unlike a model name, a tier is never
                  operator input — it comes from a registry entry — so an
                  unknown one is a bug rather than a typo, and falling back
                  would file a model under a constraint it does not have
    """
    return TIER_INFO[tier]


def require_live(tier: str, name: str, deadline_ms: float) -> Optional[str]:
    """
    Whether a model may run on a stream, and why not when it may not.

    Called before any model is warmed, so a session that cannot work is refused
    before the pod bills for loading weights.

    The refusal is worded for the operator, not the log: a studio model on a
    call produces a frozen frame while the connection, the virtual camera and
    every badge read healthy, so "it did not work" is not a message anyone can
    act on and "this one is for renders" is.

    Args:
        tier: The model's declared tier
        name: The model's name, for the message
        deadline_ms: The frame budget this preset is working to

    Returns:
        None when the model may run, else the reason it may not
    """
    info = TIER_INFO.get(tier)
    if info is None:
        return (
            '{} declares no tier, so it cannot be cleared for a live '
            'session.'.format(name))

    if not info.live_capable:
        return (
            '{} is a {} model: it takes seconds per frame against a '
            '{:.0f}ms deadline, so a call would show a frozen face while '
            'everything else looked healthy. Use it for RENDER or photos.'
            .format(name, info.label.lower(), deadline_ms))

    return None


def require_trained_artifact(tier: str, name: str, present: bool) -> Optional[str]:
    """
    Whether a trained model has the artifact it cannot run without.

    Separate from `require_live` because it fails for the opposite reason: this
    model is fast enough for a call and simply does not exist yet for this
    person. Reported as a missing input rather than as an unsuitable model, so
    nobody concludes the tier is broken.

    Args:
        tier: The model's declared tier
        name: The model's name, for the message
        present: Whether its weights were actually found

    Returns:
        None when it can run, else the reason it cannot
    """
    info = TIER_INFO.get(tier)
    if info is None or not info.needs_training:
        return None

    if not present:
        return (
            '{} is a trained model and its weights were not found. One of '
            'these is trained per person, beforehand — it cannot be built '
            'from the uploaded photographs at session time.'.format(name))

    return None
