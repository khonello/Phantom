"""
Per-identity models — one model per person, trained before the session.

The third registry, and the only one that is **scanned rather than declared**.
The other two list models that exist for everyone; this one lists whatever has
been trained for *these* people, which is a property of a directory rather than
of this repository. A hard-coded list here would be a list of strangers.

**Why this tier is worth having at all.** Every model in `swapper_models.py`
lands somewhere between the source and the target, because it has to work for
any pair it is given and has learned a manifold of plausible faces to interpolate
across. A model trained on one person has no such manifold — there is nothing
for it to regress toward but them. It is the only thing here that answers "the
target is a puppet wearing the source" rather than approximating it, and its
alignment template (`dfl_whole_face`) is a **wider crop** than arcface,
including jaw and forehead, so it moves the silhouette that every other model
leaves at the target's.

**And why it is not simply the default.** It cannot be built at session time. A
`.dfm` is hours to days of GPU training on footage of one face, done elsewhere,
before anyone opens the app. That is a product decision about onboarding, not a
setting — which is exactly why it is a tier rather than another entry in the
swap registry.

Three things about the runtime that are easy to get wrong, all of them silent:

1. **There is no source input.** The identity is in the weights. Nothing about
   embeddings, averaging, `identity_push` or the source photographs applies, and
   the source guards protect nothing here because no source is consulted.
2. **The tensor is NHWC**, not the NCHW every other model in this pipeline
   takes, and it is **BGR**, not RGB. Either mistake produces a face.
3. **The crop is sharpened first.** DeepFaceLive applies an unsharp mask before
   inference and the models were trained against that input. Skipping it is a
   softer swap that reads as a weaker model.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pipeline.services import tiers

# What a trained model can arrive as. `.dfm` is DeepFaceLive's own extension and
# the file underneath is an ONNX graph, so both load through the same session —
# the extension is a label, not a format.
IDENTITY_EXTENSIONS: Tuple[str, ...] = ('.dfm', '.onnx')

# Where to look, in priority order. Mirrors the model search path the swapper
# and enhancer already use, so a pod with a network volume and a laptop without
# one both work with no configuration.
_SEARCH = ('/workspace/identities', 'pipeline/identities')

# Environment override, because on a pod these are the one class of weight that
# is per customer rather than per deployment.
_DIR_ENV = 'IDENTITY_MODEL_DIR'


@dataclass(frozen=True)
class IdentityModel:
    """
    One trained per-identity model.

    Attributes:
        name: Registry key — the filename stem, which is whatever the person
              who trained it called it. Not validated against anything: this is
              a directory listing, and inventing a naming rule here would only
              hide files
        path: Absolute path to the weights
        template: Alignment template. `dfl_whole_face` for every DeepFaceLab
                  export, and wider than arcface — it includes jaw and forehead,
                  which is where the silhouette advantage comes from
        size_bytes: On-disk size, for the log line
    """

    name: str
    path: str
    template: str = 'dfl_whole_face'
    size_bytes: int = 0

    # Declared rather than assumed, and read by `tiers.require_live`. A trained
    # model IS fast enough for a call — it is a small GAN, not a diffusion
    # model — which is precisely why the tier cannot be a speed ladder.
    live_capable: bool = True
    needs_training: bool = True

    @property
    def tier(self) -> str:
        """
        Which tier this belongs to.

        Returns:
            Always `tiers.TRAINED`, derived rather than written down so it
            cannot disagree with the two facts above
        """
        return tiers.classify(self.live_capable, self.needs_training)


def directory() -> str:
    """
    Where trained models are kept.

    Returns:
        The configured or first existing search path. Returns the last
        candidate when none exist, so a caller has a path to name in a message
        rather than an empty string
    """
    configured = os.environ.get(_DIR_ENV, '').strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    for candidate in _SEARCH:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package, 'identities')


def scan() -> Dict[str, IdentityModel]:
    """
    Every trained model currently on disk.

    Scanned on every call rather than cached. These arrive by being copied into
    a directory — there is no download step and no restart between training a
    model and wanting to use it — so a cache would mean the answer to "is it
    there yet" was "not until you restart the pipeline".

    Returns:
        Name -> model, empty when the directory is absent or holds nothing
    """
    root = directory()
    found: Dict[str, IdentityModel] = {}
    if not os.path.isdir(root):
        return found

    for entry in sorted(os.listdir(root)):
        stem, extension = os.path.splitext(entry)
        if extension.lower() not in IDENTITY_EXTENSIONS:
            continue
        path = os.path.join(root, entry)
        if not os.path.isfile(path):
            continue
        found[stem] = IdentityModel(
            name=stem,
            path=path,
            size_bytes=os.path.getsize(path),
        )

    return found


def names() -> Tuple[str, ...]:
    """
    Every trained model name currently available.

    Returns:
        Names in directory order
    """
    return tuple(scan())


def resolve(name: str) -> IdentityModel:
    """
    Look up a trained model by name.

    Raises rather than falling back, and the reason differs from both other
    registries. `swapper_models` falls back because any swap model produces a
    swap and a typo must not take down a paid session. `studio_swappers`
    raises because a different backend is a different architecture. Here a
    different name is **a different person**, which is the confidently-wrong
    output the guards exist to prevent, arriving through a config field.

    Args:
        name: Registry key

    Returns:
        The model

    Raises:
        KeyError: If no such model is on disk
    """
    found = scan()
    try:
        return found[name]
    except KeyError:
        available: List[str] = sorted(found)
        raise KeyError(
            '{!r} is not a trained model in {}. Available: {}'.format(
                name, directory(),
                ', '.join(available) if available else 'none'))


def available(name: str) -> bool:
    """
    Whether a named trained model is present.

    Args:
        name: Registry key

    Returns:
        True if its weights are on disk
    """
    return name in scan()
