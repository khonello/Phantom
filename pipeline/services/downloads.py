"""
Which model weights this deployment is allowed to fetch.

Every weight in this pipeline downloads on first use, which was the right
default while there were four models. There are now thirteen swap models, four
restorers, an occluder and a segmenter, and between them they are several
gigabytes — so "fetch whatever is asked for" has become a policy nobody stated
and nobody can see.

**What was actually happening, stated plainly, because it is less bad than it
sounds.** Nothing bulk-downloads. A weight is fetched when the model that needs
it is selected, so a session on the default configuration pulls the default
models and nothing else. The exception was real and unconditional:
`vast/startup.sh` fetched GFPGANv1.4.pth — 340 MB — on **every** deploy, for a
restoration backend that is not the default and that most sessions never touch.

What is missing is not throttling, it is **control**: a way to say "never
download anything, this volume is already seeded" for a pod on a metered link,
and a way to pre-seed a chosen set rather than discovering the cost during a
paid session. Both are one policy.

    selected   (default)  fetch what the running configuration actually needs
    all                   fetch anything asked for — the old behaviour
    none                  fetch nothing; use what is on disk and say what is not
    <list>                fetch only these, by model or file name

`selected` and `all` differ only in the presence of a bulk pre-seed: on the live
path they behave identically, because the pipeline only ever asks for what it
is configured to use. The distinction exists so `tools/preseed_models.py` has
something to obey.

**A refusal is not a failure.** Every fetch site in this pipeline already
degrades when a weight is missing — masking falls back to the landmark hull,
restoration walks the registry, the segmenter becomes a no-op. Refusing a
download therefore lands on a path that is already tested, and it says which
variable would allow it.
"""

import os
from typing import Optional, Tuple

# Policy keywords. Anything else is read as a comma-separated allow list.
SELECTED = 'selected'
ALL = 'all'
NONE = 'none'

_POLICY_ENV = 'MODEL_DOWNLOADS'


def policy() -> str:
    """
    The configured download policy, lower-cased.

    Returns:
        `selected`, `all`, `none`, or a raw allow list
    """
    return os.environ.get(_POLICY_ENV, '').strip().lower() or SELECTED


def _allow_list(raw: str) -> Tuple[str, ...]:
    """
    Parse an allow list into comparable names.

    Args:
        raw: Comma or space separated names

    Returns:
        Lower-cased entries with any extension stripped, so `alphaface_256`,
        `alphaface_256.onnx` and `ALPHAFACE_256` are one entry
    """
    parts = [p.strip().lower() for p in raw.replace(',', ' ').split()]
    return tuple(os.path.splitext(p)[0] for p in parts if p)


def allowed(name: str, selected: bool = True) -> Tuple[bool, str]:
    """
    Whether a named weight may be downloaded now.

    Args:
        name: Model or file name, with or without an extension
        selected: Whether the running configuration actually asked for this —
                  False for a speculative or bulk fetch. Under the default
                  policy that is the whole distinction: a pod fetches what it
                  was configured to run and nothing on spec

    Returns:
        (permitted, reason). `reason` is empty when permitted, and otherwise
        names the variable that would allow it — a refusal has to be
        actionable, since the alternative is an operator reading "model not
        found" and concluding the deployment is broken
    """
    setting = policy()

    if setting == ALL:
        return True, ''

    if setting == NONE:
        return False, (
            '{}=none, so {} was not downloaded. Set it to `selected` to allow '
            'the configured models, or pre-seed the volume.'.format(
                _POLICY_ENV, name))

    if setting == SELECTED:
        if selected:
            return True, ''
        return False, (
            '{} was not downloaded: {}=selected fetches only what the running '
            'configuration needs. Name it in {} to pre-seed it.'.format(
                name, _POLICY_ENV, _POLICY_ENV))

    stem = os.path.splitext(str(name).strip().lower())[0]
    if stem in _allow_list(setting):
        return True, ''

    return False, (
        '{} is not in {}, so it was not downloaded. Add it, or set the policy '
        'to `selected`.'.format(name, _POLICY_ENV))


def refuse_reason(name: str, selected: bool = True) -> Optional[str]:
    """
    The reason a download is refused, or None when it is permitted.

    The convenience shape for a call site that only branches on refusal.

    Args:
        name: Model or file name
        selected: Whether the running configuration asked for it

    Returns:
        The reason, or None
    """
    permitted, reason = allowed(name, selected)
    return None if permitted else reason
