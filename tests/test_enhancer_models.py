"""
The face-restoration model registry.

Restoration was 68% of a frame and bought +0.03 of face/frame detail on a 101px
webcam face. That is what moved the default off CodeFormer, and these pin the
facts the decision rests on — so that if someone changes the default back, they
do it knowing what the numbers were rather than by editing a string.
"""

import pytest

from pipeline.config import FaceSwapConfig
from pipeline.services import enhancer_models
from pipeline.services.enhancer_models import ENHANCER_MODELS


# ── The registry is internally consistent ──────────────────────────────


def test_default_is_registered():
    assert enhancer_models.DEFAULT_ENHANCER_MODEL in ENHANCER_MODELS


def test_config_default_matches_the_registry_default():
    """
    Two files, one decision. A config default naming a model the registry does
    not prefer is the kind of drift nothing else would report.
    """
    assert FaceSwapConfig().enhancer_model == enhancer_models.DEFAULT_ENHANCER_MODEL


@pytest.mark.parametrize('name', sorted(ENHANCER_MODELS))
def test_every_model_is_well_formed(name):
    spec = ENHANCER_MODELS[name]
    assert spec.name == name, 'registry key must match the spec it holds'
    assert spec.backend in ('codeformer', 'gfpgan')
    assert 128 <= spec.crop <= 512
    assert spec.filename
    assert 0.0 <= spec.enhance_strength <= 1.0
    assert spec.notes, 'a model with no stated reason invites a blind swap'


@pytest.mark.parametrize('name', sorted(ENHANCER_MODELS))
def test_onnx_models_carry_a_url(name):
    """
    GFPGAN is fetched by its own package, so it alone may have no URL. Anything
    on the ONNX path has to say where it comes from or it cannot self-install
    on a fresh volume.
    """
    spec = ENHANCER_MODELS[name]
    if spec.backend == 'codeformer':
        assert spec.url.startswith('https://'), name
        assert spec.filename.endswith('.onnx')


# ── The facts the default rests on ─────────────────────────────────────


def test_the_default_is_the_small_crop_model():
    """
    The whole argument: restoration runs on a fixed crop regardless of how big
    the face is, and a 512 crop is warped straight back down into a 128-192
    aligned space. If the default ever points at a 512 model again, that was a
    decision and should fail here until it is made deliberately.
    """
    assert enhancer_models.resolve(
        enhancer_models.DEFAULT_ENHANCER_MODEL).crop == 256


def test_codeformer_is_the_only_one_with_a_fidelity_weight():
    """
    `enhancer_weight` is CodeFormer's input and nothing else's. A model marked
    `fidelity=True` without one would make a configured value silently inert.
    """
    with_weight = [n for n, s in ENHANCER_MODELS.items() if s.fidelity]
    assert with_weight == ['codeformer']


def test_look_profile_omits_the_weight():
    """
    The profile must not carry `enhancer_weight`: it means nothing to three of
    the four models, and a value that looks configured but does nothing is
    worse than an absent one.
    """
    for name, spec in ENHANCER_MODELS.items():
        assert 'enhancer_weight' not in spec.look(), name
        assert 'enhance_strength' in spec.look(), name


# ── Resolution never fails ─────────────────────────────────────────────


def test_unknown_name_falls_back_rather_than_raising():
    """A typo in .env must not take down a pod that is already being paid for."""
    spec = enhancer_models.resolve('coformer')
    assert spec.name == enhancer_models.DEFAULT_ENHANCER_MODEL


@pytest.mark.parametrize('name', sorted(ENHANCER_MODELS))
def test_every_name_resolves_to_itself(name):
    assert enhancer_models.resolve(name).name == name


def test_names_covers_the_registry():
    assert set(enhancer_models.names()) == set(ENHANCER_MODELS)


# ── The API and CLI accept exactly what the registry holds ─────────────


def test_set_realism_accepts_every_registered_model():
    """
    The validator used to hold a hard-coded pair. A registry the API cannot
    name is a registry nobody can select from at runtime.
    """
    from pipeline.api.handlers import _REALISM_FIELDS

    validator = _REALISM_FIELDS['enhancer_model']
    for name in enhancer_models.names():
        assert validator(name) == name
    assert validator('not_a_model') is None
