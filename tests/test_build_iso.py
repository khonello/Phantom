"""
What goes onto the disc image, and what must not.

An ISO is a thing that gets handed to people, so the interesting property is
not "did it build" but "what did it carry". The first version of this tool
gated `.env` by exact name while the tree also held `.env.backup-<date>` — a
real copy with a real API key — which went onto the image untouched. A gate
one spelling of the same secret walks around is not a gate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

import build_iso  # noqa: E402


# ── Secrets ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('name', [
    '.env',
    '.env.backup-20260818-234644',
    '.env.local',
    '.env.production',
    '.env.bak',
])
def test_every_env_variant_is_a_secret(name):
    assert build_iso._is_secret(name), name


def test_the_template_is_not_a_secret():
    """`.env.example` is the documented template and carries no values."""
    assert not build_iso._is_secret('.env.example')


@pytest.mark.parametrize('name', [
    'environment.py', 'envelope.txt', 'CLAUDE.md', 'pipeline.py',
])
def test_ordinary_files_are_not_secrets(name):
    assert not build_iso._is_secret(name)


def test_secrets_are_excluded_by_default():
    paths, _ = build_iso._included(with_env=False)
    leaked = [p for p in paths if build_iso._is_secret(os.path.basename(p))]
    assert leaked == [], 'a credential reached the image without --with-env'


def test_the_flag_is_what_lets_them_through():
    paths, _ = build_iso._included(with_env=True)
    assert any(os.path.basename(p) == '.env' for p in paths)


# ── Bulk that must never be carried ────────────────────────────────────


@pytest.mark.parametrize('prefix', [
    'pipeline/models/',      # 912 MB of weights, re-downloaded on first use
    'environ-orchestrator/',  # a virtualenv
    'desktop/.qtcreator/',   # a virtualenv
    'build/',                # Nuitka output, tied to the Python that made it
])
def test_excluded_trees_are_absent(prefix):
    paths, _ = build_iso._included(with_env=False)
    assert not [p for p in paths if p.startswith(prefix)], prefix


def test_bytecode_is_not_carried():
    """Stale bytecode on a machine with a different interpreter is worse than none."""
    paths, _ = build_iso._included(with_env=False)
    assert not [p for p in paths if '__pycache__' in p]


def test_an_image_never_contains_an_image():
    paths, _ = build_iso._included(with_env=False)
    assert not [p for p in paths if p.endswith('.iso')]


# ── What has to be there ───────────────────────────────────────────────


@pytest.mark.parametrize('rel', [
    'CLAUDE.md',
    'pipeline.py',
    'desktop.py',
    'desktop/main.qml',
    'pipeline/services/enhancer_models.py',
    'requirements-desktop.txt',
    'docs/SETUP_CHECKLIST.md',
    'tools/build_iso.py',
])
def test_the_project_itself_is_included(rel):
    paths, _ = build_iso._included(with_env=False)
    assert rel in paths, rel


def test_it_is_a_working_clone_not_a_snapshot():
    """`.git` earns its size: the image can be committed from and pulled into."""
    paths, _ = build_iso._included(with_env=False)
    assert any(p.startswith('.git/') for p in paths)


def test_the_iso_output_is_gitignored():
    """
    It is derived from the repository rather than part of it, and it may carry
    .env.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, '.gitignore'), encoding='utf-8') as fh:
        assert '*.iso' in fh.read()
