"""
Uploading a source, and uploading a different one over it.

The source builds the identity every frame is swapped to, so a mistake here is
not a degraded frame — it is the wrong person, on every frame, looking like it
worked. These cover the three ways that happened.
"""

import numpy as np
import pytest

from pipeline.config import FaceSwapConfig
from pipeline.services.database import FaceDatabase


class _StubFace:
    def __init__(self, tag):
        self.tag = tag
        self.normed_embedding = np.zeros(512, dtype=np.float32)


class _StubDetection:
    def __init__(self, tag):
        self.face = _StubFace(tag)


class _CountingDetector:
    """Returns a face tagged with whatever bytes the file currently holds."""

    def __init__(self):
        self.calls = 0

    def detect_one(self, frame):
        self.calls += 1
        # The stub reads the tag out of the pixel the fake image carries, so a
        # rewritten file genuinely produces a different face.
        return _StubDetection(int(frame[0][0][0]))


@pytest.fixture
def db(monkeypatch):
    import pipeline.services.database as database

    detector = _CountingDetector()
    monkeypatch.setattr(
        database.cv2, 'imread',
        lambda path: np.full((4, 4, 3), _read_tag(path), dtype=np.uint8),
    )
    return FaceDatabase(detector, FaceSwapConfig()), detector


def _read_tag(path):
    with open(path, 'rb') as fh:
        return int(fh.read().decode())


def _write(path, tag):
    with open(path, 'wb') as fh:
        fh.write(str(tag).encode())


# ── The cache must follow the file, not the name ───────────────────────


def test_same_path_new_contents_is_a_new_face(db, tmp_path):
    """
    Uploads land at `uploads/<original filename>`, and phones produce the same
    filename for everyone. Keying the cache on the path alone meant the second
    person's upload returned the first person's embedding — underneath the
    guards, which had checked the new photo.
    """
    database, detector = db
    path = str(tmp_path / 'IMG_0001.jpg')

    _write(path, 7)
    first = database._extract_from_image(path)

    # Same name, different photo — exactly what a second upload does. No
    # timestamp fiddling: the key must not depend on the clock, because two
    # uploads a moment apart can share a filesystem timestamp.
    _write(path, 9)

    second = database._extract_from_image(path)
    assert first.tag == 7
    assert second.tag == 9, 'a rewritten file must not return the cached face'


def test_unchanged_file_still_hits_the_cache(db, tmp_path):
    """The cache has to keep working, or every frame re-detects the source."""
    database, detector = db
    path = str(tmp_path / 'face.jpg')
    _write(path, 3)

    database._extract_from_image(path)
    database._extract_from_image(path)
    database._extract_from_image(path)

    assert detector.calls == 1


def test_cache_key_changes_with_contents(tmp_path):
    path = str(tmp_path / 'a.jpg')
    _write(path, 1)
    before = FaceDatabase._cache_key(path)

    _write(path, 22222)
    after = FaceDatabase._cache_key(path)

    assert before != after


def test_cache_key_survives_a_missing_file(tmp_path):
    """No worse than the old behaviour, and must not raise."""
    missing = str(tmp_path / 'gone.jpg')
    assert FaceDatabase._cache_key(missing) == missing


def test_two_different_paths_do_not_collide(tmp_path):
    a, b = str(tmp_path / 'a.jpg'), str(tmp_path / 'b.jpg')
    _write(a, 1)
    _write(b, 1)
    assert FaceDatabase._cache_key(a) != FaceDatabase._cache_key(b)


def test_key_does_not_depend_on_the_clock(tmp_path):
    """
    Identical bytes written twice must key the same, and different bytes must
    key differently, regardless of when either happened. A stat-based key got
    this wrong whenever two writes shared a filesystem timestamp tick.
    """
    a, b = str(tmp_path / 'x.jpg'), str(tmp_path / 'x2.jpg')
    _write(a, 5)
    _write(b, 5)
    assert (FaceDatabase._cache_key(a).split(':')[-1]
            == FaceDatabase._cache_key(b).split(':')[-1])


# ── Guards run on every upload, not only after a stream has started ────


def test_source_change_builds_processors_when_no_stream_has_run():
    """
    The guards live behind `_swapping_proc`, which only existed once a stream
    had started. So the first upload of a pipeline's life skipped validation
    entirely and every later one enforced it — the same photos accepted on a
    first desktop run and refused on the next, reported to the operator as a
    problem with their photo.
    """
    import inspect

    from pipeline.processing import pipeline as pipeline_module

    src = inspect.getsource(pipeline_module.ProcessingPipeline._on_config_changed)
    assert '_build_processors()' in src, (
        'a source change must be able to construct the processors that hold '
        'the guards, or validation depends on whether a stream ran first'
    )


def test_cleanup_resolves_the_upload_dir_once():
    """
    `_upload_dir()` creates the directory it returns, so calling it inside the
    `isdir` test made the test always true and created the directory a second
    time before deleting it.
    """
    import inspect

    from pipeline.api import handlers

    src = inspect.getsource(handlers.handle_cleanup_session)
    assert 'uploads = _upload_dir()' in src
    assert 'isdir(_upload_dir())' not in src
