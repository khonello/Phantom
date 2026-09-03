"""
Face-space geometry shared by the compositing stages.

These primitives were defined in `compositor.py`, which was fine while the
compositor was their only consumer. It is not any more: `texture.py` extracts in
the same canonical framing and has to fit it with the same estimator, and
`face_swapping.py` already reached for `estimate_similarity` through a deferred
import to dodge the cycle that created.

Nothing here holds state or reads config. It is the definition of where a face
sits and which band counts as texture, so that every stage that has an opinion
about either is answering from one place.
"""

from typing import Optional

import numpy as np

from pipeline.types import Face, Matrix, Points

# Five-point FFHQ alignment template, normalised to a unit square: left eye,
# right eye, nose, left mouth corner, right mouth corner. Normalised rather than
# fixed at 512 so scaling it is all a different crop size needs — the framing is
# identical, only the sampling rate changes.
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float64)

# Working resolutions the compositor steps through, and the ladder the texture
# maps are derived at. Shared so that a face moving between two compositing
# resolutions moves between two texture resolutions at the same points, rather
# than at points that nearly agree.
ALIGNED_STEPS = (128, 192, 256, 320, 384, 448, 512)

# Gaussian sigma separating "texture" from "shape", specified at 256 and scaled
# with the working resolution so that detail means the same physical thing at
# every quality preset. Shared because `_match_detail` scales this band and the
# texture layer *adds* to it — two stages describing adjacent-but-different
# bands would fight, one amplifying what the other normalises.
DETAIL_SIGMA = 1.5
DETAIL_SIGMA_REFERENCE = 256.0


def estimate_similarity(source: Points, target: Points) -> Optional[Matrix]:
    """
    Least-squares similarity transform between two point sets (Umeyama).

    Deliberately not `cv2.estimateAffinePartial2D`, whose RANSAC and LMEDS
    methods are randomized and so carry no guarantee of returning the same
    matrix for the same input. Anything that varies frame to frame here feeds
    straight back into the shimmer the compositor exists to remove. For five
    points and a similarity transform the least-squares solution is closed-form,
    exact and reproducible — and it is the same estimator InsightFace uses for
    its own alignment, so the two agree by construction.

    Note this is a similarity (4 degrees of freedom: scale, rotation,
    translation). It cannot reshape a face, so landmarks whose proportions
    differ from the target template land with a small residual. That is expected
    and correct.

    It also cannot correct pose. Composing two of these yields one of them
    exactly, so routing a warp through canonical space costs nothing and buys no
    accuracy — what canonical space buys is that the expensive half can be
    computed once. Anything needing genuine pose invariance needs a 3D fit, not
    this.

    Args:
        source: (N, 2) source points
        target: (N, 2) target points

    Returns:
        2x3 affine mapping source onto target, or None if degenerate.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 2:
        return None

    count = source.shape[0]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_demean = source - source_mean
    target_demean = target - target_mean

    covariance = (target_demean.T @ source_demean) / count
    reflection = np.ones(2)
    if np.linalg.det(covariance) < 0:
        reflection[1] = -1.0

    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    if np.linalg.matrix_rank(covariance) == 0:
        return None

    rotation = u_matrix @ np.diag(reflection) @ vt_matrix

    variance = float(source_demean.var(axis=0).sum())
    scale = 1.0 if variance < 1e-9 else float(singular @ reflection) / variance

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_mean - scale * (rotation @ source_mean)
    return matrix


def compose_affine(outer: Matrix, inner: Matrix) -> Matrix:
    """
    Compose two 2x3 affines: apply `inner` first, then `outer`.

    Args:
        outer: Transform applied second
        inner: Transform applied first

    Returns:
        2x3 affine equivalent to outer(inner(x))
    """
    outer_h = np.vstack([outer, [0.0, 0.0, 1.0]])
    inner_h = np.vstack([inner, [0.0, 0.0, 1.0]])
    composed: Matrix = (outer_h @ inner_h)[:2, :]
    return composed


def canonical_from_frame(face: Face, size: int) -> Optional[Matrix]:
    """
    Affine mapping frame space -> canonical face space at `size`.

    The same fit `FaceCompositor._ffhq_geometry` makes for the restorer, exposed
    on its own because the texture layer needs it in frame space inside `_paste`,
    where the enhancer's geometry is not in scope — and because the extractor
    needs it against a source photograph, where there is no aligned crop at all.

    Args:
        face: Detection to align to
        size: Edge length of canonical space

    Returns:
        2x3 affine, or None if the five keypoints are unusable.
    """
    kps = getattr(face, 'kps', None)
    if kps is None or len(kps) != len(FFHQ_TEMPLATE):
        return None

    matrix = estimate_similarity(
        np.asarray(kps, dtype=np.float64),
        FFHQ_TEMPLATE * size,
    )
    if matrix is None:
        return None
    return matrix.astype(np.float32)
