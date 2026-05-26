"""
Embedding: project the n×n dissimilarity matrix down to 2-D.

Two methods (as required by the task):
  * PCA — linear projection of the dissimilarity matrix rows
  * SMACOF — iterative stress-minimization (the classic MDS algorithm)

Plus a Procrustes helper to align both embeddings for visual comparison
and a Shepard-diagram helper to assess how well pairwise distances are
preserved.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

logger = logging.getLogger(__name__)


def pca_embed(D: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Project rows of D into 2-D via PCA.

    Parameters
    ----------
    D : (n, n) dissimilarity matrix (each row is a "profile").

    Returns
    -------
    coords : (n, 2) array
    explained_variance : float  (sum of first two components' ratios)
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(D)
    ev = float(pca.explained_variance_ratio_.sum())
    logger.info("PCA: explained variance (2 components) = %.2f%%", ev * 100)
    return coords, ev


def smacof_embed(
    D: np.ndarray,
    n_init: int = 10,
    max_iter: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Non-metric MDS via SMACOF (Scaling by MAjorizing a COmplicated Function).

    Parameters
    ----------
    D : (n, n) dissimilarity matrix.
    n_init : number of random restarts (best stress wins).
    max_iter : iteration cap per restart.

    Returns
    -------
    coords : (n, 2) array
    stress : float  (normalized Kruskal stress-1)
    """
    mds = MDS(
        n_components=2,
        metric=True,
        dissimilarity="precomputed",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(D)
    stress = float(mds.stress_)
    logger.info("SMACOF: final stress = %.6f  (n_init=%d)", stress, n_init)
    return coords, stress


def align_procrustes(
    reference: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Rotate/reflect/scale *target* to best match *reference* (minimizes
    sum-of-squared differences).  Useful for placing PCA and SMACOF
    embeddings in the same orientation for side-by-side comparison.

    Returns the aligned version of *target* (same shape).
    """
    # scipy.procrustes normalizes both; we want to keep the reference scale,
    # so we apply the rotation ourselves.
    ref_centered = reference - reference.mean(axis=0)
    tgt_centered = target - target.mean(axis=0)

    # Optimal rotation via SVD (Kabsch algorithm).
    H = tgt_centered.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))  # correct reflection
    S = np.diag([1.0, d])
    R = Vt.T @ S @ U.T

    # Scale to match reference spread.
    scale = np.linalg.norm(ref_centered) / np.linalg.norm(tgt_centered)
    aligned = (tgt_centered @ R) * scale + reference.mean(axis=0)
    return aligned


def shepard_data(
    D: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Shepard-diagram arrays: original dissimilarities vs
    embedded (Euclidean) distances.

    Returns (d_original, d_embedded) — both 1-D arrays of length n*(n-1)/2.
    """
    d_orig = squareform(D, checks=False)
    d_embed = pdist(coords, metric="euclidean")
    return d_orig, d_embed
