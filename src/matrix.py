"""
Co-authorship matrix construction and dissimilarity transformation.

Pipeline:
  1. build_coauthor_matrix(): raw publication data -> symmetric n×n integer
     matrix C where C[i,j] = number of joint publications between researchers
     i and j.  Diagonal is zero (self-coauthorship is not informative here).
  2. to_dissimilarity(C): C -> distance matrix D via the two-step transform
     described in the task brief:
       a) S = 1 / (C + 1)   - inverted co-authorship (similarity → small number)
       b) D[i,j] = ‖S[i,:] − S[j,:]‖₂  - Euclidean distance between
          co-authorship *profiles*.  Two researchers are close when they
          collaborate with the same people, not just with each other.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def build_coauthor_matrix(
    pub_to_authors: dict[str, set[int]],
    n: int,
) -> np.ndarray:
    """
    Parameters
    ----------
    pub_to_authors : dict mapping publication_key -> set of roster indices
        (output of parse.resolve_publications)
    n : int
        Number of researchers (rows/cols of the output matrix).

    Returns
    -------
    C : ndarray of shape (n, n), dtype int
        Symmetric co-authorship count matrix.  C[i,j] = number of unique
        publications co-authored by researchers i and j.
    """
    C = np.zeros((n, n), dtype=int)
    n_edges = 0
    for key, idx_set in pub_to_authors.items():
        idx_list = sorted(idx_set)
        for a_pos in range(len(idx_list)):
            for b_pos in range(a_pos + 1, len(idx_list)):
                i, j = idx_list[a_pos], idx_list[b_pos]
                C[i, j] += 1
                C[j, i] += 1
                n_edges += 1
    logger.info(
        "Co-author matrix: %d researchers, %d non-zero pairs, "
        "max co-authored = %d",
        n, np.count_nonzero(C) // 2, C.max(),
    )
    return C


def to_dissimilarity(C: np.ndarray) -> np.ndarray:
    """
    Transform co-authorship counts into a proper Euclidean distance matrix.

    Step 1 — Inversion:  S[i,j] = 1 / (C[i,j] + 1).
      * Many co-authored papers -> S close to 0 (high affinity).
      * Zero co-authored papers -> S = 1 (no affinity).
      * The "+1" avoids division by zero and guarantees S ∈ (0, 1].

    Step 2 — Profile distance:  D[i,j] = ‖S[i,:] − S[j,:]‖₂.
      * Two researchers are similar when they have *similar collaboration
        profiles* across the entire roster — i.e. they tend to co-author
        with the same set of colleagues.
      * This produces a true metric (satisfies the triangle inequality),
        which is a prerequisite for MDS / SMACOF convergence guarantees.
    """
    S = 1.0 / (C.astype(float) + 1.0)
    # Zero out the diagonal of S so self-similarity doesn't dominate distances.
    np.fill_diagonal(S, 0.0)
    D = squareform(pdist(S, metric="euclidean"))
    logger.info(
        "Dissimilarity matrix: min=%.4f, median=%.4f, max=%.4f",
        D[D > 0].min(), np.median(D[D > 0]), D.max(),
    )
    return D

def to_dissimilarity_direct(C: np.ndarray) -> np.ndarray:
    """
    Direct dissimilarity: D[i,j] = 1 / (C[i,j] + 1).
    
    Directly encodes co-authorship strength as proximity:
    many joint papers → small distance. Diagonal set to zero.
    """
    D = 1.0 / (C.astype(float) + 1.0)
    np.fill_diagonal(D, 0.0)
    logger.info(
        "Direct dissimilarity: min=%.4f, median=%.4f, max=%.4f",
        D[D > 0].min(), np.median(D[D > 0]), D.max(),
    )
    return D
