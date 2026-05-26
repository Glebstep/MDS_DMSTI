"""
Extended analysis: community detection, group collaboration, t-SNE.

Functions:
  * detect_communities(): Louvain community detection vs official groups
  * plot_group_collaboration(): 9×9 inter-group co-authorship matrix
  * tsne_embed(): t-SNE as a third dimensionality reduction method
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from src.plot import GROUP_ORDER, GROUP_COLORS, _initials

logger = logging.getLogger(__name__)


# ========================================================================== #
# 1. COMMUNITY DETECTION — Louvain vs official groups
# ========================================================================== #
def detect_communities(
    C: np.ndarray,
    researchers: list[dict],
    coords: np.ndarray,
    path: str | Path,
    figsize: tuple[float, float] = (22, 10),
) -> dict:
    """
    Run Louvain community detection on the co-authorship graph and compare
    the discovered communities with the official research group assignments.

    Produces a side-by-side figure: official groups (left) vs Louvain (right).
    Returns a dict with overlap statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(researchers)

    # Build weighted graph
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                G.add_edge(i, j, weight=int(C[i, j]))

    # Run Louvain
    communities = nx.algorithms.community.louvain_communities(
        G, weight="weight", resolution=1.0, seed=42
    )
    communities = sorted(communities, key=len, reverse=True)

    # Assign community labels
    community_labels = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            community_labels[node] = ci

    n_communities = len(communities)
    logger.info("Louvain found %d communities (vs %d official groups)",
                n_communities, len(set(r["group"] for r in researchers)))

    # Log community composition
    for ci, comm in enumerate(communities):
        members = [researchers[i]["display_name"] for i in sorted(comm)]
        groups_in_comm = defaultdict(int)
        for i in comm:
            groups_in_comm[researchers[i]["group"]] += 1
        dominant = max(groups_in_comm.items(), key=lambda x: x[1])
        logger.info(
            "  Community %d (%d members): dominant = %s (%d/%d)",
            ci, len(comm), dominant[0], dominant[1], len(comm),
        )

    # ------------------------------------------------------------------ #
    # Compute overlap: Normalized Mutual Information (simplified version)
    # ------------------------------------------------------------------ #
    official_labels = []
    detected_labels = []
    for i in range(n):
        official_labels.append(researchers[i]["group"])
        detected_labels.append(community_labels.get(i, -1))

    # Simple overlap metric: for each community, what fraction matches
    # its dominant official group
    total_correct = 0
    for ci, comm in enumerate(communities):
        groups_in_comm = defaultdict(int)
        for i in comm:
            groups_in_comm[researchers[i]["group"]] += 1
        dominant_count = max(groups_in_comm.values())
        total_correct += dominant_count
    overlap_ratio = total_correct / n

    logger.info("Overlap ratio (dominant group match): %.1f%%", overlap_ratio * 100)

    # ------------------------------------------------------------------ #
    # Side-by-side plot: official vs Louvain
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Normalize coords
    plot_coords = coords.copy()
    max_abs = np.abs(plot_coords).max()
    if max_abs > 0:
        plot_coords = plot_coords / max_abs

    # Left: official groups
    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]
    for group in present_groups:
        idxs = [i for i, r in enumerate(researchers) if r["group"] == group]
        xs = plot_coords[idxs, 0]
        ys = plot_coords[idxs, 1]
        color = GROUP_COLORS.get(group, (0.5, 0.5, 0.5))
        ax1.scatter(xs, ys, s=40, color=color, edgecolors="white",
                    linewidths=0.3, zorder=3)
        for idx, x, y in zip(idxs, xs, ys):
            ax1.annotate(_initials(researchers[idx]["firstname"], researchers[idx]["surname"]),
                         (x, y), textcoords="offset points", xytext=(4, 4),
                         fontsize=5, color="black", zorder=4)

    ax1.set_title("Official research groups", fontsize=11)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(False)

    # Right: Louvain communities
    comm_colors = plt.cm.Set3(np.linspace(0, 1, max(n_communities, 3)))
    for ci in range(n_communities):
        idxs = [i for i in range(n) if community_labels.get(i) == ci]
        xs = plot_coords[idxs, 0]
        ys = plot_coords[idxs, 1]
        ax2.scatter(xs, ys, s=40, color=comm_colors[ci % len(comm_colors)],
                    edgecolors="white", linewidths=0.3, zorder=3,
                    label=f"Community {ci} ({len(idxs)})")
        for idx, x, y in zip(idxs, xs, ys):
            ax2.annotate(_initials(researchers[idx]["firstname"], researchers[idx]["surname"]),
                         (x, y), textcoords="offset points", xytext=(4, 4),
                         fontsize=5, color="black", zorder=4)

    ax2.set_title(f"Louvain communities ({n_communities} found, overlap={overlap_ratio:.0%})",
                  fontsize=11)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(False)
    ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle("Official groups vs algorithmically discovered communities",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved community detection → %s", path)

    return {
        "n_communities": n_communities,
        "overlap_ratio": overlap_ratio,
        "communities": [
            {
                "id": ci,
                "size": len(comm),
                "members": [researchers[i]["display_name"] for i in sorted(comm)],
                "dominant_group": max(
                    ((researchers[i]["group"], 1) for i in comm),
                    key=lambda x: sum(1 for j in comm if researchers[j]["group"] == x[0])
                )[0],
            }
            for ci, comm in enumerate(communities)
        ],
    }


# ========================================================================== #
# 2. GROUP COLLABORATION MATRIX — 9×9 inter-group heatmap
# ========================================================================== #
def plot_group_collaboration(
    C: np.ndarray,
    researchers: list[dict],
    path: str | Path,
    figsize: tuple[float, float] = (10, 9),
) -> None:
    """
    9×9 heatmap showing total co-authorships between each pair of groups.
    Diagonal = within-group collaboration, off-diagonal = cross-group.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]
    ng = len(present_groups)

    G = np.zeros((ng, ng), dtype=int)
    for i in range(len(researchers)):
        for j in range(i + 1, len(researchers)):
            if C[i, j] > 0:
                gi = present_groups.index(researchers[i]["group"]) if researchers[i]["group"] in present_groups else -1
                gj = present_groups.index(researchers[j]["group"]) if researchers[j]["group"] in present_groups else -1
                if gi >= 0 and gj >= 0:
                    G[gi, gj] += int(C[i, j])
                    G[gj, gi] += int(C[i, j])

    # Short labels
    short_labels = [
        g.replace(" Group", "").replace(" Laboratory", " Lab")
        .replace("Research", "Res.").replace("Technologies", "Tech.")
        .replace("Engineering", "Eng.").replace("Interdisciplinary", "Interdisc.")
        .replace("Statistical", "Stat.")
        for g in present_groups
    ]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(G, cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(ng))
    ax.set_yticks(range(ng))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    # Annotate cells with values
    for i in range(ng):
        for j in range(ng):
            val = G[i, j]
            if val > 0:
                color = "white" if val > G.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Total co-authored publications")
    ax.set_title("Inter-group collaboration matrix", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved group collaboration matrix → %s", path)

    # Log summary
    logger.info("Group collaboration (top cross-group):")
    pairs = []
    for i in range(ng):
        for j in range(i + 1, ng):
            if G[i, j] > 0:
                pairs.append((G[i, j], short_labels[i], short_labels[j]))
    pairs.sort(reverse=True)
    for count, a, b in pairs[:10]:
        logger.info("  %s ↔ %s: %d joint pubs", a, b, count)


# ========================================================================== #
# 3. t-SNE — Third embedding method
# ========================================================================== #
def tsne_embed(
    D: np.ndarray,
    researchers: list[dict],
    path: str | Path,
    perplexity: float = 15.0,
    figsize: tuple[float, float] = (14, 10),
) -> np.ndarray:
    """
    t-SNE embedding as a third method alongside PCA and SMACOF.
    t-SNE is non-linear and excels at preserving local neighborhood
    structure, often revealing tighter clusters than MDS.
    """
    from sklearn.manifold import TSNE
    from src.plot import plot_embedding

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=min(perplexity, len(researchers) - 1),
        random_state=42,
        init="random",
        max_iter=2000,
    )
    coords = tsne.fit_transform(D)

    kl_divergence = float(tsne.kl_divergence_)
    logger.info("t-SNE: KL divergence = %.4f (perplexity=%.0f)", kl_divergence, perplexity)

    plot_embedding(
        coords, researchers,
        title=f"DMSTI Co-authorship — t-SNE  (KL div = {kl_divergence:.2f})",
        path=path,
    )

    return coords