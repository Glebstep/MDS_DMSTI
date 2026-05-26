"""
Advanced analysis and visualization module.

Functions:
  * plot_heatmap():      Co-authorship matrix heatmap sorted by group
  * plot_side_by_side(): PCA vs SMACOF on one figure, Procrustes-aligned
  * plot_network():      Co-authorship network with edges and group colors
  * compute_centrality(): Betweenness centrality -> bridge researchers
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

from src.plot import GROUP_ORDER, GROUP_COLORS, _initials

logger = logging.getLogger(__name__)



def plot_heatmap(
    C: np.ndarray,
    researchers: list[dict],
    path: str | Path,
    figsize: tuple[float, float] = (18, 16),
) -> None:
    """
    Heatmap of the n×n co-authorship matrix, rows and columns sorted by
    research group.  Block-diagonal structure reveals within-group
    collaboration density; off-diagonal blocks show cross-group bridges.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort researchers by group (following GROUP_ORDER), then alphabetically
    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    order = sorted(
        range(len(researchers)),
        key=lambda i: (
            group_rank.get(researchers[i]["group"], 99),
            researchers[i]["surname"],
        ),
    )

    C_sorted = C[np.ix_(order, order)]
    labels = [
        _initials(researchers[i]["firstname"], researchers[i]["surname"])
        for i in order
    ]
    full_labels = [
        f"{researchers[i]['surname']}, {researchers[i]['firstname'][0]}."
        for i in order
    ]

    fig, ax = plt.subplots(figsize=figsize)

    # Use log1p scale for better visual contrast (most values are small,
    # a few are very large like Dzemyda-Kurasova=53)
    im = ax.imshow(
        np.log1p(C_sorted),
        cmap="YlOrRd",
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(full_labels, rotation=90, fontsize=5, ha="center")
    ax.set_yticklabels(full_labels, fontsize=5)

    # Draw group separator lines
    prev_group = None
    for idx, i in enumerate(order):
        g = researchers[i]["group"]
        if prev_group is not None and g != prev_group:
            ax.axhline(idx - 0.5, color="black", linewidth=0.8)
            ax.axvline(idx - 0.5, color="black", linewidth=0.8)
        prev_group = g

    # Color bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="log(1 + co-authored publications)")

    # Group labels on the side
    group_starts = {}
    for idx, i in enumerate(order):
        g = researchers[i]["group"]
        if g not in group_starts:
            group_starts[g] = idx

    for g, start in group_starts.items():
        members = [idx for idx, i in enumerate(order) if researchers[i]["group"] == g]
        mid = (members[0] + members[-1]) / 2
        color = GROUP_COLORS.get(g, (0.5, 0.5, 0.5))
        ax.annotate(
            g.replace(" Group", "").replace(" Laboratory", " Lab"),
            xy=(len(labels) + 1, mid),
            fontsize=6,
            color=color,
            fontweight="bold",
            va="center",
            annotation_clip=False,
        )

    ax.set_title("Co-authorship matrix (sorted by research group)", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap → %s", path)


def plot_side_by_side(
    pca_coords: np.ndarray,
    smacof_coords: np.ndarray,
    researchers: list[dict],
    pca_label: str,
    smacof_label: str,
    path: str | Path,
    figsize: tuple[float, float] = (22, 10),
) -> None:
    """
    Two scatter plots side by side: PCA (left) and SMACOF (right),
    both normalized to [-1,1] for visual comparability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    def _draw(ax, coords, title):
        coords = coords.copy()
        max_abs = np.abs(coords).max()
        if max_abs > 0:
            coords = coords / max_abs

        present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]
        for group in present_groups:
            idxs = [i for i, r in enumerate(researchers) if r["group"] == group]
            if not idxs:
                continue
            xs, ys = coords[idxs, 0], coords[idxs, 1]
            color = GROUP_COLORS.get(group, (0.5, 0.5, 0.5))
            ax.scatter(xs, ys, s=50, color=color, edgecolors="white",
                       linewidths=0.3, zorder=3)
            for idx, x, y in zip(idxs, xs, ys):
                r = researchers[idx]
                ax.annotate(
                    _initials(r["firstname"], r["surname"]),
                    (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="black", zorder=4,
                )
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(False)

    _draw(ax1, pca_coords, pca_label)
    _draw(ax2, smacof_coords, smacof_label)

    # Shared legend
    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]
    handles = [
        mpatches.Patch(color=GROUP_COLORS.get(g, (0.5, 0.5, 0.5)), label=g)
        for g in present_groups
    ]
    fig.legend(
        handles=handles, title="Groups", fontsize=7, title_fontsize=8,
        loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
        framealpha=0.9,
    )

    fig.suptitle("PCA vs SMACOF — embedding comparison", fontsize=14, y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved side-by-side → %s", path)



def plot_network(
    C: np.ndarray,
    researchers: list[dict],
    coords: np.ndarray,
    path: str | Path,
    min_edge_weight: int = 2,
    figsize: tuple[float, float] = (18, 14),
) -> None:
    """
    Co-authorship network graph. Node positions from SMACOF embedding,
    edge thickness proportional to number of joint publications.
    Only edges with weight >= min_edge_weight are drawn to reduce clutter.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(researchers)
    G = nx.Graph()

    # Add nodes
    for i, r in enumerate(researchers):
        G.add_node(i, name=f"{r['firstname']} {r['surname']}", group=r["group"])

    # Add edges
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= min_edge_weight:
                G.add_edge(i, j, weight=int(C[i, j]))

    # Normalize coords for positions
    pos_coords = coords.copy()
    max_abs = np.abs(pos_coords).max()
    if max_abs > 0:
        pos_coords = pos_coords / max_abs
    pos = {i: (pos_coords[i, 0], pos_coords[i, 1]) for i in range(n)}

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges with alpha/width proportional to weight
    edges = G.edges(data=True)
    if edges:
        weights = [e[2]["weight"] for e in edges]
        max_w = max(weights) if weights else 1
        for u, v, data in edges:
            w = data["weight"]
            alpha = 0.1 + 0.5 * (w / max_w)
            width = 0.3 + 2.5 * (w / max_w)
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color="gray", alpha=alpha, linewidth=width, zorder=1,
            )

    # Draw nodes by group
    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]
    for group in present_groups:
        idxs = [i for i, r in enumerate(researchers) if r["group"] == group]
        xs = [pos[i][0] for i in idxs]
        ys = [pos[i][1] for i in idxs]
        color = GROUP_COLORS.get(group, (0.5, 0.5, 0.5))
        # Node size proportional to total co-authorships
        sizes = [min(30 + 1.5 * C[i].sum(), 120) for i in idxs]
        ax.scatter(xs, ys, s=sizes, color=color, edgecolors="white",
                   linewidths=0.5, zorder=3, label=group)

    # Labels
    for i, r in enumerate(researchers):
        lbl = _initials(r["firstname"], r["surname"])
        ax.annotate(
            lbl, pos[i], fontsize=5, color="black", zorder=4,
            ha="center", va="center",
        )

    ax.legend(title="Groups", fontsize=6, title_fontsize=7,
              loc="upper right", framealpha=0.9)
    ax.set_title(
        f"Co-authorship network  (edges ≥ {min_edge_weight} joint pubs, "
        f"{G.number_of_edges()} edges shown)",
        fontsize=12, pad=12,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved network graph → %s (%d nodes, %d edges)",
                path, G.number_of_nodes(), G.number_of_edges())



def compute_centrality(
    C: np.ndarray,
    researchers: list[dict],
    path: str | Path,
    top_n: int = 15,
) -> list[dict]:
    """
    Compute betweenness centrality on the co-authorship graph.
    High betweenness = a researcher who bridges different clusters.

    Returns top_n bridge researchers and saves a bar chart.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(researchers)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                # Invert weight: more co-authorships = stronger connection
                # = shorter path for betweenness
                G.add_edge(i, j, weight=1.0 / C[i, j])

    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)

    ranked = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for idx, score in ranked:
        r = researchers[idx]
        total_coauth = int(C[idx].sum())
        n_coauthors = int((C[idx] > 0).sum())
        results.append({
            "name": f"{r['firstname']} {r['surname']}",
            "group": r["group"],
            "betweenness": score,
            "total_coauthorships": total_coauth,
            "unique_coauthors": n_coauthors,
        })

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    names = [f"{res['name']}" for res in results]
    scores = [res["betweenness"] for res in results]
    colors = [
        GROUP_COLORS.get(res["group"], (0.5, 0.5, 0.5))
        for res in results
    ]

    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Betweenness centrality (normalized)", fontsize=10)
    ax.set_title("Top bridge researchers — connecting different groups", fontsize=12, pad=10)

    # Add group labels to bars
    for i, res in enumerate(results):
        short = res["group"].replace(" Group", "").replace(" Laboratory", " Lab")
        ax.text(
            scores[i] + max(scores) * 0.01, i,
            f"  {short}  ({res['unique_coauthors']} co-authors)",
            va="center", fontsize=7, color="gray",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved centrality chart → %s", path)

    # Log top results
    logger.info("Top %d bridge researchers:", top_n)
    for res in results:
        logger.info(
            "  %-30s %-45s bc=%.4f  (%d co-authors)",
            res["name"], res["group"], res["betweenness"], res["unique_coauthors"],
        )

    return results