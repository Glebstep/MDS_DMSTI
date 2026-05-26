"""
Visualization: styled scatter plots and quality-assessment diagrams.

Produces publication-quality PNGs that resemble the reference image
provided in the task description (dmsti_workers.png).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent color palette — nine research groups + PhD fallback.
GROUP_ORDER = [
    "Blockchain and Quantum Technologies Group",
    "Artificial Intelligence Laboratory",
    "Education Systems Group",
    "Global Optimization Group",
    "Intelligent Technologies Research Group",
    "Cyber-Social Systems Engineering Group",
    "Cognitive Computing Group",
    "Interdisciplinary Statistical Research Group",
    "Image and Signal Analysis Group",
    "PhD Students",
]
_TAB10 = plt.cm.tab10.colors
GROUP_COLORS = {g: _TAB10[i] for i, g in enumerate(GROUP_ORDER)}


def _initials(firstname: str, surname: str) -> str:
    fi = firstname[0] if firstname else "?"
    si = surname[0] if surname else "?"
    return fi + si


def plot_embedding(
    coords: np.ndarray,
    researchers: list[dict],
    title: str,
    path: str | Path,
    *,
    figsize: tuple[float, float] = (14, 10),
    point_size: int = 60,
    font_size: int = 8,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize coordinates to [-1, 1] for clean axes
    coords = coords.copy()
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs

    fig, ax = plt.subplots(figsize=figsize)

    # Collect which groups are actually present in the data
    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]

    for group in present_groups:
        idxs = [i for i, r in enumerate(researchers) if r["group"] == group]
        if not idxs:
            continue
        xs = coords[idxs, 0]
        ys = coords[idxs, 1]
        color = GROUP_COLORS.get(group, (0.5, 0.5, 0.5))
        ax.scatter(xs, ys, s=point_size, label=group, color=color,
                   edgecolors="white", linewidths=0.3, zorder=3)

        for idx, x, y in zip(idxs, xs, ys):
            r = researchers[idx]
            lbl = _initials(r["firstname"], r["surname"])
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=font_size,
                color="black",
                zorder=4,
            )

    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(
        title="Groups",
        fontsize=7,
        title_fontsize=8,
        loc="upper right",
        framealpha=0.9,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


def plot_shepard(
    d_original: np.ndarray,
    d_embedded: np.ndarray,
    title: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(d_original, d_embedded, s=2, alpha=0.3, color="steelblue")

    lo = 0
    hi = max(d_original.max(), d_embedded.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, label="ideal (y = x)")

    corr = float(np.corrcoef(d_original, d_embedded)[0, 1])
    ax.set_title(f"{title}  (r = {corr:.4f})", fontsize=12)
    ax.set_xlabel("Original dissimilarity")
    ax.set_ylabel("Embedded Euclidean distance")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Shepard diagram → %s  (r=%.4f)", path, corr)
