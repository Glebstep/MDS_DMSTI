"""
Temporal analysis of co-authorship patterns.

Functions:
  * plot_year_group():  Year × Group heatmap — when was each group most active
  * plot_year_year():   Year × Year heatmap — which years share collaborating pairs
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.parse import parse_publication, publication_key, build_researcher_index, match_author
from src.plot import GROUP_ORDER, GROUP_COLORS

logger = logging.getLogger(__name__)


def _extract_temporal_edges(
    publications_by_author: dict[str, list[str]],
    researchers: list[dict],
) -> list[tuple[int, set[int]]]:
    """
    Parse all publications and return [(year, {author_idx, ...}), ...].
    Only publications with year and ≥2 DMSTI authors are included.
    """
    by_surname = build_researcher_index(researchers)
    seen_keys: set[str] = set()
    edges: list[tuple[int, set[int]]] = []

    for author_key, raw_list in publications_by_author.items():
        for raw in raw_list:
            pub = parse_publication(raw)
            if pub is None or pub["year"] is None:
                continue
            key = publication_key(pub)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            roster_ids: set[int] = set()
            for sn, fn in pub["authors"]:
                ri = match_author(sn, fn, researchers, by_surname)
                if ri is not None:
                    roster_ids.add(ri)

            if len(roster_ids) >= 2:
                edges.append((pub["year"], roster_ids))

    return edges


def plot_year_group(
    publications_by_author: dict[str, list[str]],
    researchers: list[dict],
    path: str | Path,
    year_range: tuple[int, int] = (2000, 2025),
    figsize: tuple[float, float] = (16, 8),
) -> None:
    """
    Year × Group heatmap: number of co-authored publications per group per year.
    A publication counts for group G if at least one of its DMSTI authors
    belongs to G.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    edges = _extract_temporal_edges(publications_by_author, researchers)

    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]
    group_idx = {g: i for i, g in enumerate(present_groups)}

    years = list(range(year_range[0], year_range[1] + 1))
    year_idx = {y: i for i, y in enumerate(years)}

    M = np.zeros((len(present_groups), len(years)), dtype=int)

    for year, author_ids in edges:
        if year not in year_idx:
            continue
        yi = year_idx[year]
        groups_in_pub = set()
        for ai in author_ids:
            g = researchers[ai]["group"]
            if g in group_idx:
                groups_in_pub.add(g)
        for g in groups_in_pub:
            M[group_idx[g], yi] += 1

    # Short labels
    short_labels = [
        g.replace(" Group", "").replace(" Laboratory", " Lab")
        .replace("Research", "Res.").replace("Technologies", "Tech.")
        .replace("Engineering", "Eng.").replace("Interdisciplinary", "Interdisc.")
        .replace("Statistical", "Stat.")
        for g in present_groups
    ]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=7)
    ax.set_yticks(range(len(present_groups)))
    ax.set_yticklabels(short_labels, fontsize=9)

    # Annotate cells with values (only non-zero)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                color = "white" if M[i, j] > M.max() * 0.6 else "black"
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        fontsize=5, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Co-authored publications involving group")
    ax.set_title("Co-authorship activity by year and research group", fontsize=13, pad=12)
    ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved year×group heatmap → %s", path)


def plot_year_year(
    publications_by_author: dict[str, list[str]],
    researchers: list[dict],
    path: str | Path,
    year_range: tuple[int, int] = (2000, 2025),
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """
    Year × Year heatmap: cell (y1, y2) = number of unique researcher pairs
    who co-authored in BOTH year y1 and year y2.

    Diagonal = total unique co-author pairs active that year.
    Off-diagonal = collaboration stability: high values mean the same
    pairs keep publishing together across years.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    edges = _extract_temporal_edges(publications_by_author, researchers)

    years = list(range(year_range[0], year_range[1] + 1))
    year_idx = {y: i for i, y in enumerate(years)}

    # For each year, collect unique co-author pairs
    pairs_by_year: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for year, author_ids in edges:
        if year not in year_idx:
            continue
        ids = sorted(author_ids)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs_by_year[year].add((ids[a], ids[b]))

    # Build year×year overlap matrix
    ny = len(years)
    M = np.zeros((ny, ny), dtype=int)
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            if j >= i:
                overlap = len(pairs_by_year[y1] & pairs_by_year[y2])
                M[i, j] = overlap
                M[j, i] = overlap

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(ny))
    ax.set_yticks(range(ny))
    ax.set_xticklabels(years, rotation=90, fontsize=7)
    ax.set_yticklabels(years, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="Shared co-author pairs between years")
    ax.set_title("Collaboration stability: shared co-author pairs across years",
                 fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved year×year heatmap → %s", path)