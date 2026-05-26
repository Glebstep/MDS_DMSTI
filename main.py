#!/usr/bin/env python3
"""
Scientometric Data Mining and Visualization
Visualizes VU DMSTI research-group staff on a 2-D plane based on their
co-authored publications.

Usage:
    python main.py                  # standard run
    python main.py --refresh        # re-fetch from MII + eLABa
    python main.py --output results # write to ./results/

Pipeline:
    1. Scrape research-group roster + additional staff (lecturers, project
       employees) from MII
    2. Scrape publications for everyone from eLABa
    3. Parse, deduplicate, resolve authors -> co-authorship matrix
    4. For additional staff with DMSTI co-authorships, assign the group
       of their primary research-group co-author
    5. Filter out isolated people (no DMSTI co-authors)
    6. Transform to dissimilarity (two methods: profile-based + direct)
    7. Embed via PCA + SMACOF, produce visualizations
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np


def assign_additional_staff_groups(
    researchers: list[dict],
    C: np.ndarray,
    log: logging.Logger,
) -> None:
    """
    For people with group=None (lecturers / project employees), find the
    research-group member they co-authored with most and inherit that group.
    If no co-authorships exist, group stays None (will be filtered out).
    """
    staff_idxs = [i for i, r in enumerate(researchers) if r["group"] is not None]
    candidate_idxs = [i for i, r in enumerate(researchers) if r["group"] is None]

    assigned = 0
    for ci in candidate_idxs:
        coauth = [(C[ci, si], si) for si in staff_idxs if C[ci, si] > 0]
        coauth.sort(reverse=True)
        if coauth:
            best_count, best_si = coauth[0]
            researchers[ci]["group"] = researchers[best_si]["group"]
            log.info(
                "  %s → %s (via %s, %d joint pubs)",
                researchers[ci]["display_name"],
                researchers[best_si]["group"],
                researchers[best_si]["display_name"],
                best_count,
            )
            assigned += 1

    log.info("Assigned %d/%d additional staff to groups", assigned, len(candidate_idxs))



def filter_isolated(
    researchers: list[dict],
    C: np.ndarray,
    log: logging.Logger,
) -> tuple[list[dict], np.ndarray]:
    """Remove people with zero co-authored publications with any DMSTI member."""
    keep = [i for i in range(len(researchers)) if C[i].sum() > 0]
    removed = [researchers[i]["display_name"] for i in range(len(researchers)) if C[i].sum() == 0]
    if removed:
        log.info("Filtered %d isolated: %s", len(removed), ", ".join(removed))
    # Also remove those whose group is still None (additional staff with no DMSTI co-authors)
    keep2 = [i for i in keep if researchers[i]["group"] is not None]
    none_removed = [researchers[i]["display_name"] for i in keep if researchers[i]["group"] is None]
    if none_removed:
        log.info("Filtered %d with no group: %s", len(none_removed), ", ".join(none_removed))
    
    filtered_rs = [researchers[i] for i in keep2]
    filtered_C = C[np.ix_(keep2, keep2)]
    log.info("After filtering: %d researchers remain", len(filtered_rs))
    return filtered_rs, filtered_C


def run_embedding_pipeline(
    D: np.ndarray,
    researchers: list[dict],
    label: str,
    output_dir: Path,
    log: logging.Logger,
) -> None:
    from src.embed import pca_embed, smacof_embed, align_procrustes, shepard_data
    from src.plot import plot_embedding, plot_shepard

    out = output_dir / label
    out.mkdir(parents=True, exist_ok=True)

    pca_coords, explained_var = pca_embed(D)
    smacof_coords, stress = smacof_embed(D, n_init=10)
    smacof_aligned = align_procrustes(pca_coords, smacof_coords)

    plot_embedding(
        pca_coords, researchers,
        title=f"DMSTI Co-authorship — PCA  (explained var = {explained_var:.1%})  [{label}]",
        path=out / "pca.png",
    )
    plot_embedding(
        smacof_aligned, researchers,
        title=f"DMSTI Co-authorship — SMACOF  (stress = {stress:.4f})  [{label}]",
        path=out / "smacof.png",
    )
    plot_embedding(
        smacof_aligned, researchers,
        title=f"DMSTI staff — co-authored publications  [{label}]",
        path=out / "main_visualization.png",
    )

    d_orig_pca, d_emb_pca = shepard_data(D, pca_coords)
    plot_shepard(d_orig_pca, d_emb_pca, f"Shepard — PCA [{label}]", out / "shepard_pca.png")

    d_orig_sm, d_emb_sm = shepard_data(D, smacof_aligned)
    plot_shepard(d_orig_sm, d_emb_sm, f"Shepard — SMACOF [{label}]", out / "shepard_smacof.png")

    log.info("[%s] PCA expl.var: %.1f%%, SMACOF stress: %.4f", label, explained_var * 100, stress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

    from src.scrape import fetch_researchers, fetch_additional_staff, fetch_publications, _dedup_key

    researchers = fetch_researchers(refresh=args.refresh)
    log.info("Research group staff: %d in %d groups",
             len(researchers), len({r["group"] for r in researchers}))

    existing_keys = {_dedup_key(r["firstname"], r["surname"]) for r in researchers}
    additional = fetch_additional_staff(existing_keys, refresh=args.refresh)
    researchers.extend(additional)
    log.info("+ %d additional staff (lecturers, project employees) → %d total",
             len(additional), len(researchers))
    
    existing_keys = {_dedup_key(r["firstname"], r["surname"]) for r in researchers}
    from src.scrape import fetch_phd_students
    phd = fetch_phd_students(existing_keys, refresh=args.refresh)
    researchers.extend(phd)
    log.info("+ %d PhD students → %d total", len(phd), len(researchers))

    publications = fetch_publications(researchers, refresh=args.refresh)
    total_raw = sum(len(v) for v in publications.values())
    log.info("Loaded %d raw publication records", total_raw)


    from src.parse import resolve_publications
    from src.matrix import build_coauthor_matrix, to_dissimilarity, to_dissimilarity_direct

    pub_to_authors = resolve_publications(publications, researchers)
    C_full = build_coauthor_matrix(pub_to_authors, len(researchers))


    log.info("Assigning groups to additional staff by co-authorship:")
    assign_additional_staff_groups(researchers, C_full, log)
    # Official group overrides from group pages (scraped from
    # /structure/scientific-subdivisions/*). These take precedence
    # over automatic co-authorship assignment.
    GROUP_OVERRIDES = {
        "Ansarian Najaf Abadi Sasan": "Blockchain and Quantum Technologies Group",
        "Bagočienė Snieguolė": "Education Systems Group",
        "Bielskis Aivaras": "Intelligent Technologies Research Group",
        "Briliauskas Mantas": "Intelligent Technologies Research Group",
        "Bulavas Viktoras": "Cognitive Computing Group",
        "Dastgeer Sobia": "Image and Signal Analysis Group",
        "Dautartas Juozas": "Blockchain and Quantum Technologies Group",
        "Gervytė Miglė": "Interdisciplinary Statistical Research Group",
        "Gricius Rolandas": "Intelligent Technologies Research Group",
        "Grigaitis Saulius": "Intelligent Technologies Research Group",
        "Jurčenko Ilja": "Intelligent Technologies Research Group",
        "Juškys Raimondas": "Interdisciplinary Statistical Research Group",
        "Karlauskas Kasparas": "Image and Signal Analysis Group",
        "Kuzma Lukas": "Intelligent Technologies Research Group",
        "Mikalkėnienė Gajane": "Interdisciplinary Statistical Research Group",
        "Nakvosas Artūras": "Intelligent Technologies Research Group",
        "Petrėtis Aurimas": "Global Optimization Group",
        "Ramonaitė Justina": "Image and Signal Analysis Group",
        "Rimšelis Jonas Mindaugas": "Image and Signal Analysis Group",
        "Rizgelienė Ieva": "Intelligent Technologies Research Group",
        "Sabaliauskas Darius": "Cyber-Social Systems Engineering Group",
        "Sellapperuma Sathuta Piripun": "Global Optimization Group",
        "Skuodis Algimantas": "Cognitive Computing Group",
        "Stakauskas Brendonas": "Intelligent Technologies Research Group",
        "Surkant Roman": "Image and Signal Analysis Group",
        "Urbonaitė Neringa": "Intelligent Technologies Research Group",
        "Vaišnorė Ramunė": "Interdisciplinary Statistical Research Group",
        "Vitkauskaitė Akvilė": "Interdisciplinary Statistical Research Group",
        "Vitkauskas Jonas": "Global Optimization Group",
        "Zakševski Daniel": "Image and Signal Analysis Group",
        "Šablauskas Karolis": "Interdisciplinary Statistical Research Group",
    }
    for r in researchers:
        if r["display_name"] in GROUP_OVERRIDES:
            old = r["group"]
            r["group"] = GROUP_OVERRIDES[r["display_name"]]
            if old != r["group"]:
                log.info("Override: %s → %s (was: %s)", r["display_name"], r["group"], old)

    confirmed_names = {r["display_name"] for r in fetch_researchers()} | set(GROUP_OVERRIDES.keys())
    for r in researchers:
        if r["display_name"] not in confirmed_names:
            r["group"] = None
    log.info("Marked %d unconfirmed for removal",
             sum(1 for r in researchers if r["group"] is None))
    researchers, C = filter_isolated(researchers, C_full, log)

    np.save(args.output / "coauthor_matrix.npy", C)


    D_profile = to_dissimilarity(C)
    run_embedding_pipeline(D_profile, researchers, "profile", args.output, log)

    D_direct = to_dissimilarity_direct(C)
    run_embedding_pipeline(D_direct, researchers, "direct", args.output, log)


    from src.analysis import plot_heatmap, plot_side_by_side, plot_network, compute_centrality
    from src.embed import pca_embed, smacof_embed, align_procrustes

    # Use profile-based dissimilarity for analysis (better Shepard r)
 
    pca_coords, ev = pca_embed(D_profile)
    smacof_coords, st = smacof_embed(D_profile, n_init=10)
    smacof_aligned = align_procrustes(pca_coords, smacof_coords)

    plot_heatmap(C, researchers, args.output / "heatmap.png")
    plot_side_by_side(
        pca_coords, smacof_aligned, researchers,
        f"PCA (explained var = {ev:.1%})",
        f"SMACOF (stress = {st:.4f})",
        args.output / "pca_vs_smacof.png",
    )
    plot_network(C, researchers, smacof_aligned, args.output / "network.png")
    centrality = compute_centrality(C, researchers, args.output / "centrality.png")

    from src.interactive import build_interactive
    build_interactive(
        smacof_aligned, C, researchers,
        title="DMSTI Co-authorship Network — Interactive Explorer",
        path=args.output / "interactive.html",
    )

    from src.analysis_extra import detect_communities, plot_group_collaboration, tsne_embed

    comm_result = detect_communities(C, researchers, smacof_aligned, args.output / "communities.png")
    plot_group_collaboration(C, researchers, args.output / "group_collaboration.png")
    tsne_embed(D_profile, researchers, args.output / "tsne.png")

    from src.temporal import plot_year_group, plot_year_year
    plot_year_group(publications, researchers, args.output / "temporal_year_group.png")
    plot_year_year(publications, researchers, args.output / "temporal_year_year.png")

  
    log.info("-" * 60)
    log.info("Pipeline complete.")
    log.info("  Researchers (final): %d", len(researchers))
    log.info("  Unique publications: %d", len(pub_to_authors))
    log.info("  Outputs in:          %s/", args.output)
    log.info("-" * 60)


if __name__ == "__main__":
    main()
