# Scientometric Data Mining and Visualization - VU DMSTI

**Author:** Hlib Stepanenko  
**Task:** Visualize VU DMSTI research staff on a 2D plane based on co-authored publications  
**Supervisor:** Martynas Sabaliauskas

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py            # full pipeline (~2 min with cached data)
open output/interactive.html   # interactive exploration in browser
```

All outputs are generated in `output/`. Cached data lives in `data/` and is reused automatically; pass `--refresh` to re-scrape from MII and eLABa.

---

## 1. Data Collection

Three data sources are scraped from the MII website and eLABa repository:

| Source | URL | What we get |
|--------|-----|------------|
| MII "By Department" page | `mii.lt/en/structure/staff-2/by-departmentt` | 72 research staff in 9 groups |
| MII PhD student list | `mii.lt/en/doctoral-studies/phd-students` | 26 PhD students |
| MII research group pages | `mii.lt/en/structure/scientific-subdivisions/*` | Official group membership for additional staff |
| eLABa | `elaba.mb.vu.lt/dmsti/?aut=...` | Publication records per researcher |

The pipeline fetches 135 people total. Publications are parsed, deduplicated by DOI and normalized title, and resolved against the full roster. This yields **2,490 unique publications**, of which **720 link two or more DMSTI members** and contribute edges to the co-authorship graph.

### Scraping Challenges and Solutions

Several non-trivial issues arose during data collection that required analytical problem-solving:

- **Whitespace normalization:** The "Blockchain and Quantum Technologies Group" heading on the MII page contained a triple-space character sequence, causing it to be missed by exact string matching. Solved by normalizing all whitespace before comparison.
- **Navigation vs. content ambiguity:** Attempting to scrape official group affiliations from researcher profile pages initially returned the navigation menu group links instead of the actual "Department:" field. Solved by targeting only `<a>` tags within the Department field that link to `/scientific-subdivisions/`.
- **Lithuanian name inflection:** eLABa lists authors in citation format (sometimes with inflected Lithuanian surname endings), while MII uses display names. Author matching uses transliteration (`unidecode`) and surname + first-initial keys, tolerant of diacritics and case variations.
- **Publication deduplication:** The same paper appears under multiple co-authors' eLABa queries. Deduplication uses DOI as primary key (strongest signal) and falls back to normalized title + year for papers without DOI.
- **Surname collisions:** Two researchers named Žilinskas (Antanas and Julius) require disambiguation by first-name initial, not just surname matching.

### Group Assignment

Research group staff (72 people) receive their group directly from the MII "By Department" page headings. For PhD students and additional staff, we scrape all 9 official group pages and build a verified mapping of 31 people to their official groups. Following the supervisor's guidance, lecturers without an explicit research group affiliation on their profile are excluded from the visualization.

A notable case: **Lukas Kuzma** is listed under lecturers but his profile explicitly states affiliation with the Intelligent Technologies Research Group. By co-authorship, his strongest connection is to Martynas Sabaliauskas (Cognitive Computing Group, 4 joint publications). His position on the plot thus reflects his real collaboration pattern, while his color reflects his formal affiliation — illustrating the difference between organizational structure and actual research collaboration.

### Filtering

Following the supervisor's instructions, we remove:
- **28 isolated researchers** with zero co-authored publications with any other DMSTI member
- **18 unconfirmed staff** whose profiles show "Department: Lecturer" without explicit research group affiliation

**Final dataset: 89 researchers across 9 groups.**

---------

## 2. Co-authorship Matrix

An 89×89 symmetric integer matrix **C** where C[i,j] counts the number of publications co-authored by researchers i and j. Key statistics:

- Non-zero pairs: 495 (out of 3,916 possible)
- Maximum co-authorships: 53 (Dzemyda — Kurasova)
- Network density: 12.6%

---------

## 3. Dissimilarity Transformation

Two dissimilarity approaches are implemented and compared:

### Profile-based (recommended)

1. **Inversion:** S[i,j] = 1 / (C[i,j] + 1) — transforms co-authorship counts into affinities in (0, 1]
2. **Profile distance:** D[i,j] = ‖S[i,:] − S[j,:]‖₂ — Euclidean distance between collaboration profiles

Two researchers are close when they co-author with the **same set of colleagues**, not just with each other. This produces a proper metric satisfying the triangle inequality, which is important for MDS convergence.

### Direct

D[i,j] = 1 / (C[i,j] + 1) — directly encodes co-authorship strength as proximity. Simpler and more intuitive (frequent co-authors are placed nearby), but the resulting matrix is not a proper Euclidean distance matrix, leading to weaker embedding quality.

### Comparison

| Metric | Profile | Direct |
|--------|---------|--------|
| PCA explained variance | **89.8%** | 67.6% |
| SMACOF stress | 1660 | **304** |
| Shepard r (PCA) | **0.94** | 0.32 |
| Shepard r (SMACOF) | **0.95** | 0.69 |

The profile method achieves near-perfect distance preservation (Shepard r = 0.95). The direct method has lower stress but poor Shepard correlation due to the degenerate structure where most pairs share dissimilarity = 1.0 (zero co-authorships), creating an unresolvable column in the Shepard diagram.

Both approaches provide complementary views: the profile method is mathematically superior for MDS embedding, while the direct method offers a more intuitive visual interpretation where co-authors appear as immediate neighbors.

---

## 4. Embedding Methods

Three dimensionality reduction methods are applied to the profile-based dissimilarity matrix:

### PCA (Classical MDS)

Projects rows of the dissimilarity matrix into 2D via principal component analysis. The first two components capture 89.8% of the variance, indicating that the co-authorship structure is largely two-dimensional.

### SMACOF

Scaling by MAjorizing a COmplicated Function — iterative stress minimization with 10 random restarts. Produces a non-linear embedding that directly minimizes the mismatch between original and embedded pairwise distances.

### t-SNE

t-distributed Stochastic Neighbor Embedding with perplexity=15. A non-linear method that emphasizes local neighborhood preservation. Produces tighter, more visually separated clusters than PCA or SMACOF, at the cost of not preserving global distances.

Embeddings are aligned via Procrustes analysis (Kabsch algorithm) for visual comparability in the side-by-side comparison.

---

## 5. Visualizations

All outputs are generated by `python main.py` into the `output/` directory:

| File | Description |
|------|-------------|
| `profile/pca.png` | PCA embedding (profile dissimilarity) |
| `profile/smacof.png` | SMACOF embedding (profile dissimilarity) |
| `profile/shepard_*.png` | Shepard diagrams for profile method |
| `direct/pca.png` | PCA embedding (direct dissimilarity) |
| `direct/smacof.png` | SMACOF embedding (direct dissimilarity) |
| `direct/shepard_*.png` | Shepard diagrams for direct method |
| `heatmap.png` | 89×89 co-authorship matrix sorted by group |
| `pca_vs_smacof.png` | Side-by-side PCA vs SMACOF comparison |
| `network.png` | Co-authorship network graph (495 edges ≥ 2 pubs) |
| `centrality.png` | Top 15 bridge researchers by betweenness centrality |
| `communities.png` | Official groups vs Louvain communities |
| `group_collaboration.png` | 9×9 inter-group co-authorship matrix |
| `tsne.png` | t-SNE embedding |
| `temporal_year_group.png` | Co-authorship activity by year and group |
| `temporal_year_year.png` | Collaboration stability across years |
| `interactive.html` | **Interactive Plotly explorer** — click any researcher to highlight their co-authorship connections |

--------------

## 6. Key Findings

### Cluster Structure

The 2D embeddings reveal clear group clustering, confirming that official research groups reflect real collaboration patterns. However, three groups form a tightly interconnected core:

- **Cognitive Computing** ↔ **Image and Signal Analysis**: 276 joint publications
- **Blockchain and Quantum Technologies** ↔ **Cognitive Computing**: 250 joint publications
- **Blockchain and Quantum Technologies** ↔ **Image and Signal Analysis**: 186 joint publications

**Education Systems** is the most self-contained group (376 internal publications, relatively few cross-group connections), consistent with its distinct research focus.

### Bridge Researchers

Betweenness centrality identifies researchers who connect different clusters:

| Rank | Researcher | Group | Betweenness | Co-authors |
|------|-----------|-------|-------------|------------|
| 1 | Gintautas Dzemyda | Cognitive Computing | 0.3715 | 62 |
| 2 | Olga Kurasova | Cognitive Computing | 0.2223 | 59 |
| 3 | Martynas Sabaliauskas | Cognitive Computing | 0.1943 | 55 |
| 4 | Igoris Belovas | Intelligent Technologies | 0.1890 | 47 |
| 5 | Ernestas Filatovas | Blockchain & Quantum | 0.1633 | 56 |

Cognitive Computing Group dominates the top-3 bridge positions, reflecting its central role in the DMSTI collaboration network.

### Community Detection

Louvain community detection discovers 10 communities (vs 9 official groups) with 48% overlap. The largest detected community (34 members) merges Image and Signal Analysis, parts of Cognitive Computing, and Blockchain — reflecting the dense cross-group co-authorship between these groups. The Intelligent Technologies group (7/7 overlap) and Education Systems group (7/9 overlap) are the most self-contained.

### Temporal Trends

- Co-authorship activity has grown steadily since 2000, with a peak around 2015-2016 (Cognitive Computing, Global Optimization) and sustained high activity in 2022-2025
- The year×year stability heatmap shows that recent years (2020-2025) share many co-author pairs, indicating stable long-term collaborations
- Image and Signal Analysis shows the strongest recent growth trajectory

---

## 7. Project Structure

```
mds_dmsti/
├── .venv/
├── data/
│   ├── additional_staff.json
│   ├── fixture_sabaliauskas.json
│   ├── phd_students.json
│   ├── publications.json
│   └── researchers.json
├── output/
│   ├── direct/
│   └── profile/
├── src/
│   ├── analysis.py
│   ├── analysis_extra.py
│   ├── embed.py
│   ├── interactive.py
│   ├── matrix.py
│   ├── parse.py
│   ├── plot.py
│   ├── scrape.py
│   └── temporal.py
├── main.py
└── requirements.txt
```

### Pipeline Stages (main.py)

1. **Data** — Scrape MII roster + eLABa publications (cached in `data/`)
2. **Matrix** — Parse, deduplicate, build 135×135 co-authorship matrix
3. **Groups** — Auto-assign groups by co-authorship -> override with official groups from MII group pages → filter unconfirmed lecturers and isolated researchers -> 89 remain
4. **Embedding** — Two dissimilarity methods × (PCA + SMACOF) -> `output/profile/` and `output/direct/`
5. **Analysis** — Heatmap, network graph, centrality, interactive HTML, community detection, group collaboration matrix, t-SNE, temporal heatmaps

---

## 8. Relation to Generalized MDS (GMDS)

**Generalized Multidimensional Scaling** (GMDS), introduced by Bronstein, Bronstein, and Kimmel, extends classical MDS beyond Euclidean spaces. While classical MDS embeds objects into Euclidean space by minimizing stress (the mismatch between original and embedded distances), GMDS generalizes this to arbitrary metric spaces — it finds a mapping between two metric spaces that minimizes geodesic distance distortion. This makes GMDS particularly powerful for problems where the underlying geometry is non-Euclidean, such as comparing shapes on manifolds or analyzing networks with graph-theoretic distances.

In our project, the profile-based dissimilarity method transforms the co-authorship count matrix into a proper Euclidean distance space (via L2 norm of inverted co-authorship profiles) before applying classical MDS. This is a deliberate design choice: it ensures the dissimilarity matrix satisfies the triangle inequality, which provides convergence guarantees for SMACOF.

However, the co-authorship network is inherently a discrete graph, and Euclidean profile distances may not fully capture its topology. A natural extension using the GMDS framework would be to:

1. Compute **shortest-path (geodesic) distances** on the co-authorship graph, preserving its graph-theoretic structure
2. Apply GMDS to embed these geodesic distances into 2D, allowing the algorithm to handle the non-Euclidean nature of graph distances directly
3. Compare the GMDS embedding with our profile-based MDS to quantify how much topological information is lost in the Euclidean approximation

This could better represent peripherally connected researchers (those linked through long chains of co-authorship) whose graph distance is large but whose profile distance may be misleadingly small if they happen to share common non-collaborators.

### Analytical Decisions

Several key decisions shaped the methodology:

- **Why two dissimilarity methods?** The task suggests `1/(a+1)` as a transformation. We implement this as the "direct" method but also develop a "profile-based" method that computes Euclidean distances between co-authorship profiles. The comparison reveals that profile-based is mathematically superior (Shepard r = 0.95 vs 0.69) but direct is more intuitive. Presenting both demonstrates understanding of the trade-offs.
- **Why include PhD students and additional staff?** Research groups listed on the MII "By Department" page include only core research staff. However, PhD students and some lecturers actively co-author with group members and contribute meaningful edges to the co-authorship network. Including them (when officially affiliated with a research group) gives a more complete picture.
- **Why filter unconfirmed lecturers?** Per the supervisor's guidance, the visualization should primarily include researchers belonging to research groups. Lecturers whose MII profile shows "Department: Lecturer" (without explicit group affiliation) are excluded to avoid assigning them groups based on co-authorship guesses rather than institutional facts.
- **Why three embedding methods?** PCA and SMACOF are required by the task. We add t-SNE as a third method because it excels at preserving local neighborhood structure, often revealing cluster boundaries that linear methods miss. The KL divergence metric provides an additional quality measure.

---

## Requirements

```
requests
beautifulsoup4
lxml
numpy
scipy
scikit-learn
matplotlib
networkx
unidecode
plotly
```
