"""
Interactive Plotly visualization.

Generates a standalone HTML file where:
  * Each researcher is a colored dot (by group)
  * Hover shows: name, group, total pubs, top-5 co-authors
  * Edges between co-authors visible as faint lines
  * Click a researcher → highlights their connections in bold
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_interactive(
    coords: np.ndarray,
    C: np.ndarray,
    researchers: list[dict],
    title: str,
    path: str | Path,
    min_edge_weight: int = 2,
) -> None:
    """Generate a standalone Plotly HTML file with interactive co-authorship map."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.error("plotly not installed. Run: pip install plotly")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(researchers)

    # Normalize coords
    coords = coords.copy()
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs


    from src.plot import GROUP_ORDER, GROUP_COLORS, _initials

    hover_texts = []
    for i, r in enumerate(researchers):
        # Top co-authors
        coauth_pairs = []
        for j in range(n):
            if C[i, j] > 0:
                coauth_pairs.append((int(C[i, j]), researchers[j]))
        coauth_pairs.sort(key=lambda x: x[0], reverse=True)

        total_coauth = int(C[i].sum())
        n_coauthors = sum(1 for _, _ in coauth_pairs)

        lines = [
            f"<b>{r['firstname']} {r['surname']}</b>",
            f"Group: {r['group']}",
            f"Co-authorships: {total_coauth} (with {n_coauthors} people)",
            "",
            "<b>Top co-authors:</b>",
        ]
        for count, cr in coauth_pairs[:7]:
            lines.append(
                f"  • {cr['firstname']} {cr['surname']} "
                f"({cr['group'].split()[0]}): {count} pubs"
            )
        if len(coauth_pairs) > 7:
            lines.append(f"  ... and {len(coauth_pairs) - 7} more")

        hover_texts.append("<br>".join(lines))


    group_colors_hex = {}
    for g, rgb in GROUP_COLORS.items():
        r_val, g_val, b_val = [int(c * 255) for c in rgb[:3]]
        group_colors_hex[g] = f"rgb({r_val},{g_val},{b_val})"


    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= min_edge_weight:
                edge_x.extend([coords[i, 0], coords[j, 0], None])
                edge_y.extend([coords[i, 1], coords[j, 1], None])

    fig = go.Figure()

    # Background edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="rgba(180,180,180,0.15)", width=0.5),
        hoverinfo="skip",
        showlegend=False,
        name="co-authorship edges",
    ))


    # We'll add a separate trace per researcher for their personal edges
    # These are initially hidden and toggled via JavaScript
    personal_edges = {}
    for i in range(n):
        ex, ey, texts = [], [], []
        for j in range(n):
            if C[i, j] > 0:
                w = int(C[i, j])
                ex.extend([coords[i, 0], coords[j, 0], None])
                ey.extend([coords[i, 1], coords[j, 1], None])
        personal_edges[i] = (ex, ey)

    present_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in researchers)]

    for group in present_groups:
        idxs = [i for i, r in enumerate(researchers) if r["group"] == group]
        if not idxs:
            continue

        xs = [coords[i, 0] for i in idxs]
        ys = [coords[i, 1] for i in idxs]
        texts = [hover_texts[i] for i in idxs]
        labels = [_initials(researchers[i]["firstname"], researchers[i]["surname"]) for i in idxs]
        sizes = [min(8 + 0.15 * C[i].sum(), 35) for i in idxs]
        color = group_colors_hex.get(group, "rgb(128,128,128)")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=color,
                line=dict(color="white", width=1),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color="black"),
            hovertext=texts,
            hoverinfo="text",
            name=group,
            customdata=[i for i in idxs],
        ))

   
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        legend=dict(
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
        ),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        width=1400,
        height=950,
        margin=dict(l=20, r=20, t=60, b=20),
    )


    # Encode personal edges as JSON for JS access
    edges_json = {}
    for i in range(n):
        connections = []
        for j in range(n):
            if C[i, j] > 0:
                connections.append({
                    "target": int(j),
                    "x": [float(coords[i, 0]), float(coords[j, 0])],
                    "y": [float(coords[i, 1]), float(coords[j, 1])],
                    "weight": int(C[i, j]),
                    "name": f"{researchers[j]['firstname']} {researchers[j]['surname']}",
                })
        edges_json[i] = connections

    coords_json = [{"x": float(coords[i, 0]), "y": float(coords[i, 1])} for i in range(n)]
    names_json = [f"{r['firstname']} {r['surname']}" for r in researchers]

    # Write HTML with embedded JavaScript for click interaction
    html_content = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

    # Inject custom JavaScript before closing </body>
    custom_js = f"""
<script>
const EDGES = {json.dumps(edges_json)};
const COORDS = {json.dumps(coords_json)};
const NAMES = {json.dumps(names_json)};
let highlightTraceIdx = null;

document.addEventListener('DOMContentLoaded', function() {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) return;

    plotDiv.on('plotly_click', function(data) {{
        if (!data.points || !data.points[0]) return;
        const pt = data.points[0];
        const customdata = pt.customdata;
        if (customdata === undefined) return;

        const researcherIdx = customdata;
        const conns = EDGES[researcherIdx];
        if (!conns || conns.length === 0) return;

        // Build highlight edges
        const ex = [], ey = [];
        conns.forEach(c => {{
            ex.push(c.x[0], c.x[1], null);
            ey.push(c.y[0], c.y[1], null);
        }});

        const highlightTrace = {{
            x: ex,
            y: ey,
            mode: 'lines',
            line: {{color: 'rgba(255, 50, 50, 0.6)', width: 2}},
            hoverinfo: 'skip',
            showlegend: false,
            name: 'selected connections'
        }};

        // Remove previous highlight, add new
        const nTraces = plotDiv.data.length;
        if (highlightTraceIdx !== null) {{
            Plotly.deleteTraces(plotDiv, highlightTraceIdx);
        }}
        Plotly.addTraces(plotDiv, highlightTrace);
        highlightTraceIdx = plotDiv.data.length - 1;

        // Update title with selected researcher info
        const name = NAMES[researcherIdx];
        const nConns = conns.length;
        const totalPubs = conns.reduce((s, c) => s + c.weight, 0);
        Plotly.relayout(plotDiv, {{
            'title.text': `${{name}} — ${{nConns}} co-authors, ${{totalPubs}} joint publications (click another to change)`
        }});
    }});

    // Double-click to reset
    plotDiv.on('plotly_doubleclick', function() {{
        if (highlightTraceIdx !== null) {{
            Plotly.deleteTraces(plotDiv, highlightTraceIdx);
            highlightTraceIdx = null;
        }}
        Plotly.relayout(plotDiv, {{
            'title.text': '{title}'
        }});
    }});
}});
</script>
"""
    html_content = html_content.replace("</body>", custom_js + "</body>")

    path.write_text(html_content, encoding="utf-8")
    logger.info("Saved interactive HTML → %s (%d researchers, click to explore)", path, n)