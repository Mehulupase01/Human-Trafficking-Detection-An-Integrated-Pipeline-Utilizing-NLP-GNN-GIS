# backend/api/graph.py
# Backwards-compatible graph API that delegates to graph_build,
# plus helpers to export HTML (pyvis) and static PNG (matplotlib).

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import io

import pandas as pd
import networkx as nx

from backend.api.graph_build import build_network_from_processed

NTYPE_COLOR = {
    "Victim":       "#90CAF9",
    "Location":     "#A5D6A7",
    "Perpetrator":  "#FFAB91",
    "Chief":        "#CE93D8",
}

def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Legacy name that other pages may import."""
    return build_network_from_processed(df)

def _color_for(node_data: Dict[str, str]) -> str:
    t = node_data.get("ntype", "")
    return NTYPE_COLOR.get(t, "#CFD8DC")

def export_png(G: nx.Graph, width: int = 1400, height: int = 900) -> bytes:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(width/100, height/100), dpi=100)

    # spring layout by component (for nicer separation)
    if G.number_of_nodes() == 0:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white"); buf.seek(0)
        return buf.read()

    pos = nx.spring_layout(G, k=0.7/(len(G)**0.5 + 1), seed=42)

    # Colors/sizes
    node_colors = [_color_for(G.nodes[n]) for n in G.nodes()]
    node_sizes = []
    for n in G.nodes():
        t = G.nodes[n].get("ntype", "")
        node_sizes.append(160 if t == "Location" else 120 if t == "Victim" else 90)

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.5, edgecolors="#455A64")
    # small labels for locations & perps; suppress for large graphs
    if len(G) <= 300:
        labels = {n: n.split(":",1)[1] if ":" in n else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    # Legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=k) for k, c in NTYPE_COLOR.items()]
    plt.legend(handles=patches, loc="lower right", fontsize=8, frameon=True)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()

def export_pyvis_html(G: nx.Graph, height: str = "720px") -> str:
    from pyvis.network import Network
    net = Network(height=height, width="100%", bgcolor="#111319", font_color="#ECEFF1", directed=False)
    net.toggle_physics(True)
    net.set_options("""
    const options = {
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -6000, springLength: 120 } },
      interaction: { hover: true, multiselect: true, navigationButtons: true }
    }
    """)
    # Add nodes
    for n, data in G.nodes(data=True):
        ntype = data.get("ntype", "Node")
        label = n.split(":",1)[1] if ":" in n else n
        color = NTYPE_COLOR.get(ntype, "#CFD8DC")
        size = 18 if ntype=="Location" else 14 if ntype=="Victim" else 12
        net.add_node(n, label=label, color=color, size=size, title=f"{ntype}: {label}")
    # Add edges (style by etype)
    for u, v, data in G.edges(data=True):
        et = data.get("etype", "")
        dashes = True if et in {"route"} else False
        width = 2 if et in {"route"} else 1
        net.add_edge(u, v, title=et or "link", width=width, dashes=dashes, color="#90A4AE")
    # Legend (fixed-position corner)
    legend_nodes = [
        ("legend_v", "Victim", NTYPE_COLOR["Victim"]),
        ("legend_l", "Location", NTYPE_COLOR["Location"]),
        ("legend_p", "Perpetrator", NTYPE_COLOR["Perpetrator"]),
        ("legend_c", "Chief", NTYPE_COLOR["Chief"]),
    ]
    for nid, label, color in legend_nodes:
        net.add_node(nid, label=label, color=color, shape="box", physics=False, x=-800, y=-400)
    return net.generate_html()
