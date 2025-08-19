# backend/api/graph_export.py
from __future__ import annotations
from typing import Optional
import os
import io

import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network


GRAPH_DIR = os.path.join("frontend", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def to_pyvis_html(G: nx.Graph, height: str = "700px", physics: bool = True) -> str:
    """
    Convert NetworkX graph to an interactive PyVis HTML string.
    Adds basic styling & physics toggles.
    """
    net = Network(height=height, width="100%", bgcolor="#101014", font_color="#eee", directed=False, notebook=False)
    net.force_atlas_2based(gravity=-50) if physics else None

    # Color palette by node type
    color_map = {
        "Victim": "#5B8FF9",
        "Location": "#5AD8A6",
        "Trafficker": "#F6BD16",
        "Chief": "#E8684A",
    }

    for nid, data in G.nodes(data=True):
        ntype = data.get("ntype", "Node")
        label = data.get("label", nid)
        size = 15
        if ntype == "Location":
            size = max(10, min(40, 10 + (G.degree(nid) * 1.2)))
        net.add_node(
            nid,
            label=str(label),
            title=f"{ntype}: {label}",
            color=color_map.get(ntype, "#cccccc"),
            size=size,
        )

    for u, v, edata in G.edges(data=True):
        w = float(edata.get("weight", 1.0))
        etype = edata.get("etype", "")
        net.add_edge(u, v, value=w, title=f"{etype or 'link'} • weight={w:.1f}")

    # Add a simple legend as HTML overlay
    legend_html = """
    <div style="position:absolute; top:8px; right:8px; background:#1f2430; color:#eee; padding:10px 12px; border-radius:8px; font-family:Inter,system-ui,Arial; font-size:13px; z-index: 9999;">
      <div style="font-weight:600; margin-bottom:6px;">Legend</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#5B8FF9;border-radius:50%;margin-right:6px;"></span>Victim</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#5AD8A6;border-radius:50%;margin-right:6px;"></span>Location</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#F6BD16;border-radius:50%;margin-right:6px;"></span>Trafficker</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#E8684A;border-radius:50%;margin-right:6px;"></span>Chief</div>
    </div>
    """
    net.set_template(f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <style>body,html{{margin:0;padding:0;background:#0f1117;}}</style>
    </head>
    <body>
      {legend_html}
      {{ body }}
    </body>
    </html>
    """)

    return net.generate_html(notebook=False)


def to_png_networkx(G: nx.Graph, layout: str = "spring", dpi: int = 180) -> bytes:
    """
    Render a static PNG via matplotlib. Layout options: 'spring', 'kamada', 'circular', 'shell'
    """
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Color by node type
    color_map = []
    sizes = []
    labels = {}
    for nid, data in G.nodes(data=True):
        ntype = data.get("ntype", "Node")
        if ntype == "Victim":
            color_map.append("#5B8FF9")
            sizes.append(90)
        elif ntype == "Location":
            color_map.append("#5AD8A6")
            sizes.append(140)
        elif ntype == "Trafficker":
            color_map.append("#F6BD16")
            sizes.append(110)
        elif ntype == "Chief":
            color_map.append("#E8684A")
            sizes.append(110)
        else:
            color_map.append("#cccccc")
            sizes.append(90)
        labels[nid] = data.get("label", nid)

    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=sizes, linewidths=0.5, edgecolors="#333")
    # Keep labels light to avoid clutter
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
