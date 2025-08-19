# backend/api/hierarchy_graph.py
from __future__ import annotations
from typing import List, Tuple
import re
import io
import os

import networkx as nx
import matplotlib.pyplot as plt

# Try Graphviz via pydot or graphviz lib
_HAS_PYDOT = False
try:
    import pydot  # noqa: F401
    _HAS_PYDOT = True
except Exception:
    _HAS_PYDOT = False

# Columns
COL_HIERARCHY = "Hierarchy of Perpetrators"

_SPLIT_RE = re.compile(r"\s*(?:->|›|>|,|;|\||\band\b|\u2192)\s*", flags=re.IGNORECASE)

def parse_hierarchy_string(s: str) -> List[str]:
    if not s or not str(s).strip():
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(str(s)) if p and p.strip() and p.strip().lower() not in {"nan", "none", "null"}]
    # Title-case but keep acronyms
    out = []
    for p in parts:
        if p.isupper() and len(p) <= 5:
            out.append(p)
        else:
            out.append(p.title())
    # Remove duplicates while preserving order
    seen = []
    for p in out:
        if p not in seen:
            seen.append(p)
    return seen


def build_hierarchy_graph(df) -> nx.DiGraph:
    if COL_HIERARCHY not in df.columns:
        raise ValueError(f"DataFrame missing required column: {COL_HIERARCHY}")
    G = nx.DiGraph()
    for s in df[COL_HIERARCHY].dropna().astype(str).tolist():
        chain = parse_hierarchy_string(s)
        for a, b in zip(chain, chain[1:]):
            if not a or not b:
                continue
            if not G.has_node(a):
                G.add_node(a)
            if not G.has_node(b):
                G.add_node(b)
            if G.has_edge(a, b):
                G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
            else:
                G.add_edge(a, b, weight=1)
    return G


def to_graphviz_png(G: nx.DiGraph, dpi: int = 180) -> bytes:
    """
    Render a top-down PNG using Graphviz (if available), else fall back to matplotlib layered plot.
    """
    if _HAS_PYDOT:
        # Convert to pydot graph
        pydot_g = nx.drawing.nx_pydot.to_pydot(G)
        pydot_g.set_rankdir("TB")  # Top -> Bottom
        png_bytes = pydot_g.create_png(prog="dot")
        return png_bytes

    # Fallback: simple layered drawing with matplotlib
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if hasattr(nx, "nx_agraph") else nx.spring_layout(G, seed=7)

    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    nx.draw(G, pos, with_labels=True, node_color="#f6f8ff", edge_color="#222", node_size=1200, font_size=9, arrows=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
