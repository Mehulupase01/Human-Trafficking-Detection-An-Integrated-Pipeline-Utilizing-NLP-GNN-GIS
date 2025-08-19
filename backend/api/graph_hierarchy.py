# backend/api/graph_hierarchy.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import io

import pandas as pd
import networkx as nx

COL_SID = "Serialized ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"

def build_victim_route_dag(df: pd.DataFrame, victim_sid: str) -> nx.DiGraph:
    sub = df[df[COL_SID].astype(str) == str(victim_sid)].copy()
    if sub.empty:
        return nx.DiGraph()
    sub = sub.sort_values(COL_ROUTE, kind="stable")
    locs = sub[COL_LOC].astype(str).tolist()
    orders = sub[COL_ROUTE].astype(int).tolist()
    G = nx.DiGraph()
    for loc, ordv in zip(locs, orders):
        n = f"Step {ordv}: {loc}"
        G.add_node(n, step=int(ordv), ntype="Step")
    for i in range(len(locs)-1):
        a = f"Step {orders[i]}: {locs[i]}"
        b = f"Step {orders[i+1]}: {locs[i+1]}"
        G.add_edge(a, b, etype="next")
    return G

def export_dag_png(G: nx.DiGraph, width: int = 1400, height: int = 900) -> bytes:
    # use pygraphviz or pydot if available; else fallback to spring layout
    try:
        import pygraphviz  # noqa
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    import matplotlib.pyplot as plt
    import io as _io
    plt.figure(figsize=(width/100, height/100), dpi=100)
    nx.draw(G, pos, with_labels=True, node_color="#FFF59D", edge_color="#CFD8DC",
            node_size=1400, font_size=8, arrows=True)
    buf = _io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()
