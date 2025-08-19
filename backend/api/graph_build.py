# backend/api/graph_build.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import itertools
import networkx as nx
import pandas as pd

# Column names from processed pipeline
COL_SID = "Serialized ID"
COL_UID = "Unique ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"
COL_PERPS = "Perpetrators (NLP)"
COL_CHIEFS = "Chiefs (NLP)"

# Node type tags
TYPE_VICTIM = "Victim"
TYPE_LOCATION = "Location"
TYPE_PERP = "Trafficker"
TYPE_CHIEF = "Chief"


def _add_node_safe(G: nx.Graph, nid: str, label: str, ntype: str):
    if nid not in G:
        G.add_node(nid, label=label, ntype=ntype)


def _edge_weight_inc(G: nx.Graph, u: str, v: str, w: float = 1.0, etype: str = ""):
    if G.has_edge(u, v):
        G[u][v]["weight"] = float(G[u][v].get("weight", 1.0)) + w
    else:
        G.add_edge(u, v, weight=float(w), etype=etype)


def build_network_from_processed(
    df: pd.DataFrame,
    include_perpetrators: bool = True,
    include_chiefs: bool = True,
    connect_perp_to_location: bool = True,
    connect_chief_to_location: bool = True,
    victim_limit: Optional[int] = None,
) -> nx.Graph:
    """
    Build a heterogeneous network:
      - Victim ↔ Location edges for each trajectory row
      - Victim ↔ Perpetrator (if list available)
      - Victim ↔ Chief (if list available)
      - Optionally Perpetrator ↔ Location and Chief ↔ Location (co-occurrence per row)
    """
    required = {COL_SID, COL_LOC, COL_ROUTE}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame is missing required columns: {sorted(required - set(df.columns))}")

    G = nx.Graph()

    # Optionally limit number of victims to avoid giant graphs
    if victim_limit is not None and victim_limit > 0:
        victims = df[COL_SID].dropna().astype(str).unique().tolist()
        victims = victims[:victim_limit]
        df = df[df[COL_SID].isin(victims)]

    # Iterate rows
    for _, row in df.iterrows():
        sid = str(row[COL_SID])
        loc = str(row[COL_LOC]) if pd.notna(row[COL_LOC]) else ""
        if not sid or not loc:
            continue

        # Nodes
        v_node = f"V:{sid}"
        l_node = f"L:{loc}"
        _add_node_safe(G, v_node, label=sid, ntype=TYPE_VICTIM)
        _add_node_safe(G, l_node, label=loc, ntype=TYPE_LOCATION)

        # Victim—Location edge
        _edge_weight_inc(G, v_node, l_node, w=1.0, etype="victim_location")

        # Perpetrators
        if include_perpetrators and COL_PERPS in df.columns:
            perps = row[COL_PERPS]
            if isinstance(perps, list):
                for p in perps:
                    p = str(p).strip()
                    if not p:
                        continue
                    p_node = f"P:{p}"
                    _add_node_safe(G, p_node, label=p, ntype=TYPE_PERP)
                    _edge_weight_inc(G, v_node, p_node, w=1.0, etype="victim_perp")
                    if connect_perp_to_location:
                        _edge_weight_inc(G, p_node, l_node, w=0.5, etype="perp_location")

        # Chiefs
        if include_chiefs and COL_CHIEFS in df.columns:
            chiefs = row[COL_CHIEFS]
            if isinstance(chiefs, list):
                for c in chiefs:
                    c = str(c).strip()
                    if not c:
                        continue
                    c_node = f"C:{c}"
                    _add_node_safe(G, c_node, label=c, ntype=TYPE_CHIEF)
                    _edge_weight_inc(G, v_node, c_node, w=1.0, etype="victim_chief")
                    if connect_chief_to_location:
                        _edge_weight_inc(G, c_node, l_node, w=0.5, etype="chief_location")

    return G
