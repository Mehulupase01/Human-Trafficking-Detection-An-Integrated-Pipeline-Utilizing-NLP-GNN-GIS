# backend/graph/graph_builder.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, json, math, random
from collections import Counter, defaultdict

import pandas as pd

# pyvis for interactive vis.js graphs
from pyvis.network import Network


# ------------------------- file utils -------------------------

def _ensure_outdir() -> str:
    base = os.environ.get("APP_DATA_DIR")
    if not base:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    outdir = os.path.join(base, "tmp_graphs")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _save_html(net: Network, name: str) -> str:
    outdir = _ensure_outdir()
    path = os.path.join(outdir, f"{name}.html")
    net.show_buttons(filter_=['physics'])  # basic controls
    net.save_graph(path)
    return path


# ------------------------- helpers -------------------------

def _first_token_list(x) -> Optional[str]:
    """
    Return first token from a list-like, or None.
    """
    if isinstance(x, list) and len(x) > 0:
        return str(x[0])
    # sometimes arrays sneak in
    try:
        import numpy as np
        if isinstance(x, np.ndarray) and x.size > 0:
            return str(x.tolist()[0])
    except Exception:
        pass
    return None

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for graphing:
      - compute primary_loc = first token in Locations (NLP) or fallback to Location
      - ensure Route_Order numeric
      - drop rows with no location or no Serialized ID
    """
    d = df.copy()
    # primary location
    if "Locations (NLP)" in d.columns:
        d["primary_loc"] = d["Locations (NLP)"].apply(_first_token_list)
    else:
        d["primary_loc"] = None
    if "Location" in d.columns:
        d["primary_loc"] = d["primary_loc"].fillna(d["Location"].astype(str))
    d["primary_loc"] = d["primary_loc"].astype(str).str.strip().replace({"": None, "None": None, "nan": None})

    # Route order
    if "Route_Order" in d.columns:
        d["Route_Order"] = pd.to_numeric(d["Route_Order"], errors="coerce")
    else:
        d["Route_Order"] = pd.NA

    # must have victim & primary_loc
    if "Serialized ID" not in d.columns:
        d["Serialized ID"] = pd.NA
    d["Serialized ID"] = d["Serialized ID"].astype(str).replace({"": None, "nan": None, "None": None})

    d = d.dropna(subset=["Serialized ID", "primary_loc"])
    return d


# ------------------------- network (global) -------------------------

def build_network_pyvis(
    df: pd.DataFrame,
    *,
    include_perpetrators: bool = True,
    include_victims: bool = False,
    max_nodes: int = 1200,
    seed: int = 7,
    min_perp_nodes: int = 50,   # ✅ ensure we keep at least this many perpetrators (if available)
) -> str:
    """
    Build an interactive network:
      - Location nodes
      - Location→Location transitions weighted by # victims
      - (optional) Perpetrator nodes connected to locations where they appear
      - (optional) Victim nodes connected to their visited locations (light edges)
    Titles/hover now include richer analytics.
    """
    d = _prep_df(df)
    if d.empty:
        net = Network(height="700px", width="100%", bgcolor="#0f1117", font_color="#e5e7eb", directed=True)
        net.barnes_hut()
        net.add_node("empty", label="No data", shape="box", color="#334155")
        return _save_html(net, "network_empty")

    # ---- transitions (Location -> Location), weight by # victims
    pair_to_victims: Dict[Tuple[str, str], set] = defaultdict(set)
    loc_to_victims: Dict[str, set] = defaultdict(set)

    # perpetrator co-occurrence per location & per transition
    perp_to_loc: Dict[str, Counter] = defaultdict(Counter)

    for sid, grp in d.sort_values(["Serialized ID", "Route_Order"]).groupby("Serialized ID"):
        locs = grp["primary_loc"].tolist()
        # record victim presence per location
        for loc in set(locs):
            if loc:
                loc_to_victims[loc].add(sid)

        # collapse duplicates in path
        prev = None
        path: List[str] = []
        rows_for_step: List[dict] = []
        for _, r in grp.iterrows():
            loc = r["primary_loc"]
            if not loc:
                continue
            if loc == prev:
                # still record perp→loc even if we collapse for path
                if include_perpetrators and isinstance(r.get("Perpetrators (NLP)"), list):
                    for p in r["Perpetrators (NLP)"]:
                        if p:
                            perp_to_loc[str(p)][loc] += 1
                continue
            path.append(loc)
            rows_for_step.append(r.to_dict())
            prev = loc

        for a, b in zip(path, path[1:]):
            pair_to_victims[(a, b)].add(sid)

        if include_perpetrators:
            # also count perps for the unique steps we kept
            for r in rows_for_step:
                loc = r["primary_loc"]
                perps = r.get("Perpetrators (NLP)")
                if isinstance(perps, list):
                    for p in perps:
                        if p:
                            perp_to_loc[str(p)][loc] += 1

    edge_weights = {k: len(v) for k, v in pair_to_victims.items()}

    # location visit counts (row-level) for sizing
    loc_visit = Counter(d["primary_loc"].tolist())

    # in/out counts per location (from transitions)
    in_counts: Dict[str, int] = defaultdict(int)
    out_counts: Dict[str, int] = defaultdict(int)
    for (a, b), w in edge_weights.items():
        out_counts[a] += w
        in_counts[b] += w

    # ---- victims -> locations (light edges, heavy graphs off by default)
    victim_to_locs: Dict[str, Counter] = defaultdict(Counter)
    if include_victims:
        for sid, grp in d.groupby("Serialized ID"):
            for loc in grp["primary_loc"].unique().tolist():
                victim_to_locs[str(sid)][str(loc)] += 1

    # ---- Node sampling if too many
    loc_nodes = set([x for x in loc_visit.keys()])
    if len(loc_nodes) > max_nodes:
        keep = set([x for x, _ in loc_visit.most_common(max_nodes)])
        edge_weights = { (a,b):w for (a,b),w in edge_weights.items() if a in keep and b in keep }
        loc_nodes = keep

    # crude node accounting
    base_nodes = len(loc_nodes)
    vic_nodes  = len(victim_to_locs) if include_victims else 0
    perp_nodes = len(perp_to_loc) if include_perpetrators else 0

    # ensure we always show a meaningful number of perps
    if include_perpetrators:
        # score perpetrators by total attachments to kept locations
        perp_scores = []
        for p, cnts in perp_to_loc.items():
            score = sum(v for loc, v in cnts.items() if loc in loc_nodes)
            perp_scores.append((p, score))
        perp_scores.sort(key=lambda x: x[1], reverse=True)

        # budget remaining for perps after locations + victims
        budget = max_nodes - base_nodes - (vic_nodes if include_victims else 0)

        # ✅ enforce a minimum
        desired = max(min_perp_nodes, 0 if budget < 0 else min(budget, len(perp_scores)))
        keep_perps = {p for p, _ in perp_scores[:desired]}
        # filter perpetrator edges to kept locations only
        perp_to_loc = {p: Counter({l:c for l,c in cnts.items() if l in loc_nodes}) for p, cnts in perp_to_loc.items() if p in keep_perps}
    else:
        perp_to_loc = {}

    # if still exceeding cap with victims, trim victims by richness
    if include_victims:
        all_nodes_est = base_nodes + len(perp_to_loc) + len(victim_to_locs)
        if all_nodes_est > max_nodes:
            budget = max_nodes - base_nodes - len(perp_to_loc)
            ranks = sorted(victim_to_locs.items(), key=lambda kv: len(kv[1]), reverse=True)
            victim_to_locs = dict(ranks[:max(0, budget)])

    # ---- Build pyvis
    random.seed(seed)
    net = Network(height="700px", width="100%", bgcolor="#0f1117", font_color="#e5e7eb", directed=True, notebook=False)
    net.barnes_hut(gravity=-25000, spring_length=200)

    # legend nodes
    net.add_node("__legend_loc", label="Location", shape="dot", color="#60a5fa", size=18, physics=False, x=-700, y=-350, fixed=True)
    if include_perpetrators:
        net.add_node("__legend_perp", label="Perpetrator", shape="diamond", color="#f59e0b", size=18, physics=False, x=-700, y=-300, fixed=True)
    if include_victims:
        net.add_node("__legend_vic", label="Victim", shape="box", color="#34d399", size=18, physics=False, x=-700, y=-250, fixed=True)

    # pre-compute top perpetrators per location (for hover)
    top_perps_by_loc: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    if include_perpetrators:
        perploc_inv: Dict[str, Counter] = defaultdict(Counter)
        for p, cnts in perp_to_loc.items():
            for loc, c in cnts.items():
                perploc_inv[loc][p] += c
        for loc, ctr in perploc_inv.items():
            top_perps_by_loc[loc] = ctr.most_common(5)

    # add location nodes with rich titles
    for loc in loc_nodes:
        visits = loc_visit[loc]
        uniq_vics = len(loc_to_victims.get(loc, set()))
        in_c = in_counts.get(loc, 0)
        out_c = out_counts.get(loc, 0)
        top_perps = ", ".join(f"{p} ({c})" for p, c in top_perps_by_loc.get(loc, [])) or "—"
        size = 12 + 6 * math.log10(max(1, visits))
        title = (
            f"<b>{loc}</b>"
            f"<br/>Visits (rows): {visits}"
            f"<br/>Unique victims: {uniq_vics}"
            f"<br/>In: {in_c} • Out: {out_c}"
            f"<br/>Top perps: {top_perps}"
        )
        net.add_node(
            f"loc::{loc}",
            label=loc,
            title=title,
            color="#60a5fa",
            size=size,
            shape="dot",
        )

    # add transition edges with titles
    for (a, b), w in edge_weights.items():
        if a not in loc_nodes or b not in loc_nodes:
            continue
        width = 1 + 2 * math.log10(max(1, w))
        title = f"{a} → {b}<br/>Victims traversing: {w}"
        net.add_edge(f"loc::{a}", f"loc::{b}", value=w, width=width, color="#94a3b8", title=title, arrows="to")

    # perpetrator nodes/edges (now guaranteed min count)
    if include_perpetrators:
        for p, cnts in perp_to_loc.items():
            total = sum(cnts.values())
            uniq_locs = len(cnts)
            # estimate unique victims via co-occurrence rows near those locations
            # (conservative: not exact)
            uniq_vics = 0
            for loc in cnts.keys():
                uniq_vics += len(loc_to_victims.get(loc, set()))
            size = 10 + 4 * math.log10(max(1, total))
            title = (
                f"<b>{p}</b>"
                f"<br/>Co-occurrence rows: {total}"
                f"<br/>Attached locations: {uniq_locs}"
                f"<br/>Approx. victims (sum over locations): {uniq_vics}"
            )
            net.add_node(f"perp::{p}", label=p, title=title, color="#f59e0b", shape="diamond", size=size)
            for loc, c in cnts.items():
                if loc in loc_nodes and c > 0:
                    net.add_edge(f"perp::{p}", f"loc::{loc}", value=c, width=1 + math.log10(max(1, c)), color="#fbbf24", title=f"{p} @ {loc}<br/>Rows: {c}")

    # victim nodes/edges
    if include_victims:
        for v, cnts in victim_to_locs.items():
            title = f"<b>{v}</b><br/>Locations visited: {len(cnts)}"
            net.add_node(f"vic::{v}", label=v, color="#34d399", shape="box", size=12, title=title)
            for loc, c in cnts.items():
                if loc in loc_nodes:
                    net.add_edge(f"vic::{v}", f"loc::{loc}", value=c, width=1, color="#10b98133", title=f"{v} visited {loc}")

    return _save_html(net, "network_overview")


# ------------------------- victim traceroute -------------------------

def build_traceroute_pyvis(
    df: pd.DataFrame,
    victim_sid: str,
    *,
    collapse_repeats: bool = True,
    seed: int = 7,
) -> str:
    """
    Build a per-victim linear traceroute:
      Step N — LocationName
      tooltip shows perpetrators for that step (and chiefs if available)
    """
    d = _prep_df(df)
    d = d[d["Serialized ID"].astype(str) == str(victim_sid)].copy()
    if d.empty:
        net = Network(height="700px", width="100%", bgcolor="#0f1117", font_color="#e5e7eb", directed=True)
        net.add_node("empty", label="No data for victim", shape="box", color="#334155")
        return _save_html(net, f"traceroute_{victim_sid}")

    d = d.sort_values("Route_Order")
    steps: List[Tuple[str, List[str]]] = []  # (location, perps)
    last = None
    for _, row in d.iterrows():
        loc = row["primary_loc"]
        if not loc:
            continue
        if collapse_repeats and loc == last:
            continue
        perps = row.get("Perpetrators (NLP)", [])
        if not isinstance(perps, list):
            perps = []
        steps.append((loc, [str(p) for p in perps if p]))
        last = loc

    net = Network(height="700px", width="100%", bgcolor="#0f1117", font_color="#e5e7eb", directed=True, notebook=False)
    net.force_atlas_2based(gravity=-30, spring_length=120)

    for i, (loc, perps) in enumerate(steps, start=1):
        label = f"Step {i} — {loc}"
        tooltip = f"<b>{loc}</b><br/>Perpetrators: {', '.join(perps) if perps else '—'}"
        net.add_node(f"s::{i}", label=label, title=tooltip, color="#fde68a", shape="dot", size=16)
        if i > 1:
            net.add_edge(f"s::{i-1}", f"s::{i}", color="#cbd5e1")

    return _save_html(net, f"traceroute_{victim_sid}")
