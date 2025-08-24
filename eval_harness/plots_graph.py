from __future__ import annotations
"""
Graph plotting helpers (Altair + light NetworkX utils).

What’s here
-----------
- heuristics_bar_chart(holdout_dict): stacked bars for Hits@K / MRR / AP / ROC-AUC
- degree_cdf_chart(df): empirical CDF of node degrees (auto-detects sid/pid)
- component_cdf_chart(df): empirical CDF of component sizes (auto-detects sid/pid)

These functions are tolerant to missing data. If inputs are empty, they
return a minimal chart to avoid breaking the UI.
"""

from typing import Dict, Any, List, Sequence, Optional, Tuple
import pandas as pd
import numpy as np
import altair as alt
import networkx as nx

SID_CAND = ["sid", "subject_id", "victim_id", "case_id", "trajectory_id"]
PID_CAND = ["pid", "perp_id", "offender_id"]

# ---------------- column detection ----------------

def _pick_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def _detect_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "sid": _pick_first(df.columns, SID_CAND),
        "pid": _pick_first(df.columns, PID_CAND),
    }

# ---------------- graph builders ----------------

def _build_bipartite(df: pd.DataFrame, sid: str, pid: str) -> nx.Graph:
    g = nx.Graph()
    sub = df[[sid, pid]].dropna().astype(str).drop_duplicates()
    for _, row in sub.iterrows():
        u = f"V:{row[sid]}"
        v = f"P:{row[pid]}"
        g.add_node(u, bipartite=0)
        g.add_node(v, bipartite=1)
        g.add_edge(u, v)
    return g

# ---------------- charts ----------------

def heuristics_bar_chart(holdout_dict: Dict[str, Any]) -> alt.Chart:
    """
    Input: graph_eval.eval_link_pred(...)[\"holdout\"]
    Produces: a small-multiples bar chart with metrics × heuristic.
    """
    if not isinstance(holdout_dict, dict) or not holdout_dict:
        return alt.Chart(pd.DataFrame({"x":[0], "y":[0]})).mark_bar().encode(x="x", y="y").properties(height=120)

    rows = []
    for heur, stats in holdout_dict.items():
        if heur in ("available", "columns", "n_candidates", "reason", "skipped"):
            continue
        rows.append({
            "heuristic": heur,
            "hits@1": stats.get("hits@1", 0.0),
            "hits@3": stats.get("hits@3", 0.0),
            "hits@5": stats.get("hits@5", 0.0),
            "mrr": stats.get("mrr", 0.0),
            "ap": stats.get("ap", 0.0),
            "roc_auc": stats.get("roc_auc", 0.0),
        })
    if not rows:
        return alt.Chart(pd.DataFrame({"x":[0], "y":[0]})).mark_bar().encode(x="x", y="y").properties(height=120)

    df = pd.DataFrame(rows).melt(id_vars="heuristic", var_name="metric", value_name="score")
    # Ensure common axis
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("heuristic:N", title="Heuristic"),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            column=alt.Column("metric:N", header=alt.Header(title="Metric")),
            tooltip=["heuristic:N", "metric:N", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=260)
    )

def degree_cdf_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Compute degree distribution from df (auto sid/pid), show empirical CDF.
    """
    cols = _detect_cols(df)
    sid, pid = cols["sid"], cols["pid"]
    if sid is None or pid is None or df.empty:
        return alt.Chart(pd.DataFrame({"degree":[0], "cdf":[1.0]})).mark_line().encode(x="degree:Q", y="cdf:Q").properties(title="Degree CDF (n/a)", height=260)

    g = _build_bipartite(df, sid, pid)
    degs = np.array([d for _, d in g.degree()], dtype=float)
    if degs.size == 0:
        return alt.Chart(pd.DataFrame({"degree":[0], "cdf":[1.0]})).mark_line().encode(x="degree:Q", y="cdf:Q").properties(title="Degree CDF (empty)", height=260)
    # Empirical CDF
    s = np.sort(degs)
    cdf = np.arange(1, s.size + 1) / s.size
    cdf_df = pd.DataFrame({"degree": s, "cdf": cdf})
    return (
        alt.Chart(cdf_df)
        .mark_line()
        .encode(
            x=alt.X("degree:Q", title="Degree"),
            y=alt.Y("cdf:Q", title="Empirical CDF", scale=alt.Scale(domain=[0,1])),
            tooltip=[alt.Tooltip("degree:Q", format=".0f"), alt.Tooltip("cdf:Q", format=".2f")],
        )
        .properties(title="Degree CDF", height=260)
    )

def component_cdf_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Compute component sizes from df (auto sid/pid), show empirical CDF.
    """
    cols = _detect_cols(df)
    sid, pid = cols["sid"], cols["pid"]
    if sid is None or pid is None or df.empty:
        return alt.Chart(pd.DataFrame({"size":[0], "cdf":[1.0]})).mark_line().encode(x="size:Q", y="cdf:Q").properties(title="Component Size CDF (n/a)", height=260)

    g = _build_bipartite(df, sid, pid)
    comps = [len(c) for c in nx.connected_components(g)]
    if not comps:
        return alt.Chart(pd.DataFrame({"size":[0], "cdf":[1.0]})).mark_line().encode(x="size:Q", y="cdf:Q").properties(title="Component Size CDF (empty)", height=260)
    s = np.sort(np.asarray(comps, dtype=float))
    cdf = np.arange(1, s.size + 1) / s.size
    cdf_df = pd.DataFrame({"size": s, "cdf": cdf})
    return (
        alt.Chart(cdf_df)
        .mark_line()
        .encode(
            x=alt.X("size:Q", title="Component size (#nodes)"),
            y=alt.Y("cdf:Q", title="Empirical CDF", scale=alt.Scale(domain=[0,1])),
            tooltip=[alt.Tooltip("size:Q", format=".0f"), alt.Tooltip("cdf:Q", format=".2f")],
        )
        .properties(title="Component Size CDF", height=260)
    )
