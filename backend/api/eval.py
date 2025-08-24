from __future__ import annotations
"""
backend/api/eval.py

End-to-end automated evaluations used by the Streamlit page:
- Data completeness & ID consistency
- Duplicate suspicion (heuristic)
- Location resolution rate
- Network graph metrics (victim–perp bipartite)
- Predictive performance:
    • Next-location accuracy@{1,3}
    • Link prediction accuracy@{1,3,5}
- ETA error (MAE, days)

The code is defensive: if a required column is absent, that metric is skipped and
a zero or empty table is returned instead of raising.
"""

import io
import json
import math
import random
import hashlib
import zipfile
import datetime as dt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# Project-local imports (these exist in your repo)
from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.models.sequence_predictor import NgramSequenceModel


# ------------- add this helper near the top (after imports) -------------
def _predict_next_dist_compat(model: NgramSequenceModel, history, topk: int):
    """
    Try legacy model.predict_next_dist(prev2, prev1, topk).
    If unavailable, fall back to model.predict_next(history, topk) and
    convert to a {token: prob} dict.
    """
    prev1 = history[-1] if len(history) >= 1 else None
    prev2 = history[-2] if len(history) >= 2 else None

    meth = getattr(model, "predict_next_dist", None)
    if callable(meth):
        try:
            dist = meth(prev2, prev1, topk=topk)
            if isinstance(dist, dict) and dist:
                return dist
        except Exception:
            pass  # fallback below

    ranked = model.predict_next(history, topk=topk)  # [(tok, p)]
    return {tok: float(p) for tok, p in ranked}

# ---------- Column resolution helpers (robust to schema drift) ----------

SID_CANDIDATES = [
    "sid", "subject_id", "victim_id", "case_id", "trajectory_id", "person_id", "entity_id",
]
LOC_CANDIDATES = [
    "normalized_location", "location_norm", "location_name", "location", "loc", "city", "place",
]
ORDER_CANDIDATES = [
    "route", "route_order", "route_idx", "order", "seq", "sequence", "step", "t_index",
]
TIME_CANDIDATES = [
    "timestamp", "time", "datetime", "ts", "event_time", "created_at",
]
VICTIM_ID_CANDIDATES = [
    "victim_id", "sid", "subject_id", "person_id",
]
PERP_ID_CANDIDATES = [
    "perpetrator_id", "perp_id", "offender_id", "trafficker_id", "pid",
]
LAT_CANDS = ["lat", "latitude", "geo_lat", "geocode_lat"]
LON_CANDS = ["lon", "lng", "longitude", "geo_lon", "geocode_lon"]

ETA_TRUE_CANDS = ["eta_true_days", "true_eta_days", "eta_days_true"]
ETA_PRED_CANDS = ["eta_pred_days", "pred_eta_days", "eta_days_pred"]

TEXT_CANDS = [
    "normalized_text", "clean_text", "text", "description", "content", "body",
]
ID_UNIQUENESS_CANDS = [
    "ad_id", "post_id", "record_id", "global_id", "uid"
]


def choose_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # try relaxed matching (starts/endswith)
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if lc == cand or lc.startswith(cand) or lc.endswith(cand):
                return c
    return None


# ---------- Glue helpers ----------

def build_sequences_from_df(df: pd.DataFrame) -> List[List[str]]:
    """Build location sequences grouped by subject id, ordered by route index or time."""
    if df is None or df.empty:
        return []

    sid = choose_col(df, SID_CANDIDATES)
    loc = choose_col(df, LOC_CANDIDATES)
    order = choose_col(df, ORDER_CANDIDATES)
    timec = choose_col(df, TIME_CANDIDATES)

    if not sid or not loc:
        return []

    # choose a stable sort key
    if order and order in df.columns:
        key = order
    elif timec and timec in df.columns:
        key = timec
    else:
        # fallback: keep input order
        key = None

    seqs: List[List[str]] = []
    for _, grp in df.groupby(sid):
        if key:
            grp = grp.sort_values(key, kind="stable")
        toks = grp[loc].astype(str).str.strip()
        toks = toks[toks != ""]
        s = toks.tolist()
        if len(s) >= 2:
            seqs.append(s)
    return seqs


def _hash_fingerprint(s: str) -> str:
    s = "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace())
    s = " ".join(s.split())
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ---------- Metric blocks ----------

def compute_completeness(df: pd.DataFrame, fields: Optional[List[str]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Field", "Completeness", "NonNull", "Total"])

    if not fields:
        # pick a compact, relevant subset
        picks: List[str] = []
        for block in (SID_CANDIDATES, LOC_CANDIDATES, ORDER_CANDIDATES, TIME_CANDIDATES,
                      VICTIM_ID_CANDIDATES, PERP_ID_CANDIDATES, LAT_CANDS, LON_CANDS,
                      ID_UNIQUENESS_CANDS):
            col = choose_col(df, block)
            if col and col not in picks:
                picks.append(col)
        if not picks:
            picks = list(df.columns[:12])

        fields = picks

    rows = []
    total = len(df)
    for f in fields:
        if f not in df.columns:
            rows.append((f, 0.0, 0, total))
            continue
        nonnull = int(df[f].notna().sum())
        comp = float(nonnull) / float(total) if total else 0.0
        rows.append((f, round(comp, 4), nonnull, total))
    return pd.DataFrame(rows, columns=["Field", "Completeness", "NonNull", "Total"]).sort_values(
        "Completeness", ascending=False
    )


def compute_id_consistency(df: pd.DataFrame) -> Dict[str, object]:
    res: Dict[str, object] = {}
    if df is None or df.empty:
        return res

    sid = choose_col(df, SID_CANDIDATES)
    uniq = choose_col(df, ID_UNIQUENESS_CANDS)

    if sid:
        res["subjects_total"] = int(df[sid].nunique(dropna=True))
        dup_sids = df[sid][df[sid].duplicated(keep=False)]
        res["subjects_with_repeats"] = int(dup_sids.nunique(dropna=True))
    if uniq:
        total = len(df)
        nun = df[uniq].nunique(dropna=True)
        res["unique_id_ratio"] = round(float(nun) / float(total) if total else 0.0, 4)
        res["unique_ids"] = int(nun)
        res["rows"] = int(total)
    return res


def compute_duplicate_suspicion(df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    """
    Very lightweight duplicate heuristic:
    - Use the first available text-ish column.
    - Fingerprint text and flag buckets with size >= 2.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["group_hash", "count", "example"])

    text = choose_col(df, TEXT_CANDS)
    if not text or text not in df.columns:
        return pd.DataFrame(columns=["group_hash", "count", "example"])

    sample = df[[text]].dropna()
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)

    fps = sample[text].astype(str).map(_hash_fingerprint)
    grp = sample.assign(_fp=fps).groupby("_fp")
    dup = grp.size().reset_index(name="count")
    dup = dup[dup["count"] >= 2].sort_values("count", ascending=False)
    # attach one example row to each bucket
    if not dup.empty:
        ex_map = grp.first()[text].to_dict()
        dup["example"] = dup["_fp"].map(ex_map)
    dup = dup.rename(columns={"_fp": "group_hash"})
    return dup[["group_hash", "count", "example"]]


def compute_location_resolution(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {"resolved": 0, "total": 0, "rate": 0.0}

    lat = choose_col(df, LAT_CANDS)
    lon = choose_col(df, LON_CANDS)
    if not lat or not lon or lat not in df or lon not in df:
        return {"resolved": 0, "total": len(df), "rate": 0.0}

    ok = df[lat].notna() & df[lon].notna()
    resolved = int(ok.sum())
    total = int(len(df))
    rate = float(resolved) / float(total) if total else 0.0
    return {"resolved": resolved, "total": total, "rate": round(rate, 4)}


def compute_graph_metrics(df: pd.DataFrame) -> Tuple[Dict[str, object], nx.Graph]:
    """
    Build a bipartite graph (victims U perps) and compute simple metrics.
    """
    metrics: Dict[str, object] = {}
    G = nx.Graph()

    vcol = choose_col(df, VICTIM_ID_CANDIDATES)
    pcol = choose_col(df, PERP_ID_CANDIDATES)
    if not vcol or not pcol or vcol not in df or pcol not in df:
        return metrics, G

    # add edges
    pairs = (
        df[[vcol, pcol]]
        .dropna()
        .astype({vcol: str, pcol: str})
        .drop_duplicates()
        .values
        .tolist()
    )
    for v, p in pairs:
        G.add_node(f"V:{v}", bipartite=0)
        G.add_node(f"P:{p}", bipartite=1)
        G.add_edge(f"V:{v}", f"P:{p}")

    if G.number_of_nodes() == 0:
        return metrics, G

    comps = list(nx.connected_components(G))
    sizes = [len(c) for c in comps]
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(comps),
        "largest_component": max(sizes) if sizes else 0,
        "density": round(nx.density(G), 6),
        "avg_degree": round(np.mean([deg for _, deg in G.degree()]), 4),
    }
    return metrics, G


def benchmark_next_location(df: pd.DataFrame, topk: Tuple[int, ...] = (1, 3), min_len: int = 3) -> Dict[str, object]:
    """
    Leave-last-step-out per subject with n-gram backoff model.
    """
    res = {f"acc@{k}": 0.0 for k in topk}
    res.update({"eligible": 0})
    if df is None or df.empty:
        return res

    sid = choose_col(df, SID_CANDIDATES)
    loc = choose_col(df, LOC_CANDIDATES)
    order = choose_col(df, ORDER_CANDIDATES)
    timec = choose_col(df, TIME_CANDIDATES)
    if not sid or not loc:
        return res

    # Build sequences and bail if insufficient
    sequences = build_sequences_from_df(df)
    if not sequences:
        return res

    # Train global model (fast, counts-based)
    model = NgramSequenceModel(alpha=0.05)
    model.fit(sequences)

    eligible = 0
    hits = {k: 0 for k in topk}
    for _, grp in df.groupby(sid):
        if order and order in grp.columns:
            grp = grp.sort_values(order, kind="stable")
        elif timec and timec in grp.columns:
            grp = grp.sort_values(timec, kind="stable")
        seq = grp[loc].astype(str).str.strip().tolist()
        if len(seq) < min_len:
            continue

        eligible += 1
        history = seq[:-1]
        true_next = seq[-1]

        # Use legacy shim to match older eval code
        prev2 = history[-2] if len(history) >= 2 else None
        prev1 = history[-1] if len(history) >= 1 else None
        dist = _predict_next_dist_compat(model, history, topk=max(topk))
        ranked = [tok for tok, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)]
        for k in topk:
            if true_next in ranked[:k]:
                hits[k] += 1

    if eligible == 0:
        return res
    for k in topk:
        res[f"acc@{k}"] = round(hits[k] / float(eligible), 4)
    res["eligible"] = eligible
    return res


def benchmark_link_prediction(
    df: pd.DataFrame, max_samples: int = 300, topk: Tuple[int, ...] = (1, 3, 5)
) -> Dict[str, object]:
    """
    Simple classical link prediction on the bipartite victim–perp graph using
    Jaccard coefficient with preferential-attachment fallback. Edges are
    removed one-by-one for evaluation (local, cheap negative sampling).
    """
    results = {f"acc@{k}": 0.0 for k in topk}
    results.update({"tested": 0})
    if df is None or df.empty:
        return results

    vcol = choose_col(df, VICTIM_ID_CANDIDATES)
    pcol = choose_col(df, PERP_ID_CANDIDATES)
    if not vcol or not pcol:
        return results

    # Build base graph
    metrics, G = compute_graph_metrics(df)
    if G.number_of_edges() == 0:
        return results

    edges = list(G.edges())
    random.Random(42).shuffle(edges)
    edges = edges[: min(max_samples, len(edges))]

    hit = {k: 0 for k in topk}
    tested = 0

    for u, v in edges:
        # remove edge
        if not G.has_edge(u, v):
            continue
        G.remove_edge(u, v)

        # Candidate set: neighbors of u's two-hop excluding existing neighbors + the held-out v
        neigh_u = set(nx.neighbors(G, u))
        two_hop = set()
        for x in neigh_u:
            two_hop.update(nx.neighbors(G, x))
        candidates = {c for c in two_hop if c != u and not G.has_edge(u, c)}
        candidates.add(v)  # ensure true target present

        ebunch = [(u, c) for c in candidates]
        scores = {}
        try:
            for (x, y, s) in nx.jaccard_coefficient(G, ebunch):
                scores[y] = s
        except Exception:
            scores = {}

        # fallback: preferential attachment for any zeros
        if not scores or all(s == 0.0 for s in scores.values()):
            deg_u = G.degree(u)
            for c in candidates:
                scores[c] = deg_u * G.degree(c)

        ranked = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        for k in topk:
            if v in ranked[:k]:
                hit[k] += 1

        tested += 1
        # add edge back
        G.add_edge(u, v)

    if tested == 0:
        return results

    for k in topk:
        results[f"acc@{k}"] = round(hit[k] / float(tested), 4)
    results["tested"] = tested
    return results


def compute_eta_mae_days(df: pd.DataFrame) -> Optional[int]:
    """
    Compute MAE (days) between predicted and true ETAs if such columns exist.
    """
    if df is None or df.empty:
        return None

    y_true = choose_col(df, ETA_TRUE_CANDS)
    y_pred = choose_col(df, ETA_PRED_CANDS)
    if y_true and y_pred and y_true in df and y_pred in df:
        yt = pd.to_numeric(df[y_true], errors="coerce")
        yp = pd.to_numeric(df[y_pred], errors="coerce")
        mask = yt.notna() & yp.notna()
        if int(mask.sum()) == 0:
            return None
        mae = np.abs(yt[mask] - yp[mask]).mean()
        return int(round(float(mae)))
    return None


# ---------- Top-level orchestration ----------

def run_evaluations(ds_ids: List[str], link_max_samples: int = 300) -> Dict[str, object]:
    """
    Compose all evaluations into a single dict:
      {
        "summary": {...},
        "tables": { "completeness": df, "suspected_duplicates": df, ... },
        "details": {...}
      }
    """
    if not ds_ids:
        # still return an object so the UI doesn't crash
        return {"summary": {}, "tables": {}, "details": {}}

    df = concat_processed_frames(ds_ids)
    if df is None or df.empty:
        return {"summary": {}, "tables": {}, "details": {}}

    # Core tables/metrics
    comp = compute_completeness(df)
    idc = compute_id_consistency(df)
    dup = compute_duplicate_suspicion(df)
    loc_res = compute_location_resolution(df)
    gstats, G = compute_graph_metrics(df)
    nextloc = benchmark_next_location(df, topk=(1, 3), min_len=3)
    linkpred = benchmark_link_prediction(df, max_samples=link_max_samples, topk=(1, 3, 5))
    eta_mae = compute_eta_mae_days(df)

    # Summary for KPI tiles
    sid = choose_col(df, SID_CANDIDATES)
    loc = choose_col(df, LOC_CANDIDATES)
    summary = {
        "rows": int(len(df)),
        "victims": int(df[sid].nunique(dropna=True)) if sid else 0,
        "locations": int(df[loc].nunique(dropna=True)) if loc else 0,
        "graph_nodes": int(gstats.get("nodes", 0)),
        "nextloc_acc@1": float(nextloc.get("acc@1", 0.0)),
        "nextloc_acc@3": float(nextloc.get("acc@3", 0.0)),
        "link_acc@1": float(linkpred.get("acc@1", 0.0)),
        "link_acc@3": float(linkpred.get("acc@3", 0.0)),
        "link_acc@5": float(linkpred.get("acc@5", 0.0)),
        "eta_mae_days": int(eta_mae) if eta_mae is not None else 0,
        "location_resolution_rate": float(loc_res.get("rate", 0.0)),
    }

    return {
        "summary": summary,
        "tables": {
            "completeness": comp,
            "suspected_duplicates": dup,
        },
        "details": {
            "id_consistency": idc,
            "location_resolution": loc_res,
            "graph_metrics": gstats,
            "next_location": nextloc,
            "link_prediction": linkpred,
            "eta": {"mae_days": eta_mae},
            "sources": list(ds_ids),
        },
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }


# ---------- Save / Export helpers for the UI ----------

def _df_to_csv_bytes(df: Optional[pd.DataFrame]) -> Optional[bytes]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_report_zip(report: Dict[str, object]) -> bytes:
    """
    Produce a ZIP (in-memory) containing:
      - report.json
      - tables/*.csv (if available)
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # JSON (serialize DataFrames as dict records)
        serializable = dict(report)
        tables: Dict[str, pd.DataFrame] = serializable.get("tables", {}) or {}
        serializable["tables"] = {k: v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v
                                  for k, v in tables.items()}
        zf.writestr("report.json", json.dumps(serializable, indent=2))

        # CSVs
        for name, df in (report.get("tables") or {}).items():
            data = _df_to_csv_bytes(df)
            if data:
                zf.writestr(f"tables/{name}.csv", data)
    mem.seek(0)
    return mem.read()


def save_evaluation_report(
    title: str,
    report: Dict[str, object],
    owner: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> str:
    """
    Persist the JSON report via dataset_registry if available; otherwise
    fall back to a local file under the registry's storage root.
    Returns a generated report id.
    """
    # Generate a simple id
    rid = hashlib.sha1(
        (title + report.get("generated_at", dt.datetime.utcnow().isoformat())).encode("utf-8")
    ).hexdigest()[:12]

    meta = {
        "name": title,
        "kind": "evaluation_report",
        "owner": owner,
        "sources": sources or [],
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    # Try native registry save (present in your project)
    try:
        # Many repos expose something like: registry.save_json(kind, name, data, meta)
        # We attempt common variants and fall back to a local write.
        if hasattr(registry, "save_json"):
            registry.save_json(kind="evaluation_report", name=title, data=report, meta=meta, rid=rid)
            return rid
        if hasattr(registry, "register_json"):
            registry.register_json(kind="evaluation_report", name=title, data=report, meta=meta, rid=rid)
            return rid
    except Exception:
        pass

    # Fallback: write into a simple folder the app already reads from
    try:
        import os
        base = getattr(registry, "STORAGE_ROOT", None) or "./storage"
        path = os.path.join(base, "evaluation_reports")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{rid}.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "report": report}, f, ensure_ascii=False, indent=2)
    except Exception:
        # swallow; the UI will still show the RID
        pass

    return rid
