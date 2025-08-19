# backend/api/dataset_merge.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import copy

import pandas as pd

from backend.core import dataset_registry as registry

# Canonical fields
COL_SID    = "Serialized ID"
COL_UID    = "Unique ID"
COL_LOC    = "Location"
COL_ROUTE  = "Route_Order"
COL_PERPS  = "Perpetrators (NLP)"
COL_CHIEFS = "Chiefs (NLP)"

# Internal source tracking during merge
COL_SRC_ID   = "_Source_ID"
COL_SRC_NAME = "_Source_Name"
COL_SRC_ORD  = "_Source_Order"

MERGE_KIND   = "merged"
REPORT_KIND  = "merge_report"

# ---------------- helpers ----------------

def _victim_key(df: pd.DataFrame) -> pd.Series:
    """Prefer UID; else SID."""
    uid = df.get(COL_UID, pd.Series(index=df.index, dtype="object"))
    sid = df.get(COL_SID, pd.Series(index=df.index, dtype="object"))
    key = uid.fillna("").astype(str)
    key[key == ""] = sid.fillna("").astype(str)
    return key

def _coerce_list(x):
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            import ast
            try:
                y = ast.literal_eval(s)
                if isinstance(y, list):
                    return [str(t).strip() for t in y if str(t).strip()]
            except Exception:
                pass
        if ";" in s:
            return [t.strip() for t in s.split(";") if t.strip()]
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return []

def _normalize_lists(df: pd.DataFrame) -> pd.DataFrame:
    for c in (COL_PERPS, COL_CHIEFS):
        if c in df.columns:
            df[c] = df[c].apply(_coerce_list)
        else:
            df[c] = [[] for _ in range(len(df))]
    return df

def _concat_with_sources(dataset_ids: List[str]) -> pd.DataFrame:
    frames = []
    for i, did in enumerate(dataset_ids):
        try:
            df = registry.load_df(did).copy()
        except Exception:
            continue
        e = registry.get_entry(did)
        df[COL_SRC_ID] = did
        df[COL_SRC_NAME] = e.get("name", did)
        df[COL_SRC_ORD]  = i  # order selected by user
        frames.append(df)
    if not frames:
        raise ValueError("No datasets could be loaded.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    return _normalize_lists(df)

def _key_series(df: pd.DataFrame) -> pd.Series:
    """Group key: VictimKey|RouteOrder|Location."""
    k = _victim_key(df).astype(str)
    r = df.get(COL_ROUTE, pd.Series(index=df.index, dtype="object")).astype(str)
    l = df.get(COL_LOC, pd.Series(index=df.index, dtype="object")).astype(str)
    return k + "||r=" + r + "||l=" + l

def _find_conflict_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Conflicts = groups that have the same (victim, order, location) but
    disagree on Perpetrators/Chiefs (list content).
    """
    key = _key_series(df)
    df = df.assign(_K=key)
    groups = dict(tuple(df.groupby("_K", sort=False)))
    conflicts = {}
    for k, g in groups.items():
        if len(g) <= 1:
            continue
        # If any list differs across rows, mark conflict
        perps_set  = {tuple(x) for x in g[COL_PERPS].apply(lambda x: x if isinstance(x, list) else []).tolist()} if COL_PERPS in g.columns else set()
        chiefs_set = {tuple(x) for x in g[COL_CHIEFS].apply(lambda x: x if isinstance(x, list) else []).tolist()} if COL_CHIEFS in g.columns else set()
        if len(perps_set) > 1 or len(chiefs_set) > 1:
            conflicts[k] = g.copy()
    return conflicts

def _union_lists(rows: pd.DataFrame) -> Dict[str, List[str]]:
    perps: List[str] = []
    chiefs: List[str] = []
    if COL_PERPS in rows.columns:
        for lst in rows[COL_PERPS].tolist():
            if isinstance(lst, list):
                perps.extend(lst)
    if COL_CHIEFS in rows.columns:
        for lst in rows[COL_CHIEFS].tolist():
            if isinstance(lst, list):
                chiefs.extend(lst)
    # de-dup preserving order
    def _uniq(x):
        seen=set(); out=[]
        for t in x:
            t=str(t).strip()
            if t and t not in seen:
                seen.add(t); out.append(t)
        return out
    return {COL_PERPS: _uniq(perps), COL_CHIEFS: _uniq(chiefs)}

# ---------------- public merge API ----------------

def analyze_merge(dataset_ids: List[str]) -> Dict[str, object]:
    """
    Load + normalize sources and detect conflicts. No saving yet.
    """
    df = _concat_with_sources(dataset_ids)
    conflicts = _find_conflict_groups(df)
    return {
        "data": df,
        "conflicts": conflicts,                 # dict key -> DataFrame
        "conflict_count": len(conflicts),
        "rows": len(df),
        "unique_victims": int(_victim_key(df).nunique()),
    }

def merge_datasets(
    dataset_ids: List[str],
    owner: Optional[str] = None,
    *,
    strategy: str = "priority",                 # "priority" | "union_lists" | "keep_last"
    priority_sources: Optional[List[str]] = None,  # dataset ids in desired priority (left→right)
    manual_decisions: Optional[Dict[str, Dict]] = None, # key -> {"mode": "union"|"choose", "source_id": "..."}
    name: str = "Merged dataset",
) -> Dict[str, object]:
    """
    Perform merge with automatic strategy + optional per-conflict manual decisions.
    Returns saved merged dataset id + report id + summary and the resolved DataFrame.

    manual_decisions schema example:
      {
        "<VictimKey||r=2||l=City A>": {"mode":"union"},
        "<VictimKey||r=5||l=City B>": {"mode":"choose","source_id":"ds_172xxxx"}
      }
    """
    analysis = analyze_merge(dataset_ids)
    df = analysis["data"]
    conflicts: Dict[str, pd.DataFrame] = analysis["conflicts"]

    # Build a priority map (lower score = higher priority)
    source_order = priority_sources[:] if priority_sources else dataset_ids[:]
    prio_idx = {sid: i for i, sid in enumerate(source_order)}

    resolved_rows = []
    seen_keys = set()

    # First resolve all conflict groups
    for k, g in conflicts.items():
        seen_keys.add(k)
        decision = (manual_decisions or {}).get(k, {})
        mode = str(decision.get("mode", "")).lower()

        # Base row = best priority (or last) among the group
        g_sorted = g.sort_values(COL_SRC_ORD, kind="stable")
        # choose best by priority
        g_sorted["__prio"] = g_sorted[COL_SRC_ID].map(lambda x: prio_idx.get(x, 1_000_000))
        base = g_sorted.sort_values(["__prio", COL_SRC_ORD], kind="stable").iloc[0].copy()

        if mode == "union":
            union = _union_lists(g_sorted)
            base[COL_PERPS]  = union.get(COL_PERPS, [])
            base[COL_CHIEFS] = union.get(COL_CHIEFS, [])
        elif mode == "choose":
            sid = decision.get("source_id")
            if sid and sid in g_sorted[COL_SRC_ID].values:
                base = g_sorted[g_sorted[COL_SRC_ID] == sid].iloc[0].copy()
            else:
                # fallback to automatic strategy
                base = base
        else:
            # automatic strategy when no manual decision
            if strategy == "union_lists":
                union = _union_lists(g_sorted)
                base[COL_PERPS]  = union.get(COL_PERPS, [])
                base[COL_CHIEFS] = union.get(COL_CHIEFS, [])
            elif strategy == "priority":
                base = base
            elif strategy == "keep_last":
                base = g_sorted.sort_values(COL_SRC_ORD, ascending=False, kind="stable").iloc[0].copy()
            else:
                base = base

        # drop helper cols
        for c in ("_K", "__prio"):
            if c in base.index:
                del base[c]
        resolved_rows.append(base)

    # Now keep all non-conflicting rows as they are (one per key)
    df["_K"] = _key_series(df)
    non_conf = df[~df["_K"].isin(seen_keys)].copy()
    # For non-conf groups with multiple rows identical we still want one — keep best priority
    rest = []
    for k, g in non_conf.groupby("_K"):
        if len(g) == 1:
            r = g.iloc[0].copy()
        else:
            g["__prio"] = g[COL_SRC_ID].map(lambda x: prio_idx.get(x, 1_000_000))
            if strategy == "keep_last":
                r = g.sort_values(COL_SRC_ORD, ascending=False, kind="stable").iloc[0].copy()
            else:
                r = g.sort_values(["__prio", COL_SRC_ORD], kind="stable").iloc[0].copy()
        for c in ("_K", "__prio"):
            if c in r.index:
                del r[c]
        rest.append(r)

    resolved_df = pd.DataFrame(resolved_rows + rest).reset_index(drop=True)

    # Save artifacts
    before = len(df)
    after  = len(resolved_df)
    meta_summary = {
        "sources": dataset_ids,
        "rows_before": before,
        "rows_after": after,
        "unique_victims": int(_victim_key(resolved_df).nunique()),
        "conflict_rows": int(len(conflicts)),
        "strategy": strategy,
        "priority_sources": source_order,
        "manual_decisions_count": len(manual_decisions or {}),
    }

    mid = registry.save_df(
        name=name,
        df=resolved_df.drop(columns=[COL_SRC_ORD], errors="ignore"),
        kind=MERGE_KIND,
        owner=None,
        source=",".join(dataset_ids),
        extra_meta={"summary": meta_summary},
    )

    # Compose conflict report (compact)
    preview = []
    for k, g in list(conflicts.items())[:300]:
        preview.append({
            "key": k,
            "rows": len(g),
            "sources": g[COL_SRC_NAME].unique().tolist(),
            "example_perps": g[COL_PERPS].head(3).tolist() if COL_PERPS in g.columns else [],
            "example_chiefs": g[COL_CHIEFS].head(3).tolist() if COL_CHIEFS in g.columns else [],
            "decision": (manual_decisions or {}).get(k, {"mode":"(auto)"})
        })

    rid = registry.save_json(
        name=f"Merge report for {mid}",
        payload={"merged_id": mid, "summary": meta_summary, "conflicts_preview": preview},
        kind=REPORT_KIND,
        owner=None,
        source=",".join(dataset_ids),
    )

    return {
        "merged_id": mid,
        "report_id": rid,
        "summary": meta_summary,
        "resolved_df": resolved_df,
        "conflict_keys": list(conflicts.keys()),
    }

def list_merges() -> List[Dict]:
    return registry.list_datasets(kind=MERGE_KIND)

def delete_dataset(dataset_id: str) -> None:
    registry.delete(dataset_id)
