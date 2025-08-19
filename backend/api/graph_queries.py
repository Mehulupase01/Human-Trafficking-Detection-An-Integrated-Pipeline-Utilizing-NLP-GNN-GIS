# backend/api/graph_queries.py
"""
Query & insights helpers operating on the **processed** long-format dataset.

Assumptions (from Phase 1–2 pipeline):
- Long format with one row per (victim, location) hop.
- Columns include:
  - 'Serialized ID' (victim compact id)
  - 'Unique ID' (original id)
  - 'Location' (single exploded location per row)
  - 'Route_Order' (1..n per victim)
  - 'Perpetrators (NLP)'  (list[str], repeated per victim rows)
  - 'Chiefs (NLP)'        (list[str], repeated per victim rows)
  - Standardized 'Gender of Victim', 'Nationality of Victim'

This module provides:
- Safe value enumerators (locations, genders, nationalities, victims, perps, chiefs)
- Filters (gender = Any, location dropdown, optional text-search)
- Bidirectional queries: Victim→Perpetrators/Chiefs, Trafficker→Victims
- Trajectory per victim; origin/destination; location explorer (victims/perps/chiefs at location)
- Optional query logging to registry
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backend.core import dataset_registry as registry
from backend.core.standardize import smart_blank


# Column constants
COL_SID = "Serialized ID"
COL_UID = "Unique ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"
COL_GENDER = "Gender of Victim"
COL_NATION = "Nationality of Victim"
COL_PERPS = "Perpetrators (NLP)"
COL_CHIEFS = "Chiefs (NLP)"

REQUIRED = {COL_SID, COL_LOC, COL_ROUTE, COL_PERPS, COL_CHIEFS, COL_GENDER, COL_NATION, COL_UID}


def ensure_processed(df: pd.DataFrame) -> None:
    cols = set(df.columns)
    missing = REQUIRED - cols
    if missing:
        raise ValueError(f"DataFrame missing required processed columns: {sorted(missing)}")


def concat_processed_frames(dataset_ids: List[str]) -> pd.DataFrame:
    """
    Load and concat multiple processed/merged datasets (ignores non-tabular JSON artifacts).
    """
    frames: List[pd.DataFrame] = []
    for dsid in dataset_ids:
        meta = registry.get_entry(dsid)
        if meta.get("kind") not in {"processed", "merged"}:
            # ignore uploaded/raw or json artifacts
            continue
        df = registry.load_df(dsid)
        ensure_processed(df)
        frames.append(df)
    if not frames:
        raise ValueError("No processed/merged datasets to query.")
    full = pd.concat(frames, ignore_index=True, sort=False)
    # Normalize blanks
    for c in full.columns:
        if pd.api.types.is_object_dtype(full[c]):
            full[c] = full[c].map(smart_blank)
    return full


# ---------- Enumerators ----------

def unique_genders(df: pd.DataFrame) -> List[str]:
    vals = sorted({v for v in df[COL_GENDER].dropna().astype(str).tolist() if smart_blank(v)})
    return vals

def unique_nationalities(df: pd.DataFrame) -> List[str]:
    vals = sorted({v for v in df[COL_NATION].dropna().astype(str).tolist() if smart_blank(v)})
    return vals

def unique_locations(df: pd.DataFrame) -> List[str]:
    vals = sorted({v for v in df[COL_LOC].dropna().astype(str).tolist() if smart_blank(v)})
    return vals

def unique_victims(df: pd.DataFrame) -> List[str]:
    vals = sorted(df[COL_SID].dropna().astype(str).unique().tolist())
    return vals

def unique_perpetrators(df: pd.DataFrame) -> List[str]:
    if COL_PERPS not in df.columns:
        return []
    s = df[[COL_PERPS]].dropna()
    if s.empty:
        return []
    exploded = s[COL_PERPS].explode().dropna().astype(str)
    vals = sorted({v for v in exploded.tolist() if smart_blank(v)})
    return vals

def unique_chiefs(df: pd.DataFrame) -> List[str]:
    if COL_CHIEFS not in df.columns:
        return []
    s = df[[COL_CHIEFS]].dropna()
    if s.empty:
        return []
    exploded = s[COL_CHIEFS].explode().dropna().astype(str)
    vals = sorted({v for v in exploded.tolist() if smart_blank(v)})
    return vals


# ---------- Filters ----------

def apply_filters(
    df: pd.DataFrame,
    gender: str = "Any",
    nationality: Optional[str] = None,
    location: Optional[str] = None,
) -> pd.DataFrame:
    out = df
    if gender and gender != "Any":
        out = out[out[COL_GENDER] == gender]
    if nationality and smart_blank(nationality):
        out = out[out[COL_NATION] == nationality]
    if location and smart_blank(location):
        out = out[out[COL_LOC] == location]
    return out


# ---------- Queries ----------

def victim_to_perps_and_chiefs(df: pd.DataFrame, victim_sid: str) -> Dict[str, List[str]]:
    sub = df[df[COL_SID] == victim_sid]
    if sub.empty:
        return {"perpetrators": [], "chiefs": []}
    perps = sub[COL_PERPS].explode().dropna().astype(str).unique().tolist() if COL_PERPS in sub.columns else []
    chiefs = sub[COL_CHIEFS].explode().dropna().astype(str).unique().tolist() if COL_CHIEFS in sub.columns else []
    perps = [p for p in perps if smart_blank(p)]
    chiefs = [c for c in chiefs if smart_blank(c)]
    return {"perpetrators": sorted(perps), "chiefs": sorted(chiefs)}


def trafficker_to_victims(df: pd.DataFrame, trafficker_name: str) -> List[str]:
    if COL_PERPS not in df.columns or not smart_blank(trafficker_name):
        return []
    sub = df[df[COL_PERPS].apply(lambda lst: isinstance(lst, list) and trafficker_name in lst)]
    if sub.empty:
        return []
    sids = sorted(sub[COL_SID].astype(str).unique().tolist())
    return sids


def chief_to_victims(df: pd.DataFrame, chief_name: str) -> List[str]:
    if COL_CHIEFS not in df.columns or not smart_blank(chief_name):
        return []
    sub = df[df[COL_CHIEFS].apply(lambda lst: isinstance(lst, list) and chief_name in lst)]
    if sub.empty:
        return []
    sids = sorted(sub[COL_SID].astype(str).unique().tolist())
    return sids


def victim_trajectory(df: pd.DataFrame, victim_sid: str) -> List[str]:
    sub = df[df[COL_SID] == victim_sid].sort_values(COL_ROUTE, kind="stable")
    if sub.empty:
        return []
    return sub[COL_LOC].dropna().astype(str).tolist()


def origins_and_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a table with columns: Serialized ID | Origin | Destination
    """
    # For each victim, origin is min Route_Order, destination is max
    origin = df.loc[df.groupby(COL_SID)[COL_ROUTE].idxmin()][[COL_SID, COL_LOC]].rename(columns={COL_LOC: "Origin"})
    dest = df.loc[df.groupby(COL_SID)[COL_ROUTE].idxmax()][[COL_SID, COL_LOC]].rename(columns={COL_LOC: "Destination"})
    out = pd.merge(origin, dest, on=COL_SID, how="outer")
    return out[[COL_SID, "Origin", "Destination"]].sort_values(COL_SID, kind="stable").reset_index(drop=True)


def location_explorer(df: pd.DataFrame, location: str) -> Dict[str, List[str]]:
    sub = df[df[COL_LOC] == location]
    if sub.empty:
        return {"victims": [], "perpetrators": [], "chiefs": []}
    victims = sorted(sub[COL_SID].astype(str).unique().tolist())
    perps = sorted(sub[COL_PERPS].explode().dropna().astype(str).unique().tolist()) if COL_PERPS in sub.columns else []
    chiefs = sorted(sub[COL_CHIEFS].explode().dropna().astype(str).unique().tolist()) if COL_CHIEFS in sub.columns else []
    victims = [v for v in victims if smart_blank(v)]
    perps = [p for p in perps if smart_blank(p)]
    chiefs = [c for c in chiefs if smart_blank(c)]
    return {"victims": victims, "perpetrators": perps, "chiefs": chiefs}


# ---------- Logging (optional) ----------

def log_query(kind: str, params: Dict, owner: Optional[str] = None, source_ids: Optional[List[str]] = None) -> str:
    payload = {
        "kind": kind,
        "params": params,
        "owner": owner,
        "sources": source_ids or [],
    }
    qid = registry.save_json(name=f"Query ({kind})", payload=payload, kind="query_log", owner=owner, source=",".join(source_ids or []))
    return qid
