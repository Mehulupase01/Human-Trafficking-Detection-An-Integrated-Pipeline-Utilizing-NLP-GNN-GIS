# backend/api/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math
import pandas as pd

# Standardized (processed) column names used across the app
COL_SID      = "Serialized ID"
COL_UID      = "Unique ID"
COL_LOC      = "Location"
COL_ROUTE    = "Route_Order"
COL_GENDER   = "Gender of Victim"
COL_NATION   = "Nationality of Victim"
COL_PERPS    = "Perpetrators (NLP)"
COL_CHIEFS   = "Chiefs (NLP)"
COL_TIME     = "Time Spent in Location / Cities / Places"  # optional / messy

STD_FIELDS = [
    COL_SID, COL_UID, COL_LOC, COL_ROUTE,
    COL_GENDER, COL_NATION, COL_PERPS, COL_CHIEFS, COL_TIME
]

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def route_lengths(df: pd.DataFrame) -> pd.Series:
    if not {_ for _ in [COL_SID, COL_ROUTE, COL_LOC]}.issubset(df.columns):
        return pd.Series([], dtype=int)
    return df.groupby(COL_SID)[COL_LOC].count().astype(int)

def victims_per_location(df: pd.DataFrame) -> pd.DataFrame:
    """# victims who appear in each location (unique SIDs per loc)."""
    if not {_ for _ in [COL_SID, COL_LOC]}.issubset(df.columns):
        return pd.DataFrame(columns=["Location", "Victim Count"])
    tmp = df[[COL_SID, COL_LOC]].dropna()
    out = tmp.groupby(COL_LOC)[COL_SID].nunique().reset_index(name="Victim Count")
    return out.sort_values("Victim Count", ascending=False, kind="stable").reset_index(drop=True)

def victims_per_perpetrator(df: pd.DataFrame) -> pd.DataFrame:
    """# unique victims connected to each perpetrator (via row-level lists)."""
    if not _has(df, COL_PERPS) or not _has(df, COL_SID):
        return pd.DataFrame(columns=["Perpetrator", "Victim Count"])
    rows = []
    for _, r in df.iterrows():
        sid = str(r[COL_SID])
        perps = r[COL_PERPS] if isinstance(r[COL_PERPS], list) else []
        for p in perps:
            if not p: 
                continue
            rows.append((str(p), sid))
    if not rows:
        return pd.DataFrame(columns=["Perpetrator", "Victim Count"])
    tmp = pd.DataFrame(rows, columns=["Perpetrator", COL_SID]).drop_duplicates()
    out = tmp.groupby("Perpetrator")[COL_SID].nunique().reset_index(name="Victim Count")
    return out.sort_values("Victim Count", ascending=False, kind="stable").reset_index(drop=True)

def victims_per_chief(df: pd.DataFrame) -> pd.DataFrame:
    if not _has(df, COL_CHIEFS) or not _has(df, COL_SID):
        return pd.DataFrame(columns=["Chief", "Victim Count"])
    rows = []
    for _, r in df.iterrows():
        sid = str(r[COL_SID])
        chiefs = r[COL_CHIEFS] if isinstance(r[COL_CHIEFS], list) else []
        for c in chiefs:
            if not c:
                continue
            rows.append((str(c), sid))
    if not rows:
        return pd.DataFrame(columns=["Chief", "Victim Count"])
    tmp = pd.DataFrame(rows, columns=["Chief", COL_SID]).drop_duplicates()
    out = tmp.groupby("Chief")[COL_SID].nunique().reset_index(name="Victim Count")
    return out.sort_values("Victim Count", ascending=False, kind="stable").reset_index(drop=True)

def gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if not _has(df, COL_GENDER):
        return pd.DataFrame(columns=["Gender", "Count"])
    tmp = df[COL_GENDER].fillna("Unknown").astype(str)
    out = tmp.value_counts().reset_index()
    out.columns = ["Gender", "Count"]
    return out

def nationality_distribution(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if not _has(df, COL_NATION):
        return pd.DataFrame(columns=["Nationality", "Count"])
    tmp = df[COL_NATION].fillna("Unknown").astype(str)
    out = tmp.value_counts().reset_index()
    out.columns = ["Nationality", "Count"]
    if top_n > 0:
        return out.head(top_n).reset_index(drop=True)
    return out

def core_counts(df: pd.DataFrame) -> Dict[str, int]:
    victims = df[COL_SID].nunique() if _has(df, COL_SID) else 0
    uids    = df[COL_UID].nunique() if _has(df, COL_UID) else 0
    rows    = len(df)
    locs    = df[COL_LOC].nunique() if _has(df, COL_LOC) else 0
    # compute distinct perps/chiefs
    p_count, c_count = 0, 0
    if _has(df, COL_PERPS):
        s = set()
        for x in df[COL_PERPS].tolist():
            if isinstance(x, list):
                for p in x:
                    if p: s.add(str(p))
        p_count = len(s)
    if _has(df, COL_CHIEFS):
        s = set()
        for x in df[COL_CHIEFS].tolist():
            if isinstance(x, list):
                for c in x:
                    if c: s.add(str(c))
        c_count = len(s)
    route_len = route_lengths(df)
    med_route = int(route_len.median()) if not route_len.empty else 0
    # de-dup ratio if both ids exist
    dedup_ratio = round(uids / victims, 3) if victims > 0 and uids > 0 else None
    return {
        "rows": rows,
        "victims": victims,
        "unique_ids": uids,
        "locations": locs,
        "traffickers": p_count,
        "chiefs": c_count,
        "median_route_len": med_route,
        "dedupe_ratio_uid_per_sid": dedup_ratio,
    }

def build_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Produce all tables used by the dashboard."""
    tables: Dict[str, pd.DataFrame] = {}
    tables["Top Locations (by Victims)"] = victims_per_location(df).head(15)
    tables["Top Traffickers (by Victims)"] = victims_per_perpetrator(df).head(15)
    tables["Top Chiefs (by Victims)"] = victims_per_chief(df).head(15)
    tables["Gender Distribution"] = gender_distribution(df)
    tables["Nationality (Top 15)"] = nationality_distribution(df, top_n=15)
    # Route stats table
    rl = route_lengths(df)
    tables["Route Lengths (per Victim)"] = rl.reset_index().rename(columns={COL_SID: "Serialized ID", COL_LOC: "Stops"})
    return tables
