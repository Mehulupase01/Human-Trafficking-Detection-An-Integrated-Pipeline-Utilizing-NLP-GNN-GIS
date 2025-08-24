from __future__ import annotations
"""
Column resolver + helpers tailored to the processed dataset you shared.

What it does
------------
- Maps your schema to the evaluator's standard fields (sid, route_order, location_name, text, doc_id, actors, eta_days).
- Parses stringified list columns like "['Walid' 'Medhanie']".
- Normalizes location strings like "Asmara (Eritrea)" -> "Asmara".
- Parses fuzzy durations like "5 months and 2 weeks", "3/4 days", "2-3 days", "48 hours".
- Attaches lat/lon by joining a gazetteer dataset:
    * preferred: registry dataset with kind="gazetteer"
    * fallback: local file "data/gazetteer.csv"
    * gazetteer column detection is forgiving: name|city|location, lat|latitude|y, lon|lng|longitude|x.

Outputs
-------
resolve(df, registry) -> dict with:
    - df: augmented DataFrame (lat, lon, doc_id, text, actors, eta_days)
    - columns: mapping dict of resolved column names used
    - geocode: {"resolved": int, "total": int, "rate": float}
    - diagnostics: small stats block
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import ast
import os
import re

import numpy as np
import pandas as pd


# ---------- parsing helpers ----------

LIST_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")

STOP_ACTOR_TOKENS = {
    "yes","no","none","unknown","n/a","na","nil","-","_",
    "chief","perpetrator","perpetrators","human traffickers","place","of","and","the"
}

def parse_listish(cell: Any) -> List[str]:
    """Parse cells that look like "['A' 'B']" or "['A','B']" or even raw 'A;B'."""
    if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
        return []
    if isinstance(cell, list):
        vals = cell
    else:
        s = str(cell).strip()
        if not s or s in {"[]", "['']"}:
            return []
        # first try literal eval safely
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                vals = list(obj)
            else:
                vals = [str(obj)]
        except Exception:
            # fallback: regex quoted tokens, else split on common delimiters
            tokens = LIST_QUOTED_RE.findall(s)
            if not tokens:
                tokens = re.split(r"[;,|\u2022]", s)
            vals = [t for t in (tok.strip() for tok in tokens) if t]
    # clean
    cleaned: List[str] = []
    for v in vals:
        vv = str(v).strip().strip("'\"")
        if not vv:
            continue
        # drop trivial junk
        if vv.lower() in STOP_ACTOR_TOKENS:
            continue
        cleaned.append(vv)
    # uniquify preserving order
    seen = set()
    out = []
    for v in cleaned:
        k = v.lower()
        if k not in seen:
            out.append(v)
            seen.add(k)
    return out


def normalize_location(name: Any) -> Optional[str]:
    """Take 'City (Country)' and return 'City'; trim whitespace & noise."""
    if name is None or (isinstance(name, float) and not np.isfinite(name)):
        return None
    s = str(name).strip()
    if not s:
        return None
    # keep text before first '(' if present
    s = s.split("(")[0].strip()
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s or None


DUR_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?|\d+/\d+)(?:\s|-)?(?P<unit>day|days|week|weeks|month|months|hour|hours)",
    re.IGNORECASE,
)

def _to_float(num_str: str) -> float:
    if "/" in num_str:
        a, b = num_str.split("/", 1)
        try:
            return float(a) / float(b)
        except Exception:
            return np.nan
    try:
        return float(num_str.replace(",", "."))
    except Exception:
        return np.nan

def parse_duration_to_days(text: Any) -> Optional[float]:
    """
    Robust parser for free-text durations like:
    - "5 months and 2 weeks" -> 5*30 + 2*7
    - "3/4 days" -> 0.75
    - "2-3 days" -> 2.5
    - "48 hours" -> 2.0
    Returns None if nothing could be parsed.
    """
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return None
    s = str(text).lower()
    # normalize "2-3 days" -> "2 days 3 days"
    s = re.sub(r"(\d+)\s*-\s*(\d+)\s*(day|days)", r"\1 days \2 days", s)
    parts = DUR_RE.findall(s)
    if not parts:
        return None
    total_days = 0.0
    for num_str, unit in parts:
        val = _to_float(num_str)
        if not np.isfinite(val):
            continue
        u = unit.lower()
        if u.startswith("day"):
            total_days += val
        elif u.startswith("week"):
            total_days += val * 7.0
        elif u.startswith("month"):
            total_days += val * 30.0
        elif u.startswith("hour"):
            total_days += val / 24.0
    return float(total_days) if np.isfinite(total_days) and total_days > 0 else None


# ---------- gazetteer loading & join ----------

def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    s = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in s:
            return s[cand.lower()]
    return None

def _load_gazetteer_df(registry=None, path_fallback: str = "data/gazetteer.csv") -> Optional[pd.DataFrame]:
    # From registry kind="gazetteer"
    if registry is not None:
        try:
            items = registry.list_datasets(kind="gazetteer") or []
        except Exception:
            items = []
        if items:
            gid = items[0].get("id")
            for fn in ("load_json", "read_json"):
                try:
                    rows = getattr(registry, fn)(gid)
                    gdf = pd.DataFrame(rows)
                    if not gdf.empty:
                        return gdf
                except Exception:
                    pass
            for fn in ("load_text", "read_text"):
                try:
                    txt = getattr(registry, fn)(gid)
                    if isinstance(txt, str) and txt.strip():
                        from io import StringIO
                        gdf = pd.read_csv(StringIO(txt))
                        if not gdf.empty:
                            return gdf
                except Exception:
                    pass
    # Fallback CSV
    try:
        if os.path.exists(path_fallback):
            gdf = pd.read_csv(path_fallback)
            if not gdf.empty:
                return gdf
    except Exception:
        pass
    return None


def attach_geocodes(df: pd.DataFrame, *, name_col: str, registry=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Left join lat/lon onto df by normalized location name (case-insensitive).
    Returns (augmented_df, stats).
    """
    gdf = _load_gazetteer_df(registry)
    if gdf is None or gdf.empty:
        return df.copy(), {"resolved": 0, "total": int(df[name_col].notna().sum()), "rate": 0.0, "source": "none"}

    # detect gazetteer columns
    ncol = _pick_col(gdf.columns, ["name", "city", "location", "place", "loc", "query"])
    lat_col = _pick_col(gdf.columns, ["lat", "latitude", "y"])
    lon_col = _pick_col(gdf.columns, ["lon", "lng", "longitude", "x"])
    if ncol is None or lat_col is None or lon_col is None:
        # can't interpret gazetteer; abort gracefully
        return df.copy(), {"resolved": 0, "total": int(df[name_col].notna().sum()), "rate": 0.0, "source": "invalid_gazetteer"}

    gdf = gdf[[ncol, lat_col, lon_col]].rename(columns={ncol: "_gaz_name", lat_col: "lat", lon_col: "lon"})
    gdf["_key"] = gdf["_gaz_name"].astype(str).str.strip().str.lower()

    work = df.copy()
    work["_key"] = work[name_col].astype(str).str.strip().str.lower()
    out = work.merge(gdf[["_key", "lat", "lon"]].drop_duplicates("_key"), on="_key", how="left")
    total = int(work[name_col].notna().sum())
    resolved = int(out["lat"].notna().sum())
    rate = float(resolved / total) if total > 0 else 0.0
    out.drop(columns=["_key"], inplace=True, errors="ignore")
    return out, {"resolved": resolved, "total": total, "rate": rate, "source": "gazetteer"}


# ---------- resolver ----------

def resolve(df: pd.DataFrame, registry=None) -> Dict[str, Any]:
    """
    Produce a mapping + augmented df with standard fields present:
      sid, route_order, location_name, doc_id, text, actors (list[str]), eta_days, lat, lon
    """
    if df is None or df.empty:
        return {"df": df, "columns": {}, "geocode": {"resolved": 0, "total": 0, "rate": 0.0}, "diagnostics": {"rows": 0}}

    cols = df.columns

    # Core mappings based on your CSV
    sid = "Serialized ID" if "Serialized ID" in cols else _pick_col(cols, ["sid","subject_id","victim_id","case_id"])
    route_order = "Route_Order" if "Route_Order" in cols else _pick_col(cols, ["route_order","order","step"])
    loc_name = None
    for cand in ["City / Locations Crossed", "Location", "Locations", "Place", "City"]:
        if cand in cols:
            loc_name = cand
            break

    # Build text (for retrieval) by concatenating most informative fields available
    text_fields = [c for c in [
        "City / Locations Crossed",
        "Name of the Perpetrators involved",
        "Hierarchy of Perpetrators",
        "Human traffickers/ Chief of places",
        "Locations (NLP)",
        "Perpetrators (NLP)",
        "Chiefs (NLP)",
    ] if c in cols]
    def _mk_text(row) -> str:
        parts: List[str] = []
        for c in text_fields:
            v = row.get(c)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            parts.append(s)
        return " | ".join(parts)

    # Actors list from NLP columns if present
    actors_cols = [c for c in ["Perpetrators (NLP)", "Chiefs (NLP)"] if c in cols]

    # Duration parsing
    dur_text_col = "Time Spent in Location / Cities / Places" if "Time Spent in Location / Cities / Places" in cols else None
    dur_days_col = "Time Spent (days)" if "Time Spent (days)" in cols else None

    work = df.copy()

    # Normalize location name to a new column
    if loc_name is not None:
        work["_location_name"] = work[loc_name].map(normalize_location)
    else:
        work["_location_name"] = None

    # Synth doc_id: "{sid}-{route_order}" if both exist, else row index
    if sid and route_order and sid in cols and route_order in cols:
        work["_doc_id"] = work[sid].astype(str).str.strip() + "-" + work[route_order].astype(str).str.strip()
    elif sid and sid in cols:
        work["_doc_id"] = work[sid].astype(str).str.strip() + "-" + work.index.astype(str)
    else:
        work["_doc_id"] = work.index.astype(str)

    # Build actors list
    if actors_cols:
        work["_actors"] = (work[actors_cols]
                           .apply(lambda r: list({x for col in actors_cols for x in parse_listish(r.get(col))}), axis=1))
    else:
        work["_actors"] = [[] for _ in range(len(work))]

    # Build text
    if text_fields:
        work["_text"] = work.apply(_mk_text, axis=1)
    else:
        # fallback to any string columns
        str_cols = [c for c in cols if pd.api.types.is_string_dtype(work[c])]
        work["_text"] = work[str_cols].astype(str).agg(" | ".join, axis=1) if str_cols else ""

    # Duration to eta_days
    if dur_text_col:
        work["_eta_days"] = work[dur_text_col].map(parse_duration_to_days)
    else:
        work["_eta_days"] = None
    # fallback to numeric days if reasonable (< 2000)
    if dur_days_col and "_eta_days" in work:
        mask = work["_eta_days"].isna()
        num = pd.to_numeric(work[dur_days_col], errors="coerce")
        num = num.where(num < 2000)  # drop pathological outliers
        work.loc[mask, "_eta_days"] = num[mask]

    # Attach lat/lon via gazetteer if we have a location name
    geo_stats = {"resolved": 0, "total": 0, "rate": 0.0, "source": "none"}
    if "_location_name" in work.columns and work["_location_name"].notna().any():
        work, geo_stats = attach_geocodes(work, name_col="_location_name", registry=registry)

    mapping = {
        "sid": sid or "_row_",          # fallback; splitter will treat each row as its own group
        "route_order": route_order,
        "location_name": "_location_name",
        "doc_id": "_doc_id",
        "text": "_text",
        "actors": "_actors",
        "eta_days": "_eta_days",
        "lat": "lat" if "lat" in work.columns else None,
        "lon": "lon" if "lon" in work.columns else None,
    }

    diags = {
        "rows": int(len(work)),
        "n_subjects": int(work[sid].nunique()) if sid and sid in work.columns else None,
        "n_locations_non_null": int(work["_location_name"].notna().sum()),
        "n_with_eta_days": int(pd.Series(work["_eta_days"]).notna().sum()),
        "n_with_text": int((work["_text"].astype(str).str.len() > 0).sum()),
        "n_with_lat": int(work["lat"].notna().sum()) if "lat" in work.columns else 0,
    }

    return {"df": work, "columns": mapping, "geocode": geo_stats, "diagnostics": diags}
