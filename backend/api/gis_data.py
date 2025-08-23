# backend/api/gis_data.py
from __future__ import annotations

import ast
import datetime as dt
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.geo.geo_utils import resolve_locations_to_coords

# -----------------------------
# Robust extractors
# -----------------------------

_QSTR = re.compile(r"""['"]([^'"]+)['"]""")
_NUMS = re.compile(r"-?\d+(?:\.\d+)?")

def _as_list(obj) -> List:
    """Return obj as a Python list when it is already list-like."""
    if isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        return list(obj)
    return []

def _parse_list_of_places(cell) -> List[str]:
    """
    Accept many forms:
      - raw list objects: ['Eritrea', 'Ethiopia', ...]
      - numpy/pandas arrays/series
      - strings like:
        "['Eritrea' 'Ethiopia' 'Hitsats' 'Italy']"  ← (no commas, space-separated)
        "['Eritrea', 'Ethiopia', 'Hitsats']"
        "[Eritrea Ethiopia Hitsats Italy]" (no quotes)
        "Tripoli (Libya)" (single value)
    Returns a clean list of non-empty strings.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []

    lst = _as_list(cell)
    if lst:
        return [str(x).strip() for x in lst if str(x).strip()]

    s = str(cell).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()

        # 1) Try literal eval (best case)
        try:
            cand = ast.literal_eval(s)
            if isinstance(cand, (list, tuple)):
                return [str(x).strip() for x in cand if str(x).strip()]
        except Exception:
            pass

        # 2) Try quoted tokens
        q = _QSTR.findall(s)
        if q:
            return [t.strip() for t in q if t.strip()]

        # 3) Space/comma separation for unquoted values or
        #    python-style lists without commas: ['A' 'B'] or [A B]
        #    Normalize commas → space, then split on whitespace.
        inner = inner.replace(",", " ")
        toks = [t for t in inner.split() if t]
        return toks

    # Single token
    return [s]


def _parse_list_of_days(cell) -> List[float]:
    """
    Parse 'days' from a cell. Accepts:
      - list/array/series of numbers or strings with numbers
      - string lists (with/without commas)
      - a *single scalar number* → we return [that single number] and the caller can replicate
      - free text with numbers
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []

    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return [float(cell)]

    lst = _as_list(cell)
    if lst:
        out: List[float] = []
        for x in lst:
            try:
                out.append(float(x))
            except Exception:
                m = _NUMS.search(str(x))
                if m:
                    out.append(float(m.group(0)))
        return out

    s = str(cell).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        # try literal first
        try:
            cand = ast.literal_eval(s)
            if isinstance(cand, (list, tuple)):
                out: List[float] = []
                for x in cand:
                    try:
                        out.append(float(x))
                    except Exception:
                        m = _NUMS.search(str(x))
                        if m:
                            out.append(float(m.group(0)))
                return out
        except Exception:
            pass

        # generic extract from inside brackets
        return [float(x) for x in _NUMS.findall(s)]

    # free text: pull all numbers
    return [float(x) for x in _NUMS.findall(s)]


# ---------------------------------------
# Public: location aggregation / nodes DF
# ---------------------------------------

def compute_location_stats(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,            # unused in aggregation (kept for API compat)
    default_days_per_hop: int = 7,             # unused here
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Aggregate nodes from any 'place_col' by resolving all distinct places.

    Returns:
      nodes_df columns:
        ['location','lat','lon','count','victims','traffickers','chiefs','incoming','outgoing']
      edges_df: empty DataFrame (kept for compatibility)
      loc_to_victims: dict location -> list of victim ids (placeholder for now)
    """
    if df is None or df.empty or place_col not in df.columns:
        return (
            pd.DataFrame(columns=[
                "location","lat","lon","count","victims","traffickers","chiefs","incoming","outgoing"
            ]),
            pd.DataFrame(),
            {}
        )

    flat: List[str] = []
    for v in df[place_col].dropna().values.tolist():
        flat.extend(_parse_list_of_places(v))

    if not flat:
        return (
            pd.DataFrame(columns=[
                "location","lat","lon","count","victims","traffickers","chiefs","incoming","outgoing"
            ]),
            pd.DataFrame(),
            {}
        )

    uniq = sorted({s for s in flat if str(s).strip()})
    coord_map = resolve_locations_to_coords(uniq)

    counts: Dict[str, int] = {}
    for s in flat:
        if not str(s).strip():
            continue
        counts[s] = counts.get(s, 0) + 1

    rows = []
    for loc, cnt in counts.items():
        pt = coord_map.get(loc)
        if pt is None:
            continue
        rows.append({
            "location": loc,
            "lat": float(pt[0]),
            "lon": float(pt[1]),
            "count": int(cnt),
            "victims": [],
            "traffickers": [],
            "chiefs": [],
            "incoming": 0,
            "outgoing": 0,
        })

    nodes_df = pd.DataFrame(rows)
    edges_df = pd.DataFrame(columns=["a","b"])
    loc_to_victims: Dict[str, List[str]] = {}
    return nodes_df, edges_df, loc_to_victims


# ----------------------------------------------------------
# Public: row-wise trajectories -> Folium TimestampedGeoJson
# ----------------------------------------------------------

def build_timestamped_geojson(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,
    default_days_per_hop: int = 7,
    base_date: str = "2020-01-01",
) -> Dict:
    """
    Build a TimestampedGeoJson from row-wise trajectories.

    Each row may contain:
      - a list-like of places in `place_col`
      - optionally, a parallel list-like of days in `time_col`
        * length == len(places)-1 (per-hop), OR
        * length == len(places) (per-place), OR
        * a single scalar (use for all hops), OR
        * empty → use default_days_per_hop for each hop
    """
    if df is None or df.empty or place_col not in df.columns:
        return {"type": "FeatureCollection", "features": []}

    try:
        start0 = dt.datetime.fromisoformat(base_date)
    except Exception:
        start0 = dt.datetime(2020, 1, 1)

    features: List[Dict] = []

    for _, row in df.iterrows():
        places = _parse_list_of_places(row.get(place_col))
        if len(places) < 2:
            continue

        # Parse days
        days_raw: List[float] = []
        if time_col and (time_col in df.columns):
            days_raw = _parse_list_of_days(row.get(time_col))

        # Normalize to per-hop days
        if not days_raw:
            hop_days = [float(default_days_per_hop)] * (len(places) - 1)
        elif len(days_raw) == 1:
            # scalar → use same days for every hop
            hop_days = [float(days_raw[0])] * (len(places) - 1)
        elif len(days_raw) == len(places) - 1:
            hop_days = [float(d) for d in days_raw]
        elif len(days_raw) == len(places):
            hop_days = [float(days_raw[i]) for i in range(len(places) - 1)]
        else:
            hop_days = [float(default_days_per_hop)] * (len(places) - 1)

        # Geocode
        coord_map = resolve_locations_to_coords(places)

        # Build segments
        current = start0
        for i in range(len(places) - 1):
            a, b = places[i], places[i + 1]
            pa = coord_map.get(a)
            pb = coord_map.get(b)
            dur = float(hop_days[i])

            # Advance time even if we can't draw segment, to keep alignment
            t0 = current
            t1 = current + dt.timedelta(days=dur)
            current = t1

            if pa is None or pb is None:
                continue

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    # GeoJSON uses [lon, lat]
                    "coordinates": [[float(pa[1]), float(pa[0])], [float(pb[1]), float(pb[0])]],
                },
                "properties": {
                    "times": [t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")],
                    "style": {"color": "#66D9EF", "weight": 3, "opacity": 0.85},
                    "popup": f"{a} → {b}",
                },
            })

    return {"type": "FeatureCollection", "features": features}
