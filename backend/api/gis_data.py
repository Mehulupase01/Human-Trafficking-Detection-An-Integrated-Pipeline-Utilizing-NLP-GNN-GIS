# backend/api/gis_data.py
from __future__ import annotations

import ast
import datetime as dt
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.geo.geo_utils import resolve_locations_to_coords


# -----------------------------
# Robust list/number extractors
# -----------------------------

_QSTR = re.compile(r"""['"]([^'"]+)['"]""")
_NUMS = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_list_of_places(cell) -> List[str]:
    """
    Accepts many shapes:
      - "Tripoli (Libya)"
      - ['Eritrea' 'Ethiopia' 'Hitsats']
      - "['Eritrea', 'Ethiopia', 'Hitsats']"
      - ["Eritrea","Ethiopia","Hitsats"]
      - list objects
    Returns a clean list of non-empty strings.
    """
    if cell is None:
        return []
    if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
        return [str(x).strip() for x in list(cell) if str(x).strip()]

    s = str(cell).strip()
    if not s:
        return []

    # Looks like a Python-ish list?
    if s.startswith("[") and s.endswith("]"):
        # 1) Try literal
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
        # 2) Try “quote captures”
        found = _QSTR.findall(s)
        if found:
            return [t.strip() for t in found if t.strip()]
        # 3) Fallback: split inner by whitespace/comma
        inner = s[1:-1].strip()
        if inner:
            toks = re.split(r"[,\s]+", inner)
            toks = [t for t in toks if t]
            return toks
        return []

    # Otherwise a single token
    return [s]


def _parse_list_of_days(cell) -> List[float]:
    """
    Parse a parallel list of 'days spent' from a cell.
    Accepts: [7, 4, 2], "7,4,2", "7 4 2", or free text with numbers.
    """
    if cell is None:
        return []
    if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
        out: List[float] = []
        for x in list(cell):
            try:
                out.append(float(x))
            except Exception:
                # pull first number from string, if any
                m = _NUMS.search(str(x))
                if m:
                    out.append(float(m.group(0)))
        return out

    s = str(cell).strip()
    if not s:
        return []
    # Try Python literal
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                out: List[float] = []
                for x in list(val):
                    try:
                        out.append(float(x))
                    except Exception:
                        m = _NUMS.search(str(x))
                        if m:
                            out.append(float(m.group(0)))
                return out
        except Exception:
            pass
    # Generic extract of all numbers
    return [float(x) for x in _NUMS.findall(s)]


# ---------------------------------------
# Public: location aggregation / nodes DF
# ---------------------------------------

def compute_location_stats(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,
    default_days_per_hop: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Aggregate nodes from any 'place_col' by resolving all distinct places.
    Edges_df is returned empty for now (optional future use).

    Returns:
      nodes_df columns:
        ['location','lat','lon','count','victims','traffickers','chiefs','incoming','outgoing']
      edges_df: empty DataFrame (kept for compatibility)
      loc_to_victims: dict location -> list of victim ids (if present)
    """
    if df is None or df.empty or place_col not in df.columns:
        return pd.DataFrame(columns=["location","lat","lon","count","victims","traffickers","chiefs","incoming","outgoing"]), pd.DataFrame(), {}

    # Flatten the selected column to a list of place strings
    flat_places: List[str] = []
    for v in df[place_col].dropna().values.tolist():
        flat_places.extend(_parse_list_of_places(v))

    if not flat_places:
        return pd.DataFrame(columns=["location","lat","lon","count","victims","traffickers","chiefs","incoming","outgoing"]), pd.DataFrame(), {}

    # Resolve all unique place strings
    uniq = sorted({s for s in flat_places if str(s).strip()})
    coord_map = resolve_locations_to_coords(uniq)

    # Count occurrences and keep lightweight metadata if available
    counts: Dict[str, int] = {}
    for s in flat_places:
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
            "victims": [],        # placeholders; your pipeline can populate these later
            "traffickers": [],
            "chiefs": [],
            "incoming": 0,
            "outgoing": 0,
        })

    nodes_df = pd.DataFrame(rows)
    edges_df = pd.DataFrame(columns=["a","b"])  # placeholder
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
    Build a TimestampedGeoJson from *row-wise* trajectories.

    Each row may contain:
      - a list-like of places in `place_col`
      - optionally, a parallel list-like of days in `time_col` (length == len(places) or len(places)-1)
    We:
      1) resolve each place to (lat,lon)
      2) create segments between consecutive places
      3) assign times by cumulatively adding days, starting from base_date
    """
    if df is None or df.empty or place_col not in df.columns:
        return {"type": "FeatureCollection", "features": []}

    try:
        start = dt.datetime.fromisoformat(base_date)
    except Exception:
        start = dt.datetime(2020, 1, 1)

    features: List[Dict] = []

    for _, row in df.iterrows():
        places = _parse_list_of_places(row.get(place_col))
        if len(places) < 2:
            # a single point is not an animated segment; skip
            continue

        days_list: List[float] = []
        if time_col and (time_col in df.columns):
            days_list = _parse_list_of_days(row.get(time_col, None))

        # If we got N places, we need N-1 hop-durations.
        # Accept length N (per-place duration) or N-1 (per-hop duration) or empty (use default)
        if days_list:
            if len(days_list) == len(places) - 1:
                hop_days = days_list
            elif len(days_list) == len(places):
                # convert per-place → per-hop by pairing
                hop_days = [float(days_list[i]) for i in range(len(places) - 1)]
            else:
                hop_days = [float(default_days_per_hop)] * (len(places) - 1)
        else:
            hop_days = [float(default_days_per_hop)] * (len(places) - 1)

        # Resolve coordinates
        coord_map = resolve_locations_to_coords(places)

        # Build segments
        current_time = start
        for i in range(len(places) - 1):
            a, b = places[i], places[i + 1]
            pa = coord_map.get(a)
            pb = coord_map.get(b)
            if pa is None or pb is None:
                # skip segments that fail to geocode
                current_time += dt.timedelta(days=float(hop_days[i]))
                continue

            # Times for this segment: start -> start+hop
            t0 = current_time
            t1 = current_time + dt.timedelta(days=float(hop_days[i]))
            current_time = t1  # advance

            # Folium TimestampedGeoJson expects RFC3339 strings
            f = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(pa[1]), float(pa[0])], [float(pb[1]), float(pb[0])]],  # [lon,lat]
                },
                "properties": {
                    "times": [t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")],
                    "style": {"color": "#66D9EF", "weight": 3, "opacity": 0.8},
                    "popup": f"{a} → {b}",
                },
            }
            features.append(f)

    return {"type": "FeatureCollection", "features": features}
