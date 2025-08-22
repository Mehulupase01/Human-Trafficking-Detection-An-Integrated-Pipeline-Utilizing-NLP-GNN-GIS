# backend/api/gis_data.py
from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.geo.geo_utils import resolve_locations_to_coords


def _to_list_of_places(val) -> List[str]:
    """
    Turn a value that might be:
      - a python list
      - a string that *looks like* ['A', 'B', 'C'] or "A, B, C"
      - a single string
    into a clean list of non-empty strings.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []

    # already a list/tuple
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]

    s = str(val).strip()
    if not s:
        return []

    # Try to parse python literal list: "['A', 'B']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    # Fallback: split on comma if present, otherwise treat whole string as one token
    if "," in s:
        parts = [p.strip(" '\"\n\t") for p in s.split(",")]
        return [p for p in parts if p]

    return [s]


def _days_for_hops(hops: int, days_value, default_days_per_hop: int) -> List[int]:
    """
    For a path of length N (hops = N-1), return a list of hop-days.
    If 'days_value' is present (e.g. 'Time Spent (days)'), replicate it across hops;
    else use default_days_per_hop.
    """
    if hops <= 0:
        return []
    if days_value is None or (isinstance(days_value, float) and np.isnan(days_value)):
        return [int(default_days_per_hop)] * hops
    try:
        d = int(days_value)
        if d <= 0:
            raise ValueError
        return [d] * hops
    except Exception:
        return [int(default_days_per_hop)] * hops


def compute_location_stats(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,
    default_days_per_hop: int = 7,
    victim_col: str = "Serialized ID",
):
    """
    Build nodes and edges for mapping/animation.

    Parameters
    ----------
    df : DataFrame
        Processed dataset.
    place_col : str
        Column containing a trajectory per row (list-like or list-as-string).
    time_col : Optional[str]
        Column with a number of days spent at each row's location(s).
        If not provided, default_days_per_hop is used for each hop.
    default_days_per_hop : int
        Fallback days per hop.
    victim_col : str
        Column to use as a per-row identifier (used in popups/diagnostics).

    Returns
    -------
    nodes_df : DataFrame  (location, lat, lon, count, victims, traffickers, chiefs, incoming, outgoing)
    edges_df : DataFrame  (source, target, days)
    loc_to_victims : Dict[str, List[str]]
    """
    if place_col not in df.columns:
        raise ValueError(f"Column '{place_col}' not found in dataframe.")

    # Victim-like columns (best effort; won’t break if absent)
    victim_series = df[victim_col] if victim_col in df.columns else pd.Series(df.index.astype(str))
    traff_col = None
    for c in ["Traffickers (NLP)", "Human traffickers/ Chief of places", "Perpetrators (NLP)"]:
        if c in df.columns:
            traff_col = c
            break
    chief_col = None
    for c in ["Chiefs (NLP)", "Hierarchy of Perpetrators"]:
        if c in df.columns:
            chief_col = c
            break

    # Collect rows
    nodes: List[dict] = []
    edges: List[dict] = []
    loc_to_victims: Dict[str, List[str]] = {}

    for i, row in df.iterrows():
        traj = _to_list_of_places(row[place_col])
        if not traj:
            continue

        # coords for all used places in this row
        coords = resolve_locations_to_coords(traj)

        # victim id for this row
        victim_id = str(victim_series.get(i, i))

        # nodes — add one occurrence per row/place to allow counting
        for loc in traj:
            if loc not in coords:
                continue
            lat, lon = coords[loc]
            nodes.append({"location": loc, "lat": float(lat), "lon": float(lon), "count": 1})
            loc_to_victims.setdefault(loc, []).append(victim_id)

        # edges — pairwise hops with per-hop day estimates
        hop_days = _days_for_hops(len(traj) - 1, (row[time_col] if (time_col and time_col in df.columns) else None), default_days_per_hop)
        for a, b, d in zip(traj, traj[1:], hop_days):
            if a in coords and b in coords:
                edges.append({"source": a, "target": b, "days": int(d)})

    nodes_df = pd.DataFrame(nodes)
    if nodes_df.empty:
        # no mappable points — return empty compatible frames
        return (
            pd.DataFrame(columns=["location", "lat", "lon", "count", "victims", "traffickers", "chiefs", "incoming", "outgoing"]),
            pd.DataFrame(columns=["source", "target", "days"]),
            {},
        )

    # aggregate counts by (location,lat,lon)
    nodes_df = (
        nodes_df.groupby(["location", "lat", "lon"], as_index=False)
        .agg(count=("count", "sum"))
    )

    # attach auxiliary lists for popups (best effort)
    def _take_list(col):
        if col is None or col not in df.columns:
            return {}
        out = {}
        for i, row in df.iterrows():
            traj = _to_list_of_places(row[place_col])
            val = row[col]
            if pd.isna(val):
                continue
            as_str = str(val)
            for loc in traj:
                out.setdefault(loc, set()).add(as_str)
        return {k: sorted(v) for k, v in out.items()}

    victims_per_loc = loc_to_victims
    traff_per_loc = _take_list(traff_col)
    chief_per_loc = _take_list(chief_col)

    # compute degree (incoming/outgoing) from edges
    edges_df = pd.DataFrame(edges)
    incoming = {}
    outgoing = {}
    if not edges_df.empty:
        for _, r in edges_df.iterrows():
            outgoing[r["source"]] = outgoing.get(r["source"], 0) + 1
            incoming[r["target"]] = incoming.get(r["target"], 0) + 1

    nodes_df["victims"] = nodes_df["location"].map(lambda k: victims_per_loc.get(k, []))
    nodes_df["traffickers"] = nodes_df["location"].map(lambda k: traff_per_loc.get(k, []))
    nodes_df["chiefs"] = nodes_df["location"].map(lambda k: chief_per_loc.get(k, []))
    nodes_df["incoming"] = nodes_df["location"].map(lambda k: incoming.get(k, 0))
    nodes_df["outgoing"] = nodes_df["location"].map(lambda k: outgoing.get(k, 0))

    return nodes_df, edges_df, loc_to_victims


def build_timestamped_geojson(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,
    default_days_per_hop: int = 7,
    base_date: str = "2020-01-01",
):
    """
    Build a minimal TimestampedGeoJson payload from df.
    Each row contributes a polyline; times are expanded using days per hop.
    """
    from datetime import datetime, timedelta

    if place_col not in df.columns:
        return {"type": "FeatureCollection", "features": []}

    base = datetime.fromisoformat(base_date)
    features = []

    for _, row in df.iterrows():
        traj = _to_list_of_places(row[place_col])
        if len(traj) < 2:
            continue
        coords_map = resolve_locations_to_coords(traj)
        # filter only those with coords
        path = [(loc, coords_map[loc]) for loc in traj if loc in coords_map]
        if len(path) < 2:
            continue

        hop_days = _days_for_hops(
            len(path) - 1,
            (row[time_col] if (time_col and time_col in df.columns) else None),
            default_days_per_hop,
        )

        ts = []
        t = base
        for d in hop_days:
            t = t + timedelta(days=int(d))
            ts.append(t.isoformat())

        # assemble coordinates for the polyline
        latlon = [[p[1][1], p[1][0]] for p in path]  # (lon,lat)
        feat = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": latlon},
            "properties": {"times": ts},
        }
        features.append(feat)

    return {"type": "FeatureCollection", "features": features}
