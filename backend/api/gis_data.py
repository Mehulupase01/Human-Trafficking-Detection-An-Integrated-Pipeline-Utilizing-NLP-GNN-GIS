# backend/api/gis_data.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from backend.geo.geo_utils import resolve_locations_to_coords
# rest of the file remains EXACTLY as previously provided
import pandas as pd

from backend.geo.geo_utils import resolve_locations_to_coords

COL_SID = "Serialized ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"
COL_PERPS = "Perpetrators (NLP)"
COL_CHIEFS = "Chiefs (NLP)"
COL_TIME = "Time Spent in Location / Cities / Places"  # optional, may be messy

def _parse_days(value) -> Optional[int]:
    """
    Best-effort parser for 'Time Spent...' values.
    Accepts integers (days) or strings like '3 days', '2 weeks', '1 month'.
    """
    if value is None:
        return None
    try:
        # numeric -> days
        iv = int(float(str(value).strip()))
        return max(0, iv)
    except Exception:
        pass
    s = str(value).strip().lower()
    for unit, days in [("day",1), ("days",1), ("week",7), ("weeks",7), ("month",30), ("months",30)]:
        if unit in s:
            # find leading number
            num = 0
            for token in s.replace("-", " ").split():
                try:
                    num = float(token); break
                except Exception:
                    continue
            if num > 0:
                return int(num * days)
    return None

def compute_location_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Returns:
        nodes_df: columns [location, count, victims, traffickers, chiefs, incoming, outgoing]
        edges_df: columns [source, target, weight]
        loc_to_victims: {location -> [sid,...]} (for popup listing)
    """
    # Build victim trajectories (ordered)
    edges = defaultdict(int)
    loc_to_victims: Dict[str, set] = defaultdict(set)
    loc_to_perps: Dict[str, set] = defaultdict(set)
    loc_to_chiefs: Dict[str, set] = defaultdict(set)
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)

    for sid, grp in df.groupby(COL_SID):
        g = grp.sort_values(COL_ROUTE, kind="stable")
        path = [str(x) for x in g[COL_LOC].dropna().astype(str).tolist()]
        # Victim list per location
        for loc in path:
            loc_to_victims[loc].add(str(sid))
        # Perps/Chiefs affiliation by row's location
        if COL_PERPS in g.columns:
            for _, row in g.iterrows():
                loc = str(row[COL_LOC])
                perps = row[COL_PERPS] if isinstance(row[COL_PERPS], list) else []
                for p in perps:
                    if p: loc_to_perps[loc].add(str(p))
                if COL_CHIEFS in g.columns:
                    chiefs = row[COL_CHIEFS] if isinstance(row[COL_CHIEFS], list) else []
                    for c in chiefs:
                        if c: loc_to_chiefs[loc].add(str(c))
        # Edges
        for a, b in zip(path, path[1:]):
            if a and b:
                edges[(a,b)] += 1
                out_deg[a] += 1
                in_deg[b] += 1

    # Nodes df
    locations = sorted({loc for locs in [list(loc_to_victims.keys()), list(loc_to_perps.keys()), list(loc_to_chiefs.keys())] for loc in locs})
    counts = {loc: len(loc_to_victims.get(loc, set())) for loc in locations}
    nodes_df = pd.DataFrame({
        "location": locations,
        "count": [counts[loc] for loc in locations],
        "victims": [sorted(loc_to_victims.get(loc, set())) for loc in locations],
        "traffickers": [sorted(loc_to_perps.get(loc, set())) for loc in locations],
        "chiefs": [sorted(loc_to_chiefs.get(loc, set())) for loc in locations],
        "incoming": [in_deg.get(loc, 0) for loc in locations],
        "outgoing": [out_deg.get(loc, 0) for loc in locations],
    }).sort_values("count", ascending=False).reset_index(drop=True)

    # Edges df
    edges_df = pd.DataFrame([(a,b,w) for (a,b), w in edges.items()], columns=["source","target","weight"]).sort_values("weight", ascending=False).reset_index(drop=True)

    # Map coords (only for existing locations)
    coords_map = resolve_locations_to_coords(nodes_df["location"].tolist())
    # Keep only nodes with coordinates
    nodes_df = nodes_df[nodes_df["location"].isin(coords_map.keys())].reset_index(drop=True)
    nodes_df["lat"] = nodes_df["location"].map(lambda k: coords_map[k][0])
    nodes_df["lon"] = nodes_df["location"].map(lambda k: coords_map[k][1])

    # Filter edges to mappable nodes
    edges_df = edges_df[edges_df["source"].isin(coords_map.keys()) & edges_df["target"].isin(coords_map.keys())].reset_index(drop=True)

    # Also supply victims per location (strings)
    out_loc_to_victims = {loc: sorted(list(loc_to_victims.get(loc, set()))) for loc in nodes_df["location"].tolist()}
    return nodes_df, edges_df, out_loc_to_victims

def build_timestamped_geojson(df: pd.DataFrame, default_days_per_hop: int = 7, base_date: str = "2020-01-01") -> dict:
    """
    Construct a Folium/Leaflet-compatible TimestampedGeoJson with per-victim trajectories.
    Uses 'Time Spent...' when parseable; otherwise uses default_days_per_hop.
    """
    start_dt = datetime.fromisoformat(base_date)
    features: List[dict] = []

    for sid, grp in df.groupby(COL_SID):
        g = grp.sort_values(COL_ROUTE, kind="stable")
        locs = g[COL_LOC].astype(str).tolist()
        coords_map = resolve_locations_to_coords(locs)
        # Build a timeline for this victim
        t0 = start_dt
        # derive per-hop days, try to parse from column; else fallback
        days_series = g.get(COL_TIME, pd.Series([None]*len(g))).tolist()
        hop_days: List[int] = []
        for val in days_series[1:]:  # between steps; number of gaps = len(locs)-1
            d = _parse_days(val)
            hop_days.append(d if d is not None else default_days_per_hop)
        if len(hop_days) < max(0, len(locs)-1):
            hop_days += [default_days_per_hop] * (len(locs)-1 - len(hop_days))

        for (a, b), d in zip(zip(locs, locs[1:]), hop_days):
            if a not in coords_map or b not in coords_map:
                continue
            (lat1, lon1) = coords_map[a]; (lat2, lon2) = coords_map[b]
            t1 = t0 + timedelta(days=d)
            feat = {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                "properties": {
                    "times": [t0.isoformat(), t1.isoformat()],
                    "style": {"color": "#4FC3F7", "weight": 3, "opacity": 0.8},
                    "popup": f"{sid}: {a} → {b} ({d} days)",
                }
            }
            features.append(feat)
            t0 = t1  # next segment starts where previous ended

    return {"type": "FeatureCollection", "features": features}
