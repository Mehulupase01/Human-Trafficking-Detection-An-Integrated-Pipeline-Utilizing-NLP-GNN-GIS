# backend/api/gis_data.py
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import timedelta, date

from backend.geo.geo_utils import resolve_locations_to_coords

def _safe_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def compute_location_stats(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,
    default_days_per_hop: int = 7,
    **_ignore,
):
    """
    Build per-location aggregates and simple edges from a dataframe.
    - place_col: column that contains either a single place or a list-like string
    - time_col:  column with 'days spent' per hop (optional)
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Flatten all places and resolve coordinates
    places: List[str] = []
    for v in df[place_col].dropna().values.tolist():
        s = str(v).strip()
        if s.startswith("[") and s.endswith("]"):
            # keep textual list tokens together; split later in the page
            places.extend([s])  # will not be resolved; only used to guard empties
        else:
            places.append(s)

    # The Visualizer already parsed/flattened places before calling us and
    # uses resolve_locations_to_coords there to preview; here we compute per-node stats
    # based on the final resolved nodes in the dataset.
    # For safety, compute nodes from the set of unique strings in the selected column.
    unique_places: List[str] = []
    for v in df[place_col].dropna().values.tolist():
        unique_places.append(str(v).strip())

    # try to resolve as-is (the page already runs the stronger parser)
    coords_map = resolve_locations_to_coords(unique_places)

    rows = []
    loc_to_victims: Dict[str, List[str]] = {}
    for loc, grp in df.groupby(place_col, dropna=True):
        key = str(loc).strip()
        pt = coords_map.get(key)
        if pt is None:
            continue
        lat, lon = float(pt[0]), float(pt[1])
        victims = sorted(set(grp.get("Serialized ID", grp.index.astype(str)).astype(str).tolist()))
        traffickers = sorted(set(grp.get("Perpetrators (NLP)", []).astype(str).tolist()
                                 if "Perpetrators (NLP)" in grp.columns else []))
        chiefs = sorted(set(grp.get("Chiefs (NLP)", []).astype(str).tolist()
                            if "Chiefs (NLP)" in grp.columns else []))
        incoming = 0
        outgoing = 0
        count = len(victims)
        rows.append({
            "location": key, "lat": lat, "lon": lon, "count": count,
            "victims": victims, "traffickers": traffickers, "chiefs": chiefs,
            "incoming": incoming, "outgoing": outgoing,
        })
        loc_to_victims[key] = victims

    nodes_df = pd.DataFrame(rows)
    if nodes_df.empty:
        return nodes_df, pd.DataFrame(), {}

    # Build naive edges for animation: each row may encode a trajectory in place_col
    # The page supplies default_days_per_hop and optionally time_col for a synthetic timeline.
    edges: List[Dict] = []
    base_day = date(2020, 1, 1)
    for _, r in df.iterrows():
        seq = str(r[place_col]).strip()
        # Only connect simple "A -> B" when the cell contains a single "A" or "A (Country)" etc.
        # Complex list cells are handled in the page before calling build_timestamped_geojson.
        if "->" in seq:
            parts = [p.strip() for p in seq.split("->") if p.strip()]
        else:
            parts = [seq]

        if len(parts) < 2:
            continue

        days = default_days_per_hop
        if time_col and time_col in df.columns:
            try:
                d = int(r[time_col])
                if d > 0:
                    days = d
            except Exception:
                pass

        for a, b in zip(parts, parts[1:]):
            if a in nodes_df["location"].values and b in nodes_df["location"].values:
                edges.append({
                    "from": a, "to": b,
                    "eta_days": days,
                    "date": base_day.isoformat(),
                })
                base_day = base_day + timedelta(days=days)

    edges_df = pd.DataFrame(edges)
    return nodes_df, edges_df, loc_to_victims


def build_timestamped_geojson(
    df: pd.DataFrame,
    place_col: str,
    time_col: Optional[str] = None,
    default_days_per_hop: int = 7,
    base_date: str = "2020-01-01",
    **_ignore,
):
    """
    Convert edges into a TimestampedGeoJson FeatureCollection.
    The Visualizer calls this with the same (df, place_col, time_col, default_days_per_hop).
    """
    nodes_df, edges_df, _ = compute_location_stats(
        df=df,
        place_col=place_col,
        time_col=time_col,
        default_days_per_hop=default_days_per_hop,
    )
    if edges_df is None or edges_df.empty:
        return {"type": "FeatureCollection", "features": []}

    # map loc -> coords
    loc_to_xy = {row["location"]: (row["lat"], row["lon"]) for _, row in nodes_df.iterrows()}

    feats = []
    for _, e in edges_df.iterrows():
        a, b = e["from"], e["to"]
        if a not in loc_to_xy or b not in loc_to_xy:
            continue
        lat1, lon1 = loc_to_xy[a]
        lat2, lon2 = loc_to_xy[b]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
            "properties": {
                "times": [],  # Leaflet plugin accepts empty array for simple playback
                "style": {"color": "#FFF59D", "weight": 3, "opacity": 0.9},
                "popup": f"{a} → {b}",
            },
        })

    return {"type": "FeatureCollection", "features": feats}
