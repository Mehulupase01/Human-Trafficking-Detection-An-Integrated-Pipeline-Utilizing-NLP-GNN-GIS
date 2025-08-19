# backend/geo/geo_utils.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional, List

import pandas as pd
from rapidfuzz import fuzz

from backend.core import dataset_registry as registry
from backend.geo.gazetteer import resolve_with_gazetteer

# Highest priority: user-uploaded explicit mapping tables
def list_geo_lookups() -> List[dict]:
    return registry.find_datasets(kind="geo_lookup")

def save_geo_lookup_csv(name: str, df: pd.DataFrame, owner: Optional[str] = None) -> str:
    required = {"location", "lat", "lon"}
    if not required.issubset({c.lower() for c in df.columns}):
        raise ValueError("Geo CSV must contain columns: location, lat, lon")
    cols = {c.lower(): c for c in df.columns}
    out = df.rename(columns={cols["location"]: "location", cols["lat"]: "lat", cols["lon"]: "lon"})
    return registry.save_df(name=name or "Geo Lookup", df=out, kind="geo_lookup", owner=owner)

def _collect_user_geo_maps() -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for entry in list_geo_lookups():
        try:
            df = registry.load_df(entry["id"])
            if not {"location", "lat", "lon"}.issubset(set(df.columns)):
                continue
            for _, row in df.iterrows():
                loc = str(row["location"]).strip()
                try:
                    lat = float(row["lat"]); lon = float(row["lon"])
                except Exception:
                    continue
                if loc:
                    out[loc] = (lat, lon)
        except Exception:
            continue
    return out

# tiny emergency fallback (will rarely be used if a gazetteer is ingested)
_DEFAULT_SEED = {
    "Tripoli": (32.8872, 13.1913),
    "Khartoum": (15.5007, 32.5599),
    "Asmara": (15.3229, 38.9251),
    "Addis Ababa": (8.9806, 38.7578),
    "Cairo": (30.0444, 31.2357),
    "Agadez": (16.9733, 7.9911),
    "Tunis": (36.8065, 10.1815),
}

def resolve_locations_to_coords(locations: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    """
    Resolve using:
      1) User lookup tables
      2) Active Gazetteer (GeoNames/custom) with fuzzy & alias matching
      3) Tiny seed (last resort)
    """
    # dedupe inputs
    uniq = [str(x).strip() for x in set(locations) if isinstance(x, str) and str(x).strip()]
    out: Dict[str, Tuple[float, float]] = {}

    # 1) user tables
    u = _collect_user_geo_maps()
    for loc in uniq:
        if loc in u:
            out[loc] = u[loc]

    # 2) gazetteer for unresolved
    unresolved = [loc for loc in uniq if loc not in out]
    if unresolved:
        g_res = resolve_with_gazetteer(unresolved, score_cutoff=88)
        out.update(g_res)

    # 3) tiny seed for anything still unresolved
    for loc in unresolved:
        if loc not in out and loc in _DEFAULT_SEED:
            out[loc] = _DEFAULT_SEED[loc]

    return out
