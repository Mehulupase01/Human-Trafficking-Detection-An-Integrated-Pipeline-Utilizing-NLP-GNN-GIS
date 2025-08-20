# backend/geo/geo_utils.py
from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.core import dataset_registry as registry

# Optional: RapidFuzz gives much better fuzzy scores if available.
try:
    from rapidfuzz import fuzz, process  # type: ignore
    _HAVE_RF = True
except Exception:  # pragma: no cover
    import difflib
    _HAVE_RF = False

# Optional: country name → ISO2 mapping
try:
    import pycountry  # type: ignore
    _HAVE_PYCOUNTRY = True
except Exception:  # pragma: no cover
    _HAVE_PYCOUNTRY = False

# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------

_STOPWORDS = {
    "city", "province", "governorate", "state", "region", "prefecture",
    "district", "wilaya", "municipality", "town", "village", "county",
    "of", "the"
}

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _norm_tokenize(s: str) -> List[str]:
    s = _strip_accents(s.lower())
    s = re.sub(r"[^\w\s,/-]+", " ", s)      # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if t and t not in _STOPWORDS]
    return toks

def _norm_key(s: str) -> str:
    return " ".join(_norm_tokenize(s))

def _n(s: Optional[str]) -> str:
    return _norm_key(str(s or ""))

# ------------------------------------------------------------------
# Gazetteer / explicit lookup loading
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _active_gaz_df() -> pd.DataFrame:
    """
    Load active gazetteer as columns:
    name, lat, lon, country, admin, population
    plus normalized helper columns: name_n, country_n, admin_n.
    """
    from backend.geo.gazetteer import get_active_gazetteer_id  # lazy import
    gid = get_active_gazetteer_id()
    if not gid:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","name_n","country_n","admin_n"])
    try:
        df = registry.load_df(gid)
    except Exception:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","name_n","country_n","admin_n"])

    keep = [c for c in ["name","lat","lon","country","admin","population"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","name_n","country_n","admin_n"])

    df = df[keep].copy()
    df["name"] = df["name"].astype(str)
    df["country"] = df.get("country", pd.Series("", index=df.index)).fillna("").astype(str)
    df["admin"]   = df.get("admin",   pd.Series("", index=df.index)).fillna("").astype(str)

    df["name_n"]    = df["name"].map(_n)
    df["country_n"] = df["country"].map(_n)
    df["admin_n"]   = df["admin"].map(_n)
    return df

@lru_cache(maxsize=1)
def _explicit_lookup_tables() -> List[pd.DataFrame]:
    """Load explicit user CSV lookups (location,lat,lon). Newest first."""
    try:
        items = list_geo_lookups()
    except Exception:
        items = []
    out: List[pd.DataFrame] = []
    for it in items:
        try:
            df = registry.load_df(it["id"])
            cols = {c.lower(): c for c in df.columns}
            if not {"location","lat","lon"}.issubset(set(cols.keys())):
                continue
            d2 = pd.DataFrame({
                "location": df[cols["location"]].astype(str),
                "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
                "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
            }).dropna()
            d2["loc_n"] = d2["location"].map(_n)
            out.append(d2)
        except Exception:
            continue
    return out

def clear_geo_caches() -> None:
    """Call this if you switch the active gazetteer to refresh caches."""
    _active_gaz_df.cache_clear()
    _explicit_lookup_tables.cache_clear()
    _resolve_one.cache_clear()

# ------------------------------------------------------------------
# Country helpers
# ------------------------------------------------------------------

@lru_cache(maxsize=1_000)
def _country_from_text(norm_text: str) -> Optional[str]:
    """Try to detect a country ISO2 code in normalized text."""
    toks = norm_text.split()
    # ISO2 token, e.g., 'ly', 'sd'
    for t in toks:
        if len(t) == 2 and t.isalpha():
            return t.upper()

    if _HAVE_PYCOUNTRY:
        try:
            # try whole string
            c = pycountry.countries.get(name=norm_text.title())
            if c and getattr(c, "alpha_2", None):
                return c.alpha_2
        except Exception:
            pass
        # try every token as a country name
        for t in toks:
            try:
                c = pycountry.countries.get(name=t.title())
                if c and getattr(c, "alpha_2", None):
                    return c.alpha_2
            except Exception:
                continue
    return None

def _best_city_in_country(iso2: str) -> Optional[Tuple[float, float]]:
    """Fallback centroid: pick most populous gazetteer entry for country."""
    g = _active_gaz_df()
    if g.empty:
        return None
    sub = g[(g["country"].str.upper() == iso2.upper())]
    if sub.empty:
        return None
    if "population" in sub.columns:
        sub = sub.sort_values("population", ascending=False, na_position="last")
    r = sub.iloc[0]
    try:
        return float(r["lat"]), float(r["lon"])
    except Exception:
        return None

# ------------------------------------------------------------------
# Fuzzy matching
# ------------------------------------------------------------------

def _score(a: str, b: str) -> float:
    if _HAVE_RF:
        return float(fuzz.token_set_ratio(a, b))
    else:  # pragma: no cover
        return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()

def _best_match_idx(query_norm: str, cand_norms: pd.Series, min_score: float) -> Optional[int]:
    if _HAVE_RF:
        res = process.extractOne(query_norm, cand_norms.tolist(), scorer=fuzz.WRatio, score_cutoff=min_score)
        if res:
            _, _, idx = res
            # map back to original index
            return cand_norms.index[idx]
        return None
    # Fallback: manual scan
    best_idx = None
    best = -1.0
    for idx, val in cand_norms.items():
        sc = _score(query_norm, val)
        if sc > best:
            best, best_idx = sc, idx
    return best_idx if best >= min_score else None

# ------------------------------------------------------------------
# Single-item resolution (cached)
# Accepts either a string "Tripoli" or a dict:
#   {"location": "Tripoli", "country": "Libya", "admin": "Tripoli"}
# ------------------------------------------------------------------

@lru_cache(maxsize=8192)
def _resolve_one(raw: str, ctry_hint: str = "", adm_hint: str = "") -> Optional[Tuple[float, float]]:
    """Resolve one item to (lat, lon)."""
    if not raw or not str(raw).strip():
        return None
    q_norm = _n(str(raw))
    if not q_norm:
        return None

    # 1) explicit lookup CSVs (priority)
    for tbl in _explicit_lookup_tables():
        hit = tbl.loc[tbl["loc_n"] == q_norm]
        if not hit.empty:
            r = hit.iloc[0]
            return float(r["lat"]), float(r["lon"])

    gaz = _active_gaz_df()
    if gaz.empty:
        return None

    # 2) exact in gazetteer (normalized)
    exact = gaz.loc[gaz["name_n"] == q_norm]
    if not exact.empty:
        # prefer country/admin match if hints present
        if ctry_hint or adm_hint:
            c_n = _n(ctry_hint)
            a_n = _n(adm_hint)
            sub = exact
            if ctry_hint:
                sub = sub[sub["country_n"] == c_n] if not sub.empty else sub
            if adm_hint:
                sub = sub[sub["admin_n"] == a_n] if not sub.empty else sub
            if not sub.empty:
                r = sub.iloc[0]
                return float(r["lat"]), float(r["lon"])
        r = exact.iloc[0]
        return float(r["lat"]), float(r["lon"])

    # Try to detect a country code from the text or hints (helps centroid fallback)
    iso2 = _country_from_text(q_norm) or _country_from_text(_n(ctry_hint)) if ctry_hint else None

    # 3) country-aware fuzzy: prefer matches in that country/admin
    if ctry_hint:
        c_n = _n(ctry_hint)
        sub = gaz[gaz["country_n"] == c_n]
        if not sub.empty:
            # if admin hint exists, try admin-filtered first
            if adm_hint:
                a_n = _n(adm_hint)
                sub2 = sub[sub["admin_n"] == a_n]
                idx = _best_match_idx(q_norm, sub2["name_n"], min_score=84.0) if not sub2.empty else None
                if idx is not None:
                    r = sub2.loc[idx]
                    return float(r["lat"]), float(r["lon"])
            idx = _best_match_idx(q_norm, sub["name_n"], min_score=84.0)
            if idx is not None:
                r = sub.loc[idx]
                return float(r["lat"]), float(r["lon"])

    # 4) general fuzzy on name only
    idx = _best_match_idx(q_norm, gaz["name_n"], min_score=90.0)
    if idx is not None:
        r = gaz.loc[idx]
        return float(r["lat"]), float(r["lon"])

    # 5) if the string is likely a country (or we extracted an ISO2), pick a centroid
    if iso2 is not None:
        centroid = _best_city_in_country(iso2)
        if centroid is not None:
            return centroid

    return None

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

Item = Union[str, Dict[str, str]]

def resolve_locations_to_coords(items: Iterable[Item]) -> Dict[str, Tuple[float, float]]:
    """
    Resolve items to (lat,lon).
    - If item is a string, it's treated as a location name.
    - If item is a dict, it may include {"location", "country", "admin"} for better disambiguation.
    Returns a dict keyed by the original *location string* (for dicts, uses the 'location' field).
    """
    out: Dict[str, Tuple[float, float]] = {}
    for it in items:
        if isinstance(it, dict):
            loc = str(it.get("location", "")).strip()
            if not loc:
                continue
            pt = _resolve_one(loc, str(it.get("country","")), str(it.get("admin","")))
            if pt is not None:
                out[loc] = pt
        else:
            loc = str(it)
            pt = _resolve_one(loc, "", "")
            if pt is not None:
                out[loc] = pt
    return out

def match_report(items: Iterable[Item]) -> Dict[str, int]:
    """Return {'total', 'matched', 'unmatched'} for quick diagnostics."""
    total = 0
    hit = 0
    for it in items:
        total += 1
        if isinstance(it, dict):
            loc = str(it.get("location",""))
            ok = _resolve_one(loc, str(it.get("country","")), str(it.get("admin",""))) is not None if loc else False
        else:
            ok = _resolve_one(str(it), "", "") is not None
        if ok:
            hit += 1
    return {"total": total, "matched": hit, "unmatched": total - hit}

# ------------------------------------------------------------------
# Explicit lookup helpers (used by the page)
# ------------------------------------------------------------------

def save_geo_lookup_csv(name: str, df: pd.DataFrame, owner: str | None = None):
    cols = {c.lower(): c for c in df.columns}
    need = {"location", "lat", "lon"}
    if not need.issubset(set(cols)):
        raise ValueError("Geo lookup CSV must have columns: location, lat, lon")
    slim = pd.DataFrame({
        "location": df[cols["location"]].astype(str),
        "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
    }).dropna()
    return registry.save_df(name=name, df=slim, kind="geo_lookup", owner=owner)

def list_geo_lookups():
    return registry.find_datasets(kind="geo_lookup")
