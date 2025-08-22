from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.core import dataset_registry as registry

# Optional: RapidFuzz gives much better fuzzy scores if available.
try:
    from rapidfuzz import fuzz  # type: ignore
    _HAVE_RF = True
except Exception:
    import difflib  # fallback
    _HAVE_RF = False

# Optional: country name → ISO2 mapping
try:
    import pycountry  # type: ignore
    _HAVE_PYCOUNTRY = True
except Exception:
    _HAVE_PYCOUNTRY = False

# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------

_STOPWORDS = {
    "city", "province", "governorate", "state", "region", "prefecture",
    "district", "wilaya", "municipality", "town", "village", "county",
    "office", "camp", "unhcr", "iom",
    "of", "the", "and", "in", "at"
}

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _norm_tokenize(s: str) -> List[str]:
    s = _strip_accents(str(s).lower())
    s = re.sub(r"[^\w\s,/-]+", " ", s)      # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if t and t not in _STOPWORDS]
    return toks

def _norm_key(s: str) -> str:
    return " ".join(_norm_tokenize(s))

# ------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------

def list_geo_lookups() -> List[Dict]:
    """
    Return registry entries saved as kind='geo_lookup'.
    Each should be a CSV saved via the app with columns location,lat,lon.
    """
    try:
        return registry.find_datasets(kind="geo_lookup")
    except Exception:
        return []

# ------------------------------------------------------------------
# Gazetteer / explicit lookup loading
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _active_gaz_df() -> pd.DataFrame:
    """
    Load active gazetteer as (name,lat,lon,country,admin,population) with normalized keys.
    """
    try:
        from backend.geo.gazetteer import get_active_gazetteer_id  # lazy import
    except Exception:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","norm","norm_cty"])

    gid = None
    try:
        gid = get_active_gazetteer_id()
    except Exception:
        pass

    if not gid:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","norm","norm_cty"])

    try:
        df = registry.load_df(gid)
    except Exception:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","norm","norm_cty"])

    keep = [c for c in ["name","lat","lon","country","admin","population"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["name","lat","lon","country","admin","population","norm","norm_cty"])

    df = df[keep].copy()
    # Coerce + normalize
    df["name"] = df["name"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["norm"] = df["name"].map(_norm_key)

    if "country" not in df.columns:
        df["country"] = ""
    df["country"] = df["country"].fillna("").astype(str)

    df["norm_cty"] = np.where(
        df["country"] != "",
        (df["norm"] + " " + df["country"].str.lower()),
        df["norm"]
    )
    return df

@lru_cache(maxsize=1)
def _explicit_lookup_tables() -> List[pd.DataFrame]:
    """
    Load explicit user CSV lookups (location,lat,lon). Newest first.
    """
    items = list_geo_lookups()
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
            d2["norm"] = d2["location"].map(_norm_key)
            out.append(d2)
        except Exception:
            continue
    return out

# ------------------------------------------------------------------
# Country helpers
# ------------------------------------------------------------------

@lru_cache(maxsize=1_000)
def _country_from_text(norm_text: str) -> Optional[str]:
    """
    Try to detect a country ISO2 code in normalized text.
    - Detects existing ISO2 tokens (e.g., 'ly', 'sd')
    - Uses pycountry when available to map full names
    """
    toks = norm_text.split()
    # ISO2 token, e.g., 'ly', 'sd'
    for t in toks:
        if len(t) == 2 and t.isalpha():
            return t.upper()

    if _HAVE_PYCOUNTRY:
        # try the whole string
        try:
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
        # token_set_ratio is robust to swapped/extra words
        return float(fuzz.token_set_ratio(a, b))
    else:
        return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()

def _best_match(query_norm: str, cand_norms: pd.Series, min_score: float) -> Optional[int]:
    best_idx = None
    best = -1.0
    for idx, val in cand_norms.items():
        sc = _score(query_norm, val)
        if sc > best:
            best, best_idx = sc, idx
    return best_idx if best >= min_score else None

# ------------------------------------------------------------------
# Single-string resolution (cached)
# ------------------------------------------------------------------

@lru_cache(maxsize=8192)
def _resolve_one(raw_location: str) -> Optional[Tuple[float, float]]:
    """
    Resolve one raw string to (lat, lon):
      1) explicit CSV exact (normalized)
      2) exact gazetteer match
      3) country-aware fuzzy (name + country)
      4) fuzzy on name alone
      5) country-only fallback → most populous city as centroid
    """
    if not raw_location or not str(raw_location).strip():
        return None
    q_norm = _norm_key(str(raw_location))
    if not q_norm:
        return None

    # 1) explicit lookup CSVs (priority)
    for tbl in _explicit_lookup_tables():
        hit = tbl.loc[tbl["norm"] == q_norm]
        if not hit.empty:
            r = hit.iloc[0]
            return float(r["lat"]), float(r["lon"])

    gaz = _active_gaz_df()
    if gaz.empty:
        return None

    # 2) exact in gazetteer (normalized)
    exact = gaz.loc[gaz["norm"] == q_norm]
    if not exact.empty:
        r = exact.iloc[0]
        return float(r["lat"]), float(r["lon"])

    # Try to detect a country code from the text (helps both 3 and 5)
    iso2 = _country_from_text(q_norm)

    # 3) country-aware fuzzy: prefer matches in that country
    if iso2 is not None:
        sub = gaz[gaz["country"].str.upper() == iso2.upper()]
        if not sub.empty:
            idx = _best_match(q_norm, sub["norm_cty"], min_score=84.0)
            if idx is not None:
                r = sub.loc[idx]
                return float(r["lat"]), float(r["lon"])

    # 4) general fuzzy on name only
    idx = _best_match(q_norm, gaz["norm"], min_score=90.0)
    if idx is not None:
        r = gaz.loc[idx]
        return float(r["lat"]), float(r["lon"])

    # 5) if likely a country, pick a centroid
    if iso2 is not None:
        centroid = _best_city_in_country(iso2)
        if centroid is not None:
            return centroid

    return None

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def resolve_locations_to_coords(locations: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    """Resolve an iterable of strings to (lat,lon). Uses cache for speed."""
    out: Dict[str, Tuple[float, float]] = {}
    for s in locations:
        try:
            pt = _resolve_one(str(s))
            if pt is not None:
                out[str(s)] = pt
        except Exception:
            continue
    return out

def match_report(locations: Iterable[str]) -> Dict[str, int]:
    """Return {'total', 'matched', 'unmatched'} for quick diagnostics."""
    total = 0
    hit = 0
    for s in locations:
        total += 1
        if _resolve_one(str(s)) is not None:
            hit += 1
    return {"total": total, "matched": hit, "unmatched": total - hit}

# ------------------------------------------------------------------
# Cache management helper (for front-end refresh)
# ------------------------------------------------------------------

def bust_geo_caches() -> None:
    """Clear all cached gazetteer and resolution state."""
    try:
        _active_gaz_df.cache_clear()
    except Exception:
        pass
    try:
        _explicit_lookup_tables.cache_clear()
    except Exception:
        pass
    try:
        _resolve_one.cache_clear()
    except Exception:
        pass
    try:
        _country_from_text.cache_clear()
    except Exception:
        pass
