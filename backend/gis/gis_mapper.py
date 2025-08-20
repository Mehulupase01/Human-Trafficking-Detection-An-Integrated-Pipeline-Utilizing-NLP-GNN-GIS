# backend/gis/gis_mapper.py
from __future__ import annotations
import io, zipfile
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from backend.core import dataset_registry as registry

# ---- column candidates (case-insensitive)
_NAME_CAND = [
    "name","asciiname","city","place","town","label","label en",
    "Name","City","Place","Town","LABEL EN"
]
_LAT_CAND  = ["lat","latitude","Lat","Latitude","y","Y"]
_LON_CAND  = ["lon","lng","longitude","Lon","Longitude","x","X"]

# Accept a single "Coordinates" column like "11.50021, 125.49811"
_COORD_CAND = ["coordinates", "coord", "coords", "Coordinates", "Coord", "Coords"]

# Country / admin variants commonly seen in GeoNames exports and portals
_COUNTRY_CAND = [
    "country","country code","country_code","Country","cc","iso2","ISO2",
    "country name","country name en","Country name EN","LABEL EN",
    "country_name","country_name_en"
]
_ADMIN1_CAND  = [
    "admin","admin1","admin1_code","adm1","state","region",
    "Admin1", "admin name", "admin1 name", "admin1_name", "Admin name", "Admin Name"
]
_POP_CAND  = ["population","pop","Population"]

def _pick_col(cols: List[str], cand: List[str]) -> Optional[str]:
    """Pick first matching column from candidates (case-insensitive)."""
    low = {str(c).strip().lower(): c for c in cols}
    for k in cand:
        if k.lower() in low:
            return low[k.lower()]
    return None

def _s(v) -> str:
    """Safe to-string + strip for any scalar."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    try:
        return str(v).strip()
    except Exception:
        return ""

def _f(v) -> Optional[float]:
    """Coerce to float if possible; else None."""
    try:
        return float(v)
    except Exception:
        try:
            return float(_s(v))
        except Exception:
            return None

def _split_coordinates(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Split a 'Coordinates' style column 'lat, lon' into two numeric series."""
    # Use astype(str) then split once on the first comma
    parts = series.astype(str).str.split(",", n=1, expand=True)
    if parts.shape[1] == 2:
        lat = parts[0].str.strip().map(_f)
        lon = parts[1].str.strip().map(_f)
    else:
        lat = pd.Series([None] * len(series))
        lon = pd.Series([None] * len(series))
    return lat, lon

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce schema and validity: name, lat, lon, (country, admin, population)."""
    need = ["name","lat","lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}'")

    out = df.copy()

    # Coerce numerics (handles strings from coordinates)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    # Valid lat/lon ranges
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 180))]

    # Clean names
    out["name"] = out["name"].astype(str).str.strip()
    out = out[out["name"] != ""]

    # Fill optional fields
    if "country" not in out.columns:
        out["country"] = np.nan
    if "admin" not in out.columns:
        out["admin"] = np.nan
    if "population" not in out.columns:
        out["population"] = np.nan

    # Final order
    return out[["name","lat","lon","country","admin","population"]].reset_index(drop=True)

# ------------------------ PUBLIC API ------------------------

def ingest_geonames_zip(upload, min_population: int = 0, title: str = "GeoNames") -> str:
    """
    Ingest cities5000.zip / cities15000.zip / cities1000.zip / allCountries.zip, etc.
    Reads the inner TXT as tab-separated (no header), maps official GeoNames positions.
    """
    # normalize bytes
    if hasattr(upload, "getvalue"):
        data = upload.getvalue()
    elif hasattr(upload, "read"):
        data = upload.read()
    else:
        data = upload  # bytes

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not inner:
            raise ValueError("ZIP does not contain a .txt GeoNames file.")
        # pick first .txt
        with zf.open(inner[0]) as fh:
            # Tab-separated, variable width; force strings; no NA coercion
            df_raw = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                dtype=str,
                na_filter=False,
                engine="python"
            )

    # Helper to get Nth column safely
    def col(i: int) -> pd.Series:
        try:
            return df_raw.iloc[:, i]
        except Exception:
            return pd.Series([], dtype="string")

    # GeoNames positions (0-based)
    # 1: name, 2: asciiname, 4: latitude, 5: longitude, 8: country code, 10: admin1 code, 14: population
    name_col = col(1)
    if name_col.empty or (name_col.astype(str).str.len().max() if len(name_col) else 0) == 0:
        name_col = col(2)  # fallback to asciiname

    df = pd.DataFrame({
        "name": name_col.map(_s),
        "lat":  col(4).map(_f),
        "lon":  col(5).map(_f),
        "country": col(8).map(_s),    # country code
        "admin":   col(10).map(_s),   # admin1 code
        "population": pd.to_numeric(col(14), errors="coerce")
    })

    if min_population and min_population > 0:
        df = df[df["population"].fillna(0) >= float(min_population)]

    df = _finalize(df)
    return registry.save_df(
        name=f"{title} (ingested)",
        df=df,
        kind="gazetteer",
        extra_meta={"rows": len(df)}
    )

def ingest_custom_gazetteer_csv(upload, title: str = "Custom gazetteer") -> str:
    """
    Ingest any CSV with at least (name, lat, lon) or a single 'coordinates' column.
    Auto-detect delimiter; skip bad lines; tolerate unusual headers.
    """
    # normalize bytes
    if hasattr(upload, "getvalue"):
        data = upload.getvalue()
    elif hasattr(upload, "read"):
        data = upload.read()
    else:
        data = upload

    bio = io.BytesIO(data)
    df_raw = pd.read_csv(
        bio,
        engine="python",      # tolerant tokenizing
        sep=None,             # infer delimiter
        on_bad_lines="skip",  # skip malformed lines
        dtype=str,
        na_filter=False
    )
    if df_raw.empty:
        raise ValueError("CSV appears empty/unreadable.")

    cols = df_raw.columns.tolist()
    name = _pick_col(cols, _NAME_CAND)
    lat  = _pick_col(cols, _LAT_CAND)
    lon  = _pick_col(cols, _LON_CAND)
    coord = _pick_col(cols, _COORD_CAND)
    cty  = _pick_col(cols, _COUNTRY_CAND)
    adm  = _pick_col(cols, _ADMIN1_CAND)
    pop  = _pick_col(cols, _POP_CAND)

    if not name:
        raise ValueError("Could not identify a 'name' column.")

    # If lat/lon not present, try to split a coordinates column
    lat_series = df_raw[lat] if lat else None
    lon_series = df_raw[lon] if lon else None
    if (lat_series is None or lon_series is None) and coord:
        lat_series, lon_series = _split_coordinates(df_raw[coord])

    if lat_series is None or lon_series is None:
        raise ValueError(
            f"Could not find required columns (name/lat/lon). "
            f"Detected: name={name}, lat={lat or coord}, lon={lon or coord}"
        )

    df = pd.DataFrame({
        "name": df_raw[name].map(_s),
        "lat":  pd.Series(lat_series).map(_f),
        "lon":  pd.Series(lon_series).map(_f),
        "country": (df_raw[cty].map(_s) if cty else np.nan),
        "admin":   (df_raw[adm].map(_s) if adm else np.nan),
        "population": pd.to_numeric(df_raw[pop], errors="coerce") if pop else np.nan
    })

    df = _finalize(df)
    return registry.save_df(
        name=f"{title} (ingested)",
        df=df,
        kind="gazetteer",
        extra_meta={"rows": len(df)}
    )
