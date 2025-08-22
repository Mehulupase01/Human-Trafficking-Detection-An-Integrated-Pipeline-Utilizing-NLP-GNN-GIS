# backend/gis/gis_mapper.py
from __future__ import annotations
import io, re, zipfile
from typing import List, Optional
import numpy as np
import pandas as pd
from backend.core import dataset_registry as registry

# ---- column candidates (case‑insensitive)
_NAME_CAND = ["name","asciiname","city","place","town","Name","City","Place","Town"]
_LAT_CAND  = ["lat","latitude","Lat","Latitude","y","Y"]
_LON_CAND  = ["lon","lng","longitude","Lon","Longitude","x","X"]
_COUNTRY_CAND = ["country","country_code","Country","cc","iso2","ISO2"]
_ADMIN1_CAND  = ["admin","admin1","admin1_code","adm1","state","region","Admin1"]
_POP_CAND  = ["population","pop","Population"]

def _pick_col(cols: List[str], cand: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in cand:
        if k.lower() in low:
            return low[k.lower()]
    return None

def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    try:
        return str(v).strip()
    except Exception:
        return ""

# -------- robust coordinate parsing --------
def _coord_from_string(v, kind: str) -> Optional[float]:
    """
    Parse latitude/longitude from messy strings.

    Handles:
      - normal floats: "32.8967", "-13.1"
      - thousand-separated decimals: "328.966.720" -> 32.8966720 (lat), "131.777.923" -> 131.777923 (lon)
      - comma decimals: "35,6895"
      - stray characters: "  32.8967  (Libya)  "
    """
    limit = 90.0 if kind == "lat" else 180.0
    if v is None:
        return None

    s = str(v).strip()
    if not s or s.upper() in {"N/A", "NA", "NULL", "NONE"}:
        return None

    # unify commas to dot, keep only digits/dot/sign for last resorts
    s_norm = s.replace(",", ".").strip()

    # 1) straight float first
    try:
        val = float(s_norm)
        if abs(val) <= limit:
            return val
    except Exception:
        pass

    # 2) "three-dot" pattern like 328.966.720 or -11.019.081
    if s_norm.count(".") >= 2:
        sign = -1.0 if "-" in s_norm else 1.0
        digits = "".join(ch for ch in s_norm if ch.isdigit())
        if digits:
            try:
                num = int(digits)
            except Exception:
                num = None
            if num is not None:
                # Try scales that yield a plausible coord.
                # For lat we prefer 2 deg + 6–7 decimals -> 1e7 then 1e6.
                # For lon we prefer 3 deg + 6 decimals -> 1e6 then 1e7.
                scales = (1e7, 1e6, 1e5, 1e4) if kind == "lat" else (1e6, 1e7, 1e5, 1e4)
                for sc in scales:
                    val = sign * (num / sc)
                    if abs(val) <= limit:
                        return val

    # 3) last resort: strip junk, keep first dot only, then normalize by dividing until in range
    s_junkless = re.sub(r"[^0-9.\-]+", "", s_norm)
    try:
        val = float(s_junkless)
        while abs(val) > limit:
            val /= 10.0
        return val
    except Exception:
        return None

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    need = ["name", "lat", "lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}'")
    out = df.copy()

    # numeric coercion after our robust parser
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])

    out["name"] = out["name"].astype(str).str.strip()
    out = out[out["name"] != ""]

    for k in ["country", "admin", "population"]:
        if k not in out.columns:
            out[k] = np.nan

    return out[["name", "lat", "lon", "country", "admin", "population"]].reset_index(drop=True)

# ------------------------ PUBLIC API ------------------------

def ingest_geonames_zip(upload, min_population: int = 0, title: str = "GeoNames") -> str:
    """Ingest cities5000.zip / cities1000.zip / cities15000.zip / allCountries.zip."""
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
        with zf.open(inner[0]) as fh:
            df_raw = pd.read_csv(
                fh, sep="\t", header=None, dtype=str, na_filter=False, engine="python"
            )

    # Map by official positions (see GeoNames readme)
    def col(i):
        try:
            return df_raw.iloc[:, i]
        except Exception:
            return pd.Series([], dtype="string")

    name = col(1) if 1 in df_raw.columns else pd.Series("", dtype="string")
    if name.empty or (getattr(name, "str", None) and name.str.len().max() == 0):
        name = col(2)  # fallback to asciiname

    df = pd.DataFrame({
        "name": name.map(_s),
        "lat":  col(4).map(lambda x: _coord_from_string(x, "lat")),
        "lon":  col(5).map(lambda x: _coord_from_string(x, "lon")),
        "country": col(8).map(_s),   # country code
        "admin":   col(10).map(_s),  # admin1 code
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
    """Ingest any CSV with at least (name, lat, lon). Auto‑detect delimiter; skip bad lines."""
    if hasattr(upload, "getvalue"):
        data = upload.getvalue()
    elif hasattr(upload, "read"):
        data = upload.read()
    else:
        data = upload

    bio = io.BytesIO(data)
    df_raw = pd.read_csv(
        bio,
        engine="python",     # avoid C‑engine tokenizer errors
        sep=None,            # infer delimiter
        on_bad_lines="skip", # skip malformed lines
        dtype=str,
        na_filter=False
    )
    if df_raw.empty:
        raise ValueError("CSV appears empty/unreadable.")

    cols = df_raw.columns.tolist()
    name = _pick_col(cols, _NAME_CAND)
    lat  = _pick_col(cols, _LAT_CAND)
    lon  = _pick_col(cols, _LON_CAND)
    cty  = _pick_col(cols, _COUNTRY_CAND)
    adm  = _pick_col(cols, _ADMIN1_CAND)
    pop  = _pick_col(cols, _POP_CAND)
    if not (name and lat and lon):
        raise ValueError(f"Could not find required columns (name/lat/lon). Detected: name={name}, lat={lat}, lon={lon}")

    df = pd.DataFrame({
        "name": df_raw[name].map(_s),
        "lat":  df_raw[lat].map(lambda x: _coord_from_string(x, "lat")),
        "lon":  df_raw[lon].map(lambda x: _coord_from_string(x, "lon")),
        "country": df_raw[cty].map(_s) if cty else np.nan,
        "admin":   df_raw[adm].map(_s) if adm else np.nan,
        "population": pd.to_numeric(df_raw[pop], errors="coerce") if pop else np.nan
    })

    df = _finalize(df)
    return registry.save_df(
        name=f"{title} (ingested)",
        df=df,
        kind="gazetteer",
        extra_meta={"rows": len(df)}
    )
