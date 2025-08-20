# backend/gis/gis_mapper.py
from __future__ import annotations
import io, zipfile, re
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from backend.core import dataset_registry as registry

# ---- column candidates (case-insensitive)
_NAME_CAND = ["name","asciiname","city","place","town","Name","City","Place","Town"]
_LAT_CAND  = ["lat","latitude","Lat","Latitude","y","Y"]
_LON_CAND  = ["lon","lng","longitude","Lon","Longitude","x","X"]
_COORDS_CAND = ["coordinates","coord","Coord","Coordinates"]
_COUNTRY_CAND = ["country","country_code","Country","cc","iso2","ISO2","country name","Country name EN"]
_ADMIN1_CAND  = ["admin","admin1","admin1_code","adm1","state","region","Admin1","admin1 name","admin name EN"]
_POP_CAND  = ["population","pop","Population"]

def _pick_col(cols: List[str], cand: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in cand:
        if k.lower() in low:
            return low[k.lower()]
    return None

def _s(v) -> str:
    if v is None: return ""
    if isinstance(v, str): return v.strip()
    try: return str(v).strip()
    except Exception: return ""

_DEC_SEP = re.compile(r"[ \u00A0]")  # spaces incl. nbsp

def _to_float(value) -> Optional[float]:
    """
    Robust numeric parsing:
    - strips spaces/nbsp
    - handles decimal comma (1.234,56 or 1,234.56)
    - drops thousand-separators
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "n/a":
        return None
    s = _DEC_SEP.sub("", s)          # remove spaces
    s = s.replace("−", "-")          # minus sign variant
    # If both ',' and '.' present, assume last symbol is decimal; remove the other.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        # If only comma, treat as decimal comma
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        # else: already decimal point
    try:
        return float(s)
    except Exception:
        return None

def _split_coords(raw: str) -> Tuple[Optional[float], Optional[float]]:
    """Accepts 'lat, lon' or 'lon, lat' or space separated. Returns (lat, lon)."""
    if raw is None:
        return (None, None)
    s = _s(raw)
    if not s:
        return (None, None)
    # common separators
    s = s.replace(";", ",").replace("|", ",")
    pieces = [t for t in re.split(r"[,\s]+", s) if t]
    if len(pieces) < 2:
        return (None, None)
    a = _to_float(pieces[0])
    b = _to_float(pieces[1])
    if a is None or b is None:
        return (None, None)
    # Heuristic: swap if first looks like lon and second like lat
    if abs(a) > 90 and abs(b) <= 90:
        a, b = b, a
    return (a, b)

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    need = ["name","lat","lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}'")
    out = df.copy()
    out["lat"] = out["lat"].map(_to_float)
    out["lon"] = out["lon"].map(_to_float)

    # drop impossible
    out = out[(out["lat"].abs() <= 90) & (out["lon"].abs() <= 180)]
    out = out.dropna(subset=["lat","lon"])
    out["name"] = out["name"].astype(str).str.strip()
    out = out[out["name"] != ""]
    for k in ["country","admin","population"]:
        if k not in out.columns: out[k] = np.nan
    # normalize strings
    out["country"] = out["country"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    out["admin"]   = out["admin"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    return out[["name","lat","lon","country","admin","population"]].reset_index(drop=True)

# ------------------------ PUBLIC API ------------------------

def ingest_geonames_zip(upload, min_population: int = 0, title: str = "GeoNames") -> str:
    """Ingest citiesXXX.zip / allCountries.zip."""
    # normalize bytes
    if hasattr(upload, "getvalue"):
        data = upload.getvalue()
    elif hasattr(upload, "read"):
        data = upload.read()
    else:
        data = upload  # bytes

    import pandas as pd
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not inner:
            raise ValueError("ZIP does not contain a .txt GeoNames file.")
        with zf.open(inner[0]) as fh:
            df_raw = pd.read_csv(fh, sep="\t", header=None, dtype=str, na_filter=False, engine="python")

    def col(i):
        try: return df_raw.iloc[:, i]
        except Exception: return pd.Series([], dtype="string")

    name = col(1) if 1 in df_raw.columns else pd.Series("", dtype="string")
    if name.empty or name.str.len().max() == 0:
        name = col(2)  # fallback to asciiname

    df = pd.DataFrame({
        "name": name.map(_s),
        "lat":  col(4).map(_to_float),
        "lon":  col(5).map(_to_float),
        "country": col(8).map(_s),   # country code
        "admin":   col(10).map(_s),  # admin1 code
        "population": pd.to_numeric(col(14), errors="coerce")
    })

    if min_population and min_population > 0:
        df = df[df["population"].fillna(0) >= float(min_population)]

    df = _finalize(df)
    return registry.save_df(name=f"{title} (ingested)", df=df, kind="gazetteer", extra_meta={"rows": len(df)})

def ingest_custom_gazetteer_csv(upload, title: str = "Custom gazetteer") -> str:
    """
    Ingest CSV with at least (name, lat, lon). Also supports a single 'coordinates' column.
    Handles thousand-separators, decimal commas, and swapped lon/lat.
    """
    if hasattr(upload, "getvalue"):
        data = upload.getvalue()
    elif hasattr(upload, "read"):
        data = upload.read()
    else:
        data = upload

    bio = io.BytesIO(data)
    df_raw = pd.read_csv(
        bio,
        engine="python",
        sep=None,            # infer delimiter
        on_bad_lines="skip",
        dtype=str,
        na_filter=False
    )
    if df_raw.empty:
        raise ValueError("CSV appears empty/unreadable.")

    cols = df_raw.columns.tolist()
    name = _pick_col(cols, _NAME_CAND)
    lat  = _pick_col(cols, _LAT_CAND)
    lon  = _pick_col(cols, _LON_CAND)
    coords = _pick_col(cols, _COORDS_CAND)
    cty  = _pick_col(cols, _COUNTRY_CAND)
    adm  = _pick_col(cols, _ADMIN1_CAND)
    pop  = _pick_col(cols, _POP_CAND)

    if not name:
        raise ValueError("Could not find a 'name' column.")

    if not (lat and lon) and not coords:
        raise ValueError("Need either (lat & lon) columns or a single 'coordinates' column.")

    if coords and (lat is None or lon is None):
        # Split coordinates
        lat_vals, lon_vals = zip(*df_raw[coords].map(_split_coords))
        tmp = pd.DataFrame({"name": df_raw[name].map(_s), "lat": lat_vals, "lon": lon_vals})
    else:
        tmp = pd.DataFrame({"name": df_raw[name].map(_s),
                            "lat":  df_raw[lat].map(_to_float),
                            "lon":  df_raw[lon].map(_to_float)})

    if cty: tmp["country"] = df_raw[cty].map(_s)
    if adm: tmp["admin"] = df_raw[adm].map(_s)
    if pop: tmp["population"] = pd.to_numeric(df_raw[pop], errors="coerce")

    df = _finalize(tmp)
    return registry.save_df(name=f"{title} (ingested)", df=df, kind="gazetteer", extra_meta={"rows": len(df)})
