# backend/geo/gazetteer.py
from __future__ import annotations
import io
import os
import re
import zipfile
import unicodedata
from typing import Dict, List, Tuple, Optional

import pandas as pd
from rapidfuzz import process, fuzz

from backend.core import dataset_registry as registry

# --------- Public API ---------
# Ingest sources:
#   ingest_geonames_zip(uploaded_file, min_population=0)
#   ingest_geonames_tsv(uploaded_file, min_population=0)
#   ingest_custom_csv(uploaded_file)
# Manage:
#   list_gazetteers()
#   set_active_gazetteer(gid)
# Resolve (used by geo_utils):
#   load_active_gazetteer()
#   resolve_with_gazetteer(unique_locations: List[str]) -> Dict[str, Tuple[float,float]]

GAZETTEER_KIND = "gazetteer"
GAZETTEER_META_KEY = "gazetteer_active_id"

# ---- util: normalize names ----
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s-]", re.UNICODE)

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _norm_name(s: str) -> str:
    s = (s or "").strip()
    s = _strip_accents(s)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip().lower()
    return s

def _maybe_country_from_string(s: str) -> Optional[str]:
    # very light heuristic: if "City, Country" keep the country token as hint
    if "," in s:
        tail = s.split(",")[-1].strip()
        if 2 <= len(tail) <= 64:
            return _norm_name(tail)
    return None

# ---- registry helpers for a single key/value (active gazetteer id) ----
def _load_meta() -> Dict:
    # reuse dataset index file to stash a tiny settings dict
    meta_path = os.path.join(os.environ.get("APP_DATA_DIR", "data"), "datasets", "_gazetteer_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        return pd.read_json(meta_path, typ="series").to_dict()
    except Exception:
        return {}

def _save_meta(d: Dict) -> None:
    meta_path = os.path.join(os.environ.get("APP_DATA_DIR", "data"), "datasets", "_gazetteer_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    pd.Series(d).to_json(meta_path)

def set_active_gazetteer(gid: str) -> None:
    m = _load_meta()
    m[GAZETTEER_META_KEY] = gid
    _save_meta(m)

def get_active_gazetteer_id() -> Optional[str]:
    return _load_meta().get(GAZETTEER_META_KEY)

def list_gazetteers() -> List[Dict]:
    items = registry.find_datasets(kind=GAZETTEER_KIND)
    # newest first
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items

# ---- ingestors ----
_GZ_COLUMNS = [
    "geonameid","name","asciiname","alternatenames","latitude","longitude","feature class","feature code",
    "country code","cc2","admin1 code","admin2 code","admin3 code","admin4 code",
    "population","elevation","dem","timezone","modification date"
]
def _gz_frame(cols) -> pd.DataFrame:
    # ensure we always have a uniform schema
    base = {c: [] for c in _GZ_COLUMNS}
    for c in cols:
        if c not in base:
            base[c] = []
    return pd.DataFrame(columns=list(base.keys()))

def _postprocess_geonames(df: pd.DataFrame, min_population: int = 0) -> pd.DataFrame:
    keep_cols = ["name","asciiname","alternatenames","latitude","longitude","country code","admin1 code","population"]
    df = df[keep_cols].copy()
    df.rename(columns={
        "country code":"country",
        "admin1 code":"admin1",
        "latitude":"lat",
        "longitude":"lon",
        "alternatenames":"alts",
        "population":"population",
    }, inplace=True)
    # normalize & split alternates
    df["name_norm"] = df["name"].map(_norm_name)
    df["alts"] = df["alts"].fillna("").astype(str)
    df["alt_list"] = df["alts"].apply(lambda s: [a for a in ( _norm_name(a) for a in s.split(",") ) if a])
    # numeric
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    # drop rows without coords
    df = df.dropna(subset=["lat","lon"])
    if min_population > 0:
        df = df[df["population"] >= int(min_population)]
    # build bucket key = first letter of name_norm
    df["bucket"] = df["name_norm"].map(lambda s: s[:1] if s else "#")
    return df.reset_index(drop=True)

def ingest_geonames_tsv(file_like, name: Optional[str]=None, min_population: int = 0, owner: Optional[str]=None) -> str:
    # file_like: UploadedFile or file object; read as TSV without header
    df = pd.read_csv(file_like, sep="\t", header=None, names=_GZ_COLUMNS, dtype=str, low_memory=False, quoting=3)
    df = _postprocess_geonames(df, min_population=min_population)
    return registry.save_df(name=name or "GeoNames TSV", df=df, kind=GAZETTEER_KIND, owner=owner)

def ingest_geonames_zip(zip_file, inner_filename: Optional[str]=None, name: Optional[str]=None, min_population: int = 0, owner: Optional[str]=None) -> str:
    zf = zipfile.ZipFile(zip_file)
    # pick the largest .txt if inner not specified
    candidate = inner_filename
    if candidate is None:
        txts = [zi for zi in zf.infolist() if zi.filename.lower().endswith(".txt")]
        if not txts:
            raise ValueError("Zip does not contain any .txt files.")
        candidate = sorted(txts, key=lambda zi: zi.file_size, reverse=True)[0].filename
    with zf.open(candidate) as f:
        df = pd.read_csv(f, sep="\t", header=None, names=_GZ_COLUMNS, dtype=str, low_memory=False, quoting=3)
    df = _postprocess_geonames(df, min_population=min_population)
    return registry.save_df(name=name or f"GeoNames {os.path.basename(candidate)}", df=df, kind=GAZETTEER_KIND, owner=owner)

def ingest_custom_csv(file_like, name: Optional[str]=None, owner: Optional[str]=None) -> str:
    # Expect at least: name, lat, lon. Optional: country, admin1, population
    df = pd.read_csv(file_like)
    cols = {c.lower(): c for c in df.columns}
    req = {"name","lat","lon"}
    if not req.issubset(set(cols.keys())):
        raise ValueError("Custom CSV must include columns: name, lat, lon")
    df = df.rename(columns={
        cols["name"]:"name",
        cols["lat"]:"lat",
        cols["lon"]:"lon",
        **({cols.get("country"):"country"} if "country" in cols else {}),
        **({cols.get("admin1"):"admin1"} if "admin1" in cols else {}),
        **({cols.get("population"):"population"} if "population" in cols else {}),
    })
    # Fill optional columns
    for c in ["country","admin1","population"]:
        if c not in df.columns:
            df[c] = ""
    df["name_norm"] = df["name"].map(_norm_name)
    df["alt_list"] = [[] for _ in range(len(df))]
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    df["bucket"] = df["name_norm"].map(lambda s: s[:1] if s else "#")
    return registry.save_df(name=name or "Custom Gazetteer", df=df, kind=GAZETTEER_KIND, owner=owner)

# ---- active gazetteer loader + resolver ----
_cached_gid: Optional[str] = None
_cached_df: Optional[pd.DataFrame] = None
_cached_names: Optional[List[str]] = None
_cached_buckets: Optional[Dict[str, List[int]]] = None
_cached_alias_map: Optional[Dict[str, List[int]]] = None

def load_active_gazetteer() -> Optional[pd.DataFrame]:
    global _cached_gid, _cached_df, _cached_names, _cached_buckets, _cached_alias_map
    gid = get_active_gazetteer_id()
    if gid is None:
        # fallback: newest available
        items = list_gazetteers()
        if not items:
            return None
        gid = items[0]["id"]
        set_active_gazetteer(gid)

    if _cached_df is not None and _cached_gid == gid:
        return _cached_df

    df = registry.load_df(gid)
    # Build simple indexes
    _cached_gid = gid
    _cached_df = df
    _cached_names = df["name_norm"].astype(str).tolist()
    # bucket: first letter -> list of row indices (to prune fuzzy search)
    _cached_buckets = {}
    for idx, b in enumerate(df["bucket"].astype(str).tolist()):
        _cached_buckets.setdefault(b or "#", []).append(idx)
    # alias map: alt_norm -> list of row indices
    _cached_alias_map = {}
    for idx, alts in enumerate(df["alt_list"]):
        for a in (alts or []):
            _cached_alias_map.setdefault(a, []).append(idx)
    return df

def resolve_with_gazetteer(locations: List[str], score_cutoff: int = 88) -> Dict[str, Tuple[float, float]]:
    """
    Resolve arbitrary location strings to coords using the active gazetteer.
    Strategy: exact name_norm -> alias exact -> fuzzy within letter-bucket.
    """
    df = load_active_gazetteer()
    if df is None:
        return {}

    out: Dict[str, Tuple[float, float]] = {}

    names = _cached_names
    buckets = _cached_buckets
    alias_map = _cached_alias_map

    for raw in set(locations):
        if not isinstance(raw, str) or not raw.strip():
            continue
        norm = _norm_name(raw)
        if not norm:
            continue

        # 1) exact match on primary name
        matches = df.index[df["name_norm"] == norm].tolist()
        if matches:
            i = matches[0]
            out[raw] = (float(df.at[i,"lat"]), float(df.at[i,"lon"]))
            continue

        # 2) exact match on aliases
        if norm in alias_map:
            i = alias_map[norm][0]
            out[raw] = (float(df.at[i,"lat"]), float(df.at[i,"lon"]))
            continue

        # 3) fuzzy within same first-letter bucket (or global if empty)
        bucket_key = norm[:1] if norm else "#"
        candidate_idxs = buckets.get(bucket_key, [])
        if not candidate_idxs:
            candidate_idxs = list(range(len(df)))  # degenerate, but rare

        # Build candidate name list (primary + a few alternates) for those indices
        candidate_names = [names[i] for i in candidate_idxs]
        # Fuzzy match: pick best
        best = process.extractOne(norm, candidate_names, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if best:
            _, score, pos = best  # candidate_names[pos]
            i = candidate_idxs[pos]
            out[raw] = (float(df.at[i,"lat"]), float(df.at[i,"lon"]))
            continue

        # No match -> skip
    return out
