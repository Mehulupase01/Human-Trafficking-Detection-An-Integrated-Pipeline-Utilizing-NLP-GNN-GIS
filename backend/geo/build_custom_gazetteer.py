from __future__ import annotations
import re
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Iterable, Tuple, List, Dict, Optional

import pandas as pd

from backend.core import dataset_registry as registry

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _split_coords(s: str) -> Tuple[Optional[float], Optional[float]]:
    if s is None or pd.isna(s):
        return None, None
    parts = re.split(r"[,; ]+", str(s))
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except Exception:
            pass
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

def _best_match(place: str, cand_idxs: List[int], name_norm: pd.Series,
                gn2: pd.DataFrame) -> Tuple[Optional[int], float]:
    """
    Fuzzy match a single place token against candidate GeoNames rows.
    Uses difflib ratio; slight boost for higher-population places.
    """
    nplace = _norm(place)
    if not nplace:
        return None, 0.0
    # a tiny speed-up: first-letter bucket
    narrowed = [i for i in cand_idxs if name_norm[i][:1] == nplace[:1]] or cand_idxs
    best = None
    best_score = 0.0
    for idx in narrowed:
        score = SequenceMatcher(None, nplace, name_norm[idx]).ratio()
        pop = gn2.at[idx, "population"]
        if pd.notna(pop):
            score += min(0.05, float(pop) / 5e7)  # small bump for big places
        if score > best_score:
            best_score = score
            best = idx
    return best, best_score

def _read_geonames_csv(path: str) -> pd.DataFrame:
    """
    Accepts the 'geonames-all-cities-with-a-population-1000.csv' (or similar).
    Tries to infer the relevant columns and returns a normalized df with:
    ['name','country','lat','lon','population']
    """
    gn = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", engine="python")
    cols = {c.lower(): c for c in gn.columns}

    def find(*cands):
        for c in cands:
            cl = c.lower()
            if cl in cols:
                return cols[cl]
        return None

    name_col = find("name", "asciiname", "label en", "label_en") or gn.columns[0]
    country_col = find("country name en", "country", "country_name_en", "countrynameen")
    coord_col = find("coordinates", "coord", "latlon", "lat_lon")
    pop_col = find("population")

    out = pd.DataFrame()
    out["name"] = gn[name_col].astype(str).str.strip()
    out["country"] = gn[country_col].astype(str).str.strip() if country_col else ""

    if coord_col:
        latlon = gn[coord_col].apply(_split_coords)
        out["lat"] = [a for a, b in latlon]
        out["lon"] = [b for a, b in latlon]
    else:
        lat_c = find("lat", "latitude")
        lon_c = find("lon", "longitude", "lng")
        out["lat"] = pd.to_numeric(gn[lat_c], errors="coerce") if lat_c else pd.NA
        out["lon"] = pd.to_numeric(gn[lon_c], errors="coerce") if lon_c else pd.NA

    out["population"] = pd.to_numeric(gn[pop_col], errors="coerce") if pop_col else pd.NA
    return out.reset_index(drop=True)

def build_gazetteer_from_token_file(
    token_file: str,
    geonames_csv: str,
    save_to_registry: bool = True,
    registry_name: str = "Custom Gazetteer (from list)"
) -> Tuple[pd.DataFrame, Optional[str], Dict[str, int]]:
    """
    token_file: text file with lines like  ['Khartoum' 'Sudan' '...']
    geonames_csv: path to offline GeoNames CSV (your 27MB file)

    Returns: (df, dataset_id_if_saved, summary)
    """
    # --- 1) Parse token lines
    lines = open(token_file, "r", encoding="utf-8", errors="ignore").read().splitlines()

    def parse_line(s: str) -> List[str]:
        inside = re.findall(r"\[(.*?)\]", s)
        if not inside:
            return []
        chunk = inside[0]
        toks = re.findall(r"'([^']+)'|\"([^\"]+)\"", chunk)
        vals = [a or b for (a, b) in toks]
        if not vals:  # fallback: split spaces
            vals = [p.strip() for p in chunk.split() if p.strip() not in ("[", "]")]
        vals = [re.sub(r"\s+", " ", v.strip()) for v in vals if v and v.strip()]
        return vals

    token_rows = [parse_line(s) for s in lines]
    token_rows = [r for r in token_rows if r]

    # --- 2) Load & prep GeoNames
    gn2 = _read_geonames_csv(geonames_csv)
    countries = set(gn2["country"].dropna().unique().tolist())
    name_norm = gn2["name"].astype(str).map(_norm)

    # index by country for quick filtering
    country_index = defaultdict(list)
    for idx, c in gn2["country"].items():
        country_index[c].append(idx)

    resolved = []
    unresolved = 0

    for tokens in token_rows:
        # Separate country tokens (using the country names that exist in your file)
        tok_countries = [t for t in tokens if t in countries]
        places = [t for t in tokens if t not in countries]
        if not places:
            unresolved += 1
            continue

        # Candidate rows: within selected country (if present) else whole table
        if tok_countries:
            cand = list({i for c in tok_countries for i in country_index.get(c, [])})
        else:
            cand = list(range(len(gn2)))

        # Choose the strongest match among the place tokens
        best_idx, best_score, best_place = None, 0.0, None
        for p in places:
            idx, score = _best_match(p, cand, name_norm, gn2)
            if idx is not None and score > best_score:
                best_idx, best_score, best_place = idx, score, p

        if best_idx is None or pd.isna(gn2.at[best_idx, "lat"]) or pd.isna(gn2.at[best_idx, "lon"]):
            unresolved += 1
            continue

        resolved.append({
            "name": gn2.at[best_idx, "name"],
            "lat": float(gn2.at[best_idx, "lat"]),
            "lon": float(gn2.at[best_idx, "lon"]),
            "country": gn2.at[best_idx, "country"],
            "admin": "",  # admin1 not available in this CSV; leave blank
            "population": gn2.at[best_idx, "population"] if pd.notna(gn2.at[best_idx, "population"]) else "",
            "source_tokens": " | ".join(tokens),
            "matched_from": best_place,
            "score": round(best_score, 3),
        })

    df = pd.DataFrame(resolved).dropna(subset=["lat", "lon"])
    df = df.drop_duplicates(subset=["name", "country"]).reset_index(drop=True)

    dataset_id = None
    if save_to_registry and not df.empty:
        dataset_id = registry.save_df(
            name=registry_name,
            df=df[["name", "lat", "lon", "country", "admin", "population"]],
            kind="gazetteer",
            extra_meta={"rows": len(df)},
        )

    summary = {
        "input_lines": len(token_rows),
        "resolved": len(df),
        "unresolved": unresolved,
    }
    return df, dataset_id, summary
