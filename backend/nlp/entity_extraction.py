# nlp/entity_extraction.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Iterable
import re, hashlib
import numpy as np
import pandas as pd
from backend.core import dataset_registry as registry

# --- Canonical columns ---
COL_SID      = "Serialized ID"
COL_UID      = "Unique ID"
COL_LOC      = "Location"
COL_ROUTE    = "Route_Order"
COL_PERPS    = "Perpetrators (NLP)"
COL_CHIEFS   = "Chiefs (NLP)"
COL_LOCS_NLP = "Locations (NLP)"
COL_GENDER   = "Gender of Victim"
COL_NATION   = "Nationality of Victim"
COL_TIME_SRC = "Time Spent in Location / Cities / Places"
COL_TIME_D   = "Time Spent (days)"
COL_TIME_RAW = "Time Spent (raw)"
COL_INT_DATE = "Date of Interview"
COL_LEAVE_YR = "Left Home Country Year"
PROCESS_KIND = "processed"

# ---------------- Gazetteer (optional) ----------------
def _load_gazetteer_words() -> Optional[set]:
    try:
        from backend.geo.gazetteer import current_gazetteer_df
        g = current_gazetteer_df()
        if g is None or g.empty: return None
        names = g["name"].dropna().astype(str).tolist() if "name" in g.columns else []
        toks = set()
        for n in names:
            for w in re.findall(r"[A-Za-z]+", n):
                toks.add(w.lower())
        return toks or None
    except Exception:
        return None

_GAZ_TOKS = _load_gazetteer_words()
def _looks_like_place_token(tok: str) -> bool:
    if not tok or len(tok) < 2: return False
    if _GAZ_TOKS is not None: return tok.lower() in _GAZ_TOKS
    return bool(re.fullmatch(r"[A-Z][a-z]+", tok))

# ---------------- Tokenizers / Normalizers ----------------
_WORDS = re.compile(r"[A-Za-z]+")
_CAPWORD = re.compile(r"[A-Z][a-z]+")
STOP_PERSON = {s.lower() for s in {
    "Unknown","None","No","Nil","N/A","Boss","Chief","Police","Officer","Man","Woman",
    "Him","Her","They","He","She","I","Me","You","We","And","But","The","A","An",
    "Someone","Anybody","Everybody","Person","People","Kidnapper","Guard","Smuggler"
}}
NATIONALITY_MAP = {
    "eritrea":"Eritrean","eritrean":"Eritrean","ethiopia":"Ethiopian","ethiopian":"Ethiopian",
    "sudan":"Sudanese","sudanese":"Sudanese","somalia":"Somali","somali":"Somali",
    "libya":"Libyan","libyan":"Libyan","tunisia":"Tunisian","tunisian":"Tunisian",
    "morocco":"Moroccan","moroccan":"Moroccan","egypt":"Egyptian","egyptian":"Egyptian",
    "italy":"Italian","italian":"Italian","france":"French","french":"French",
    "germany":"German","german":"German","spain":"Spanish","spanish":"Spanish",
    "nigeria":"Nigerian","nigerian":"Nigerian",
}

def _split_units(s: str) -> List[str]: return re.split(r"[,\;/\|]| to |->|–|-{1,2}", s)
def _one_word_tokens_from_phrase(p: str) -> List[str]: return _WORDS.findall(p)
def _unique_preserve(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

def _extract_location_tokens(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    bag: List[str] = []
    for p in _split_units(text):
        for w in _one_word_tokens_from_phrase(p):
            if _looks_like_place_token(w): bag.append(w.capitalize())
    return _unique_preserve(bag)

def _extract_person_tokens(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    toks: List[str] = []
    for w in _one_word_tokens_from_phrase(text):
        if _CAPWORD.fullmatch(w) and w.lower() not in STOP_PERSON:
            toks.append(w)
    return _unique_preserve(toks)

def _normalize_gender(x) -> str:
    """
    Normalize free-text gender field into one of:
    - Male
    - Female
    - Unknown
    """
    if x is None:
        return "Unknown"
    s = str(x).strip().lower()

    if not s or s in {"none", "nan", "n/a", "unk", "unknown"}:
        return "Unknown"

    # check female FIRST to avoid "male" substring issue
    female_patterns = {"f", "female", "woman", "girl", "lady", "women"}
    male_patterns   = {"m", "male", "man", "boy", "gentleman", "men"}

    if s in female_patterns:
        return "Female"
    if s in male_patterns:
        return "Male"

    # regex fallback with word boundaries
    import re
    if re.search(r"\bf(emale)?\b", s):
        return "Female"
    if re.search(r"\bm(ale)?\b", s):
        return "Male"

    return "Unknown"


def _normalize_nationality(x) -> str:
    s = str(x or "").strip()
    if not s: return ""
    cands = _one_word_tokens_from_phrase(s)
    for w in cands:
        k = w.lower()
        if k in NATIONALITY_MAP: return NATIONALITY_MAP[k]
    for w in cands:
        if _CAPWORD.fullmatch(w): return NATIONALITY_MAP.get(w.lower(), w)
    return NATIONALITY_MAP.get(s.lower(), s.capitalize())

_FRAC = re.compile(r"(\d+)\s*/\s*(\d+)")
_NUM  = re.compile(r"\b(\d+(?:\.\d+)?)\b")
def _parse_duration_days(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s.strip(): return None
    t = s.lower()
    frac = _FRAC.search(t); frac_val = None
    if frac:
        a, b = frac.groups()
        try: frac_val = float(a)/float(b)
        except Exception: pass
    nums = [float(x) for x in _NUM.findall(t)]
    days = 0.0
    if "month" in t:
        n = nums[0] if nums else 1.0
        days += n * 30.0
        if "week" in t: days += (nums[1] if len(nums)>1 else 0.0) * 7.0
        if "day"  in t: days += (nums[2] if len(nums)>2 else 0.0)
        return int(max(1, round(days)))
    if "week" in t:
        n = nums[0] if nums else 1.0
        days = n * 7.0 + (nums[1] if len(nums)>1 else 0.0)
        return int(max(1, round(days)))
    if "day" in t:
        days = frac_val if frac_val is not None else (nums[0] if nums else 1.0)
        return int(max(1, round(days)))
    return None

def _compact_time_raw(s: str) -> str:
    if not isinstance(s, str) or not s.strip(): return ""
    toks = re.findall(r"[A-Za-z]+|\d+", s.lower())
    toks = [t for t in toks if t != "and"]
    return "_".join(toks)

# ---------------- Base column ensure/fill ----------------
def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in [COL_SID, COL_UID, COL_LOC, COL_ROUTE, COL_PERPS, COL_CHIEFS,
              COL_LOCS_NLP, COL_GENDER, COL_NATION, COL_TIME_D, COL_TIME_RAW]:
        if c not in df.columns:
            if c in (COL_PERPS, COL_CHIEFS, COL_LOCS_NLP): df[c] = [[] for _ in range(len(df))]
            else: df[c] = pd.NA
    return df

def _fill_ids_and_route(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_SID] = df.get(COL_SID, pd.Series(index=df.index)).astype(str)
    df[COL_UID] = df.get(COL_UID, pd.Series(index=df.index)).astype(str)
    sid_empty = df[COL_SID].isin(["<NA>","None","nan",""])
    uid_empty = df[COL_UID].isin(["<NA>","None","nan",""])
    df.loc[sid_empty & ~uid_empty, COL_SID] = df.loc[sid_empty & ~uid_empty, COL_UID]
    df.loc[uid_empty & ~sid_empty, COL_UID] = df.loc[uid_empty & ~sid_empty, COL_SID]

    if COL_ROUTE not in df.columns: df[COL_ROUTE] = pd.NA
    df[COL_ROUTE] = pd.to_numeric(df[COL_ROUTE], errors="coerce")
    if df[COL_ROUTE].isna().any():
        df[COL_ROUTE] = df.groupby(COL_SID, dropna=False).cumcount() + 1
    df[COL_ROUTE] = df[COL_ROUTE].fillna(0).astype(int)
    return df

def _apply_targeted_extraction(df: pd.DataFrame) -> pd.DataFrame:
    # Locations (NLP)
    location_source_cols = [c for c in df.columns if c.lower().startswith("city / locations crossed")
                            or c.lower().startswith("final location")
                            or "borders crossed" in c.lower()
                            or c.lower() == "location"]
    locs_all: List[List[str]] = []
    for _, row in df.iterrows():
        bag: List[str] = []
        for c in location_source_cols:
            bag.extend(_extract_location_tokens(str(row.get(c, ""))))
        locs_all.append(_unique_preserve(bag))
    if locs_all: df[COL_LOCS_NLP] = locs_all

    # Perpetrators
    src_perp = next((c for c in df.columns if c.lower().startswith("name of the perpetrators")), None)
    if src_perp is None: src_perp = next((c for c in df.columns if "perpetrator" in c.lower()), None)
    df[COL_PERPS] = [ _extract_person_tokens(str(row.get(src_perp, ""))) if src_perp else [] for _,row in df.iterrows() ]

    # Chiefs
    src_chief = next((c for c in df.columns if "chief" in c.lower()), None)
    df[COL_CHIEFS] = [ _extract_person_tokens(str(row.get(src_chief, ""))) if src_chief else [] for _,row in df.iterrows() ]

    # Gender / Nationality
    df[COL_GENDER] = (df[COL_GENDER] if COL_GENDER in df.columns else "Unknown").apply(_normalize_gender) \
                     if COL_GENDER in df.columns else pd.Series(["Unknown"]*len(df))
    df[COL_NATION] = df.get(COL_NATION, pd.Series(index=df.index)).apply(_normalize_nationality)

    # Time Spent
    src_time = COL_TIME_SRC if COL_TIME_SRC in df.columns else None
    time_days, time_raw = [], []
    for _, row in df.iterrows():
        s = str(row.get(src_time, "")) if src_time else ""
        d = _parse_duration_days(s)
        time_days.append(d if d is not None else pd.NA)
        time_raw.append(_compact_time_raw(s) if s else "")
    df[COL_TIME_D] = pd.Series(time_days, index=df.index, dtype="Int64")
    df[COL_TIME_RAW] = time_raw

    # Date & Year
    if COL_INT_DATE in df.columns:
        dtv = pd.to_datetime(df[COL_INT_DATE], errors="coerce", utc=False)
        df[COL_INT_DATE] = dtv.dt.date.astype(str).replace("NaT","")
    if COL_LEAVE_YR in df.columns:
        def _yr(x):
            s = str(x or ""); m = re.search(r"\b(19|20)\d{2}\b", s)
            return int(m.group(0)) if m else pd.NA
        df[COL_LEAVE_YR] = df[COL_LEAVE_YR].apply(_yr).astype("Int64")
    return df

# ---------------- NEW: Assign HTV1, HTV2… (robust & stable) ----------------
def _norm_id_str(x: object) -> str:
    """Normalize ID-like strings; treat placeholders as empty."""
    s = "" if x is None else str(x).strip()
    if s in {"", "<NA>", "nan", "None"}:
        return ""
    return s

def _row_signature(row: pd.Series) -> str:
    """
    Fallback signature to tie multiple rows of the same victim together
    when both Unique ID and Serialized ID are missing.
    Uses a few relatively stable fields.
    """
    parts = [
        _norm_id_str(row.get(COL_INT_DATE, "")),
        _norm_id_str(row.get(COL_NATION, "")),
        _norm_id_str(row.get(COL_GENDER, "")),
    ]
    # add first location token if present
    locs = row.get(COL_LOCS_NLP, [])
    if isinstance(locs, (list, tuple, set, np.ndarray)) and len(locs) > 0:
        first_loc = list(locs)[0]
        parts.append(str(first_loc))
    else:
        parts.append(_norm_id_str(row.get(COL_LOC, "")))
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return h

def _assign_htv_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a stable mapping of victim -> HTV{n} across the dataset.
    Priority key: Unique ID (clean) -> original Serialized ID (clean) -> fallback signature.
    Order: first appearance in the dataset.
    """
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))

    # Build victim key per row
    uid = df.get(COL_UID, pd.Series([""] * len(df))).apply(_norm_id_str)
    sid = df.get(COL_SID, pd.Series([""] * len(df))).apply(_norm_id_str)

    # compute fallback signatures only where both missing
    need_fallback = (uid == "") & (sid == "")
    sig = pd.Series([""] * len(df), index=df.index, dtype="object")
    if need_fallback.any():
        sig.loc[need_fallback] = df[need_fallback].apply(_row_signature, axis=1)

    victim_key = uid.where(uid != "", other=sid.where(sid != "", other=sig))
    df["_VictimKey"] = victim_key

    # Map to HTV{n} in order of first appearance
    first_idx = df.groupby("_VictimKey")["_orig_idx"].min().sort_values(kind="stable")
    key_order = list(first_idx.index)
    mapping = {k: f"HTV{i+1}" for i, k in enumerate(key_order)}
    df[COL_SID] = df["_VictimKey"].map(mapping)

    # Re-number Route_Order within HTV groups:
    # sort by existing Route_Order (if valid) then by original index
    if COL_ROUTE in df.columns:
        # ensure numeric
        df[COL_ROUTE] = pd.to_numeric(df[COL_ROUTE], errors="coerce")
        df["_sort_val"] = df[COL_ROUTE].fillna(df["_orig_idx"])
    else:
        df["_sort_val"] = df["_orig_idx"]

    df = df.sort_values(["Serialized ID", "_sort_val", "_orig_idx"], kind="stable")
    df[COL_ROUTE] = df.groupby(COL_SID).cumcount() + 1
    df = df.sort_values("_orig_idx", kind="stable")

    # cleanup
    df.drop(columns=["_orig_idx", "_VictimKey", "_sort_val"], inplace=True, errors="ignore")
    return df

# ---------------- Save helpers (unchanged) ----------------
def _hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(("|".join(df.columns)).encode("utf-8"))
    h.update(str(len(df)).encode("utf-8"))
    h.update(df.head(200).to_csv(index=False).encode("utf-8"))
    return h.hexdigest()

def _coerce_types_for_save(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    list_cols = {COL_PERPS, COL_CHIEFS, COL_LOCS_NLP}
    for c in out.columns:
        if c in list_cols: continue
        if pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            continue
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d"); continue
        out[c] = out[c].astype(str).replace({"<NA>":"","nan":"","None":""})
    return out

def save_processed(df: pd.DataFrame, name: str, owner: Optional[str] = None, source: Optional[str] = None) -> str:
    safe = _coerce_types_for_save(df)
    content_hash = _hash_df(safe)
    existing = registry.find_datasets(kind=PROCESS_KIND, content_hash=content_hash)
    if existing: return existing[0]["id"]
    did = registry.save_df(name=name, df=safe, kind=PROCESS_KIND, owner=owner, source=source,
                           extra_meta={"content_hash": content_hash})
    return did

# ---------------- Public API ----------------
def standardize_to_processed(
    raw_df: pd.DataFrame,
    *,
    extract_from_text: bool = True,
    overwrite_entities: bool = True,
) -> pd.DataFrame:
    df = raw_df.copy()
    df = _ensure_base_columns(df)
    df = _fill_ids_and_route(df)
    if extract_from_text:
        df = _apply_targeted_extraction(df)
    # Assign HTV ids (new)
    df = _assign_htv_ids(df)

    front = [COL_SID, COL_UID, COL_LOC, COL_ROUTE,
             COL_PERPS, COL_CHIEFS, COL_LOCS_NLP,
             COL_GENDER, COL_NATION, COL_TIME_D, COL_TIME_RAW,
             COL_INT_DATE, COL_LEAVE_YR]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]
