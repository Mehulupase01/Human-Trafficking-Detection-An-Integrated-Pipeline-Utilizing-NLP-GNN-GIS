# backend/api/query.py
from __future__ import annotations
from typing import Iterable, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ---------- Utilities used by pages ----------

def explode_list_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Explode a list-like column (lists, tuples, sets, ndarrays).
    Scalars become one-element lists; NaN/None become [].
    """
    if col not in df.columns:
        out = df.copy()
        out[col] = [[] for _ in range(len(out))]
        return out.explode(col, ignore_index=True)

    out = df.copy()

    def _norm(v):
        # Missing
        if v is None:
            return []
        if isinstance(v, float) and pd.isna(v):
            return []
        # Already list-like
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if not _is_na_like(x)]
        if isinstance(v, np.ndarray):
            return [str(x) for x in v.tolist() if not _is_na_like(x)]
        # Scalar
        return [str(v)]

    out[col] = out[col].apply(_norm)
    return out.explode(col, ignore_index=True)


def _is_na_like(x) -> bool:
    """Return True for scalar NaN/None; never raises for arrays."""
    if x is None:
        return True
    # numpy/pandas scalar NaN
    if isinstance(x, float):
        return pd.isna(x)
    try:
        # Only call pd.isna on scalars; arrays produce array-of-bool
        if np.isscalar(x):
            return pd.isna(x)
    except Exception:
        pass
    return False


def list_contains_any(series: pd.Series, tokens: Iterable[str]) -> pd.Series:
    """
    True if a row's list-like cell contains ANY of the tokens (case-insensitive).
    Accepts lists/tuples/sets/ndarrays/scalars.
    """
    toks = [t.strip() for t in tokens if str(t).strip()]
    if not toks:
        return pd.Series([True] * len(series), index=series.index)

    toks_lower = {t.lower() for t in toks}

    def _hit(v):
        if isinstance(v, (list, tuple, set)):
            vals = {str(x).lower() for x in v if not _is_na_like(x)}
            return not toks_lower.isdisjoint(vals)
        if isinstance(v, np.ndarray):
            vals = {str(x).lower() for x in v.tolist() if not _is_na_like(x)}
            return not toks_lower.isdisjoint(vals)
        if _is_na_like(v):
            return False
        return str(v).lower() in toks_lower

    return series.apply(_hit)


def join_for_display(v) -> str:
    """Render list columns nicely in tables; arrays treated as lists."""
    if isinstance(v, (list, tuple, set)):
        return " · ".join(str(x) for x in v if not _is_na_like(x))
    if isinstance(v, np.ndarray):
        return " · ".join(str(x) for x in v.tolist() if not _is_na_like(x))
    return "" if _is_na_like(v) else str(v)


def join_for_csv(v) -> str:
    """Render list columns in CSV downloads; arrays treated as lists."""
    if isinstance(v, (list, tuple, set)):
        return ";".join(str(x) for x in v if not _is_na_like(x))
    if isinstance(v, np.ndarray):
        return ";".join(str(x) for x in v.tolist() if not _is_na_like(x))
    return "" if _is_na_like(v) else str(v)


def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Collect clean uniques for dropdowns."""
    out: Dict[str, List[str]] = {}
    if "Gender of Victim" in df.columns:
        out["genders"] = sorted(x for x in df["Gender of Victim"].dropna().astype(str).unique().tolist() if x)
    else:
        out["genders"] = []

    if "Nationality of Victim" in df.columns:
        out["nationalities"] = sorted(x for x in df["Nationality of Victim"].dropna().astype(str).unique().tolist() if x)
    else:
        out["nationalities"] = []

    if "Locations (NLP)" in df.columns:
        loc_long = explode_list_column(df, "Locations (NLP)")
        out["locations"] = sorted(x for x in loc_long["Locations (NLP)"].dropna().astype(str).unique().tolist() if x)
    else:
        out["locations"] = []

    if "Serialized ID" in df.columns:
        out["victims"] = sorted(df["Serialized ID"].dropna().astype(str).unique().tolist())
    else:
        out["victims"] = []

    return out


# ---------- Filters (backwards compatible) ----------

def apply_filters(
    df: pd.DataFrame,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Backwards-compatible filter entry point.

    Legacy signature:
        apply_filters(df, nationality, gender, year_range, location_substr)

    New signature (kwargs):
        apply_filters(
            df,
            gender=...,
            nationality=...,
            locations=[...],              # matches ANY in `Locations (NLP)`
            perpetrators_any=[...],
            chiefs_any=[...],
            victim_sid='HTV12',
            year_range=(2010, 2020),
            location_substr="tripoli",    # legacy raw-text match
        )
    """
    if args and not kwargs:
        try:
            nationality = args[0] if len(args) >= 1 else None
            gender      = args[1] if len(args) >= 2 else None
            year_range  = args[2] if len(args) >= 3 else None
            loc_substr  = args[3] if len(args) >= 4 else None

            return _apply_filters_new(
                df=df,
                gender=gender or None,
                nationality=nationality or None,
                locations=None,
                perpetrators_any=None,
                chiefs_any=None,
                victim_sid=None,
                year_range=year_range if isinstance(year_range, (list, tuple)) else None,
                location_substr=str(loc_substr) if loc_substr else None,
            )
        except Exception:
            pass

    return _apply_filters_new(
        df=df,
        gender=kwargs.get("gender"),
        nationality=kwargs.get("nationality"),
        locations=kwargs.get("locations"),
        perpetrators_any=kwargs.get("perpetrators_any"),
        chiefs_any=kwargs.get("chiefs_any"),
        victim_sid=kwargs.get("victim_sid"),
        year_range=kwargs.get("year_range"),
        location_substr=kwargs.get("location_substr"),
    )


def _apply_filters_new(
    df: pd.DataFrame,
    *,
    gender: Optional[str] = None,
    nationality: Optional[str] = None,
    locations: Optional[Iterable[str]] = None,
    perpetrators_any: Optional[Iterable[str]] = None,
    chiefs_any: Optional[Iterable[str]] = None,
    victim_sid: Optional[str] = None,
    year_range: Optional[Tuple[int, int]] = None,
    location_substr: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    mask = pd.Series([True] * len(df), index=df.index)

    if gender and "Gender of Victim" in df.columns and gender != "Any":
        mask &= df["Gender of Victim"].astype(str).eq(gender)

    if nationality and "Nationality of Victim" in df.columns and nationality != "(Any)":
        mask &= df["Nationality of Victim"].astype(str).eq(nationality)

    if locations and "Locations (NLP)" in df.columns:
        mask &= list_contains_any(df["Locations (NLP)"], locations)

    if perpetrators_any and "Perpetrators (NLP)" in df.columns:
        mask &= list_contains_any(df["Perpetrators (NLP)"], perpetrators_any)

    if chiefs_any and "Chiefs (NLP)" in df.columns:
        mask &= list_contains_any(df["Chiefs (NLP)"], chiefs_any)

    if victim_sid and "Serialized ID" in df.columns and victim_sid != "(Any)":
        mask &= df["Serialized ID"].astype(str).eq(victim_sid)

    if year_range and "Left Home Country Year" in df.columns:
        try:
            lo, hi = int(year_range[0]), int(year_range[1])
            yrs = pd.to_numeric(df["Left Home Country Year"], errors="coerce")
            mask &= (yrs >= lo) & (yrs <= hi)
        except Exception:
            pass

    if location_substr:
        sub = str(location_substr).strip().lower()
        if sub:
            cols = [c for c in df.columns if c.lower() in {"location", "city / locations crossed", "final location"}]
            if cols:
                m2 = pd.Series([False] * len(df), index=df.index)
                for c in cols:
                    m2 |= df[c].astype(str).str.lower().str.contains(sub, na=False)
                mask &= m2

    return df[mask].copy()
