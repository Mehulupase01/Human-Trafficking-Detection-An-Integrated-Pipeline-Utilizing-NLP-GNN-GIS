# backend/api/temporal.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import re
import math
from collections import defaultdict

import numpy as np
import pandas as pd

# -------- helpers to robustly handle NA/strings --------

_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
    re.IGNORECASE,
)

UNIT_TO_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

def _is_scalar_na(x) -> bool:
    """True only if x is a scalar NA/None. Never raises for arrays/lists."""
    if x is None:
        return True
    try:
        # Only check NA-ness for scalars. pd.isna(list) returns array -> avoid.
        if np.isscalar(x):
            return pd.isna(x)
    except Exception:
        pass
    return False

def _first_token(val) -> Optional[str]:
    """First token from list/array; else clean string; else None."""
    if isinstance(val, list) and val:
        v = val[0]
        s = "" if _is_scalar_na(v) else str(v).strip()
        return s or None
    try:
        if hasattr(val, "size") and getattr(val, "size", 0) > 0:
            v = val.tolist()[0]
            s = "" if _is_scalar_na(v) else str(v).strip()
            return s or None
    except Exception:
        pass
    if _is_scalar_na(val):
        return None
    s = str(val).strip()
    return s or None

def _parse_days_from_text(s: str) -> Optional[float]:
    """Parse '3 weeks', '10 days', '1.5 months' → days."""
    if not isinstance(s, str) or not s.strip():
        return None
    m = _DURATION_RE.search(s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    days = val * UNIT_TO_DAYS.get(unit, 1)
    return float(days) if days > 0 else None

def _to_pos_float_days(x) -> Optional[float]:
    """
    Coerce any input to a positive finite float days value; return None if impossible.
    Handles pd.NA/None/strings/np types safely.
    """
    if _is_scalar_na(x):
        return None
    # already numeric?
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        if math.isfinite(v) and v > 0:
            return v
        return None
    # string like '3 weeks'
    if isinstance(x, str):
        v = _parse_days_from_text(x)
        return v if (v is not None and v > 0) else None
    # last-chance cast
    try:
        v = float(x)
        if math.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return None


class TemporalETA:
    """
    Learn medians for hop durations using:
      - 'Time Spent (days)' if present/valid
      - else parse text from 'Time Spent in Location / Cities / Places'
    Fallback chain when predicting:
      median(A→B) → median(B) → global median → UI fallback.
    """

    def __init__(self, fallback_days: float = 7.0):
        self.fallback_days = float(fallback_days)
        self.trans_medians: Dict[Tuple[str, str], float] = {}
        self.loc_medians: Dict[str, float] = {}
        self.global_median: float = self.fallback_days
        self.fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        victim_col: str = "Serialized ID",
        loc_col: str = "Locations (NLP)",
        time_days_col: str = "Time Spent (days)",
        time_text_col: str = "Time Spent in Location / Cities / Places",
    ) -> None:
        if df is None or df.empty or victim_col not in df.columns:
            self.fitted = True
            self.trans_medians.clear()
            self.loc_medians.clear()
            self.global_median = self.fallback_days
            return

        d = df.copy()

        # derive primary step location (first token from Locations(NLP), else Location)
        d["_step_loc"] = d[loc_col].apply(_first_token) if loc_col in d.columns else None
        if "Location" in d.columns:
            mask = pd.isna(d["_step_loc"])
            if mask.any():
                d.loc[mask, "_step_loc"] = d.loc[mask, "Location"].apply(_first_token)

        # build numeric days column with robust coercion
        d["_days"] = np.nan
        if time_days_col in d.columns:
            d["_days"] = d[time_days_col].apply(_to_pos_float_days)
        if time_text_col in d.columns:
            # only fill where numeric is missing
            missing = d["_days"].isna()
            if missing.any():
                d.loc[missing, "_days"] = d.loc[missing, time_text_col].apply(_to_pos_float_days)

        # ensure route order for ordering
        if "Route_Order" in d.columns:
            d["Route_Order"] = pd.to_numeric(d["Route_Order"], errors="coerce")
            d = d.sort_values([victim_col, "Route_Order"], kind="stable")

        trans: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        by_loc: Dict[str, List[float]] = defaultdict(list)
        all_durations: List[float] = []

        for vid, g in d.groupby(victim_col):
            locs = g["_step_loc"].tolist()
            days = g["_days"].tolist()

            # collapse consecutive duplicate locations; keep arrival-day alignment
            cleaned_locs: List[str] = []
            cleaned_days: List[Optional[float]] = []
            last = None
            for loc, dur in zip(locs, days):
                loc = _first_token(loc)
                if not isinstance(loc, str) or not loc:
                    continue
                if loc == last:
                    # Still skip consecutive dup for transitions; arrival day remains attached to the later step.
                    last = loc
                    continue
                # store day value as float or NaN — never pd.NA
                v = _to_pos_float_days(dur)
                cleaned_locs.append(loc)
                cleaned_days.append(v if v is not None else np.nan)
                last = loc

            # accumulate medians per transition and per arrival location
            for i in range(1, len(cleaned_locs)):
                prev, curr = cleaned_locs[i - 1], cleaned_locs[i]
                dur = cleaned_days[i]  # duration associated with arrival at curr
                if isinstance(dur, (int, float)) and np.isfinite(dur) and dur > 0:
                    trans[(prev, curr)].append(float(dur))
                    by_loc[curr].append(float(dur))
                    all_durations.append(float(dur))

        # finalize medians
        self.trans_medians = {k: float(np.median(v)) for k, v in trans.items() if v}
        self.loc_medians = {k: float(np.median(v)) for k, v in by_loc.items() if v}
        self.global_median = float(np.median(all_durations)) if all_durations else self.fallback_days
        self.fitted = True

    def predict_path(self, path: List[str], start_date: Optional[str] = None) -> List[Dict[str, object]]:
        """
        Given a list of locations [L0, L1, L2, ...], return ETA rows per hop:
        L0→L1, L1→L2, ...
        """
        if not self.fitted or not path:
            return []

        results: List[Dict[str, object]] = []
        cur_date = pd.to_datetime(start_date) if start_date else None
        cum = 0.0

        for i in range(1, len(path)):
            prev, curr = str(path[i - 1]), str(path[i])
            dur = self.trans_medians.get((prev, curr),
                  self.loc_medians.get(curr, self.global_median))
            # dur is always a float here; guard anyway
            dur = float(dur) if (isinstance(dur, (int, float)) and math.isfinite(dur) and dur > 0) else self.fallback_days

            cum += dur
            arrival = (cur_date + pd.to_timedelta(cum, unit="D")) if cur_date is not None else None

            results.append({
                "Step": i,
                "Predicted Location": curr,
                "ETA (days)": round(dur, 2),
                "ETA (weeks)": round(dur / 7.0, 2),
                "Cumulative days": round(cum, 2),
                "Arrival date": arrival,
            })

        return results
