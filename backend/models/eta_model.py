# backend/models/eta_model.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

COL_SID = "Serialized ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"
COL_TIME = "Time Spent in Location / Cities / Places"


def parse_time_to_days(value) -> Optional[int]:
    """
    Parse loose values like '3', '3 days', '2 weeks', '1 month' → days (int).
    Returns None if unparseable.
    """
    if value is None:
        return None
    try:
        # numeric → days
        iv = int(float(str(value).strip()))
        return max(0, iv)
    except Exception:
        pass

    s = str(value).strip().lower()
    # standardize separators
    for ch in ",;|":
        s = s.replace(ch, " ")
    tokens = s.split()
    num = None
    for t in tokens:
        try:
            num = float(t)
            break
        except Exception:
            continue
    if num is None:
        return None

    if "week" in s:
        return int(num * 7)
    if "month" in s:
        return int(num * 30)
    if "day" in s:
        return int(num)
    # fall through: unknown unit
    return None


def build_duration_stats(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build robust duration stats from processed long-format data.

    We interpret the 'Time Spent...' value on row j as the duration between
    Location[j-1] → Location[j]. (Matches the GIS animation logic.)
    """
    if not {COL_SID, COL_LOC, COL_ROUTE}.issubset(df.columns):
        raise ValueError("DataFrame missing required columns for ETA modeling.")

    # Per (A,B) transition durations
    pair_days: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    # Per-location stay (at B)
    loc_days: Dict[str, List[int]] = defaultdict(list)
    global_days: List[int] = []

    for sid, grp in df.groupby(COL_SID):
        g = grp.sort_values(COL_ROUTE, kind="stable")
        locs = g[COL_LOC].astype(str).tolist()
        # Align durations with transitions: value at step j describes hop (j-1)->(j)
        time_vals = g.get(COL_TIME, pd.Series([None] * len(g))).tolist()
        if len(locs) <= 1:
            continue
        hops = list(zip(locs, locs[1:], time_vals[1:]))  # (A, B, time_at_B)

        for a, b, val in hops:
            d = parse_time_to_days(val)
            if d is None:
                continue
            pair_days[(a, b)].append(d)
            loc_days[b].append(d)
            global_days.append(d)

    # Aggregate medians
    pair_median = {k: int(pd.Series(v).median()) for k, v in pair_days.items() if v}
    loc_median = {k: int(pd.Series(v).median()) for k, v in loc_days.items() if v}
    global_median = int(pd.Series(global_days).median()) if global_days else None

    return {
        "pair_median": pair_median,     # (A,B) -> days
        "loc_median": loc_median,       # B -> days
        "global_median": global_median  # days
    }


def estimate_path_durations(
    history: List[str],
    next_locs: List[str],
    stats: Dict[str, object],
    default_days: int = 7,
) -> List[int]:
    """
    For each transition history[-1]→next1, next1→next2, ... select duration:
      1) (A,B) median if available
      2) B location median if available
      3) global median if available
      4) default_days
    Returns a list of ETA days per predicted hop, same length as next_locs.
    """
    pair_median = stats.get("pair_median", {})
    loc_median = stats.get("loc_median", {})
    global_median = stats.get("global_median", None)

    durations: List[int] = []
    prev = history[-1] if history else None
    for nxt in next_locs:
        d = None
        if prev is not None and (prev, nxt) in pair_median:
            d = pair_median[(prev, nxt)]
        elif nxt in loc_median:
            d = loc_median[nxt]
        elif global_median is not None:
            d = global_median
        else:
            d = default_days
        durations.append(int(max(0, d)))
        prev = nxt
    return durations


def cumulative_arrival_dates(
    start_date: date,
    durations_days: List[int]
) -> List[date]:
    """
    Construct arrival dates by cumulatively adding durations to the start_date.
    """
    out: List[date] = []
    t = start_date
    for d in durations_days:
        t = t + timedelta(days=int(d))
        out.append(t)
    return out
