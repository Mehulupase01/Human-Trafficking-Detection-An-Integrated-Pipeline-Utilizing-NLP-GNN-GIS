import math
import numpy as np
import pandas as pd

from eval_harness.split_manager import build_splits
from eval_harness.components.gis_eval import eval_all as gis_eval


def _toy_gis_df(n_subjects=6, points_per_subj=4, seed=9):
    """
    Build a small dataframe with:
      - sid (trajectory id), lat/lon grid around Amsterdam
      - timestamp (monotonic per sid)
      - start_ts / end_ts (for ETA)
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_lat, base_lon = 52.37, 4.90
    ts0 = pd.Timestamp("2024-02-01 00:00:00")
    for s in range(n_subjects):
        sid = f"S{s:02d}"
        for k in range(points_per_subj):
            lat = base_lat + 0.01 * ((s + k) % 5)
            lon = base_lon + 0.02 * ((s + 2 * k) % 5)
            ts = ts0 + pd.Timedelta(hours=3 * k)  # monotonic
            # ETA pair varies by subj/step
            start = ts
            end = ts + pd.Timedelta(days=((k % 3) + 1))
            rows.append(
                {
                    "sid": sid,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": ts.isoformat(),
                    "start_ts": start.isoformat(),
                    "end_ts": end.isoformat(),
                }
            )
    # add a few rows with missing coords to exercise geocode rate logic
    for j in range(3):
        rows.append(
            {"sid": f"Sx{j}", "lat": None, "lon": None, "timestamp": ts0.isoformat(), "start_ts": None, "end_ts": None}
        )
    return pd.DataFrame(rows)


def _is_prob(x):
    try:
        x = float(x)
        return 0.0 - 1e-9 <= x <= 1.0 + 1e-9
    except Exception:
        return False


def test_gis_eval_smoke_and_bounds():
    df = _toy_gis_df()
    splits = build_splits(df, seed=7, k=3, test_frac=0.30)

    res = gis_eval(df, splits)
    assert res.get("available", True) is True, f"GIS unavailable: {res}"

    # Geocode rate in [0,1]
    ge = res.get("geocode", {})
    assert ge.get("available", True) is True
    assert _is_prob(ge.get("rate", 0.0))

    # Trajectory stats present and non-negative
    tr = res.get("trajectories", {})
    assert tr.get("available", True) is True
    assert int(tr.get("trajectories", 0)) >= 1
    assert float(tr.get("median_hops", 0.0)) >= 0.0
    assert float(tr.get("gap_hours_median", 0.0)) >= 0.0
    assert float(tr.get("gap_hours_p90", 0.0)) >= 0.0

    # Next-location: metrics in [0,1] (zeros allowed)
    nxh = res.get("nextloc", {}).get("holdout", {})
    if nxh.get("available", True):
        assert _is_prob(nxh.get("acc@1", 0.0))
        assert _is_prob(nxh.get("acc@3", 0.0))

    # ETA MAE: finite and >= 0
    eta = res.get("eta", {}).get("holdout", {})
    if eta.get("available", True):
        mae = float(eta.get("mae_days", 0.0))
        assert math.isfinite(mae) and mae >= 0.0
