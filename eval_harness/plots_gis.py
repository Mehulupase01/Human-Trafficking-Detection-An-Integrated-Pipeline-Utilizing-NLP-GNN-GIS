from __future__ import annotations
"""
GIS plotting helpers (Folium + Altair).

What’s here
-----------
- make_map(df): builds a Folium map with an optional HeatMap & MarkerCluster,
  auto-detecting lat/lon; safely returns a minimal map if columns unavailable.
- render_map(st, fmap): embeds Folium map in Streamlit (uses streamlit-folium
  if present; otherwise raw HTML fallback).
- gaps_histogram(df, sid_col, time_col): Altair histogram of per-hop time gaps.
- clusters_histogram(labels): Altair histogram of cluster labels (if available).

These helpers are decoupled from the evaluator so you can reuse them in multiple pages.
"""

from typing import Optional, Sequence, List, Dict, Any
import numpy as np
import pandas as pd
import altair as alt

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
except Exception:  # Folium not installed
    folium = None
    HeatMap = None
    MarkerCluster = None

LAT_CAND = ["lat", "latitude", "geo_lat", "y"]
LON_CAND = ["lon", "longitude", "geo_lon", "x"]
SID_CAND = ["sid", "subject_id", "victim_id", "case_id", "trajectory_id"]
TIME_CAND = ["ts", "timestamp", "time", "date", "datetime"]

def _pick_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None

# ---------------- Folium map helpers ----------------

def _center_of_points(df: pd.DataFrame, lat: str, lon: str) -> List[float]:
    sub = df[[lat, lon]].dropna()
    if sub.empty:
        return [0.0, 0.0]
    return [float(sub[lat].mean()), float(sub[lon].mean())]

def make_map(df: pd.DataFrame, *, add_heat: bool = True, add_cluster: bool = True, max_points: int = 5000):
    """
    Build a Folium map from a dataframe with lat/lon columns (auto-detected).
    Safely returns a bare Map if Folium or columns are missing.
    """
    if folium is None or df is None or df.empty:
        return None

    lat = _pick_first(df.columns, LAT_CAND)
    lon = _pick_first(df.columns, LON_CAND)
    if lat is None or lon is None:
        # no map possible
        return folium.Map(location=[0, 0], zoom_start=2)

    sub = df[[lat, lon]].dropna().copy()
    if sub.empty:
        return folium.Map(location=[0, 0], zoom_start=2)

    sub = sub.iloc[:max_points]
    center = _center_of_points(sub, lat, lon)
    fmap = folium.Map(location=center, zoom_start=3, control_scale=True)

    if add_heat and HeatMap is not None:
        HeatMap(sub[[lat, lon]].values.tolist(), radius=7, blur=6, max_zoom=12).add_to(fmap)

    if add_cluster and MarkerCluster is not None:
        mc = MarkerCluster(name="Samples")
        for _, row in sub.iterrows():
            try:
                folium.CircleMarker(
                    location=[float(row[lat]), float(row[lon])],
                    radius=2, fill=True, fill_opacity=0.6, opacity=0.6
                ).add_to(mc)
            except Exception:
                continue
        mc.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap

def render_map(st, fmap) -> None:
    """
    Render a Folium map within Streamlit.
    Uses streamlit-folium if available, else HTML fallback.
    """
    if fmap is None:
        st.caption("Map not available (missing Folium or data).")
        return
    # Try streamlit-folium
    try:
        from streamlit_folium import st_folium
        st_folium(fmap, width=None, height=500)
        return
    except Exception:
        pass
    # Fallback to raw HTML
    try:
        html = fmap._repr_html_()
        st.components.v1.html(html, height=520, scrolling=True)
    except Exception:
        st.caption("Unable to render map.")

# ---------------- Altair histograms ----------------

def gaps_histogram(df: pd.DataFrame, *, sid_col: Optional[str] = None, time_col: Optional[str] = None) -> alt.Chart:
    """
    Build a histogram of per-hop time gaps (hours) within subject trajectories.
    Auto-detects columns if not provided.
    """
    if df is None or df.empty:
        return alt.Chart(pd.DataFrame({"x":[0]})).mark_bar().encode(x="x").properties(height=200)

    sid = sid_col or _pick_first(df.columns, SID_CAND)
    tcol = time_col or _pick_first(df.columns, TIME_CAND)
    if sid is None:
        return alt.Chart(pd.DataFrame({"x":[0]})).mark_bar().encode(x="x").properties(title="Time gaps (n/a)", height=200)

    work = pd.DataFrame({sid: df[sid]})
    if tcol and tcol in df.columns:
        t = df[tcol]
        if np.issubdtype(t.dtype, np.number):
            work["_ts"] = t
        else:
            work["_ts"] = pd.to_datetime(t, errors="coerce").astype("int64") // 10**9
    else:
        work["_ts"] = np.arange(len(work))

    gaps = []
    for _, sub in work.dropna(subset=["_ts"]).sort_values([sid, "_ts"]).groupby(sid):
        ts = sub["_ts"].to_numpy(dtype=float)
        if ts.size >= 2:
            dt_hours = np.diff(ts) / 3600.0
            gaps += [float(x) for x in dt_hours if np.isfinite(x)]
    if not gaps:
        return alt.Chart(pd.DataFrame({"x":[0]})).mark_bar().encode(x="x").properties(title="Time gaps (empty)", height=200)

    dfh = pd.DataFrame({"gap_hours": gaps})
    return (
        alt.Chart(dfh)
        .mark_bar()
        .encode(
            x=alt.X("gap_hours:Q", bin=alt.Bin(maxbins=30), title="Gap (hours)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q"), alt.Tooltip("gap_hours:Q", format=".1f")]
        )
        .properties(title="Per-hop time gaps", height=220)
    )

def clusters_histogram(labels: Sequence[int]) -> alt.Chart:
    """
    Histogram of cluster labels (e.g., from DBSCAN). -1 denotes noise.
    """
    if labels is None or len(labels) == 0:
        return alt.Chart(pd.DataFrame({"x":[0]})).mark_bar().encode(x="x").properties(height=200)
    df = pd.DataFrame({"label": list(map(int, labels))})
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="Cluster label (-1 = noise)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["label:N", alt.Tooltip("count():Q")],
        )
        .properties(title="Cluster label distribution", height=220)
    )
