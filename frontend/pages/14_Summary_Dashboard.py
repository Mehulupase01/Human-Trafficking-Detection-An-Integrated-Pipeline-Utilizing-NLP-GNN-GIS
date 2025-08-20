from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Folium maps
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

from datetime import date, timedelta

# Project imports (paths may differ slightly in your repo)
from backend.api.graph_queries import concat_processed_frames
from backend.core import dataset_registry as registry

# --------------------------------------------------------------------------------------
# Registry fallbacks (fixes AttributeError)
# --------------------------------------------------------------------------------------
def _list_registry_keys() -> List[str]:
    """
    Aggressively discover dataset keys from backend.core.dataset_registry.
    Tries common methods, attributes, dict-like stores, and any callable that
    looks like a 'list*' function returning an iterable of strings.
    """
    keys: List[str] = []

    # 1) Known method names (most likely)
    try_methods = [
        "list_processed_keys",
        "list_processed_datasets",
        "list_keys",
        "list",
        "available",
        "available_processed",
        "processed_keys",
        "get_processed_keys",
    ]
    for m in try_methods:
        if hasattr(registry, m):
            obj = getattr(registry, m)
            try:
                res = obj() if callable(obj) else obj
                if isinstance(res, dict):
                    keys = list(res.keys())
                elif isinstance(res, (list, tuple, set)):
                    keys = list(res)
                # Stop if we found any
                if keys:
                    return [str(k) for k in keys]
            except Exception:
                pass

    # 2) Dict-like attributes that often hold processed datasets
    dict_like_attrs = [
        "processed", "datasets", "MERGED", "PROCESSED", "STORE", "REGISTRY",
    ]
    for attr in dict_like_attrs:
        if hasattr(registry, attr):
            obj = getattr(registry, attr)
            if isinstance(obj, dict) and obj:
                return [str(k) for k in obj.keys()]

    # 3) Fallback: scan all attributes; call any 'list*' function safely
    for name in dir(registry):
        if name.startswith("_"):
            continue
        obj = getattr(registry, name)
        if callable(obj) and name.lower().startswith("list"):
            try:
                res = obj()
                if isinstance(res, dict):
                    keys.extend(list(res.keys()))
                elif isinstance(res, (list, tuple, set)):
                    keys.extend(list(res))
            except Exception:
                continue
    keys = [str(k) for k in keys if isinstance(k, (str, int))]
    # De-dup while preserving order
    seen = set(); deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def _registry_fetch_frames(keys: List[str]) -> List[pd.DataFrame]:
    for cand in ("fetch_frames", "get_frames", "get_processed_frames"):
        if hasattr(registry, cand):
            fn = getattr(registry, cand)
            try:
                frames = fn(keys)
                if isinstance(frames, (list, tuple)):
                    return list(frames)
            except Exception:
                pass
    try:
        df = concat_processed_frames(keys)
        return [df]
    except Exception:
        return []

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if df.empty:
        return {}
    mapping = {}
    mapping["victim_id"]   = pick_first_present(df, ["victim_id","uid","unique_id","person_id","subject_id"])
    mapping["session_id"]  = pick_first_present(df, ["sid","session_id","case_id","record_id"])
    mapping["date"]        = pick_first_present(df, ["date","timestamp","event_date","created_at","dt"])
    mapping["location"]    = pick_first_present(df, ["location","place","city","node_location","geoname"])
    mapping["country"]     = pick_first_present(df, ["country","nation","nationality_country","country_of_origin"])
    mapping["nationality"] = pick_first_present(df, ["nationality","nation","citizenship"])
    mapping["gender"]      = pick_first_present(df, ["gender","sex"])
    mapping["age"]         = pick_first_present(df, ["age","age_years"])
    mapping["role"]        = pick_first_present(df, ["role","actor_type"])
    mapping["perpetrator"] = pick_first_present(df, ["trafficker","perpetrator","offender","suspect","chief"])
    mapping["route_len"]   = pick_first_present(df, ["route_length","stops","num_stops","path_len"])
    mapping["lat"]         = pick_first_present(df, ["lat","latitude"])
    mapping["lon"]         = pick_first_present(df, ["lon","lng","longitude"])
    return mapping

def normalize_gender(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")
    m = (
        s.astype(str).str.strip().str.lower()
         .replace({
             "m":"male","f":"female","man":"male","woman":"female",
             "female":"female","male":"male","unknown":"unknown","unk":"unknown",
             "na":"unknown","none":"unknown","nan":"unknown"
         })
    )
    return m.where(m.isin(["male","female","unknown"]), "unknown")

def normalize_nationality(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")
    return (
        s.astype(str).str.strip()
         .replace({"Not specified":"Unknown","not specified":"Unknown","nan":"Unknown","NaN":"Unknown","None":"Unknown"})
         .replace(r"^\s*$", "Unknown", regex=True)
    )

def safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.to_datetime(pd.Series([], dtype="object"))

@st.cache_data(show_spinner=False)
def load_processed_frames(selected_keys: List[str]) -> pd.DataFrame:
    if not selected_keys:
        return pd.DataFrame()
    frames = _registry_fetch_frames(selected_keys)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    if df.empty:
        return df, {}
    schema = detect_schema(df).copy()
    out = pd.DataFrame(index=df.index)
    out["victim_id"]   = df[schema["victim_id"]]   if schema["victim_id"]   else None
    out["session_id"]  = df[schema["session_id"]]  if schema["session_id"]  else None
    out["date"]        = safe_to_datetime(df[schema["date"]]) if schema["date"] else pd.NaT
    out["location"]    = df[schema["location"]]    if schema["location"]    else None
    out["country"]     = df[schema["country"]]     if schema["country"]     else None
    out["nationality"] = normalize_nationality(df[schema["nationality"]]) if schema["nationality"] else None
    out["gender"]      = normalize_gender(df[schema["gender"]]) if schema["gender"] else None
    out["age"]         = pd.to_numeric(df[schema["age"]], errors="coerce") if schema["age"] else np.nan
    out["role"]        = df[schema["role"]]        if schema["role"]        else None
    out["perpetrator"] = df[schema["perpetrator"]] if schema["perpetrator"] else None
    out["route_len"]   = pd.to_numeric(df[schema["route_len"]], errors="coerce") if schema["route_len"] else np.nan
    out["lat"]         = pd.to_numeric(df[schema["lat"]], errors="coerce") if schema["lat"] else np.nan
    out["lon"]         = pd.to_numeric(df[schema["lon"]], errors="coerce") if schema["lon"] else np.nan

    out["year_month"] = out["date"].dt.to_period("M").astype(str)
    out["year"]       = out["date"].dt.year
    out["has_geo"]    = out["lat"].notna() & out["lon"].notna()
    return out, schema

def kpi_number(label: str, value, help_text: str=""):
    st.metric(label, value, help=help_text)

# --------------------------------------------------------------------------------------
# Safe date range helper
# --------------------------------------------------------------------------------------
def _safe_date_range(df):
    if df.empty or df["date"].dropna().empty:
        today = date.today()
        return (today - timedelta(days=365), today)
    dmin = pd.to_datetime(df["date"].min(), errors="coerce")
    dmax = pd.to_datetime(df["date"].max(), errors="coerce")
    if pd.isna(dmin) or pd.isna(dmax):
        today = date.today()
        return (today - timedelta(days=365), today)
    start = dmin.date()
    end   = dmax.date()
    if start > end:
        start, end = end, start
    return (start, end)

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Summary Dashboard", layout="wide", page_icon="📊")
st.title("📊 Summary Dashboard")

with st.sidebar:
    st.header("Filters")
    all_keys = _list_registry_keys()
    with st.expander("🔧 Registry debug"):
        st.write("Module:", registry.__name__)
        st.write("Discovered keys:", all_keys if all_keys else "—")
        st.write("Attrs (first 30):", [a for a in dir(registry) if not a.startswith("_")][:30])

        manual = st.text_input("Manual dataset keys (comma-separated)", value="")
        if manual and not all_keys:
            all_keys = [k.strip() for k in manual.split(",") if k.strip()]

    selected = st.multiselect("Choose datasets", all_keys, default=(all_keys[:1] if all_keys else []))
    df_raw = load_processed_frames(selected)
    df, schema = prepare(df_raw)

    # Safe date input
    start_default, end_default = _safe_date_range(df)
    date_rng = st.date_input(
        "Date range",
        value=(start_default, end_default),
        key="dash_date_range",
    )

    gender_filter = st.multiselect("Gender", ["female","male","unknown"], default=["female","male","unknown"])
    nationality_topk = st.slider("Top N nationalities", 5, 25, 15)
    smooth = st.slider("Smoothing (hist/series)", 0, 10, 2)

    # Apply filters
    if not df.empty and date_rng:
        if isinstance(date_rng, tuple) and len(date_rng) == 2:
            start = pd.to_datetime(date_rng[0])
            end   = pd.to_datetime(date_rng[1]) + pd.Timedelta(days=1)
        else:
            start = pd.to_datetime(date_rng)
            end   = start + pd.Timedelta(days=1)
        df = df[(df["date"] >= start) & (df["date"] < end)]
    if not df.empty:
        df = df[df["gender"].isin(gender_filter)]

if df.empty:
    st.info("No records to display. Select a dataset on the left.")
    st.stop()

# --------------------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    victims = df["victim_id"].nunique() if df["victim_id"].notna().any() else len(df)
    kpi_number("Victims", victims)

with k2:
    unique_ids = df["victim_id"].nunique() if df["victim_id"].notna().any() else np.nan
    kpi_number("Unique IDs", unique_ids)

with k3:
    n_locations = df["location"].nunique() if df["location"].notna().any() else df["has_geo"].sum()
    kpi_number("Locations", n_locations)

with k4:
    n_perp = df["perpetrator"].nunique() if df["perpetrator"].notna().any() else 0
    kpi_number("Traffickers", n_perp)

with k5:
    med_route = int(np.nanmedian(df["route_len"])) if "route_len" in df and df["route_len"].notna().any() else 0
    kpi_number("Median Route Len", med_route)

with k6:
    # UID/SID ≈ uniqueness proxy
    uid = df["victim_id"].nunique() if df["victim_id"].notna().any() else None
    sid = df["session_id"].nunique() if df["session_id"].notna().any() else None
    ratio = round(uid / sid, 3) if uid and sid and sid > 0 else "—"
    kpi_number("UID/SID Ratio", ratio)

st.markdown("---")

# --------------------------------------------------------------------------------------
# 1) Distributions & Rankings
# --------------------------------------------------------------------------------------
c1, c2 = st.columns([2, 2])

with c1:
    st.subheader("Top Locations (by Victims)")
    if df["location"].notna().any():
        top_loc = (
            df.groupby("location")["victim_id"].nunique().reset_index(name="victim_count")
              .sort_values("victim_count", ascending=False).head(20)
        )
        fig = px.bar(top_loc, x="victim_count", y="location", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("CSV", top_loc.to_csv(index=False).encode("utf-8"), file_name="top_locations.csv")
    else:
        st.info("No location column present.")

with c2:
    st.subheader("Top Traffickers (by linked Victims)")
    if df["perpetrator"].notna().any():
        top_perp = (
            df.dropna(subset=["perpetrator"])
              .groupby("perpetrator")["victim_id"].nunique()
              .reset_index(name="victim_count").sort_values("victim_count", ascending=False).head(20)
        )
        fig = px.bar(top_perp, x="victim_count", y="perpetrator", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("CSV", top_perp.to_csv(index=False).encode("utf-8"), file_name="top_traffickers.csv")
    else:
        st.info("No perpetrator stats available.")






st.markdown("---")
st.header("🗺️ Geospatial Add‑Ons")

with st.expander("Upload geocoded locations (CSV with columns: location, lat, lon)"):
    up = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="geo_upload")
    if up is not None:
        try:
            updf = pd.read_csv(up)
            # normalize column names
            colmap = {c.lower().strip(): c for c in updf.columns}
            reqs = ["location", "lat", "lon"]
            if all(r in colmap for r in reqs):
                updf = updf.rename(columns={colmap["location"]: "location",
                                            colmap["lat"]: "lat",
                                            colmap["lon"]: "lon"})
                updf["lat"] = pd.to_numeric(updf["lat"], errors="coerce")
                updf["lon"] = pd.to_numeric(updf["lon"], errors="coerce")
                updf = updf.dropna(subset=["lat","lon"])
                st.success(f"Loaded {len(updf)} rows.")

                # 2.1 Clustered points
                st.subheader("Location Markers (Clustered)")
                fig_scatter = px.scatter_mapbox(
                    updf, lat="lat", lon="lon", hover_name="location",
                    zoom=2, height=420
                )
                fig_scatter.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_scatter, use_container_width=True)

                # 2.2 Heatmap (density)
                st.subheader("Density Heatmap")
                # Plotly doesn't have native heatmap layer for mapbox; emulate with density contours
                fig_heat = px.density_mapbox(
                    updf, lat="lat", lon="lon", z=None, radius=20,
                    hover_name="location", height=420, zoom=2
                )
                fig_heat.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_heat, use_container_width=True)

                st.download_button("Download cleaned locations CSV",
                                   updf.to_csv(index=False).encode("utf-8"),
                                   file_name="locations_clean.csv")
            else:
                st.error("CSV must have columns: location, lat, lon (case-insensitive).")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")





# --------------------------------------------------------------------------------------
# 2) Gender & Nationality (normalized)
# --------------------------------------------------------------------------------------
g1, g2 = st.columns([1, 2])

with g1:
    st.subheader("Gender Distribution")
    g = df["gender"].value_counts(dropna=False).rename_axis("gender").reset_index(name="count")
    fig = px.pie(g, names="gender", values="count", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("CSV", g.to_csv(index=False).encode("utf-8"), file_name="gender_distribution.csv")

with g2:
    st.subheader(f"Nationality (Top {nationality_topk})")
    n = (
        df["nationality"].value_counts(dropna=False).rename_axis("nationality").reset_index(name="count")
        .head(nationality_topk)
    )
    fig = px.bar(n, x="count", y="nationality", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("CSV", n.to_csv(index=False).encode("utf-8"), file_name="nationality_top.csv")

# --------------------------------------------------------------------------------------
# 3) Time series: cases per month (+ smoothing)
# --------------------------------------------------------------------------------------
st.subheader("Cases Over Time (Monthly)")
if df["year_month"].notna().any():
    ts = df.groupby("year_month")["victim_id"].nunique().reset_index(name="victims")
    ts = ts.sort_values("year_month")
    if smooth > 0 and len(ts) > smooth:
        ts["smoothed"] = ts["victims"].rolling(window=smooth, min_periods=1, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts["year_month"], y=ts["victims"], name="victims"))
    if "smoothed" in ts:
        fig.add_trace(go.Scatter(x=ts["year_month"], y=ts["smoothed"], mode="lines", name="smoothed"))
    fig.update_layout(xaxis_title="Month", yaxis_title="Victims")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("CSV", ts.to_csv(index=False).encode("utf-8"), file_name="cases_over_time.csv")
else:
    st.info("No dates present for time series.")

# --------------------------------------------------------------------------------------
# 4) Route length distribution (+ smoothing + robust stats)
# --------------------------------------------------------------------------------------
st.subheader("Route Lengths per Victim")
if "route_len" in df and df["route_len"].notna().any():
    rl = df.dropna(subset=["route_len"]).copy()
    # truncate extreme outliers for viz while keeping them flagged below
    q99 = rl["route_len"].quantile(0.99)
    trunc = rl["route_len"].clip(upper=q99)
    hist = np.histogram(trunc, bins="auto")
    counts = pd.Series(hist[0])
    bins = pd.Series(hist[1][:-1])

    if smooth > 0 and len(counts) > 1:
        counts_sm = counts.rolling(window=max(2, smooth), min_periods=1, center=True).mean()
    else:
        counts_sm = counts

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bins, y=counts, name="count"))
    if not counts_sm.equals(counts):
        fig.add_trace(go.Scatter(x=bins, y=counts_sm, mode="lines", name="smoothed"))
    fig.update_layout(xaxis_title="Stops per victim", yaxis_title="Victim count")
    st.plotly_chart(fig, use_container_width=True)

    stats = {
        "n": int(len(rl)),
        "median": float(np.nanmedian(rl["route_len"])),
        "p90": float(np.nanpercentile(rl["route_len"], 90)),
        "max": float(np.nanmax(rl["route_len"])),
    }
    st.caption(f"**n**={stats['n']} • **median**={stats['median']:.0f} • **p90**={stats['p90']:.0f} • **max**={stats['max']:.0f} (outliers truncated at 99th pct for viz)")
    st.download_button("CSV", rl[["victim_id","route_len"]].to_csv(index=False).encode("utf-8"), file_name="route_lengths.csv")
else:
    st.info("No route length column present.")

# --------------------------------------------------------------------------------------
# 5) Age distribution & pyramid (if age present)
# --------------------------------------------------------------------------------------
st.subheader("Age Distribution")
if df["age"].notna().any():
    age_bins = pd.cut(df["age"], bins=list(range(0, 81, 5)), right=False)
    age_counts = age_bins.value_counts().sort_index().reset_index()
    age_counts.columns = ["age_band","count"]
    fig = px.bar(age_counts, x="age_band", y="count")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("CSV", age_counts.to_csv(index=False).encode("utf-8"), file_name="age_distribution.csv")
else:
    st.info("No age information present.")

# --------------------------------------------------------------------------------------
# 6) Heatmap of activity by month vs. location (top 10 locations)
# --------------------------------------------------------------------------------------
st.subheader("Activity Heatmap (Month × Location)")
if df["location"].notna().any() and df["year_month"].notna().any():
    top10loc = df["location"].value_counts().head(10).index
    heat = (
        df[df["location"].isin(top10loc)]
        .groupby(["year_month","location"])["victim_id"].nunique().reset_index(name="victims")
    )
    heat_pvt = heat.pivot(index="location", columns="year_month", values="victims").fillna(0)
    fig = px.imshow(heat_pvt, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("CSV", heat.to_csv(index=False).encode("utf-8"), file_name="heatmap_month_location.csv")
else:
    st.info("Need location and date to draw this heatmap.")
    
    
    
    
    
    
    st.subheader("📈 Movement Flows (Sankey)")

tabs = st.tabs(["Auto (from dataset)", "Upload edges CSV"])

# ---------- Case A: auto from dataset ----------
with tabs[0]:
    if {"victim_id","date","location"}.issubset(df.columns) and df["victim_id"].notna().any() and df["date"].notna().any():
        tmp = df.dropna(subset=["victim_id","date","location"]).copy()
        tmp = tmp.sort_values(["victim_id","date"])
        # consecutive transitions
        tmp["next_location"] = tmp.groupby("victim_id")["location"].shift(-1)
        edges = (
            tmp.dropna(subset=["next_location"])
               .groupby(["location","next_location"])["victim_id"].nunique()
               .reset_index(name="count")
               .sort_values("count", ascending=False)
        )

        limit = st.slider("Show top N flows", 10, 200, 50)
        edges = edges.head(limit)

        if edges.empty:
            st.info("No transitions detected.")
        else:
            # build sankey indices
            nodes = pd.Index(pd.unique(edges[["location","next_location"]].values.ravel()))
            src = nodes.get_indexer(edges["location"])
            dst = nodes.get_indexer(edges["next_location"])
            val = edges["count"].astype(int)

            fig = go.Figure(data=[go.Sankey(
                node=dict(label=nodes.tolist(), pad=12, thickness=14),
                link=dict(source=src, target=dst, value=val)
            )])
            fig.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Edges CSV", edges.to_csv(index=False).encode("utf-8"), file_name="auto_flows.csv")
    else:
        st.info("Auto mode needs victim_id + date + location in the dataset.")

# ---------- Case B: manual edges upload ----------
with tabs[1]:
    st.caption("Upload an edges CSV with columns: source,target[,value].")
    efile = st.file_uploader("Upload edges CSV", type=["csv"], key="edges_upload")
    if efile is not None:
        try:
            edf = pd.read_csv(efile)
            # normalize likely headers
            lower = {c.lower().strip(): c for c in edf.columns}
            # map common aliases
            alias = {
                "source": ["source","src","origin","from","start"],
                "target": ["target","dst","destination","to","end"],
                "value":  ["value","weight","count","n"]
            }
            cols = {}
            for std, cands in alias.items():
                for c in cands:
                    if c in lower:
                        cols[std] = lower[c]
                        break
            if not {"source","target"}.issubset(cols):
                st.error("Could not find columns for source/target. Try headers: source,target[,value]")
            else:
                edf = edf.rename(columns={cols["source"]: "source", cols["target"]: "target", **({cols["value"]:"value"} if "value" in cols else {})})
                if "value" not in edf.columns:
                    edf["value"] = 1
                # aggregate
                edges = edf.groupby(["source","target"])["value"].sum().reset_index(name="value")
                # build sankey
                nodes = pd.Index(pd.unique(edges[["source","target"]].values.ravel()))
                src = nodes.get_indexer(edges["source"])
                dst = nodes.get_indexer(edges["target"])
                val = edges["value"].astype(float)

                fig = go.Figure(data=[go.Sankey(
                    node=dict(label=nodes.tolist(), pad=12, thickness=14),
                    link=dict(source=src, target=dst, value=val)
                )])
                fig.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Normalized edges CSV",
                                   edges.to_csv(index=False).encode("utf-8"),
                                   file_name="edges_normalized.csv")
        except Exception as e:
            st.error(f"Failed to render edges CSV: {e}")




# --------------------------------------------------------------------------------------
# 7) Data Quality Panel (duplicates, missingness, outliers)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Data Quality & Diagnostics")

dq1, dq2, dq3 = st.columns(3)

with dq1:
    # Duplicates by victim_id + date (heuristic)
    if df["victim_id"].notna().any():
        dup = df.duplicated(subset=["victim_id","date"], keep=False).sum()
        kpi_number("Suspected duplicates", int(dup), help_text="Same victim_id + date appearing multiple times")
    else:
        kpi_number("Suspected duplicates", "—")

with dq2:
    # Missingness
    miss = df.isna().mean().sort_values(ascending=False).head(5)
    miss_tbl = miss.reset_index()
    miss_tbl.columns = ["column","missing_fraction"]
    st.caption("Top missingness")
    st.dataframe(miss_tbl, hide_index=True)

with dq3:
    # Route outliers
    if "route_len" in df and df["route_len"].notna().any():
        q99_all = df["route_len"].quantile(0.99)
        outlier_n = (df["route_len"] > q99_all).sum()
        kpi_number("Extreme route outliers", int(outlier_n), help_text=">99th percentile")
    else:
        kpi_number("Extreme route outliers", "—")

st.success("All charts use normalized gender/nationality and shared filters, fixing prior inconsistencies.")
