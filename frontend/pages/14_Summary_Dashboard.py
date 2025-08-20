# frontend/pages/14_Summary_Dashboard.py
from __future__ import annotations

# --- MUST be the first Streamlit call on the page ---
import streamlit as st
st.set_page_config(page_title="Summary Dashboard", page_icon="📊", layout="wide")

# --- Rest of imports ---
import json, re
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.metrics import STD_FIELDS, core_counts, build_summary_tables

# ============================================================
# Helpers: minimal, robust schema detection for add-on charts
# (We still use build_summary_tables for the classic charts.)
# ============================================================
_CAND = {
    "victim_id":   ["victim_id","uid","unique_id","person_id","subject_id","trajectory_id","traj_id","entity_id","case_person_id","global_id"],
    "session_id":  ["session_id","sid","case_id","record_id","visit_id"],
    "date":        ["date","timestamp","event_date","created_at","dt","datetime","time"],
    "location":    ["location","place","city","node_location","geoname","loc","site"],
    "lat":         ["lat","latitude"],
    "lon":         ["lon","lng","longitude"],
}

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df.empty: return None
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in lower: return lower[n]
    for c in df.columns:
        cl = c.lower().replace("-", "_").replace(" ", "_")
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", cl):
                return c
    return None

def unify_for_addons(df_in: pd.DataFrame) -> (pd.DataFrame, Dict[str, Optional[str]], List[str]):
    """Return df with safe standard columns for add-ons + schema map + notes."""
    notes: List[str] = []
    df = df_in.copy()
    cols = {k: _find_col(df, v) for k, v in _CAND.items()}

    # victim id
    victim_key = cols["victim_id"] or _find_col(df, ["uid","unique_id","person_id","subject_id","trajectory_id","traj_id"]) or cols["session_id"]
    if victim_key:
        if "victim_id" not in df.columns: df["victim_id"] = df[victim_key].astype(str)
    else:
        df["victim_id"] = np.arange(len(df)).astype(str)
        notes.append("No victim identifier found; using synthetic per-row id (row-level proxy).")

    # location
    if cols["location"] and "location" not in df.columns:
        df["location"] = df[cols["location"]]

    # dates
    if cols["date"]:
        df["date"] = pd.to_datetime(df[cols["date"]], errors="coerce", utc=True)
        df["year"] = df["date"].dt.year
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
    else:
        notes.append("No usable date column detected.")

    # coords
    for c in ("lat","lon"):
        col = cols[c]
        if col and c not in df.columns:
            df[c] = pd.to_numeric(df[col], errors="coerce")

    return df, cols, notes

# ============================================================
# Page
# ============================================================
st.title("📊 Summary Dashboard")

st.markdown(
    "This page matches the **old working logic** for core charts and adds robust, "
    "victim-aware insights. No date/time filter. Each chart has its **own** smoothing/bins control."
)

# 1) Dataset selection ---------------------------------------------------
st.subheader("1) Choose datasets")
processed = registry.list_datasets(kind="processed") or []
merged    = registry.list_datasets(kind="merged") or []
queryable = processed + merged

def _fmt(e: dict) -> str:
    return f"{e.get('name')}  •  {e.get('kind')}  •  {e.get('id')}"

if not queryable:
    st.info("No processed or merged datasets are available.")
    st.stop()

selected = st.multiselect("Datasets:", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()

ds_ids = [e["id"] if isinstance(e, dict) else e for e in selected]

with st.spinner("Loading and aggregating..."):
    df = concat_processed_frames(ds_ids)

st.caption(f"**Rows:** {len(df):,} • **Columns:** {len(df.columns)}")

# Build tables exactly like the old page (these power the core charts)
tables = build_summary_tables(df)

# Also prep a stable frame for new add-ons
df_add, schema_map, notes = unify_for_addons(df)
with st.expander("🔎 Add-on schema detection (debug)"):
    st.write(schema_map)
    for n in notes:
        st.warning(n)

# 2) KPIs ---------------------------------------------------------------
st.subheader("2) KPIs")
kpis = core_counts(df)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Victims", f"{kpis['victims']:,}")
c2.metric("Unique IDs", f"{kpis['unique_ids']:,}")
c3.metric("Locations", f"{kpis['locations']:,}")
c4.metric("Traffickers", f"{kpis['traffickers']:,}")
c5.metric("Chiefs", f"{kpis['chiefs']:,}")
c6.metric("Median Route Len", f"{kpis['median_route_len']}")
c7.metric("UID/SID Ratio", f"{kpis['dedupe_ratio_uid_per_sid']}" if kpis["dedupe_ratio_uid_per_sid"] is not None else "—")

st.divider()

# 3) Your requested core charts (powered by build_summary_tables) -------
st.subheader("3) Distributions & Rankings")

# 3.1 Gender pie/donut
t_gender = tables.get("Gender Distribution", pd.DataFrame())
st.markdown("**Gender Distribution (victim-level)**")
if not t_gender.empty:
    chart = alt.Chart(t_gender).mark_arc(innerRadius=60).encode(
        theta="Count:Q",
        color=alt.Color("Gender:N"),
        tooltip=["Gender:N", "Count:Q"]
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", t_gender.to_csv(index=False).encode("utf-8"),
                       file_name="gender_distribution.csv", mime="text/csv")
else:
    st.info("No gender stats available in these datasets.")

# 3.2 Nationality top-15
t_nat = tables.get("Nationality (Top 15)", pd.DataFrame())
st.markdown("**Top 15 Nationalities of Victims**")
if not t_nat.empty:
    n_top = t_nat.head(15)
    chart = alt.Chart(n_top).mark_bar().encode(
        x=alt.X("Count:Q"),
        y=alt.Y("Nationality:N", sort='-x'),
        tooltip=["Nationality:N", "Count:Q"]
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", n_top.to_csv(index=False).encode("utf-8"),
                       file_name="nationality_top15.csv", mime="text/csv")
else:
    st.info("No nationality stats available.")

# 3.3 Top 15 Perpetrators (by victims)
t_perp = tables.get("Top Traffickers (by Victims)", pd.DataFrame())
st.markdown("**Top 15 Perpetrators (by victims)**")
if not t_perp.empty:
    p_top = t_perp.head(15)
    chart = alt.Chart(p_top).mark_bar().encode(
        x=alt.X("Victim Count:Q"),
        y=alt.Y("Perpetrator:N", sort='-x'),
        tooltip=["Perpetrator:N", "Victim Count:Q"]
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", p_top.to_csv(index=False).encode("utf-8"),
                       file_name="top_perpetrators.csv", mime="text/csv")
else:
    st.info("No perpetrator stats available.")

# 3.4 Top 15 Locations by traffic (unique victims)
t_loc = tables.get("Top Locations (by Victims)", pd.DataFrame())
st.markdown("**Top 15 Locations by traffic (unique victims who passed through)**")
if not t_loc.empty:
    l_top = t_loc.head(15)
    chart = alt.Chart(l_top).mark_bar().encode(
        x=alt.X("Victim Count:Q", title="Victims"),
        y=alt.Y("Location:N", sort='-x'),
        tooltip=["Location:N", "Victim Count:Q"]
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", l_top.to_csv(index=False).encode("utf-8"),
                       file_name="top_locations_by_victims.csv", mime="text/csv")
else:
    st.info("No location stats available.")

st.divider()

# 4) New Insights -------------------------------------------------------

# 4.1 Dwell time distribution (victim-location stays)
st.subheader("4) Time Spent at Locations — Distribution (victim-location stays)")
st.caption("Consecutive events per victim define stays. We compute (next_date − date) in **days**; last open-ended stays are dropped.")
if {"victim_id","date","location"}.issubset(df_add.columns) and df_add["date"].notna().any():
    seq = (df_add.dropna(subset=["victim_id","date","location"])
                 .sort_values(["victim_id","date"])
                 .loc[:, ["victim_id","date","location"]])
    seq["next_date"] = seq.groupby("victim_id")["date"].shift(-1)
    stay = seq.dropna(subset=["next_date"]).copy()
    stay["days"] = (stay["next_date"] - stay["date"]).dt.total_seconds() / 86400.0
    stay = stay[stay["days"] >= 0]

    bins_dwell = st.slider("Histogram bins (days)", 10, 100, 40, key="bins_dwell")
    hist, edges = np.histogram(stay["days"].values, bins=bins_dwell)
    centers = (edges[:-1] + edges[1:]) / 2
    dwell_df = pd.DataFrame({"days": centers, "VictimCount": hist})

    smooth_dwell = st.slider("Smoothing (moving avg over bins)", 0, 15, 3, key="smooth_dwell")
    if smooth_dwell > 0 and len(dwell_df) > smooth_dwell:
        dwell_df["Smoothed"] = dwell_df["VictimCount"].rolling(window=smooth_dwell, center=True, min_periods=1).mean()

    base = alt.Chart(dwell_df).encode(x=alt.X("days:Q", title="Days"), tooltip=["days","VictimCount"])
    bars = base.mark_bar().encode(y=alt.Y("VictimCount:Q", title="# victim stays"))
    chart = bars if "Smoothed" not in dwell_df.columns else bars + alt.Chart(dwell_df).mark_line().encode(y="Smoothed:Q")
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", stay[["victim_id","location","date","next_date","days"]].to_csv(index=False).encode("utf-8"),
                       file_name="dwell_time_per_victim_location.csv", mime="text/csv")
else:
    st.info("Need victim_id + date + location for dwell time distribution.")

# 4.2 Trend — victims per year (victim’s first observed year)
st.subheader("5) Trend — victims per year (victim’s first observed year)")
if {"victim_id","date"}.issubset(df_add.columns) and df_add["date"].notna().any():
    first_seen = (df_add.dropna(subset=["victim_id","date"])
                         .sort_values(["victim_id","date"])
                         .groupby("victim_id")["date"].min().dt.year.value_counts().sort_index()
                         .reset_index())
    first_seen.columns = ["Year","NewVictims"]

    smooth_year = st.slider("Smoothing (years)", 0, 5, 2, key="smooth_year")
    plot_df = first_seen.copy()
    if smooth_year > 0 and len(plot_df) > smooth_year:
        plot_df["SMA"] = plot_df["NewVictims"].rolling(window=smooth_year, center=True, min_periods=1).mean()

    base = alt.Chart(plot_df).encode(x=alt.X("Year:O"))
    bars = base.mark_bar().encode(y=alt.Y("NewVictims:Q", title="Victims"))
    chart = bars if "SMA" not in plot_df.columns else bars + alt.Chart(plot_df).mark_line(point=True).encode(y="SMA:Q")
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", first_seen.to_csv(index=False).encode("utf-8"),
                       file_name="victims_per_year_first_seen.csv", mime="text/csv")
else:
    st.info("No usable dates to compute first-seen year.")

# 4.3 Month × Location heatmap (Top-10 locations by unique victims)
st.subheader("6) Month × Location heatmap (Top-10 locations by unique victims)")
if {"victim_id","location","year_month"}.issubset(df_add.columns) and df_add["location"].notna().any():
    top10 = df_add.groupby("location")["victim_id"].nunique().sort_values(ascending=False).head(10).index
    heat = (df_add[df_add["location"].isin(top10)]
             .groupby(["year_month","location"])["victim_id"].nunique()
             .reset_index(name="Victims"))
    heatmap = alt.Chart(heat).mark_rect().encode(
        x=alt.X("year_month:N", title="Month"),
        y=alt.Y("location:N", title="Location"),
        color=alt.Color("Victims:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["year_month","location","Victims"]
    ).properties(height=320)
    st.altair_chart(heatmap, use_container_width=True)
    st.download_button("⬇️ CSV", heat.to_csv(index=False).encode("utf-8"),
                       file_name="month_location_heatmap.csv", mime="text/csv")
else:
    st.info("Need (location, date) for this heatmap.")

st.divider()

# 5) Map & Flow Add-Ons --------------------------------------------------
st.subheader("7) Map & Flow Add-Ons")

# Gazetteer upload
with st.expander("Upload Gazetteer (CSV: location, lat, lon)", expanded=False):
    gzu = st.file_uploader("Gazetteer CSV", type=["csv"], key="gazetteer")
    gaz = None
    if gzu is not None:
        try:
            gaz = pd.read_csv(gzu)
            gaz.columns = [c.strip().lower() for c in gaz.columns]
            if not {"location","lat","lon"}.issubset(set(gaz.columns)):
                st.error("CSV must contain: location, lat, lon")
                gaz = None
        except Exception as e:
            st.error(f"Failed to read gazetteer: {e}")

geo_df = df_add.copy()
if gaz is not None and "location" in geo_df.columns:
    geo_df = geo_df.merge(gaz[["location","lat","lon"]], on="location", how="left", suffixes=("", "_gz"))
    if "lat_gz" in geo_df.columns:
        geo_df["lat"] = geo_df["lat"].where(geo_df["lat"].notna(), pd.to_numeric(geo_df["lat_gz"], errors="coerce"))
        geo_df["lon"] = geo_df["lon"].where(geo_df["lon"].notna(), pd.to_numeric(geo_df["lon_gz"], errors="coerce"))

if {"lat","lon"}.issubset(geo_df.columns) and geo_df[["lat","lon"]].notna().any().any():
    pts = geo_df.dropna(subset=["lat","lon"]).copy()
    pts["lat"] = pd.to_numeric(pts["lat"], errors="coerce")
    pts["lon"] = pd.to_numeric(pts["lon"], errors="coerce")
    pts = pts.dropna(subset=["lat","lon"])
    if not pts.empty:
        st.markdown("**Location markers**")
        fig_scatter = px.scatter_mapbox(pts, lat="lat", lon="lon", hover_name="location", zoom=2, height=420)
        fig_scatter.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("**Density (heat) map**")
        fig_heat = px.density_mapbox(pts, lat="lat", lon="lon", radius=18, zoom=2, height=420)
        fig_heat.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No coordinates to map yet. Upload a gazetteer above to enrich.")

st.markdown("**Movement Flow (Origin → Destination)**")
if {"victim_id","date","location"}.issubset(df_add.columns) and df_add["date"].notna().any():
    seq = (df_add.dropna(subset=["victim_id","date","location"])
                  .sort_values(["victim_id","date"])
                  .loc[:, ["victim_id","date","location"]])
    seq["next_location"] = seq.groupby("victim_id")["location"].shift(-1)
    flows = seq.dropna(subset=["next_location"]).groupby(["location","next_location"]).size().reset_index(name="count")
    topN = st.slider("Show top N flows", 10, 300, 60, key="flows_topN")
    flows = flows.sort_values("count", ascending=False).head(topN)
    if not flows.empty:
        nodes = pd.Index(pd.unique(flows[["location","next_location"]].values.ravel())).tolist()
        node_index = {n:i for i,n in enumerate(nodes)}
        sankey = go.Figure(data=[go.Sankey(
            node=dict(label=nodes, pad=12, thickness=16),
            link=dict(
                source=flows["location"].map(node_index),
                target=flows["next_location"].map(node_index),
                value=flows["count"]
            )
        )])
        sankey.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(sankey, use_container_width=True)
        st.download_button("⬇️ CSV", flows.to_csv(index=False).encode("utf-8"),
                           file_name="origin_destination_flows.csv", mime="text/csv")
    else:
        st.info("Not enough sequential data to infer flows.")
else:
    st.info("Need victim_id + date + location to build flows.")

st.divider()

# 6) Standardized fields reference & export (unchanged from old page) ---
st.subheader("Standardized fields used")
st.code(json.dumps(STD_FIELDS, indent=2), language="json")

st.subheader("Export all tables")
if st.button("📦 Export ZIP (CSV)", use_container_width=True):
    import zipfile, tempfile, os
    tmp = tempfile.TemporaryDirectory()
    zpath = os.path.join(tmp.name, "summary_export.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, tdf in tables.items():
            safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            csv_bytes = tdf.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{safe}.csv", csv_bytes)
    with open(zpath, "rb") as f:
        st.download_button("⬇️ Download ZIP", data=f.read(), file_name="summary_export.zip",
                           mime="application/zip", use_container_width=True)
