# frontend/pages/14_Summary_Dashboard.py
from __future__ import annotations
import json
import io

import pandas as pd
import altair as alt
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.metrics import (
    STD_FIELDS, core_counts, build_summary_tables,
)

st.set_page_config(page_title="Summary Dashboard", page_icon="📊", layout="wide")
st.title("📊 Summary Dashboard")

st.markdown("""
A quick, standardized overview across your **Processed** / **Merged** datasets:
- Core KPIs (victims, unique IDs, locations, traffickers, chiefs, route lengths)
- Top locations / traffickers / chiefs
- Gender & Nationality breakdowns
- Route length distribution per victim
""")

# --------- Data selection ----------
st.subheader("1) Choose datasets")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
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

ds_ids = [e["id"] for e in selected]

with st.spinner("Loading and aggregating..."):
    df = concat_processed_frames(ds_ids)

st.caption(f"**Rows:** {len(df):,} • **Columns:** {len(df.columns)}")

# --------- KPIs ----------
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

# --------- Tables & Charts ----------
st.subheader("3) Distributions & Rankings")

tables = build_summary_tables(df)

# Top Locations
t_top_loc = tables["Top Locations (by Victims)"]
left, right = st.columns(2)
with left:
    st.markdown("**Top Locations (by Victims)**")
    if not t_top_loc.empty:
        chart = alt.Chart(t_top_loc).mark_bar().encode(
            x=alt.X("Victim Count:Q"),
            y=alt.Y("Location:N", sort='-x'),
            tooltip=["Location:N", "Victim Count:Q"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.download_button("⬇️ CSV", t_top_loc.to_csv(index=False).encode("utf-8"),
                           file_name="top_locations.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No location stats available.")

# Top Traffickers
t_top_perp = tables["Top Traffickers (by Victims)"]
with right:
    st.markdown("**Top Traffickers (by Victims)**")
    if not t_top_perp.empty:
        chart = alt.Chart(t_top_perp).mark_bar().encode(
            x=alt.X("Victim Count:Q"),
            y=alt.Y("Perpetrator:N", sort='-x'),
            tooltip=["Perpetrator:N", "Victim Count:Q"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.download_button("⬇️ CSV", t_top_perp.to_csv(index=False).encode("utf-8"),
                           file_name="top_traffickers.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No perpetrator stats available.")

# Gender Distribution
g = tables["Gender Distribution"]
l2, r2 = st.columns(2)
with l2:
    st.markdown("**Gender Distribution**")
    if not g.empty:
        chart = alt.Chart(g).mark_arc(innerRadius=40).encode(
            theta="Count:Q",
            color=alt.Color("Gender:N"),
            tooltip=["Gender:N", "Count:Q"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.download_button("⬇️ CSV", g.to_csv(index=False).encode("utf-8"),
                           file_name="gender_distribution.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No gender field present.")

# Nationality
nat = tables["Nationality (Top 15)"]
with r2:
    st.markdown("**Nationality (Top 15)**")
    if not nat.empty:
        chart = alt.Chart(nat).mark_bar().encode(
            x=alt.X("Count:Q"),
            y=alt.Y("Nationality:N", sort='-x'),
            tooltip=["Nationality:N", "Count:Q"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.download_button("⬇️ CSV", nat.to_csv(index=False).encode("utf-8"),
                           file_name="nationality_top15.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No nationality field present.")

# Route lengths
st.markdown("**Route Lengths per Victim**")
rl = tables["Route Lengths (per Victim)"]
if not rl.empty:
    chart = alt.Chart(rl).mark_bar().encode(
        x=alt.X("Stops:Q", bin=alt.Bin(maxbins=30), title="Stops per victim"),
        y=alt.Y("count()", title="Victim count"),
        tooltip=[alt.Tooltip("count()", title="Victims")]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
    st.download_button("⬇️ CSV", rl.to_csv(index=False).encode("utf-8"),
                       file_name="route_lengths_per_victim.csv", mime="text/csv", use_container_width=True)
else:
    st.info("No route length information available.")

st.divider()

# --------- Standardized fields reference ----------
st.subheader("4) Standardized fields used")
st.code(json.dumps(STD_FIELDS, indent=2), language="json")

# --------- Export everything ----------
st.subheader("5) Export all tables")
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
