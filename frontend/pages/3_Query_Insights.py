# frontend/pages/3_Query_Insights.py
from __future__ import annotations

# Ensure project root path / env
from _bootstrap import *  # noqa

import streamlit as st
import pandas as pd
import altair as alt

from backend.core import dataset_registry as registry
from backend.api.query import (explode_list_column, list_contains_any,
    join_for_display, join_for_csv,
)

st.set_page_config(page_title="Query Builder & Insights", page_icon="🔎", layout="wide")
st.title("🔎 Query Builder & Insights")

st.markdown("""
- **Processed/Merged** datasets only (standardized fields)
- **Any-gender**, searchable **Locations (NLP)**, text filters for **Perpetrators/Chiefs**
- **Saved queries**, **column picker**, and **insight charts** (Nationality, Route lengths, Top Locations)
""")

# Refresh button to reload registry data on demand
refresh = st.button("🔄 Refresh data from registry", help="Clear cache and reload")
if refresh:
    st.cache_data.clear()
    import time
    st.session_state["registry_bump"] = time.time()
@st.cache_data(show_spinner=False)

def _load_concat(ids, bump):
    frames = []
    for i in ids:
        try:
            frames.append(registry.load_df(i))
        except Exception as e:
            st.warning(f"Failed to load {i}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# ---------------- Load datasets ----------------
processed = registry.list_datasets(kind="processed")
merged    = registry.list_datasets(kind="merged")
queryable = processed + merged
def _fmt(e): return f"{e.get('name')} • {e.get('kind')} • {e.get('id')}"

if not queryable:
    st.info("No processed/merged datasets available.")
    st.stop()

selected = st.multiselect("Datasets to query:", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()

src_ids = [e["id"] for e in selected]

@st.cache_data(show_spinner=False)
def _load_concat(ids):
    frames = []
    for i in ids:
        try:
            frames.append(registry.load_df(i))
        except Exception as e:
            st.warning(f"Failed to load {i}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

with st.spinner("Loading data..."):
    df = _load_concat(src_ids)

if df.empty:
    st.info("No rows in selected datasets.")
    st.stop()

# Canonical fields present?
HAS_LOCS = "Locations (NLP)" in df.columns
HAS_PERP = "Perpetrators (NLP)" in df.columns
HAS_CHIE = "Chiefs (NLP)" in df.columns
HAS_GEND = "Gender of Victim" in df.columns
HAS_NAT  = "Nationality of Victim" in df.columns
HAS_SID  = "Serialized ID" in df.columns

# ---------------- Filters ----------------
st.subheader("Filters")

colA, colB, colC = st.columns([1,1,1])
with colA:
    # Always offer full set; filtering uses this value
    gender_options = ["Any", "Male", "Female", "Unknown"]
    chosen_gender = st.selectbox("Gender", options=gender_options, index=0)

with colB:
    nations = ["(Any)"]
    if HAS_NAT:
        nations += sorted(x for x in df["Nationality of Victim"].dropna().astype(str).unique().tolist() if x)
    chosen_nat = st.selectbox("Nationality", options=nations)

with colC:
    if HAS_LOCS:
        # build unique token list from Locations (NLP)
        loc_long = explode_list_column(df, "Locations (NLP)")
        toks = sorted(t for t in loc_long["Locations (NLP)"].dropna().astype(str).unique().tolist() if t)
        pick_locs = st.multiselect("Locations (NLP) (searchable)", options=toks)
    else:
        st.caption("No `Locations (NLP)` in data."); pick_locs = []

colD, colE, colF = st.columns([1,1,1])
with colD:
    perp_text = st.text_input("Perpetrators (comma-separated, match any)", value="")
with colE:
    chief_text = st.text_input("Chiefs (comma-separated, match any)", value="")
with colF:
    vic = st.selectbox("Victims (Serialized ID)", options=["(Any)"] + (sorted(df["Serialized ID"].astype(str).unique().tolist()) if HAS_SID else []))

# ---------------- Apply filters ----------------
mask = pd.Series([True]*len(df), index=df.index)

if HAS_GEND and chosen_gender != "Any":
    mask &= df["Gender of Victim"].astype(str).eq(chosen_gender)

if HAS_NAT and chosen_nat != "(Any)":
    mask &= df["Nationality of Victim"].astype(str).eq(chosen_nat)

if pick_locs and HAS_LOCS:
    # a row passes if its Locations (NLP) contains ANY of the selected tokens
    mask &= list_contains_any(df["Locations (NLP)"], pick_locs)

if perp_text.strip() and HAS_PERP:
    toks = [t.strip() for t in perp_text.split(",") if t.strip()]
    mask &= list_contains_any(df["Perpetrators (NLP)"], toks)

if chief_text.strip() and HAS_CHIE:
    toks = [t.strip() for t in chief_text.split(",") if t.strip()]
    mask &= list_contains_any(df["Chiefs (NLP)"], toks)

if HAS_SID and vic != "(Any)":
    mask &= df["Serialized ID"].astype(str).eq(vic)

filtered = df[mask].copy()
st.caption(f"Filtered rows: **{len(filtered):,}** / {len(df):,}")

st.divider()

# ---------------- Insights ----------------
st.subheader("Insights")

# Top Locations (explode tokens)
if HAS_LOCS:
    loc_long = explode_list_column(filtered, "Locations (NLP)")
    loc_counts = (loc_long["Locations (NLP)"]
                  .dropna()
                  .astype(str)
                  .value_counts()
                  .reset_index())
    loc_counts.columns = ["Location", "Count"]
else:
    loc_counts = pd.DataFrame(columns=["Location","Count"])

# Gender distribution
if HAS_GEND:
        g_counts = (
        df.groupby("Serialized ID")["Gender of Victim"].first()
        .value_counts()
        .reset_index()
    )
        g_counts.columns = ["Gender", "Count"]
       
else:
    g_counts = pd.DataFrame(columns=["Gender","Count"])

# Nationality top-20
if HAS_NAT:
    n_counts = (
        df.groupby("Serialized ID")["Nationality of Victim"].first()
        .value_counts()
        .head(20)
        .reset_index()
    )
    n_counts.columns = ["Nationality", "Count"]
else:
    n_counts = pd.DataFrame(columns=["Nationality","Count"])

# Route lengths per victim (bins)
if HAS_SID and "Route_Order" in filtered.columns:
    lens = (filtered.groupby("Serialized ID")["Route_Order"].max().reset_index(name="Stops"))
else:
    lens = pd.DataFrame(columns=["Serialized ID","Stops"])

c1, c2 = st.columns([1.2,1])
with c1:
    st.markdown("**Top Locations (by token frequency)**")
    if not loc_counts.empty:
        xmax = int(loc_counts["Count"].max() * 1.05)
        chart = alt.Chart(loc_counts.head(30)).mark_bar().encode(
            x=alt.X("Count:Q", title="Count", scale=alt.Scale(domain=[0, xmax], nice=True)),
            y=alt.Y("Location:N", sort='-x', title="Location"),
            tooltip=["Location:N","Count:Q"],
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No location tokens found.")

with c2:
    st.markdown("**Gender**")
    if not g_counts.empty:
        total = int(g_counts["Count"].sum())
        pie = alt.Chart(g_counts).mark_arc(innerRadius=60).encode(
            theta="Count:Q",
            color=alt.Color("Gender:N", legend=alt.Legend(title="Gender")),
            tooltip=["Gender:N","Count:Q"],
        ).properties(height=360, width=360)
        st.altair_chart(pie, use_container_width=True)
    else:
        st.caption("No gender data.")

c3, c4 = st.columns([1,1])
with c3:
    st.markdown("**Nationality (Top 20)**")
    if not n_counts.empty:
        xmax = int(n_counts["Count"].max() * 1.05)
        chart = alt.Chart(n_counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Count", scale=alt.Scale(domain=[0, xmax], nice=True)),
            y=alt.Y("Nationality:N", sort='-x', title="Nationality"),
            tooltip=["Nationality:N","Count:Q"],
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No nationality data.")

with c4:
    st.markdown("**Route lengths per victim**")
    if not lens.empty:
        chart = alt.Chart(lens).mark_bar().encode(
            x=alt.X("Stops:Q", bin=alt.Bin(maxbins=25), title="Stops (binned)"),
            y=alt.Y("count()", title="Victims"),
            tooltip=[alt.Tooltip("count()", title="Victims")],
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No route length data.")

st.divider()

# ---------------- Results ----------------
st.subheader("Results")

# Column picker defaults
all_cols = list(filtered.columns)
default_cols = [c for c in [
    "Serialized ID","Unique ID","Location","Route_Order",
    "Perpetrators (NLP)","Chiefs (NLP)","Locations (NLP)",
    "Gender of Victim","Nationality of Victim","Time Spent (days)","Time Spent (raw)"
] if c in all_cols]

cols_to_show = st.multiselect("Columns to display", options=all_cols, default=default_cols)

# Pagination
rpp = st.selectbox("Rows per page", options=[25,50,100,200], index=1)
max_page = max(1, (len(filtered) + rpp - 1) // rpp)
page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)

# Pretty rendering for list columns
disp = filtered.loc[:, cols_to_show].copy()
for c in ["Perpetrators (NLP)","Chiefs (NLP)","Locations (NLP)"]:
    if c in disp.columns:
        disp[c] = disp[c].apply(join_for_display)

start = (page-1)*rpp
st.write(f"Page {page} / {max_page}")
st.dataframe(disp.iloc[start:start+rpp], use_container_width=True, hide_index=True)

# Downloads
csv_df = filtered.loc[:, cols_to_show].copy()
for c in ["Perpetrators (NLP)","Chiefs (NLP)","Locations (NLP)"]:
    if c in csv_df.columns:
        csv_df[c] = csv_df[c].apply(join_for_csv)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("📥 Download (CSV)", data=csv_df.to_csv(index=False).encode("utf-8"),
                       file_name="query_results.csv", mime="text/csv", use_container_width=True)
with col_dl2:
    st.download_button("📥 Download (JSON)", data=filtered.loc[:, cols_to_show].to_json(orient="records").encode("utf-8"),
                       file_name="query_results.json", mime="application/json", use_container_width=True)
