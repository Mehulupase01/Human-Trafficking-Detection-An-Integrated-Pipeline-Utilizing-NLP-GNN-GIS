# frontend/pages/7_Trend_Graph_Explorer.py
from __future__ import annotations
import io
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames

st.set_page_config(page_title="Country-wise Victim Trends", page_icon="📈", layout="wide")
st.title("📈 Country‑wise Victim Trends Over Time")

# ---------------------- helpers ----------------------

def _first_token(x: object) -> Optional[str]:
    """Return the first meaningful token from list/array/str; None if empty."""
    if x is None:
        return None
    if isinstance(x, list) and x:
        v = x[0]
        s = "" if (pd.isna(v) if np.isscalar(v) else False) else str(v).strip()
        return s or None
    try:
        # numpy array
        if hasattr(x, "size") and getattr(x, "size", 0) > 0:
            v = x.tolist()[0]
            s = "" if (pd.isna(v) if np.isscalar(v) else False) else str(v).strip()
            return s or None
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip()
        return s or None
    if np.isscalar(x) and not pd.isna(x):
        s = str(x).strip()
        return s or None
    return None

def _derive_year_col(df: pd.DataFrame) -> pd.Series:
    """
    Robust year derivation:
      1) parse 'Date of Interview' -> .dt.year
      2) else 'Left Home Country Year' numeric
    """
    year = pd.Series([np.nan] * len(df), index=df.index, dtype="float")
    if "Date of Interview" in df.columns:
        d = pd.to_datetime(df["Date of Interview"], errors="coerce")
        year = d.dt.year.astype("float")
    if "Left Home Country Year" in df.columns:
        # fill missing with numeric LHCY
        y2 = pd.to_numeric(df["Left Home Country Year"], errors="coerce").astype("float")
        year = year.fillna(y2)
    return year

def _get_group_key(df: pd.DataFrame, by: str) -> pd.Series:
    """
    Group key:
      - Nationality -> 'Nationality of Victim' (cleaned string)
      - Location    -> first token from 'Locations (NLP)', fallback to 'Location'
    """
    if by == "Nationality":
        s = df.get("Nationality of Victim", pd.Series([None] * len(df)))
        return s.astype("string").str.strip().replace({"": pd.NA}).astype("string")
    # Location
    if "Locations (NLP)" in df.columns:
        s = df["Locations (NLP)"].apply(_first_token)
    else:
        s = pd.Series([None] * len(df))
    if s.isna().any() and "Location" in df.columns:
        mask = s.isna()
        s.loc[mask] = df.loc[mask, "Location"].apply(_first_token)
    return s.astype("string")

@st.cache_data(show_spinner=False)
def _aggregate_unique_victims(
    df: pd.DataFrame,
    by: str,
    year_min: Optional[int],
    year_max: Optional[int],
    choices: Optional[List[str]],
) -> pd.DataFrame:
    """
    Returns long table with columns: Year, Group, Victims (unique count).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Year", "Group", "Victims"])

    # columns we rely on
    sid_col = "Serialized ID"
    if sid_col not in df.columns:
        # Try a fallback if needed
        if "SerializedID" in df.columns:
            df = df.rename(columns={"SerializedID": sid_col})
        else:
            return pd.DataFrame(columns=["Year", "Group", "Victims"])

    # derive year
    year = _derive_year_col(df)
    df = df.assign(_Year=year)

    # group key
    df = df.assign(_Group=_get_group_key(df, by))

    # filter year range
    if year_min is not None:
        df = df[df["_Year"].fillna(-9e9) >= float(year_min)]
    if year_max is not None:
        df = df[df["_Year"].fillna(9e9) <= float(year_max)]

    # drop missing essentials
    df = df.dropna(subset=["_Year", sid_col])
    df["_Year"] = df["_Year"].astype(int)

    # optional filter by selected groups
    if choices:
        sel = pd.Series(choices, dtype="string").dropna().str.strip()
        df = df[df["_Group"].isin(sel)]

    if df.empty:
        return pd.DataFrame(columns=["Year", "Group", "Victims"])

    # unique victim count per (Year, Group)
    agg = (
        df.groupby(["_Year", "_Group"])[sid_col]
        .nunique(dropna=True)
        .reset_index(name="Victims")
        .rename(columns={"_Year": "Year", "_Group": "Group"})
        .sort_values(["Group", "Year"], kind="stable")
    )
    return agg

def _download_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _download_bytes_json(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_json(buf, orient="records", force_ascii=False, indent=2)
    return buf.getvalue().encode("utf-8")

# ---------------------- data source ----------------------

processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
available = processed + merged

def _fmt(ds: dict) -> str:
    return f"{ds.get('name')} • {ds.get('kind')} • {ds.get('id')}"

if not available:
    st.info("Please upload or merge datasets first.")
    st.stop()

picks = st.multiselect("Datasets to analyze:", options=available, format_func=_fmt)
if not picks:
    st.info("Select at least one dataset to proceed.")
    st.stop()

src_ids = [d["id"] for d in picks]
with st.spinner("Loading datasets..."):
    df = concat_processed_frames(src_ids)

if df is None or df.empty:
    st.warning("Selected datasets are empty after loading/merge.")
    st.stop()

# ---------------------- controls ----------------------

st.markdown("### Controls")

left, right = st.columns([2, 3])

with left:
    group_by = st.radio(
        "Trend by",
        options=["Nationality", "Location"],
        index=0,
        horizontal=True,
        help="Nationality: 'Nationality of Victim'. Location: first token from 'Locations (NLP)' (fallback to 'Location').",
    )

    # derive available groups
    groups_series = _get_group_key(df, group_by).dropna().astype("string")
    unique_groups = sorted(g for g in groups_series.unique() if g and g.lower() != "none")

    # search + multi select
    selected_groups = st.multiselect(
        f"Select {group_by}(s) (leave empty for all)",
        options=unique_groups,
    )

    # year range from data
    years_series = _derive_year_col(df).dropna().astype(int)
    if years_series.empty:
        st.error("No usable year information found (neither 'Date of Interview' nor 'Left Home Country Year').")
        st.stop()
    y_min, y_max = int(years_series.min()), int(years_series.max())
    yr1, yr2 = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max), step=1)

with right:
    smoothing = st.select_slider(
        "Smoothing (rolling mean)",
        options=[0, 3, 5],
        value=0,
        help="Apply a centered rolling mean over years (0 disables smoothing).",
    )
    cumulative = st.toggle("Cumulative totals", value=False)
    yscale = st.selectbox("Y‑axis scale", options=["linear", "log"], index=0)

# ---------------------- aggregate ----------------------

agg = _aggregate_unique_victims(df, by=group_by, year_min=yr1, year_max=yr2, choices=selected_groups)

if agg.empty:
    st.info("No data matched your filters.")
    st.stop()

# smoothing/cumulative
plot = agg.copy()
plot = plot.sort_values(["Group", "Year"], kind="stable")

if smoothing and smoothing > 0:
    plot["Victims"] = (
        plot.groupby("Group", group_keys=False)["Victims"]
        .apply(lambda s: s.rolling(window=smoothing, center=True, min_periods=1).mean())
        .round(2)
    )

if cumulative:
    plot["Victims"] = plot.groupby("Group", group_keys=False)["Victims"].cumsum()

# ---------------------- chart ----------------------

st.markdown("### Trends")

# Limit groups for clarity if none selected (top 10 by total victims in range)
if not selected_groups:
    totals = agg.groupby("Group")["Victims"].sum().sort_values(ascending=False)
    keep = set(totals.head(10).index)
    plot_display = plot[plot["Group"].isin(keep)]
    note = f"Showing top {min(10, len(totals))} {group_by.lower()}s by total victims in selected range."
    st.caption(note)
else:
    plot_display = plot

# Build Altair chart
base = alt.Chart(plot_display).encode(
    x=alt.X("Year:O", axis=alt.Axis(title="Year")),
    y=alt.Y("Victims:Q", scale=alt.Scale(type=yscale), axis=alt.Axis(title="Unique victims")),
    color=alt.Color("Group:N", legend=alt.Legend(title=group_by)),
    tooltip=[
        alt.Tooltip("Group:N", title=group_by),
        alt.Tooltip("Year:O"),
        alt.Tooltip("Victims:Q", format=",.0f"),
    ],
)

line = base.mark_line(point=True).properties(height=430)
st.altair_chart(line, use_container_width=True)

# ---------------------- table + downloads ----------------------

st.markdown("### Aggregated table")
st.dataframe(
    plot_display.sort_values(["Group", "Year"], kind="stable"),
    use_container_width=True,
    hide_index=True,
    height=360,
)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download table (CSV)",
        data=_download_bytes_csv(plot_display),
        file_name="victim_trends.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "⬇️ Download table (JSON)",
        data=_download_bytes_json(plot_display),
        file_name="victim_trends.json",
        mime="application/json",
        use_container_width=True,
    )

# ---------------------- notes ----------------------
with st.expander("Notes & definitions"):
    st.markdown(
        """
- **Unique victims** are counted by `Serialized ID`, not by rows.
- **Nationality** uses the standardized field *Nationality of Victim*.  
- **Location** uses the first token from *Locations (NLP)*; if absent, from *Location*.  
- **Year** prefers *Date of Interview → year*; fallback to *Left Home Country Year*.  
- If you leave the group selection empty, the chart shows the **top 10** groups in the selected year range.
"""
    )
