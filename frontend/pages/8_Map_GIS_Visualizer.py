# frontend/pages/8_Map_GIS_Visualizer.py
from __future__ import annotations

import ast
import json
import re
from typing import Iterable, List, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import Figure
from folium.plugins import Fullscreen, HeatMap, MarkerCluster, TimestampedGeoJson
from streamlit.components.v1 import html as st_html

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.gis_data import compute_location_stats, build_timestamped_geojson

# --- Resolver + diagnostics
from backend.geo.geo_utils import resolve_locations_to_coords
try:
    from backend.geo.geo_utils import match_report, bust_geo_caches  # type: ignore
except Exception:
    def match_report(locs: Iterable[str]):
        locs = list(locs or [])
        return {"total": len(locs), "matched": 0, "unmatched": len(locs)}
    def bust_geo_caches():
        pass

# --- Gazetteer ingestion (robust)
from backend.gis.gis_mapper import (
    ingest_geonames_zip as robust_ingest_zip,
    ingest_custom_gazetteer_csv as robust_ingest_csv,
)
from backend.geo.gazetteer import (
    list_gazetteers, ingest_geonames_tsv,
    set_active_gazetteer, get_active_gazetteer_id,
)

# ---------------------- Page ----------------------
st.set_page_config(page_title="GIS Map & Spatio-Temporal Visualizer",
                   page_icon="🗺️", layout="wide")
st.title("🗺️ GIS Map & Spatio-Temporal Visualizer")

with st.expander("📚 Gazetteer Manager (GeoNames/custom)", expanded=False):
    c1, c2, c3 = st.columns([2, 2, 3])

    with c1:
        gz_zip = st.file_uploader("GeoNames ZIP", type=["zip"], key="gz_zip")
        min_pop = st.number_input("Min population filter (optional)", min_value=0, value=0, step=1000)
        if gz_zip is not None and st.button("Ingest ZIP", use_container_width=True):
            try:
                gid = robust_ingest_zip(gz_zip, min_population=int(min_pop), title=f"GeoNames {gz_zip.name}")
                set_active_gazetteer(gid); bust_geo_caches(); st.success(f"Ingested & active: {gid}"); st.rerun()
            except Exception as e:
                st.error(f"Failed to ingest ZIP: {e}")

        st.caption("— or —")
        gz_tsv = st.file_uploader("GeoNames TXT/TSV", type=["txt", "tsv"], key="gz_tsv")
        if gz_tsv is not None and st.button("Ingest TXT/TSV", use_container_width=True):
            try:
                gid = ingest_geonames_tsv(gz_tsv, name=f"GeoNames {gz_tsv.name}", min_population=int(min_pop))
                set_active_gazetteer(gid); bust_geo_caches(); st.success(f"Ingested & active: {gid}"); st.rerun()
            except Exception as e:
                st.error(f"Failed to ingest TXT/TSV: {e}")

    with c2:
        custom = st.file_uploader("Custom Gazetteer CSV (name,lat,lon[,country,admin1,population])", type=["csv"], key="gz_csv")
        if custom is not None and st.button("Ingest Custom CSV", use_container_width=True):
            try:
                gid = robust_ingest_csv(custom, title=f"Custom {custom.name}")
                set_active_gazetteer(gid); bust_geo_caches(); st.success(f"Ingested & active: {gid}"); st.rerun()
            except Exception as e:
                st.error(f"Failed to ingest custom CSV: {e}")

    with c3:
        gaz_list = list_gazetteers()
        current_gid = get_active_gazetteer_id()
        st.markdown("**Available gazetteers:**")
        if not gaz_list:
            st.info("No gazetteers yet — upload one.")
        else:
            ids = [g["id"] for g in gaz_list]
            idx = ids.index(current_gid) if current_gid in ids else 0
            choice = st.selectbox("Active gazetteer", options=ids, index=idx,
                                  format_func=lambda gid: f"{[g for g in gaz_list if g['id']==gid][0]['name']} • {gid}")
            if st.button("Set active", use_container_width=True):
                set_active_gazetteer(choice); bust_geo_caches(); st.success(f"Active: {choice}"); st.rerun()

st.divider()

# ---------------------- Data sources ----------------------
st.subheader("1) Data sources")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged
if not queryable:
    st.info("No processed/merged datasets available."); st.stop()

selected = st.multiselect(
    "Select dataset(s) to visualize:",
    options=queryable,
    format_func=lambda e: f"{e.get('name')} • {e.get('kind')} • {e.get('id')}"
)
if not selected:
    st.warning("Select at least one dataset."); st.stop()

src_ids = [e["id"] for e in selected]
with st.spinner("Loading data..."):
    df = concat_processed_frames(src_ids)
if df is None or df.empty:
    st.error("Loaded dataframe is empty."); st.stop()

# ---------------------- Map options ----------------------
st.subheader("2) Map options")

CAND_LOC_COLS = [
    "City / Locations Crossed",
    "Locations (NLP)",
    "Location",
    "Final Location",
    "City/ Locations Crossed",
    "City / Locations",
    "City / Locations crossed",
]
cols_present = [c for c in CAND_LOC_COLS if c in df.columns]
default_loc_col = cols_present[0] if cols_present else df.columns[0]

place_col = st.selectbox(
    "Pick the column that contains places (city/country/region):",
    options=list(df.columns),
    index=list(df.columns).index(default_loc_col),
)

time_col = st.selectbox(
    "Column with 'days spent' per hop (optional):",
    options=["(none)"] + list(df.columns),
    index=(["(none)"] + list(df.columns)).index("Time Spent (days)") if "Time Spent (days)" in df.columns else 0,
)
default_days = st.number_input("Default days per hop (used when missing)", min_value=1, max_value=90, value=7, step=1)
animate = st.toggle("Animate trajectories", value=True)

# ---------------------- Lightweight preview helpers ----------------------
_QSTR = re.compile(r"""['"]([^'"]+)['"]""")

def _extract_preview(cell) -> List[str]:
    """Fast local preview extractor (mirrors backend parser behavior)."""
    from backend.api.gis_data import _parse_list_of_places  # reuse exact logic
    return _parse_list_of_places(cell)

with st.expander("🍪 Preview locations to resolve (first 30)", expanded=False):
    try:
        sample_vals = df[place_col].head(30).astype(str).tolist()
    except Exception:
        sample_vals = [str(df[place_col].head(30).values)]
    st.code(json.dumps(sample_vals, indent=2), language="json")

all_places: List[str] = []
for v in df[place_col].dropna().values.tolist():
    all_places.extend(_extract_preview(v))

with st.expander("🪴 Resolution sample (first 20)", expanded=True):
    sample = all_places[:20]
    res_rows: List[Tuple[str, float, float]] = []
    for s in sample:
        pt = resolve_locations_to_coords([s]).get(s)
        if pt is None:
            res_rows.append((s, np.nan, np.nan))
        else:
            res_rows.append((s, float(pt[0]), float(pt[1])))

    if res_rows:
        st.dataframe(pd.DataFrame(res_rows, columns=["location", "lat", "lon"]),
                     use_container_width=True, hide_index=True)
    rep = match_report(all_places)
    st.caption(f"Resolved locations (direct fuzzy): {rep['matched']} / {rep['total']} (unmatched: {rep['unmatched']}).")

# ---------------------- Build map layers ----------------------
if not all_places:
    st.warning(
        "No mappable locations found.\n\n"
        "• Verify the selected column has real place names or lists.\n"
        "• Ensure a gazetteer is ingested & active."
    )
    st.stop()

with st.spinner("Computing map layers..."):
    nodes_df, edges_df, _ = compute_location_stats(
        df=df,
        place_col=place_col,
        time_col=(None if time_col == "(none)" else time_col),
        default_days_per_hop=int(default_days),
    )

if nodes_df is None or nodes_df.empty:
    st.warning("No resolved coordinates after gazetteer matching.")
    st.stop()

min_lat, max_lat = nodes_df["lat"].min(), nodes_df["lat"].max()
min_lon, max_lon = nodes_df["lon"].min(), nodes_df["lon"].max()
center_lat = float((min_lat + max_lat) / 2.0)
center_lon = float((min_lon + max_lon) / 2.0)

fig = Figure(width="100%", height="760px")
m = folium.Map(location=[center_lat, center_lon], zoom_start=4,
               tiles="CartoDB dark_matter", control_scale=True)
fig.add_child(m)
Fullscreen().add_to(m)

# Marker cluster toggle
use_cluster = st.toggle("Marker cluster", value=True)
show_heatmap = st.toggle("Heatmap layer", value=False)

layer = MarkerCluster(name="Markers").add_to(m) if use_cluster else folium.FeatureGroup(name="Markers").add_to(m)

# Size by count
cmin, cmax = int(nodes_df["count"].min()), int(nodes_df["count"].max())
def _r(cnt: int) -> int:
    if cmax == cmin:
        return 8
    return int(8 + 20 * ((cnt - cmin) / max(1, (cmax - cmin))))

for _, row in nodes_df.iterrows():
    loc = row["location"]; lat = float(row["lat"]); lon = float(row["lon"]); cnt = int(row.get("count", 1))
    folium.CircleMarker(
        location=(lat, lon), radius=_r(cnt),
        color="#88F7C3", weight=1, fill=True, fill_opacity=0.35, fill_color="#88F7C3",
        tooltip=f"{loc} • {cnt} occurrences",
    ).add_to(layer)

if show_heatmap:
    heat_pts = [[float(r["lat"]), float(r["lon"]), float(r["count"])] for _, r in nodes_df.iterrows()]
    HeatMap(heat_pts, name="Heatmap", min_opacity=0.3, radius=25, blur=18, max_zoom=8).add_to(m)

# Animation
if animate:
    with st.spinner("Building animated trajectories..."):
        tj = build_timestamped_geojson(
            df=df,
            place_col=place_col,
            time_col=(None if time_col == "(none)" else time_col),
            default_days_per_hop=int(default_days),
            base_date="2020-01-01",
        )
    if tj and tj.get("features"):
        TimestampedGeoJson(
            data=tj,
            period="P1D",            # 1 day time step
            add_last_point=False,
            duration="P7D",          # visible trail length
            transition_time=200,
            loop=False,
            auto_play=False,
            max_speed=5,
            loop_button=True,
            date_options="YYYY-MM-DD",
            time_slider_drag_update=True,
        ).add_to(m)
    else:
        st.info("No trajectory segments to animate (insufficient geocodes or no timing info).")

folium.LayerControl(collapsed=False).add_to(m)
try:
    m.fit_bounds([[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]])
except Exception:
    pass

html = m.get_root().render()
st_html(html, height=780, scrolling=True)

st.download_button(
    "⬇️ Download map (HTML)",
    data=html.encode("utf-8"),
    file_name="gis_map.html",
    mime="text/html",
    use_container_width=True,
)
