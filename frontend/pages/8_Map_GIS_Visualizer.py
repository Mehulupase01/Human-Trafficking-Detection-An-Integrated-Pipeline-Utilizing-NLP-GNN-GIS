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
from branca.element import Element, Figure, JavascriptLink, MacroElement, Template
from folium.plugins import Fullscreen, HeatMap, MarkerCluster, TimestampedGeoJson
from streamlit.components.v1 import html as st_html

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.gis_data import compute_location_stats, build_timestamped_geojson

# --- Fuzzy resolver + diagnostics (imports are safe even if extra functions are missing)
from backend.geo.geo_utils import resolve_locations_to_coords
try:
    from backend.geo.geo_utils import match_report, bust_geo_caches  # type: ignore
except Exception:
    def match_report(locs: Iterable[str]):
        locs = list(locs or [])
        return {"total": len(locs), "matched": 0, "unmatched": len(locs)}
    def bust_geo_caches():
        pass

# --- Robust gazetteer ingesters
from backend.gis.gis_mapper import (
    ingest_geonames_zip as robust_ingest_zip,
    ingest_custom_gazetteer_csv as robust_ingest_csv,
)

# --- Gazetteer list/set + optional TXT/TSV ingest
from backend.geo.gazetteer import (
    list_gazetteers,
    ingest_geonames_tsv,
    set_active_gazetteer,
    get_active_gazetteer_id,
)

# --- Explicit lookup helpers
try:
    from backend.geo.geo_utils import save_geo_lookup_csv, list_geo_lookups  # type: ignore
except Exception:  # pragma: no cover
    def list_geo_lookups():
        return registry.find_datasets(kind="geo_lookup")

    def save_geo_lookup_csv(name: str, df: pd.DataFrame, owner: str | None = None):
        cols = {c.lower(): c for c in df.columns}
        need = ["location", "lat", "lon"]
        for c in need:
            if c not in cols:
                raise ValueError("Geo lookup CSV must have columns: location, lat, lon")
        slim = pd.DataFrame({
            "location": df[cols["location"]].astype(str),
            "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
            "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
        }).dropna()
        return registry.save_df(name=name, df=slim, kind="geo_lookup", owner=owner)

# ---------------------- Page ----------------------
st.set_page_config(page_title="GIS Map & Spatio-Temporal Visualizer",
                   page_icon="🗺️", layout="wide")
st.title("🗺️ GIS Map & Spatio-Temporal Visualizer")

# ---------------------- Gazetteer Manager ----------------------
with st.expander("📚 Gazetteer Manager (GeoNames/custom)", expanded=False):
    left, mid, right = st.columns([2, 2, 3])

    with left:
        gz_zip = st.file_uploader("GeoNames ZIP", type=["zip"], key="gz_zip")
        min_pop = st.number_input("Min population filter (optional)", min_value=0, value=0, step=1000)
        if gz_zip is not None and st.button("Ingest ZIP", use_container_width=True):
            try:
                gid = robust_ingest_zip(
                    gz_zip,
                    min_population=int(min_pop),
                    title=f"GeoNames {gz_zip.name}",
                )
                set_active_gazetteer(gid)
                bust_geo_caches()
                st.success(f"Ingested and set active: {gid}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to ingest ZIP: {e}")

        st.markdown("— or —")
        gz_tsv = st.file_uploader("GeoNames TXT/TSV", type=["txt", "tsv"], key="gz_tsv")
        if gz_tsv is not None and st.button("Ingest TXT/TSV", use_container_width=True):
            try:
                gid = ingest_geonames_tsv(
                    gz_tsv,
                    name=f"GeoNames {gz_tsv.name}",
                    min_population=int(min_pop),
                )
                set_active_gazetteer(gid)
                bust_geo_caches()
                st.success(f"Ingested and set active: {gid}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to ingest TSV: {e}")

    with mid:
        custom = st.file_uploader("Custom Gazetteer CSV", type=["csv"], key="gz_csv")
        if custom is not None and st.button("Ingest Custom CSV", use_container_width=True):
            try:
                gid = robust_ingest_csv(custom, title=f"Custom {custom.name}")
                set_active_gazetteer(gid)
                bust_geo_caches()
                st.success(f"Ingested and set active: {gid}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to ingest custom CSV: {e}")

    with right:
        gaz_list = list_gazetteers()
        current_gid = get_active_gazetteer_id()
        st.markdown("**Available gazetteers (newest first):**")
        if not gaz_list:
            st.info("No gazetteers yet — upload one on the left.")
        else:
            ids = [g["id"] for g in gaz_list]
            try:
                default_index = ids.index(current_gid) if current_gid in ids else 0
            except Exception:
                default_index = 0
            choice = st.selectbox(
                "Active gazetteer",
                options=ids,
                index=default_index,
                format_func=lambda gid: f"{[g for g in gaz_list if g['id']==gid][0]['name']} • {gid}",
            )
            if st.button("Set active", use_container_width=True):
                set_active_gazetteer(choice)
                bust_geo_caches()
                st.success(f"Active gazetteer set to: {choice}")
                st.rerun()

st.divider()

# ---------------------- Data sources ----------------------
st.subheader("1) Data sources")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged
if not queryable:
    st.info("No processed or merged datasets available. Go process or merge data first.")
    st.stop()

selected = st.multiselect("Select dataset(s) to visualize:", options=queryable,
                          format_func=lambda e: f"{e.get('name')} • {e.get('kind')} • {e.get('id')}")
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()
src_ids = [e["id"] for e in selected]

with st.spinner("Loading data..."):
    df = concat_processed_frames(src_ids)

if df is None or df.empty:
    st.error("Loaded dataframe is empty.")
    st.stop()

# ---------------------- Column selection ----------------------
st.subheader("2) Map options")
CAND_LOC_COLS = [
    "Location",
    "Locations (NLP)",
    "City / Locations Crossed",
    "Final Location",
]
cols_present = [c for c in CAND_LOC_COLS if c in df.columns]
default_loc_col = cols_present[0] if cols_present else df.columns[0]
place_col = st.selectbox("Pick the column that contains places (city/country/region):",
                         options=list(df.columns),
                         index=list(df.columns).index(default_loc_col))

# Timing
time_col = st.selectbox("Column with 'days spent' per hop (optional):",
                        options=["(none)"] + list(df.columns),
                        index=(["(none)"] + list(df.columns)).index("Time Spent (days)") if "Time Spent (days)" in df.columns else 0)
default_days = st.number_input("Default days per hop (used when missing)", min_value=1, max_value=90, value=7, step=1)
animate = st.toggle("Animate trajectories", value=True)

# ---------------------- Local parser (_extract) ----------------------
_QSTR = re.compile(r"""['"]([^'"]+)['"]""")

def _extract(cell):
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
        found = _QSTR.findall(s)
        if found:
            return [t.strip() for t in found if t.strip()]
        inner = s[1:-1].replace(",", " ").strip()
        return [t for t in inner.split() if t]
    return [s]

# ---------------------- Preview ----------------------
with st.expander("🍪 Preview locations to resolve (first 30)", expanded=False):
    try:
        sample_vals = df[place_col].head(30).astype(str).tolist()
    except Exception:
        sample_vals = [str(df[place_col].head(30).values)]
    st.code(json.dumps(sample_vals, indent=2), language="json")

all_places: List[str] = []
for v in df[place_col].dropna().values.tolist():
    all_places.extend(_extract(v))

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

# ---------------------- Map generation ----------------------
if not all_places:
    st.warning("No mappable locations found.")
    st.stop()

with st.spinner("Computing map layers..."):
    nodes_df, edges_df, loc_to_victims = compute_location_stats(
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

fig = Figure(width="100%", height="720px")
m = folium.Map(location=[center_lat, center_lon],
               zoom_start=4,
               tiles="CartoDB dark_matter",
               control_scale=True)
fig.add_child(m)
Fullscreen().add_to(m)

# Heatmap + cluster
cluster = MarkerCluster(name="Markers").add_to(m) if True else folium.FeatureGroup(name="Markers").add_to(m)

for _, row in nodes_df.iterrows():
    loc = row["location"]
    lat = float(row["lat"])
    lon = float(row["lon"])
    count = int(row.get("count", 1))
    marker = folium.CircleMarker(
        location=(lat, lon),
        radius=6,
        color="#88F7C3",
        weight=1,
        fill=True,
        fill_opacity=0.35,
        fill_color="#88F7C3",
        tooltip=f"{loc} • {count} victims",
    )
    marker.add_to(cluster)

if animate:
    try:
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
                period="P1D",
                add_last_point=False,
                duration="P7D",
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
    except Exception as e:
        st.warning(f"Animation disabled: {e}")

folium.LayerControl(collapsed=False).add_to(m)
try:
    m.fit_bounds([[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]])
except Exception:
    pass

html = m.get_root().render()
st_html(html, height=760, scrolling=True)
st.download_button("⬇️ Download map (HTML)", data=html.encode("utf-8"),
                   file_name="gis_map.html", mime="text/html", use_container_width=True)
