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

# Fuzzy resolver + diagnostics (safe fallbacks)
from backend.geo.geo_utils import resolve_locations_to_coords
try:
    from backend.geo.geo_utils import match_report, bust_geo_caches  # type: ignore
except Exception:
    def match_report(locs: Iterable[str]):
        locs = list(locs or [])
        return {"total": len(locs), "matched": 0, "unmatched": len(locs)}
    def bust_geo_caches():
        pass

# Robust gazetteer ingesters
from backend.gis.gis_mapper import (
    ingest_geonames_zip as robust_ingest_zip,
    ingest_custom_gazetteer_csv as robust_ingest_csv,
)

# Gazetteer list/set + optional TXT/TSV
from backend.geo.gazetteer import (
    list_gazetteers,
    ingest_geonames_tsv,
    set_active_gazetteer,
    get_active_gazetteer_id,
)

# Explicit lookup helpers — import if available; else provide fallbacks
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

st.markdown(
    """
**Now powered by an offline Gazetteer** 🚀  
Upload **GeoNames** (zip/txt) or a **custom CSV** (`name,lat,lon[,country,admin1,population]`).  
The app resolves locations automatically across any dataset — no fixed lists.
"""
)

# ───────────────────────── Gazetteer Manager ─────────────────────────
with st.expander("📚 Gazetteer Manager (GeoNames/custom)", expanded=False):
    left, mid, right = st.columns([2, 2, 3])

    # --- GeoNames ZIP/TXT
    with left:
        st.markdown("**Upload GeoNames** (`allCountries.zip`, `cities1000.zip`, `cities15000.zip`, etc., or an inner `.txt`):")
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

    # --- Custom CSV
    with mid:
        st.markdown("**Upload Custom CSV** (`name, lat, lon [, country, admin1, population]`):")
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

    # --- Available gazetteers + quick controls
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
            c1, c2 = st.columns([1,1])
            if c1.button("Set active", use_container_width=True):
                set_active_gazetteer(choice)
                bust_geo_caches()
                st.success(f"Active gazetteer set to: {choice}")
                st.rerun()
            if c2.button("Refresh caches", use_container_width=True):
                bust_geo_caches()
                st.success("Gazetteer caches cleared.")

st.divider()

# ───────────────────────── Data sources ─────────────────────────
st.subheader("1) Data sources")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged

def _fmt(e: dict) -> str:
    return f"{e.get('name')}  •  {e.get('kind')}  •  {e.get('id')}"

if not queryable:
    st.info("No processed or merged datasets available. Go process or merge data first.")
    st.stop()

selected = st.multiselect("Select dataset(s) to visualize:", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()
src_ids = [e["id"] for e in selected]

with st.spinner("Loading data..."):
    df = concat_processed_frames(src_ids)

if df is None or df.empty:
    st.error("Loaded dataframe is empty.")
    st.stop()

# ───────────────────────── Column selection for places ─────────────────────────
st.subheader("2) Map options")

CAND_LOC_COLS = [
    "City / Locations Crossed",
    "Location",
    "Locations (NLP)",
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

# Timing / animation
with st.expander("🌸 Trajectory timing (when you enable animation)", expanded=False):
    st.caption("If you don't have dates, we build a synthetic timeline by cumulatively adding 'Time Spent (days)'.")
    time_col = st.selectbox(
        "Column with 'days spent' per hop (optional):",
        options=["(none)"] + list(df.columns),
        index=(["(none)"] + list(df.columns)).index("Time Spent (days)") if "Time Spent (days)" in df.columns else 0,
    )
    default_days = st.number_input("Default days per hop (used when missing)", min_value=1, max_value=90, value=7, step=1)
    animate = st.toggle("Animate trajectories", value=True, help="Disable if you only want static markers/heatmap.")

# ── Very robust place extractor ───────────────────────────────────────────────
_QSTR = re.compile(r"""['"]([^'"]+)['"]""")

def extract_places(cell) -> List[str]:
    """
    Handles:
      - "Tripoli (Libya)"
      - ['Eritrea' 'Ethiopia' 'Hitsats']   ← *no commas*
      - "['Eritrea', 'Ethiopia', 'Hitsats']"
      - ["Eritrea","Ethiopia","Hitsats"]
      - stray free text → single token
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []

    # List-looking payloads
    if s.startswith("[") and s.endswith("]"):
        # normalize odd " ' '" gaps into "','"
        t = s.replace("' '", "', '").replace('" "', '", "')
        try:
            val = ast.literal_eval(t)
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
        # quoted tokens
        found = _QSTR.findall(t)
        if found:
            return [f.strip() for f in found if f.strip()]
        # last‑ditch: split inner on whitespace/commas
        inner = t[1:-1]
        inner = re.sub(r"[;,|]+", " ", inner)
        toks = [tok.strip() for tok in inner.split() if tok.strip()]
        return toks

    # Single token / free text
    return [s]

# Debug expander
with st.expander("🔎 Data debug (click to expand)", expanded=False):
    st.write("Dataframe shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(10), use_container_width=True)

# Preview raw values
with st.expander("🍪 Preview locations to resolve (first 30)", expanded=False):
    try:
        sample_vals = df[place_col].head(30).astype(str).tolist()
    except Exception:
        sample_vals = [str(df[place_col].head(30).values)]
    st.code(json.dumps(sample_vals, indent=2), language="json")

# Build a flat list of distinct places from the chosen column
all_places: List[str] = []
for v in df[place_col].dropna().values.tolist():
    all_places.extend(extract_places(v))

# Resolution sample table (first 20)
with st.expander("🪴 Resolution sample (first 20)", expanded=True):
    sample = all_places[:20]
    res_rows: List[Tuple[str, float, float]] = []
    for s in sample:
        pt = resolve_locations_to_coords([s]).get(s)
        if pt is None:
            res_rows.append((s, None, None))
        else:
            res_rows.append((s, float(pt[0]), float(pt[1])))
    if res_rows:
        st.dataframe(pd.DataFrame(res_rows, columns=["location", "lat", "lon"]),
                     use_container_width=True, hide_index=True)
    rep = match_report(all_places)
    st.caption(f"Resolved locations (direct fuzzy): {rep['matched']} / {rep['total']} (unmatched: {rep['unmatched']}).")

# Layer toggles
c1, c2 = st.columns([1, 1])
with c1:
    use_cluster = st.toggle("Marker cluster", value=True)
with c2:
    show_heatmap = st.toggle("Heatmap layer", value=False)

# If nothing resolvable, stop early
if not all_places:
    st.warning("No mappable locations found.\n\n• Check the Data debug expander above — verify the selected column actually contains place names or lists.\n• Ensure a gazetteer is ingested and active, or upload an explicit (location,lat,lon) CSV in the Gazetteer Manager.")
    st.stop()

# ───────────────────────── Build nodes/edges and (optionally) animation ─────────────────────────
with st.spinner("Computing map layers..."):
    # NOTE: do NOT pass unsupported kwargs; keep signature stable.
    nodes_df, edges_df, loc_to_victims = compute_location_stats(
        df=df,
        place_col=place_col,
        time_col=(None if time_col == "(none)" else time_col),
        default_days_per_hop=int(default_days),
    )

if nodes_df is None or nodes_df.empty:
    st.warning("No resolved coordinates after gazetteer matching. Double‑check the active gazetteer and the chosen column.")
    st.stop()

# Fit to bounds
min_lat, max_lat = nodes_df["lat"].min(), nodes_df["lat"].max()
min_lon, max_lon = nodes_df["lon"].min(), nodes_df["lon"].max()
center_lat = float((min_lat + max_lat) / 2.0)
center_lon = float((min_lon + max_lon) / 2.0)

# Base map
fig = Figure(width="100%", height="720px")
m = folium.Map(location=[center_lat, center_lon],
               zoom_start=4,
               tiles="CartoDB dark_matter",
               control_scale=True)
fig.add_child(m)

# Fullscreen
Fullscreen().add_to(m)

# Leaflet EasyPrint (PNG/PDF)
fig.header.add_child(JavascriptLink("https://unpkg.com/leaflet-easyprint@2.1.9/dist/bundle.js"))
easyprint_btn = Element(
    """
<script>
function addEasyPrint(map){
  L.easyPrint({
    title: 'Export map',
    position: 'topleft',
    sizeModes: ['Current', 'A4Landscape', 'A4Portrait'],
    exportOnly: true,
    hideControlContainer: false
  }).addTo(map);
}
</script>
"""
)
fig.header.add_child(easyprint_btn)
run_after = MacroElement()
run_after._template = Template(
    """
{% macro script(this, kwargs) %}
addEasyPrint({{this._parent.get_name()}});
{% endmacro %}
"""
)
m.add_child(run_after)

# Cluster or simple layer
cluster = MarkerCluster(name="Markers").add_to(m) if use_cluster else folium.FeatureGroup(name="Markers").add_to(m)

# Scale marker radius by victim count
min_c = int(nodes_df["count"].min())
max_c = int(nodes_df["count"].max())

def scale_radius(cnt: int) -> int:
    if max_c == min_c:
        return 8
    return int(8 + 20 * ((cnt - min_c) / max(1, (max_c - min_c))))

# Markers
for _, row in nodes_df.iterrows():
    loc = row["location"]
    lat = float(row["lat"])
    lon = float(row["lon"])
    victims = row.get("victims", []) or []
    perps = row.get("traffickers", []) or []
    chiefs = row.get("chiefs", []) or []
    incoming = int(row.get("incoming", 0))
    outgoing = int(row.get("outgoing", 0))
    count = int(row.get("count", 1))

    popup_html = f"""
    <div style="font-family:Inter,system-ui,Arial; font-size:12px; color:#eee; background:#1f2430; padding:10px; border-radius:8px; max-width:380px;">
      <div style="font-weight:600; font-size:13px; margin-bottom:6px;">{loc}</div>
      <div><b>Victims</b> ({len(victims)}): {', '.join(victims[:12])}{' ...' if len(victims) > 12 else ''}</div>
      <div><b>Traffickers</b> ({len(perps)}): {', '.join(perps[:12])}{' ...' if len(perps) > 12 else ''}</div>
      <div><b>Chiefs</b> ({len(chiefs)}): {', '.join(chiefs[:12])}{' ...' if len(chiefs) > 12 else ''}</div>
      <div style="margin-top:6px;"><b>Incoming</b>: {incoming} &nbsp; | &nbsp; <b>Outgoing</b>: {outgoing}</div>
      <div style="margin-top:6px;"><b>Victim count (node size)</b>: {count}</div>
    </div>
    """
    marker = folium.CircleMarker(
        location=(lat, lon),
        radius=scale_radius(count),
        color="#88F7C3",
        weight=1,
        fill=True,
        fill_opacity=0.35,
        fill_color="#88F7C3",
        tooltip=f"{loc} • {count} victims",
    )
    marker.add_child(folium.Popup(popup_html, max_width=420))
    marker.add_to(cluster)

# Heatmap
if show_heatmap:
    heat_pts = [[float(r["lat"]), float(r["lon"]), float(r["count"])] for _, r in nodes_df.iterrows()]
    HeatMap(heat_pts, name="Heatmap", min_opacity=0.3, radius=25, blur=18, max_zoom=8).add_to(m)

# Animated trajectories (TimestampedGeoJson)
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
    except TypeError as e:
        st.warning(f"Animation disabled: {e}")

# Controls & bounds
folium.LayerControl(collapsed=False).add_to(m)
try:
    m.fit_bounds([[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]])
except Exception:
    pass

# Render
html = m.get_root().render()
st_html(html, height=760, scrolling=True)

# Download HTML
st.download_button(
    "⬇️ Download map (HTML)",
    data=html.encode("utf-8"),
    file_name="gis_map.html",
    mime="text/html",
    use_container_width=True,
)
