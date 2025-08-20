# frontend/pages/8_Map_GIS_Visualizer.py
from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen, HeatMap, MarkerCluster, TimestampedGeoJson
from streamlit.components.v1 import html as st_html
from branca.element import Figure, JavascriptLink, MacroElement, Template, Element

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.gis_data import compute_location_stats, build_timestamped_geojson

# Fuzzy resolver + diagnostics
from backend.geo.geo_utils import resolve_locations_to_coords, match_report

# Robust gazetteer ingesters (fixes float.strip / tokenizing errors)
from backend.gis.gis_mapper import (
    ingest_geonames_zip as robust_ingest_zip,
    ingest_custom_gazetteer_csv as robust_ingest_csv,
)

# List/set active gazetteers; optional TXT/TSV ingester
from backend.geo.gazetteer import (
    list_gazetteers,
    ingest_geonames_tsv,
    set_active_gazetteer,
    get_active_gazetteer_id,
)

# Explicit lookup helpers — import if available; else provide safe fallbacks
try:
    from backend.geo.geo_utils import save_geo_lookup_csv, list_geo_lookups  # type: ignore
except Exception:  # pragma: no cover
    def list_geo_lookups():
        """Fallback: pull any saved geo lookups from the registry."""
        return registry.find_datasets(kind="geo_lookup")

    def save_geo_lookup_csv(name: str, df: pd.DataFrame, owner: str | None = None):
        """Fallback: save a minimal (location,lat,lon) CSV into the registry as kind=geo_lookup."""
        cols = {c.lower(): c for c in df.columns}
        need = ["location", "lat", "lon"]
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError("Geo lookup CSV must have columns: location, lat, lon")
        slim = pd.DataFrame({
            "location": df[cols["location"]].astype(str),
            "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
            "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
        }).dropna()
        return registry.save_df(name=name, df=slim, kind="geo_lookup", owner=owner)

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
with st.expander("📚 Gazetteer Manager (GeoNames/custom)", expanded=True):
    left, mid, right = st.columns([2, 2, 3])

    with left:
        st.markdown("**Upload GeoNames** (`allCountries.zip`, `cities1000.zip`, `cities15000.zip`, etc., or an inner `.txt`):")
        gz_zip = st.file_uploader("GeoNames ZIP", type=["zip"], key="gz_zip")
        min_pop = st.number_input("Min population filter (optional)", min_value=0, value=0, step=1000)

        if gz_zip is not None and st.button("Ingest ZIP"):
            try:
                gid = robust_ingest_zip(
                    gz_zip,
                    min_population=int(min_pop),
                    title=f"GeoNames {gz_zip.name}",
                )
                set_active_gazetteer(gid)
                st.success(f"Ingested and set active: {gid}")
            except Exception as e:
                st.error(f"Failed to ingest ZIP: {e}")

        st.markdown("— or —")
        gz_tsv = st.file_uploader("GeoNames TXT/TSV", type=["txt", "tsv"], key="gz_tsv")
        if gz_tsv is not None and st.button("Ingest TXT/TSV"):
            try:
                gid = ingest_geonames_tsv(
                    gz_tsv,
                    name=f"GeoNames {gz_tsv.name}",
                    min_population=int(min_pop),
                )
                set_active_gazetteer(gid)
                st.success(f"Ingested and set active: {gid}")
            except Exception as e:
                st.error(f"Failed to ingest TSV: {e}")

    with mid:
        st.markdown("**Upload Custom CSV** (`name, lat, lon [, country, admin1, population]`):")
        custom = st.file_uploader("Custom Gazetteer CSV", type=["csv"], key="gz_csv")
        if custom is not None and st.button("Ingest Custom CSV"):
            try:
                gid = robust_ingest_csv(custom, title=f"Custom {custom.name}")
                set_active_gazetteer(gid)
                st.success(f"Ingested and set active: {gid}")
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
            if st.button("Set active gazetteer"):
                set_active_gazetteer(choice)
                st.success(f"Active gazetteer set to: {choice}")

st.divider()

st.markdown(
    """
**Map features**
- Dark basemap, zoom/pan, fullscreen
- Markers sized by **victim count**, with popups listing **Victims**, **Traffickers**, **Chiefs**, and **incoming/outgoing** counts
- **Animated trajectories** (time axis) and **heatmap / clustering** layers
- **In-browser PNG/PDF export** (Leaflet EasyPrint)
- **Overlay saved predictions & ETA runs**
"""
)

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

# ── Prediction overlay (optional)
st.subheader("Prediction overlay (optional)")
pred_runs = registry.find_datasets(kind="prediction_run")
pred_options = []
for pr in pred_runs:
    try:
        payload = registry.load_json(pr["id"])
        victim = payload.get("victim")
        pred_locs = payload.get("predicted_next_locations", [])
        pred_str = " → ".join([p.get("location", "?") for p in pred_locs[:3]])
        pred_options.append((f"{victim} : {pred_str}  •  {pr['id']}", pr["id"]))
    except Exception:
        continue

if pred_options:
    pred_lookup = {pid: label for (label, pid) in pred_options}
    sel_pred_ids = st.multiselect(
        "Select prediction runs to overlay:",
        options=list(pred_lookup.keys()),
        format_func=lambda pid: pred_lookup.get(pid, pid),
    )
else:
    sel_pred_ids = []
    st.caption("No saved prediction runs found yet.")

# ── ETA overlay (optional)
st.subheader("ETA overlay (optional)")
eta_runs = registry.find_datasets(kind="eta_run")
eta_options = []
for er in eta_runs:
    try:
        payload = registry.load_json(er["id"])
        victim = payload.get("victim")
        steps = payload.get("steps")
        eta_str = " + ".join([str(s.get("eta_days", "?")) for s in payload.get("steps_detail", [])][:3])
        eta_options.append((f"{victim} : {steps} steps • {eta_str} days  •  {er['id']}", er["id"]))
    except Exception:
        continue

if eta_options:
    eta_lookup = {pid: label for (label, pid) in eta_options}
    sel_eta_ids = st.multiselect(
        "Select ETA runs to overlay:",
        options=list(eta_lookup.keys()),
        format_func=lambda pid: eta_lookup.get(pid, pid),
    )
else:
    sel_eta_ids = []
    st.caption("No saved ETA runs found yet.")

# ───────────────────────── Explicit geo lookup (optional) ─────────────────────────
with st.expander("📍 (Optional) Upload an explicit Location → Lat/Lon mapping", expanded=False):
    st.markdown("Upload a CSV with columns: **location, lat, lon** (these override gazetteer matches).")
    up = st.file_uploader("Upload geo lookup CSV", type=["csv"])
    owner = st.text_input("Owner email (optional)", value="")
    if up is not None:
        try:
            df_geo = pd.read_csv(up)
            gid = save_geo_lookup_csv(name=f"Geo Lookup ({up.name})", df=df_geo, owner=(owner or None))
            st.success(f"Saved geo lookup: {gid}")
        except Exception as e:
            st.error(f"Failed to save geo lookup: {e}")
    prev = list_geo_lookups()
    if prev:
        st.caption(f"Found {len(prev)} explicit lookup tables (newest first). These take priority over gazetteer matches.")

st.divider()

# ───────────────────────── Map options ─────────────────────────
st.subheader("2) Map options")
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    use_cluster = st.toggle("Marker cluster", value=True)
with c2:
    show_heatmap = st.toggle("Heatmap layer", value=True)
with c3:
    default_days = st.number_input("Default days per hop (if unknown)", min_value=1, max_value=90, value=7, step=1)
with c4:
    animate = st.toggle("Animate trajectories", value=True)

st.caption("Use the **Export map** button (top-left) for PNG/PDF, or the **Download map (HTML)** button below.")

# ───────────────────────── Build data ─────────────────────────
with st.spinner("Loading & aggregating data..."):
    df = concat_processed_frames(src_ids)
    nodes_df, edges_df, loc_to_victims = compute_location_stats(df)

# quick feedback about resolution quality
rep = match_report(nodes_df["location"].tolist()) if not nodes_df.empty else {"total": 0, "matched": 0, "unmatched": 0}
st.caption(f"Resolved locations: {rep['matched']} / {rep['total']} (unmatched: {rep['unmatched']}).")

if nodes_df.empty:
    st.warning("No mappable locations found. Upload a gazetteer (GeoNames/custom) above, or an explicit lookup CSV.")
    st.stop()

# Fit to bounds
min_lat, max_lat = nodes_df["lat"].min(), nodes_df["lat"].max()
min_lon, max_lon = nodes_df["lon"].min(), nodes_df["lon"].max()
center_lat = (min_lat + max_lat) / 2.0
center_lon = (min_lon + max_lon) / 2.0

# Base map
fig = Figure(width="100%", height="720px")
m = folium.Map(location=[center_lat, center_lon],
               zoom_start=5,
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
if use_cluster:
    cluster = MarkerCluster(name="Markers").add_to(m)
else:
    cluster = folium.FeatureGroup(name="Markers").add_to(m)

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
    victims = row["victims"]
    perps = row["traffickers"]
    chiefs = row["chiefs"]
    incoming = int(row["incoming"])
    outgoing = int(row["outgoing"])
    count = int(row["count"])

    popup_html = f"""
    <div style="font-family:Inter,system-ui,Arial; font-size:12px; color:#eee; background:#1f2430; padding:10px; border-radius:8px; max-width:380px;">
      <div style="font-weight:600; font-size:13px; margin-bottom:6px;">{loc}</div>
      <div><b>Victims</b> ({len(victims)}): {', '.join(victims[:20])}{' ...' if len(victims) > 20 else ''}</div>
      <div><b>Traffickers</b> ({len(perps)}): {', '.join(perps[:20])}{' ...' if len(perps) > 20 else ''}</div>
      <div><b>Chiefs</b> ({len(chiefs)}): {', '.join(chiefs[:20])}{' ...' if len(chiefs) > 20 else ''}</div>
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

# Animated trajectories
if animate:
    with st.spinner("Building animated trajectories..."):
        tj = build_timestamped_geojson(df, default_days_per_hop=int(default_days), base_date="2020-01-01")
    if tj["features"]:
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
            name="Animated Trajectories",
        ).add_to(m)
    else:
        st.info("No trajectory segments to animate (insufficient geocodes).")

# Overlay predictions
if sel_pred_ids:
    pred_group = folium.FeatureGroup(name="Predicted Paths", show=True).add_to(m)
    for pid in sel_pred_ids:
        try:
            payload = registry.load_json(pid)
        except Exception:
            continue
        victim = payload.get("victim")
        seq = [d.get("location") for d in payload.get("predicted_next_locations", []) if isinstance(d, dict)]
        if not victim or not seq:
            continue
        sub = df[df["Serialized ID"] == victim].sort_values("Route_Order", kind="stable")
        if sub.empty:
            continue
        last_loc = str(sub.iloc[-1]["Location"])
        full_path = [last_loc] + [s for s in seq if isinstance(s, str) and s.strip()]
        try:
            coords_map = resolve_locations_to_coords(full_path)
        except Exception:
            coords_map = {}
        for a, b in zip(full_path, full_path[1:]):
            if a not in coords_map or b not in coords_map:
                continue
            lat1, lon1 = coords_map[a]
            lat2, lon2 = coords_map[b]
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color="#FFE082",
                weight=3,
                opacity=0.85,
                dash_array="6,6",
                tooltip=f"Predicted: {victim} • {a} → {b}",
            ).add_to(pred_group)

# Overlay ETA runs
if sel_eta_ids:
    eta_group = folium.FeatureGroup(name="ETA Paths", show=True).add_to(m)
    for eid in sel_eta_ids:
        try:
            payload = registry.load_json(eid)
        except Exception:
            continue
        victim = payload.get("victim")
        detail = payload.get("steps_detail", [])
        if not victim or not detail:
            continue
        sub = df[df["Serialized ID"] == victim].sort_values("Route_Order", kind="stable")
        if sub.empty:
            continue
        last_loc = str(sub.iloc[-1]["Location"])
        seq = [last_loc] + [d.get("to") for d in detail if isinstance(d, dict) and d.get("to")]
        try:
            coords_map = resolve_locations_to_coords(seq)
        except Exception:
            coords_map = {}
        for (a, b), step in zip(zip(seq, seq[1:]), detail):
            if a not in coords_map or b not in coords_map:
                continue
            lat1, lon1 = coords_map[a]
            lat2, lon2 = coords_map[b]
            days = step.get("eta_days", "?")
            weeks = step.get("eta_weeks", "?")
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color="#FFF59D",
                weight=4,
                opacity=0.9,
                dash_array="2,8",
                tooltip=f"ETA: {victim} • {a} → {b} ≈ {days}d (~{weeks}w)",
            ).add_to(eta_group)

# Controls & bounds
folium.LayerControl(collapsed=False).add_to(m)
try:
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
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
