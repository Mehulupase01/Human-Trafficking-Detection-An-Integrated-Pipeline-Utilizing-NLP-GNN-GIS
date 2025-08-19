# frontend/pages/4_Hierarchy_Viewer.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from streamlit.components.v1 import html as st_html

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.graph_build import build_network_from_processed
from backend.api.graph import export_png, export_pyvis_html
from backend.api.graph_hierarchy import build_victim_route_dag, export_dag_png

st.set_page_config(page_title="Network Graphs", page_icon="🕸️", layout="wide")
st.title("🕸️ Network Graphs (interactive)")

st.markdown("""
- **Interactive graph** with physics, tooltips, navigation buttons  
- **Legend**, **fullscreen** (browser’s full-screen), and **HTML/PNG export**  
- **Hierarchy (per-victim route)** view with PNG export
""")

# 1) choose datasets
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged
def _fmt(e): return f"{e.get('name')} • {e.get('kind')} • {e.get('id')}"
if not queryable:
    st.info("No processed/merged datasets available.")
    st.stop()
selected = st.multiselect("Datasets:", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()
src_ids = [e["id"] for e in selected]

with st.spinner("Loading data..."):
    df = concat_processed_frames(src_ids)

tab_net, tab_hier = st.tabs(["🌐 Network", "🧭 Victim Route Hierarchy"])

with tab_net:
    st.subheader("Network view")
    st.caption("Nodes: Victims, Locations, Perpetrators, Chiefs. Edges: visits, routes, links.")
    # Filters (light)
    colA, colB = st.columns([1,1])
    with colA:
        max_nodes = st.slider("Max nodes to render (sample if larger)", 100, 5000, 1500, 100)
    with colB:
        export_width = st.selectbox("PNG width", [1000, 1400, 1800], index=1)

    # Build graph
    G = build_network_from_processed(df)
    if G.number_of_nodes() == 0:
        st.info("Empty graph.")
    else:
        # sample if too big (preserve largest component)
        if G.number_of_nodes() > max_nodes:
            import networkx as nx
            comps = sorted(nx.connected_components(G), key=len, reverse=True)
            H = G.subgraph(comps[0]).copy()
            st.caption(f"Graph too large ({G.number_of_nodes()} nodes). Showing largest component with {H.number_of_nodes()} nodes.")
        else:
            H = G

        # Interactive HTML
        html = export_pyvis_html(H, height="740px")
        st_html(html, height=760, scrolling=True)

        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            png = export_png(H, width=export_width, height=900)
            st.download_button("⬇️ Download PNG", data=png, file_name="network.png", mime="image/png", use_container_width=True)
        with col2:
            st.download_button("⬇️ Download HTML", data=html.encode("utf-8"),
                               file_name="network.html", mime="text/html", use_container_width=True)

with tab_hier:
    st.subheader("Victim route hierarchy")
    victims = sorted(df["Serialized ID"].dropna().astype(str).unique().tolist()) if "Serialized ID" in df.columns else []
    if not victims:
        st.info("No victims available.")
    else:
        v = st.selectbox("Victim (Serialized ID)", options=victims)
        if v:
            H = build_victim_route_dag(df, v)
            if H.number_of_nodes() == 0:
                st.info("No route found for this victim.")
            else:
                png = export_dag_png(H, width=1200, height=800)
                st.image(png, caption="Route DAG (PNG preview)", use_column_width=True)
                st.download_button("⬇️ Download DAG PNG", data=png, file_name=f"route_{v}.png", mime="image/png", use_container_width=True)
