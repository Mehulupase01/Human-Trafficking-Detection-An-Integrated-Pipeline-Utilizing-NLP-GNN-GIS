# frontend/pages/4_Network_Graphs.py
from __future__ import annotations

from _bootstrap import *  # ensures PYTHONPATH + APP_DATA_DIR

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from backend.core import dataset_registry as registry
from backend.graph.graph_builder import build_network_pyvis, build_traceroute_pyvis

st.set_page_config(page_title="Network Graphs (interactive)", page_icon="🕸️", layout="wide")
st.title("🕸️ Network Graphs (interactive)")

st.markdown("""
- **Interactive graph** with physics, tooltips, and navigation controls.
- **Legend**, **fullscreen** (browser’s full-screen), and **HTML/PNG export** from the embedded viewer.
- **🧭 Victim Traceroute** shows the ordered path of **Locations (NLP)** with perpetrators per step.
""")

# ---------------- datasets ----------------
processed = registry.list_datasets(kind="processed")
merged    = registry.list_datasets(kind="merged")
choices = processed + merged

def _fmt(e): return f"{e.get('name')} • {e.get('kind')}"

if not choices:
    st.info("No processed/merged datasets available.")
    st.stop()

picked = st.multiselect("Datasets:", options=choices, format_func=_fmt, default=[choices[0]])
if not picked:
    st.warning("Select at least one dataset above.")
    st.stop()

ids = [e["id"] for e in picked]

@st.cache_data(show_spinner=False)
def _load(ids, bump):
    frames = []
    for i in ids:
        try:
            frames.append(registry.load_df(i))
        except Exception as err:
            st.warning(f"Failed to load {i}: {err}")
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

df = _load(ids, st.session_state.get("registry_bump", 0))

tab_net, tab_trace = st.tabs(["🌐 Network", "🧭 Victim Traceroute"])

# ---------------- 🌐 Network ----------------
with tab_net:
    st.subheader("Network view")
    st.caption("Nodes: **Locations** (and optionally Perpetrators/Victims). Edges: **Location→Location** transitions (weighted by # victims).")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        max_nodes = st.slider("Max nodes to render (sample if larger)", min_value=100, max_value=5000, value=1000, step=50)
    with c2:
        inc_perp = st.toggle("Include perpetrators", value=True)
    with c3:
        inc_vics = st.toggle("Include victims (heavy)", value=False)

    if df.empty:
        st.info("No rows found.")
    else:
        with st.spinner("Building network..."):
            html_path = build_network_pyvis(
                df,
                include_perpetrators=inc_perp,
                include_victims=inc_vics,
                max_nodes=max_nodes,
            )
        st.success("Graph ready.")
        # Use the new param (no deprecation)
        components.html(open(html_path, "r", encoding="utf-8").read(), height=720, scrolling=False)

# ---------------- 🧭 Victim Traceroute ----------------
with tab_trace:
    st.subheader("Victim traceroute")

    if "Serialized ID" not in df.columns:
        st.warning("Selected datasets have no `Serialized ID` column.")
    else:
        victims = sorted(df["Serialized ID"].astype(str).dropna().unique().tolist())
        sid = st.selectbox("Victim (Serialized ID)", options=victims)

        colA, colB = st.columns(2)
        with colA:
            collapse = st.toggle("Collapse repeated consecutive locations", value=True)
        with colB:
            pass

        if sid:
            with st.spinner("Building victim route..."):
                html_path = build_traceroute_pyvis(df, victim_sid=sid, collapse_repeats=collapse)
            st.success("Traceroute ready.")
            components.html(open(html_path, "r", encoding="utf-8").read(), height=720, scrolling=False)
