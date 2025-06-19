# /frontend/pages/8_Map_GIS_Visualizer.py

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from backend.api.gis import create_gis_map
from backend.api.nlp import run_nlp_pipeline

st.set_page_config(page_title="🗺️ GIS Map Viewer", layout="wide")
st.title("🗺️ Trafficking Routes GIS Map")

if st.session_state.get("role") not in ["Admin", "Researcher"]:
    st.warning("Only Admin and Researcher roles can access this feature.")
    st.stop()

# Dataset source toggle
dataset_source = st.radio("Select Dataset Source", ["Uploaded Dataset", "Merged Dataset"])
df = None
if dataset_source == "Uploaded Dataset" and "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
elif dataset_source == "Merged Dataset" and "merged_df" in st.session_state:
    df = st.session_state["merged_df"]

if df is None:
    st.info("Please upload or merge a dataset first.")
    st.stop()

st.markdown("Click the button below to generate a geographic visualization of victim routes based on structured entity extraction.")

if st.button("Generate GIS Map"):
    structured = run_nlp_pipeline(df)
    map_path = create_gis_map(structured)
    st.success("Map generated successfully!")
    components.html(open(map_path, "r", encoding="utf-8").read(), height=600)
