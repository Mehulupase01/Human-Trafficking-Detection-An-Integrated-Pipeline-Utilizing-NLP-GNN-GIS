import streamlit as st
import pandas as pd
from backend.api.graph_queries import (
    build_victim_trafficker_map,
    get_victims_by_trafficker,
    get_traffickers_by_victim,
    build_victim_trajectory,
    get_countries_crossed,
    get_origin_and_destination
)

st.set_page_config(page_title="🔍 Graph Query Insights", layout="wide")
st.title("🔍 Query Network: Victims, Traffickers & Routes")

if st.session_state.get("role") not in ["Admin", "Researcher", "Viewer"]:
    st.warning("Access restricted to Admin, Researcher, and Viewer roles.")
    st.stop()

if "uploaded_df" not in st.session_state:
    st.info("Please upload a dataset first.")
    st.stop()

df = st.session_state["uploaded_df"]
vt_map, tv_map = build_victim_trafficker_map(df)
trajectories = build_victim_trajectory(df)
country_crosses = get_countries_crossed(df)
origins, destinations = get_origin_and_destination(df)

# Select Query Type
query_type = st.radio("Choose Query Type", [
    "Victim → Traffickers",
    "Trafficker → Victims",
    "Victim → Trajectory",
    "Victim → Borders/Countries Crossed",
    "Victim → Origin & Destination"
])

if query_type == "Victim → Traffickers":
    selected_vid = st.selectbox("Select Victim ID", list(vt_map.keys()))
    result = get_traffickers_by_victim(vt_map, selected_vid)
    st.success(f"Traffickers associated with victim {selected_vid}:")
    st.write(result)

elif query_type == "Trafficker → Victims":
    selected_tr = st.selectbox("Select Trafficker Name", sorted(tv_map.keys()))
    result = get_victims_by_trafficker(tv_map, selected_tr)
    st.success(f"Victims associated with trafficker '{selected_tr}':")
    st.write(result)

elif query_type == "Victim → Trajectory":
    selected_vid = st.selectbox("Select Victim ID", list(trajectories.keys()))
    st.success(f"Trajectory of victim {selected_vid}:")
    st.write(trajectories[selected_vid])

elif query_type == "Victim → Borders/Countries Crossed":
    selected_vid = st.selectbox("Select Victim ID", list(country_crosses.keys()))
    st.success(f"Countries or borders crossed by victim {selected_vid}:")
    st.write(country_crosses[selected_vid])

elif query_type == "Victim → Origin & Destination":
    selected_vid = st.selectbox("Select Victim ID", list(origins.keys()))
    st.success(f"Origin: {origins[selected_vid]}")
    st.success(f"Destination: {destinations[selected_vid]}")
