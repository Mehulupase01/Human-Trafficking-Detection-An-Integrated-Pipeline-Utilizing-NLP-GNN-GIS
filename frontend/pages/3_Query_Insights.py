import streamlit as st
import pandas as pd
from datetime import datetime
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

dataset_source = st.radio("Select Dataset Source", ["Uploaded Dataset", "Merged Dataset"])
df = None
if dataset_source == "Uploaded Dataset" and "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
elif dataset_source == "Merged Dataset" and "merged_df" in st.session_state:
    df = st.session_state["merged_df"]

if df is None:
    st.info("Please upload or merge a dataset first.")
    st.stop()
vt_map, tv_map = build_victim_trafficker_map(df)
trajectories = build_victim_trajectory(df)
country_crosses = get_countries_crossed(df)
origins, destinations = get_origin_and_destination(df)

# Initialize query history if not present
if "query_history" not in st.session_state:
    st.session_state.query_history = []

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

    st.session_state.query_history.append({
        "query_type": "Victim to Trafficker",
        "parameters": f"Victim ID: {selected_vid}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": st.session_state.get("email")
    })

elif query_type == "Trafficker → Victims":
    selected_tr = st.selectbox("Select Trafficker Name", sorted(tv_map.keys()))
    result = get_victims_by_trafficker(tv_map, selected_tr)
    st.success(f"Victims associated with trafficker '{selected_tr}':")
    st.write(result)

    st.session_state.query_history.append({
        "query_type": "Trafficker to Victims",
        "parameters": f"Trafficker: {selected_tr}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": st.session_state.get("email")
    })

elif query_type == "Victim → Trajectory":
    selected_vid = st.selectbox("Select Victim ID", list(trajectories.keys()))
    st.success(f"Trajectory of victim {selected_vid}:")
    st.write(trajectories[selected_vid])

    st.session_state.query_history.append({
        "query_type": "Victim Trajectory",
        "parameters": f"Victim ID: {selected_vid}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": st.session_state.get("email")
    })

elif query_type == "Victim → Borders/Countries Crossed":
    selected_vid = st.selectbox("Select Victim ID", list(country_crosses.keys()))
    st.success(f"Countries or borders crossed by victim {selected_vid}:")
    st.write(country_crosses[selected_vid])

    st.session_state.query_history.append({
        "query_type": "Borders Crossed",
        "parameters": f"Victim ID: {selected_vid}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": st.session_state.get("email")
    })

elif query_type == "Victim → Origin & Destination":
    selected_vid = st.selectbox("Select Victim ID", list(origins.keys()))
    st.success(f"Origin: {origins[selected_vid]}")
    st.success(f"Destination: {destinations[selected_vid]}")

    st.session_state.query_history.append({
        "query_type": "Origin & Destination",
        "parameters": f"Victim ID: {selected_vid}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": st.session_state.get("email")
    })
