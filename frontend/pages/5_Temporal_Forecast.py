import streamlit as st
import pandas as pd
from backend.api.time_location_predict import predict_time_location

st.set_page_config(page_title="🔮 Temporal Forecast", layout="wide")
st.title("🔮 Predict Time of Arrival at Location")

if st.session_state.get("role") not in ["Admin", "Researcher"]:
    st.warning("Only Admin and Researcher roles can access this feature.")
    st.stop()

# Dataset toggle
dataset_source = st.radio("Select Dataset Source", ["Uploaded Dataset", "Merged Dataset"])
df = None
if dataset_source == "Uploaded Dataset" and "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
elif dataset_source == "Merged Dataset" and "merged_df" in st.session_state:
    df = st.session_state["merged_df"]

if df is None:
    st.info("Please upload or merge a dataset first.")
    st.stop()

st.markdown("""
This tool estimates the likely year a victim could be present at various locations, based on patterns in your data.
""")

predictions = predict_time_location(df)

if isinstance(predictions, str):
    st.error(predictions)
else:
    st.success("Prediction complete.")
    st.dataframe(predictions.reset_index(drop=True))
