# /frontend/pages/6_Temporal_Forecast.py
import streamlit as st
import pandas as pd
from backend.api.time_location_predict import predict_time_location

st.set_page_config(page_title="🔮 Temporal Forecast", layout="wide")
st.title("🔮 Predict Time of Arrival at Location")

if st.session_state.get("role") not in ["Admin", "Researcher"]:
    st.warning("Only Admin and Researcher roles can access this feature.")
    st.stop()

if "uploaded_df" not in st.session_state:
    st.info("Please upload a dataset first.")
    st.stop()

st.markdown("""
This tool estimates the likely year a victim could be present at various locations, based on patterns in your data.
""")

predictions = predict_time_location(st.session_state["uploaded_df"])

if isinstance(predictions, str):
    st.error(predictions)
else:
    st.success("Prediction complete.")
    st.dataframe(predictions.reset_index(drop=True))
