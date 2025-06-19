import streamlit as st
from backend.api.upload_tracker import get_upload_history
import pandas as pd

st.set_page_config(page_title="🗂️ Upload History", layout="centered")
st.title("🗂️ Dataset Upload History")

if st.session_state.get("role") not in ["Admin", "Researcher", "Viewer"]:
    st.warning("Access denied.")
    st.stop()

history = get_upload_history()

if history:
    df = pd.DataFrame(history)
    st.dataframe(df)
else:
    st.info("No upload history available.")