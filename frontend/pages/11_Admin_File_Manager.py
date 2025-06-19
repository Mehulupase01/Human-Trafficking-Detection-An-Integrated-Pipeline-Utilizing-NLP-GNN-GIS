# /frontend/pages/11_Admin_File_Manager.py

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="🗂️ Admin File Manager", layout="wide")
st.title("🗂️ Admin File Manager")

# Role restriction
if st.session_state.get("role") != "Admin":
    st.warning("Access denied: This page is only for Admins.")
    st.stop()

# Initialize upload history if not already present
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

# Display current uploads
if st.session_state.upload_history:
    st.markdown("### 📄 Uploaded Datasets")
    df = pd.DataFrame(st.session_state.upload_history)
    st.dataframe(df)

    selected_files = st.multiselect("Select files to delete", df["file_name"].tolist())

    if st.button("Delete Selected Files"):
        st.session_state.upload_history = [
            log for log in st.session_state.upload_history if log["file_name"] not in selected_files
        ]
        st.success(f"Deleted {len(selected_files)} file(s).")
        st.experimental_rerun()
else:
    st.info("No uploaded files found.")
