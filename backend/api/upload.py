import streamlit as st
import pandas as pd
from backend.core.schema_check import validate_schema

def process_upload(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "File format not supported. Please upload .csv or .xlsx"

        valid, message = validate_schema(df)
        if not valid:
            return None, message

        return df, "File uploaded and validated successfully."
    except Exception as e:
        return None, f"Upload failed: {str(e)}"

import pandas as pd
from backend.api.upload import process_upload

if "role" in st.session_state:
    if st.session_state["role"] in ["Admin", "Data Owner"]:
        st.header("📤 Upload Cleaned Victim Interview Dataset")

        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file:
            df, status = process_upload(uploaded_file)
            if df is not None:
                st.success(status)
                st.dataframe(df.head())
            else:
                st.error(status)