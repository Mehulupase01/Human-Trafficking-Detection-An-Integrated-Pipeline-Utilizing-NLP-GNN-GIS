import streamlit as st
import pandas as pd
from backend.core.schema_check import validate_schema

REQUIRED_COLUMNS = [
    "Unique ID",
    "Interviewer Name",
    "Date of Interview",
    "Gender of Victim",
    "Nationality of Victim",
    "Left Home Country Year",
    "Borders Crossed",
    "City / Locations Crossed",
    "Final Location",
    "Name of the Perpetrators involved",
    "Hierarchy of Perpetrators",
    "Human traffickers/ Chief of places",
    "Time Spent in Location / Cities / Places"
]

def validate_schema(df):
    df.columns = [c.strip() for c in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    return True, "Schema validated."

def process_upload(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        if "Unique ID" in df.columns:
            df["Unique ID"] = df["Unique ID"].astype(str)
        else:
            return None, "❌ Unsupported file format. Upload .csv or .xlsx only."

        df.columns = [col.strip() for col in df.columns]
        valid, message = validate_schema(df)
        if not valid:
            return None, message

        return df, "✅ File uploaded and validated successfully."
    except Exception as e:
        return None, f"❌ Upload failed: {str(e)}"
