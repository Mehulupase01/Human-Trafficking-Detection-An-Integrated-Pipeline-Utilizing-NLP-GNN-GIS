# frontend/pages/11_Admin_File_Manager.py
from __future__ import annotations
import io
import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames

st.set_page_config(page_title="Admin File Manager", page_icon="🗃️", layout="wide")
st.title("🗃️ Admin File Manager")

st.caption("Quick overview & download helpers for datasets saved in the registry.")

processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")

if not (processed or merged):
    st.info("No uploaded files found.")
    st.stop()

tab1, tab2 = st.tabs(["Processed", "Merged"])

def _download_df_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

with tab1:
    if not processed:
        st.caption("— none —")
    else:
        # pick and preview
        choice = st.selectbox(
            "Pick a processed dataset",
            options=processed,
            format_func=lambda d: f"{d.get('name')} • {d.get('id')}",
        )
        if choice:
            try:
                df = concat_processed_frames([choice["id"]])
            except Exception:
                df = None
            if df is None or df.empty:
                st.warning("Could not load or empty.")
            else:
                st.dataframe(df.head(200), use_container_width=True, height=360)
                st.download_button(
                    "⬇️ Download full CSV",
                    data=_download_df_bytes(df),
                    file_name=f"{choice.get('name','processed')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

with tab2:
    if not merged:
        st.caption("— none —")
    else:
        choice = st.selectbox(
            "Pick a merged dataset",
            options=merged,
            format_func=lambda d: f"{d.get('name')} • {d.get('id')}",
        )
        if choice:
            try:
                df = concat_processed_frames([choice["id"]])
            except Exception:
                df = None
            if df is None or df.empty:
                st.warning("Could not load or empty.")
            else:
                st.dataframe(df.head(200), use_container_width=True, height=360)
                st.download_button(
                    "⬇️ Download full CSV",
                    data=_download_df_bytes(df),
                    file_name=f"{choice.get('name','merged')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
