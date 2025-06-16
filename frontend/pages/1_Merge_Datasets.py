import streamlit as st
from backend.api.dataset_merge import merge_datasets

st.set_page_config(page_title="📂 Merge Datasets", layout="wide")
st.title("📂 Merge Multiple Datasets Chronologically")

uploaded_files = st.file_uploader("Upload multiple datasets", accept_multiple_files=True, type=["csv", "xlsx"])

if uploaded_files:
    if st.button("Merge and Preview"):
        merged_df = merge_datasets(uploaded_files)
        st.session_state["merged_df"] = merged_df
        st.success(f"Merged {len(uploaded_files)} files. Sorted by 'Left Home Country Year'.")
        st.dataframe(merged_df.head())

        st.download_button("📥 Download Merged Dataset", data=merged_df.to_csv(index=False).encode(), file_name="merged_dataset.csv", mime="text/csv")
