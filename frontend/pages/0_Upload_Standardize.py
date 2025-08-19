# frontend/pages/0_Upload_Standardize.py
from __future__ import annotations
import time
from _bootstrap import *  # noqa
# --- Make sure project root is on sys.path BEFORE any backend imports ---
import os, sys
THIS_DIR = os.path.abspath(os.path.dirname(__file__))                   # .../frontend/pages
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))      # repo root
FRONTEND_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))            # .../frontend

for p in (PROJECT_ROOT, FRONTEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional: load .env if you use it
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import streamlit as st
import pandas as pd

# Try imports now that sys.path is fixed
try:
    from backend.api.upload import process_and_save, read_files, process_dataframe
    from backend.core import dataset_registry as registry
except Exception as e:
    st.error(f"Import error: {e}")
    # Helpful diagnostics
    import importlib.util
    def _find(name):
        spec = importlib.util.find_spec(name)
        return spec.origin if spec else "NOT FOUND"
    with st.expander("Diagnostics"):
        st.write("PROJECT_ROOT:", PROJECT_ROOT)
        st.write("FRONTEND_DIR:", FRONTEND_DIR)
        st.write("backend ->", _find("backend"))
        st.write("backend.api.upload ->", _find("backend.api.upload"))
        st.write("backend.nlp.entity_extraction ->", _find("backend.nlp.entity_extraction"))
        st.write("nlp.entity_extraction ->", _find("nlp.entity_extraction"))
        st.write("sys.path:", sys.path)
    st.stop()

st.set_page_config(page_title="Upload & Standardize", page_icon="📤", layout="wide")
st.title("📤 Upload & Standardize")

st.markdown("""
Upload your CSV / Excel / JSON files, **standardize** them into the app's canonical *Processed* schema,
and save them to the registry for use across all pages (Query, Graphs, GIS, Predictions, etc.).
""")

# -------- Settings --------
colA, colB = st.columns([1,1])
with colA:
    dataset_name = st.text_input("Dataset name", value="Uploaded dataset")
    owner = st.text_input("Owner email (optional)", value="")
with colB:
    already_processed = st.checkbox("Files are already processed (skip entity extraction)", value=False)
    extract_from_text = st.checkbox("Extract entities from narrative text (heuristic)", value=not already_processed, disabled=already_processed)
    overwrite_entities = st.checkbox("Overwrite existing entity columns if present", value=True, disabled=already_processed)


st.divider()

# -------- Files --------
files = st.file_uploader(
    "Choose one or more files (CSV, XLSX/XLS, JSON)",
    type=["csv", "xlsx", "xls", "json"],
    accept_multiple_files=True,
    key="upload_files",
)
if not files:
    st.info("Select at least one file to continue.")
    st.stop()

# -------- Preview (client-side) --------
with st.spinner("Reading file(s)..."):
    try:
        raw = read_files(files)
    except Exception as e:
        st.error(str(e))
        st.stop()

st.caption(f"Loaded **{len(raw):,}** rows from **{len(files)}** file(s).")
st.dataframe(raw.head(50), use_container_width=True, hide_index=True, height=300)

# Preview standardized (without saving yet)

with st.expander("🔍 Preview standardized output (first 100 rows)", expanded=False):
    try:
        prev = process_dataframe(
            raw,
            already_processed=already_processed,
            extract_from_text=extract_from_text,
            overwrite_entities=overwrite_entities,
        )
        st.dataframe(prev.head(100), use_container_width=True, hide_index=True, height=360)
        st.download_button(
            "⬇️ Download preview (CSV)",
            data=prev.to_csv(index=False).encode("utf-8"),
            file_name="standardized_preview.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Standardization preview failed: {e}")

st.divider()

with st.expander("✅ NLP check — single-word tokens (preview)", expanded=True):
    cols = [c for c in [
        "Locations (NLP)", "Perpetrators (NLP)", "Chiefs (NLP)",
        "Gender of Victim", "Nationality of Victim",
        "Time Spent (days)", "Time Spent (raw)"
    ] if c in prev.columns]

    if cols:
        df_show = prev.loc[:, cols].head(20)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.caption("No NLP columns found.")

# -------- Save --------
if st.button("💾 Save as Processed dataset", type="primary", use_container_width=True):
    with st.spinner("Standardizing and saving..."):
        try:
            res = process_and_save(
                files,
                dataset_name=dataset_name or "Uploaded dataset",
                owner=(owner or None),
                already_processed=already_processed,
                extract_from_text=extract_from_text,
                overwrite_entities=overwrite_entities,
            )
        except Exception as e:
            st.error(f"Save failed: {e}")
            st.stop()

    # ✅ NEW: make saved data visible across pages immediately
    st.cache_data.clear()
    st.session_state["registry_bump"] = time.time()

    st.success(f"Saved processed dataset: **{res['dataset_id']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{res['rows']:,}")
    c2.metric("Victims", f"{res['victims']:,}")
    c3.download_button(
        "⬇️ Download saved CSV",
        data=res["preview"].to_csv(index=False).encode("utf-8"),
        file_name=f"{res['dataset_id']}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()

# -------- Registry glance --------
st.subheader("📦 Processed datasets in registry")
proc = registry.list_datasets(kind="processed")
if not proc:
    st.caption("No processed datasets yet.")
else:
    def _fmt(e): return f"{e.get('name')} • {e.get('id')} • {e.get('created_at')}"
    choice = st.selectbox("Select a processed dataset", options=proc, format_func=_fmt)
    if choice:
        try:
            df = registry.load_df(choice["id"])
            st.metric("Rows", len(df))
            st.metric("Victims", df["Serialized ID"].nunique() if "Serialized ID" in df.columns else 0)
            st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{choice['id']}.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Load failed: {e}")
