# frontend/pages/10_Upload_History.py
from __future__ import annotations
import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames

st.set_page_config(page_title="Upload History", page_icon="🗂️", layout="wide")
st.title("🗂️ Dataset Upload History")

def _fmt(ds: dict) -> dict:
    return {
        "id": ds.get("id",""),
        "name": ds.get("name",""),
        "kind": ds.get("kind",""),
        "owner": ds.get("owner",""),
        "created_at": ds.get("created_at") or ds.get("ts") or "",
        "rows": ds.get("rows") or "",
    }

processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
raws = []
if hasattr(registry, "list_datasets"):
    try:
        raws = registry.list_datasets(kind="raw")
    except Exception:
        pass

blocks = [
    ("Processed datasets", processed),
    ("Merged datasets", merged),
    ("Raw uploads", raws),
]

any_data = any(len(b[1]) for b in blocks)
if not any_data:
    st.info("No upload history available.")
    st.stop()

for title, items in blocks:
    with st.expander(title, expanded=True):
        if not items:
            st.caption("— none —")
            continue

        # normalize
        rows = [_fmt(x) for x in items]
        df = pd.DataFrame(rows)

        # try to compute row counts if missing (lightweight best-effort)
        if "rows" not in df.columns or df["rows"].isna().all():
            try:
                # Careful: avoid loading huge data. Just attempt first few for a sample count.
                ids = df["id"].tolist()
                if ids:
                    from backend.api.graph_queries import load_processed_frame  # may or may not exist
                    sample_id = ids[0]
                    try:
                        sdf = load_processed_frame(sample_id)
                    except Exception:
                        sdf = None
                    if sdf is not None:
                        df.loc[df["id"] == sample_id, "rows"] = len(sdf)
            except Exception:
                pass

        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
