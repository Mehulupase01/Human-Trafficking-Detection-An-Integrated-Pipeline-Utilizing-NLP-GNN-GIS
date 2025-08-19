# frontend/pages/9_Query_History.py
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry

st.set_page_config(page_title="Query History", page_icon="🧾", layout="wide")
st.title("🧾 Query History (Admin & Researcher Only)")

def _try_list(kind: str) -> List[Dict[str, Any]]:
    """
    Try multiple registry entrypoints to fetch stored JSON artifacts.
    We’ve been saving runs with registry.save_json(..., kind=...).
    """
    for fn in ("list_artifacts", "list_json", "list_runs"):
        if hasattr(registry, fn):
            try:
                out = getattr(registry, fn)(kind=kind)  # expected: list of dicts
                if isinstance(out, list) and out:
                    return out
            except Exception:
                pass
    return []

def _rows_for(kind: str) -> pd.DataFrame:
    items = _try_list(kind)
    if not items:
        return pd.DataFrame(columns=["id","name","kind","owner","source","created_at"])
    # normalize
    rows = []
    for it in items:
        rows.append({
            "id": it.get("id") or it.get("_id") or "",
            "name": it.get("name") or it.get("title") or kind,
            "kind": it.get("kind") or kind,
            "owner": it.get("owner") or "",
            "source": it.get("source") or it.get("sources") or "",
            "created_at": it.get("created_at") or it.get("ts") or "",
            "_raw": it,
        })
    return pd.DataFrame(rows)

# What do we try to show?
CANDIDATE_KINDS = [
    "saved_query",             # if we ever saved query builder states
    "prediction_run",          # next-locations (n-gram)
    "perp_prediction_run",     # baseline perpetrators
    "eta_run",                 # temporal ETA runs
]

tabs = st.tabs([k.replace("_"," ").title() for k in CANDIDATE_KINDS])

for tab, kind in zip(tabs, CANDIDATE_KINDS):
    with tab:
        df = _rows_for(kind)
        if df.empty:
            st.info(f"No {kind.replace('_',' ')} logged yet.")
            continue

        # Display table
        show = df.drop(columns=["_raw"], errors="ignore")
        st.dataframe(show, use_container_width=True, hide_index=True, height=360)

        # Download
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download table (CSV)",
                data=show.to_csv(index=False).encode("utf-8"),
                file_name=f"{kind}_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ Download (JSON)",
                data=json.dumps(df["_raw"].tolist(), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{kind}_history.json",
                mime="application/json",
                use_container_width=True,
            )

        # Peek a single item
        st.markdown("#### Inspect a record")
        sel = st.selectbox("Pick id", options=["(select)"] + show["id"].tolist())
        if sel and sel != "(select)":
            row = df[df["id"] == sel].iloc[0]["_raw"]
            st.json(row)
