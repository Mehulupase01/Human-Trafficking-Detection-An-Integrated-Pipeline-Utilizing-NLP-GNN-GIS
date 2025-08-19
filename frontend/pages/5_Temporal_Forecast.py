# frontend/pages/5_Temporal_Forecast.py
from __future__ import annotations
import json
from datetime import date
from typing import List, Tuple

import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames, unique_victims
# Saving utility (keep your existing saver)
from backend.api.eta import save_eta_run

# New: sequence predictor & ETA builder
from backend.models.sequence_predictor import (
    NgramSequenceModel,
    build_sequences_from_df,
    last_context_for_victim,
)
from backend.api.temporal import TemporalETA

# -------------------------------------------------------
st.set_page_config(page_title="Temporal Forecast (ETA: days/weeks)", page_icon="⏱️", layout="wide")
st.title("⏱️ Temporal Forecast — Predict Time of Arrival (days/weeks)")

st.markdown("""
This page predicts **Time of Arrival (ETA)** for the next locations per victim:

- Next locations via an **order‑2 n‑gram** model with backoff (to order‑1 and global).
- ETAs from learned **transition medians** (A→B), backoff to **arrival‑location medians** and **global median**,
  else using your **fallback days** slider.
""")

# ---------- Dataset selection ----------
st.subheader("1) Choose datasets (Processed or Merged)")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged

def _fmt(e: dict) -> str:
    return f"{e.get('name')}  •  {e.get('kind')}  •  {e.get('id')}"

if not queryable:
    st.info("No processed or merged datasets available.")
    st.stop()

selected = st.multiselect("Select dataset(s):", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()

ds_ids = [e["id"] for e in selected]

# ---------- Load data ----------
with st.spinner("Loading data..."):
    df = concat_processed_frames(ds_ids)

# Short‑circuit if empty
if df is None or df.empty:
    st.info("Selected datasets are empty.")
    st.stop()

# ---------- Controls ----------
st.subheader("2) Forecast settings")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    victim_sid = st.selectbox("Victim (Serialized ID)", options=["(Select)"] + unique_victims(df), index=0)
with c2:
    steps = st.slider("Predict next N locations", min_value=1, max_value=5, value=3, step=1)
with c3:
    fallback_days = st.number_input("Fallback days (if unknown)", min_value=1, max_value=60, value=7, step=1)

c4, c5 = st.columns([1, 2])
with c4:
    use_start = st.toggle("Use a start date", value=True)
with c5:
    start_date_val = st.date_input("Start date (for arrival timeline)", value=date.today()) if use_start else None

owner_email = st.text_input("Owner email (optional)", value="")
run = st.button("🚀 Predict ETA", type="primary", disabled=(victim_sid == "(Select)"))

# ---------- Helpers ----------
def _clean_loc(v):
    """Normalize list-like tokens such as ['Tripoli'] -> 'Tripoli'."""
    if isinstance(v, list) and v:
        return str(v[0])
    return "" if pd.isna(v) or v is None else str(v)

# ---------- Run ----------
if run and victim_sid != "(Select)":
    # 1) Train the next‑location model on all victims
    seqs = build_sequences_from_df(df)
    model = NgramSequenceModel(alpha=0.05)
    model.fit(seqs)

    # 2) Get the victim's last context (order‑2) and predict N next locations
    history = last_context_for_victim(df, victim_sid, order=2)
    preds: List[Tuple[str, float]] = model.predict_path(history, steps=int(steps))

    if not preds:
        st.info("No prediction available for this victim (insufficient history after cleaning).")
        st.stop()

    # 3) Build the path to score for ETAs: last known step + predicted steps
    base = list(history)[:]  # may be 1–2 tokens depending on data
    predicted_locs = [_clean_loc(loc) for (loc, _p) in preds]
    path_for_eta = (base[-1:] if base else []) + predicted_locs  # transitions: last_known -> pred1 -> pred2 ...

    # 4) Fit ETA model (learn medians) and score the predicted path
    eta = TemporalETA(fallback_days=float(fallback_days))
    eta.fit(df)  # learns (A->B), per‑location, and global medians from the *whole* dataset

    # Need a full path including the last known hop origin
    if base:
        full_path = base[-1:] + predicted_locs  # origin is last known step
    else:
        # If we don't have a last step, assume transitions only between predictions
        full_path = predicted_locs[:]

    rows_eta = eta.predict_path(full_path, start_date=(start_date_val if use_start else None))

    # 5) Merge scores + ETAs into a single table for the UI
    # rows_eta contains only hops (len = steps - maybe fewer if path short)
    # We align by step number
    result_rows = []
    cum_days = 0.0
    for i, hop in enumerate(rows_eta, start=1):
        loc = hop.get("Predicted Location", predicted_locs[i - 1] if i - 1 < len(predicted_locs) else "")
        # probability from next‑location model (same order)
        score = float(preds[i - 1][1]) if i - 1 < len(preds) else 0.0
        eta_days = float(hop.get("ETA (days)", fallback_days))
        cum_days = float(hop.get("Cumulative days", 0.0))
        result_rows.append({
            "Step": i,
            "Predicted Location": loc,
            "Score": round(score, 4),
            "ETA (days)": round(eta_days, 2),
            "ETA (weeks)": round(eta_days / 7.0, 2),
            "Cumulative days": round(cum_days, 2),
            "Arrival date": hop.get("Arrival date", ""),
        })

    if not result_rows:
        st.info("Could not compute ETAs for the predicted path.")
        st.stop()

    st.subheader("3) Results")
    table = pd.DataFrame(result_rows)
    st.dataframe(table, use_container_width=True, hide_index=True, height=380)

    colA, colB, colC = st.columns(3)
    with colA:
        st.download_button(
            "⬇️ Download (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"eta_{victim_sid}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with colB:
        out_json = {
            "victim": victim_sid,
            "steps": int(steps),
            "fallback_days": float(fallback_days),
            "start_date": (start_date_val.isoformat() if (use_start and start_date_val) else None),
            "results": result_rows,
            "sources": ds_ids,
        }
        st.download_button(
            "⬇️ Download (JSON)",
            data=json.dumps(out_json, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"eta_{victim_sid}.json",
            mime="application/json",
            use_container_width=True,
        )
    with colC:
        if st.button("💾 Save ETA run", use_container_width=True):
            rid = save_eta_run(
                sources=ds_ids,
                owner=(owner_email or None),
                victim_sid=victim_sid,
                next_locs=[r["Predicted Location"] for r in result_rows],
                eta_days=[r["ETA (days)"] for r in result_rows],
                start_date_iso=(start_date_val.isoformat() if (use_start and start_date_val) else None),
            )
            st.success(f"Saved ETA run id: {rid}")

    with st.expander("Details"):
        # Quick stats for transparency
        stats = {
            "history_used": history,
            "predicted_locations": predicted_locs,
            "eta_fallback_days_slider": float(fallback_days),
        }
        st.json(stats)
