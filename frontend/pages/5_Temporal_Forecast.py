# frontend/pages/5_Temporal_Forecast.py
from __future__ import annotations
import json
from datetime import date
import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames, unique_victims
from backend.api.eta import predict_eta_for_victim, save_eta_run

st.set_page_config(page_title="Temporal Forecast (ETA: days/weeks)", page_icon="⏱️", layout="wide")
st.title("⏱️ Temporal Forecast — Predict Time of Arrival (days/weeks)")

st.markdown("""
This page predicts **Time of Arrival (ETA)** for the next locations per victim:

- Next locations via an order-2 n-gram model (with backoff)
- ETAs from learned **transition medians** (A→B), backoff to **location medians** and **global median**, else a default
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

# ---------- Controls ----------
st.subheader("2) Forecast settings")
c1, c2, c3 = st.columns([2,1,1])
with c1:
    victim_sid = st.selectbox("Victim (Serialized ID)", options=["(Select)"] + unique_victims(df), index=0)
with c2:
    steps = st.slider("Predict next N locations", min_value=1, max_value=5, value=3, step=1)
with c3:
    default_days = st.number_input("Fallback days (if unknown)", min_value=1, max_value=60, value=7, step=1)

c4, c5 = st.columns([1,2])
with c4:
    use_start = st.toggle("Use a start date", value=True)
with c5:
    start_date_val = st.date_input("Start date (for arrival timeline)", value=date.today()) if use_start else None

owner_email = st.text_input("Owner email (optional)", value="")

run = st.button("🚀 Predict ETA", type="primary", disabled=(victim_sid == "(Select)"))

if run and victim_sid != "(Select)":
    with st.spinner("Predicting next locations and ETAs..."):
        result = predict_eta_for_victim(
            df, victim_sid=victim_sid, steps=steps,
            default_days=int(default_days),
            start_date=start_date_val if use_start else None,
        )

    next_locs = result["predicted"]
    eta_days = result["eta_days"]
    eta_weeks = [round(d / 7.0, 2) for d in eta_days]
    arrivals = result.get("arrival_dates", [])

    if not next_locs:
        st.info("No prediction available for this victim (insufficient history).")
    else:
        st.subheader("3) Results")
        rows = []
        cum_days = 0
        for i, (loc, d, w) in enumerate(zip(next_locs, eta_days, eta_weeks), start=1):
            cum_days += d
            rows.append({
                "Step": i,
                "Predicted Location": loc,
                "ETA (days)": int(d),
                "ETA (weeks)": w,
                "Cumulative days": cum_days,
                "Arrival date": arrivals[i-1] if arrivals else "",
            })
        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True, hide_index=True, height=360)

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
                "steps": steps,
                "default_days": int(default_days),
                "start_date": (start_date_val.isoformat() if (use_start and start_date_val) else None),
                "results": rows,
                "sources": ds_ids,
                "stats": result.get("stats_summary", {}),
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
                    next_locs=next_locs,
                    eta_days=eta_days,
                    start_date_iso=(start_date_val.isoformat() if (use_start and start_date_val) else None),
                )
                st.success(f"Saved ETA run id: {rid}")

        with st.expander("Details"):
            st.json(result.get("stats_summary", {}))
