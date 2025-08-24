from __future__ import annotations
import json
import streamlit as st
import altair as alt
import pandas as pd

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.api.eval import run_evaluations, save_evaluation_report, export_report_zip

st.set_page_config(page_title="Automated Evaluations & Benchmarks", page_icon="🧪", layout="wide")
st.title("🧪 Automated Evaluations & Benchmarks")

st.markdown("""
Run **end-to-end quality & performance checks** across your processed/merged data:

- Data completeness & **ID consistency**
- **Duplicate suspicion** (heuristic)
- **Location** resolution rate
- **Network graph** metrics
- Predictive performance: **next location** (acc@1/3), **link prediction** (acc@1/3/5)
- **ETA** error (MAE, days)

You can **save** a versioned JSON report or **download** a ZIP (JSON + CSV tables).
""")

# -------------------- dataset picker --------------------
st.subheader("1) Choose datasets (Processed or Merged)")
datasets = registry.list_datasets() or []
processed = [d for d in datasets if d.get("kind") in {"processed", "merged"}]
if not processed:
    st.info("No processed or merged datasets are available yet.")
    st.stop()

def _fmt(opt):
    return f"{opt.get('name','<unnamed>')} • {opt.get('id','?')} • {opt.get('kind','?')}"

selected = st.multiselect("Datasets", options=processed, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset to evaluate.")
    st.stop()
ds_ids = [e["id"] for e in selected]

owner_email = st.text_input("Owner email (optional)", value="")

# -------------------- Predictive configuration + RUN (slider moved here) --------------------
st.divider()
st.subheader("Predictive Benchmarks (configure & run)")

conf_col, run_col = st.columns([3,1])
with conf_col:
    # The slider is now right next to the predictive section where it is used.
    link_max = st.slider(
        "Max victim–perp edges to sample for link-prediction benchmark",
        min_value=50, max_value=1000, value=300, step=50, help="Only used for the Link Prediction acc@k metric."
    )
with run_col:
    if st.button("🚀 Run Evaluations", type="primary", use_container_width=True):
        with st.spinner("Computing metrics, graph stats, and predictive benchmarks..."):
            report = run_evaluations(ds_ids, link_max_samples=int(link_max))
        st.session_state["_auto_eval_report"] = report
        st.success("Evaluations complete.")

# rehydrate report if already ran
report = st.session_state.get("_auto_eval_report")
if not report:
    st.stop()

summary = report.get("summary", {})
details = report.get("details", {})
tables = report.get("tables", {})

# -------------------- KPIs --------------------
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Rows", f"{summary.get('rows',0):,}")
kpi2.metric("Victims", f"{summary.get('victims',0):,}")
kpi3.metric("Locations", f"{summary.get('locations',0):,}")
kpi4.metric("Graph Nodes", f"{summary.get('graph_nodes',0):,}")
kpi5.metric("NextLoc acc@1", f"{summary.get('nextloc_acc@1',0.0):.3f}")
kpi6.metric("Link acc@3", f"{summary.get('link_acc@3',0.0):.3f}")

kpi7, kpi8, kpi9, kpi10, kpi11, kpi12 = st.columns(6)
kpi7.metric("NextLoc acc@3", f"{summary.get('nextloc_acc@3',0.0):.3f}")
kpi8.metric("Link acc@1", f"{summary.get('link_acc@1',0.0):.3f}")
kpi9.metric("Link acc@5", f"{summary.get('link_acc@5',0.0):.3f}")
kpi10.metric("ETA MAE (days)", f"{summary.get('eta_mae_days',0) or 0}")
kpi11.metric("Loc Resolution Rate", f"{summary.get('location_resolution_rate',0.0):.3f}")
kpi12.metric("Components", f"{details.get('graph_metrics',{}).get('components',0)}")

st.divider()

# -------------------- Data completeness & ID consistency --------------------
st.subheader("Data Completeness & ID Consistency")
comp_df = tables.get("completeness")
idc = details.get("id_consistency", {})

left, right = st.columns([3,2], gap="large")
with left:
    if isinstance(comp_df, pd.DataFrame) and not comp_df.empty:
        chart = alt.Chart(comp_df).mark_bar().encode(
            x=alt.X('Field:N', sort='-y', title='Field'),
            y=alt.Y('Completeness:Q', title='Completeness (0–1)'),
            tooltip=['Field', alt.Tooltip('Completeness:Q', format='.2f'), 'NonNull', 'Total']
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No completeness table available.")

with right:
    st.json(idc, expanded=False)

# -------------------- Duplicate suspicion --------------------
st.subheader("Duplicate Suspicion (Heuristic)")
dup_df = tables.get("suspected_duplicates")
if isinstance(dup_df, pd.DataFrame) and not dup_df.empty:
    st.dataframe(dup_df, use_container_width=True, height=240)
else:
    st.caption("No high-likelihood duplicates found.")

# -------------------- Location resolution --------------------
st.subheader("Location Resolution")
st.json(details.get("location_resolution", {}), expanded=False)

# -------------------- Graph metrics --------------------
st.subheader("Network Graph Metrics")
st.json(details.get("graph_metrics", {}), expanded=False)

# -------------------- Predictive Benchmarks (results) --------------------
st.subheader("Predictive Benchmarks")
cols = st.columns(3)
with cols[0]:
    st.metric("Next Location acc@1", f"{summary.get('nextloc_acc@1',0.0):.3f}")
    st.metric("Next Location acc@3", f"{summary.get('nextloc_acc@3',0.0):.3f}")
with cols[1]:
    st.metric("Link Prediction acc@1", f"{summary.get('link_acc@1',0.0):.3f}")
    st.metric("Link Prediction acc@3", f"{summary.get('link_acc@3',0.0):.3f}")
with cols[2]:
    st.metric("Link Prediction acc@5", f"{summary.get('link_acc@5',0.0):.3f}")
    st.metric("ETA MAE (days)", f"{summary.get('eta_mae_days',0) or 0}")

# sample tables if available
nl_table = tables.get("next_location_examples")
lp_table = tables.get("link_prediction_examples")
if isinstance(nl_table, pd.DataFrame) and not nl_table.empty:
    st.markdown("**Next-location: sample predictions**")
    st.dataframe(nl_table, use_container_width=True, height=230)
if isinstance(lp_table, pd.DataFrame) and not lp_table.empty:
    st.markdown("**Link-prediction: sample predictions**")
    st.dataframe(lp_table, use_container_width=True, height=230)

# -------------------- Save / Export --------------------
st.divider()
save_col, dl_col = st.columns([1,1])
with save_col:
    if st.button("💾 Save Report", use_container_width=True):
        rid = save_evaluation_report("Automated Evaluation", report, owner=(owner_email or None), sources=ds_ids)
        st.success(f"Saved evaluation report id: {rid}")
with dl_col:
    if st.button("📦 Download ZIP (JSON + CSV)", use_container_width=True):
        data = export_report_zip(report)
        st.download_button("⬇️ Click to download", data=data, file_name="evaluation_report.zip", mime="application/zip", use_container_width=True)

# -------------------- Past reports --------------------
st.divider()
st.subheader("📜 Past Evaluation Reports")
past = registry.list_datasets(kind="evaluation_report")
if not past:
    st.caption("No saved reports yet.")
else:
    choice = st.selectbox("Select a report", options=past, format_func=lambda e: f"{e.get('name')} • {e.get('id')} • {e.get('created_at')}")
    if choice:
        st.json(registry.load_json(choice["id"]))
