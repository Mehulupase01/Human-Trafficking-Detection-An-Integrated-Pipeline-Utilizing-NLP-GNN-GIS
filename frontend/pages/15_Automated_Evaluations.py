# frontend/pages/15_Automated_Evaluations.py
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
- Data completeness & ID consistency
- Duplicate suspicion (heuristic)
- Location resolution rate (via gazetteer/overrides)
- Network graph metrics
- Predictive performance: **next location** (acc@1/3), **link prediction** (acc@1/3/5)
- ETA error (**MAE days**)
Export everything as a ZIP or save a versioned JSON report.
""")

# ---------- Data selection ----------
st.subheader("1) Choose datasets (Processed or Merged)")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged

def _fmt(e: dict) -> str:
    return f"{e.get('name')}  •  {e.get('kind')}  •  {e.get('id')}"

if not queryable:
    st.info("No processed or merged datasets available.")
    st.stop()

selected = st.multiselect("Datasets:", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset.")
    st.stop()

ds_ids = [e["id"] for e in selected]
owner_email = st.text_input("Owner email (optional)", value="")
link_max = st.slider("Max victim–perp edges to sample for link-pred benchmark", 50, 1000, 300, 50)

# ---------- Run ----------
if st.button("🚀 Run Evaluations", type="primary"):
    with st.spinner("Computing metrics, graph stats, and predictive benchmarks..."):
        report = run_evaluations(ds_ids, link_max_samples=int(link_max))

    st.success("Evaluations complete.")

    # ---------- Summary KPIs ----------
    st.subheader("2) Summary KPIs")
    s = report["summary"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{s['rows']:,}")
    c2.metric("Victims", f"{s['victims']:,}")
    c3.metric("Locations", f"{s['locations']:,}")
    c4.metric("Loc Resolution", f"{100*s['location_resolution_rate']:.1f}%")
    d1,d2,d3 = st.columns(3)
    d1.metric("NextLoc acc@1", f"{100*s['nextloc_acc@1']:.1f}%")
    d2.metric("LinkPred acc@1", f"{100*s['link_acc@1']:.1f}%")
    d3.metric("ETA MAE (days)", f"{s['eta_mae_days'] if s['eta_mae_days'] is not None else '—'}")

    # ---------- Completeness ----------
    st.subheader("3) Data Completeness")
    comp = report["tables"]["completeness"]
    if not comp.empty:
        chart = alt.Chart(comp).mark_bar().encode(
            x=alt.X("Coverage:Q", axis=alt.Axis(format="%")),
            y=alt.Y("Field:N", sort='-x'),
            tooltip=["Field:N", alt.Tooltip("Coverage:Q", format=".1%"), "Non-null:Q", "Total Rows:Q"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(comp, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No completeness data (unexpected).")

    # ---------- Suspected duplicates ----------
    st.subheader("4) Suspected Duplicates (heuristic)")
    dup = report["tables"]["suspected_duplicates"]
    if not dup.empty:
        st.dataframe(dup, use_container_width=True, hide_index=True, height=260)
        st.caption("Heuristic key: Nationality|Gender|FirstLoc|LastLoc|RouteLen — review manually.")
    else:
        st.caption("No duplicate groups flagged by the heuristic.")

    # ---------- Location resolution ----------
    st.subheader("5) Location Resolution")
    loc_det = report["details"]["location_resolution"]
    left, right = st.columns([2,1])
    with left:
        st.metric("Unique Locations", f"{loc_det.get('unique_locations',0):,}")
        st.metric("Resolved", f"{loc_det.get('resolved',0):,}")
    with right:
        st.metric("Resolution Rate", f"{100*loc_det.get('rate',0.0):.1f}%")
    unresolved_tbl = report["tables"]["top_unresolved_locations_sample"]
    if not unresolved_tbl.empty:
        st.dataframe(unresolved_tbl, use_container_width=True, hide_index=True, height=220)

    # ---------- Graph metrics ----------
    st.subheader("6) Graph Metrics")
    g = report["details"]["graph"]
    if "error" in g:
        st.error(g["error"])
    else:
        g1,g2,g3,g4 = st.columns(4)
        g1.metric("Nodes", g["nodes"])
        g2.metric("Edges", g["edges"])
        g3.metric("Components", g["components"])
        g4.metric("Largest component", g["largest_component_size"])
        st.json({"avg_degree": g.get("avg_degree"), "avg_clustering": g.get("avg_clustering"), "nodes_by_type": g.get("nodes_by_type", {})})

    # ---------- Predictive benchmarks ----------
    st.subheader("7) Predictive Benchmarks")
    b1,b2 = st.columns(2)
    with b1:
        nl = report["details"]["next_location"]
        st.markdown("**Next Location (leave-last-step-out)**")
        st.json(nl)
    with b2:
        lp = report["details"]["link_prediction"]
        st.markdown("**Link Prediction (edge removal sampling)**")
        st.json(lp)

    # ---------- ETA benchmark ----------
    st.subheader("8) ETA Benchmark")
    st.json(report["details"]["eta"])

    # ---------- Export / Save ----------
    st.subheader("9) Export")
    zbytes = export_report_zip(report)
    st.download_button("⬇️ Download Report (ZIP)", data=zbytes, file_name="evaluation_report.zip", mime="application/zip", use_container_width=True)

    if st.button("💾 Save Report", use_container_width=True):
        rid = save_evaluation_report("Automated Evaluation", report, owner=(owner_email or None), sources=ds_ids)
        st.success(f"Saved evaluation report id: {rid}")

# Past reports
st.divider()
st.subheader("📜 Past Evaluation Reports")
past = registry.list_datasets(kind="evaluation_report")
if not past:
    st.caption("No saved reports yet.")
else:
    choice = st.selectbox("Select a report", options=past, format_func=lambda e: f"{e.get('name')} • {e.get('id')} • {e.get('created_at')}")
    if choice:
        st.json(registry.load_json(choice["id"]))
