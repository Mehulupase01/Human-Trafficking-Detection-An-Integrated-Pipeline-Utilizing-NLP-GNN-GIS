from __future__ import annotations
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Core runner + registry
from eval_harness.runner import run_all, RunnerConfig
from eval_harness.data_access import DataAccess
from backend.core import dataset_registry as registry

# Save/Export adapter
from backend.api.eval import save_evaluation_report, export_report_zip, list_past_reports, load_report

# Plot helpers
from eval_harness.plots_nlp import pr_curve_chart, roc_curve_chart, reliability_chart, confusion_heatmap
from eval_harness.plots_graph import heuristics_bar_chart, degree_cdf_chart, component_cdf_chart
from eval_harness.plots_gis import make_map, render_map, gaps_histogram
from eval_harness.plots_query import query_metrics_bar, latency_box

st.set_page_config(page_title="Automated Evaluations", page_icon="🧪", layout="wide")
st.title("🧪 Automated Evaluations (Hold-out 30% + K-fold CV)")

st.markdown("""
Evaluate **NLP**, **Network Graph**, **GIS**, and **Query/Retrieval** with a fixed **30% test split** and **K-fold CV** on the remaining 70%.
If a section lacks required columns, the UI will show a clear reason.
""")

# ---------------- Dataset selector ----------------

def _list_processed():
    try:
        items = registry.list_datasets(kind="processed") or []
        items += registry.list_datasets(kind="merged") or []
        return items
    except Exception:
        try:
            items = registry.list_datasets() or []
        except Exception:
            try:
                items = registry.list_all() or []
            except Exception:
                items = []
        return [x for x in items if (x.get("kind") or x.get("type")) in {"processed", "merged"}]

def _fmt_row(it: Dict[str, Any]) -> str:
    return f"{it.get('name','<unnamed>')} • {it.get('id','?')} • {it.get('kind','?')}"

ds_options = _list_processed()
if not ds_options:
    st.warning("No processed/merged datasets found in the registry.")
    st.stop()

selected = st.multiselect("Datasets", options=ds_options, format_func=_fmt_row)
if not selected:
    st.info("Select at least one dataset to evaluate.")
    st.stop()

ds_ids = [s["id"] for s in selected]

# ---------------- Configuration ----------------

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
seed = c1.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
kfold = c2.number_input("K-folds (CV)", min_value=2, max_value=10, value=5, step=1)
test_frac = c3.slider("Test fraction", min_value=0.2, max_value=0.4, value=0.30, step=0.05)
graph_samples = c4.slider("Graph link-pred samples", min_value=50, max_value=1000, value=300, step=50)
query_k = c5.slider("Query top-k", min_value=5, max_value=50, value=10, step=5)

run_btn = st.button("🚀 Run Evaluations", type="primary", use_container_width=True)

if run_btn:
    cfg = RunnerConfig(seed=int(seed), k=int(kfold), test_frac=float(test_frac),
                       graph_max_samples=int(graph_samples), query_top_k=int(query_k))
    with st.spinner("Running evaluation on selected datasets..."):
        report = run_all(ds_ids, registry, cfg=cfg)
    st.session_state["eval_report"] = report
    st.success("Done.")

report = st.session_state.get("eval_report")
if not report:
    st.stop()

# ---------------- Schema Mapping & Geocode (NEW) ----------------
with st.expander("🔎 Schema Mapping & Geocode (detected)", expanded=False):
    cols = report.get("resolved_columns", {})
    st.write("**Resolved columns**", cols or "(none)")
    gis = report.get("sections", {}).get("gis", {})
    ge = (gis or {}).get("geocode", {})
    if ge:
        rate = ge.get("rate", 0.0)
        st.markdown(f"- **Geocode rate:** {rate:.3f}  &nbsp;&nbsp; *(resolved {ge.get('resolved',0)}/{ge.get('total',0)} via {ge.get('source','?')})*")
    st.caption("If something looks wrong, check your processed CSV column names; we can override mappings if needed.")

# Load processed df for charts/maps
_da = DataAccess(registry)
_df_proc = _da.load_processed(ds_ids)

# ---------------- Splits summary ----------------

st.divider()
st.subheader("Splits & Leakage Checks")
spl = report.get("splits", {}).get("summary", {})
if spl:
    cA, cB = st.columns([2,3])
    with cA:
        st.json({k: v for k, v in spl.items() if k not in ("folds",)})
    with cB:
        folds_tbl = pd.DataFrame(spl.get("folds", []))
        if not folds_tbl.empty:
            st.dataframe(folds_tbl, use_container_width=True, height=220)
else:
    st.caption(report.get("splits", {}).get("reason", "No splits info available."))

# ---------------- KPI tiles ----------------

def _safe_get(d, *path, default=0.0):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

st.divider()
st.subheader("Key KPIs (Hold-out)")

sec = report.get("sections", {})
nlp_kpi_f1   = _safe_get(sec, "nlp", "holdout", "f1", default=0.0)
nlp_kpi_ap   = _safe_get(sec, "nlp", "holdout", "ap", default=0.0)
graph_j_hit1 = _safe_get(sec, "graph", "holdout", "jaccard", "hits@1", default=0.0)
graph_j_mrr  = _safe_get(sec, "graph", "holdout", "jaccard", "mrr", default=0.0)
geo_rate     = _safe_get(sec, "gis", "geocode", "rate", default=0.0)
traj_count   = _safe_get(sec, "gis", "trajectories", "trajectories", default=0)
next_acc1    = _safe_get(sec, "gis", "nextloc", "holdout", "acc@1", default=0.0)
eta_mae      = _safe_get(sec, "gis", "eta", "holdout", "mae_days", default=0.0)
q_ndcg10     = _safe_get(sec, "query", "holdout", "metrics", "ndcg@10", default=0.0)

t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.columns(9)
t1.metric("NLP F1", f"{nlp_kpi_f1:.3f}")
t2.metric("NLP AP", f"{nlp_kpi_ap:.3f}")
t3.metric("Graph Hits@1 (Jaccard)", f"{graph_j_hit1:.3f}")
t4.metric("Graph MRR (Jaccard)", f"{graph_j_mrr:.3f}")
t5.metric("Geo Resolution", f"{geo_rate:.3f}")
t6.metric("Trajectories", f"{int(traj_count):,}")
t7.metric("Next-loc Acc@1", f"{next_acc1:.3f}")
t8.metric("ETA MAE (days)", f"{eta_mae:.2f}")
t9.metric("Query nDCG@10", f"{q_ndcg10:.3f}")

# ---------------- Tabs ----------------

tab_nlp, tab_graph, tab_gis, tab_query = st.tabs(["NLP", "Graph", "GIS", "Query"])

# --- NLP tab ---
with tab_nlp:
    nlp = sec.get("nlp", {})
    if not nlp.get("available", True) and "reason" in nlp:
        st.warning(nlp.get("reason"))
    if "holdout" in nlp:
        st.subheader("Hold-out (30% Test)")
        h = nlp["holdout"]
        c1,c2,c3 = st.columns([1,1,1])
        c1.metric("F1", f"{h.get('f1',0):.3f}")
        c2.metric("AP (PR AUC)", f"{h.get('ap',0):.3f}")
        c3.metric("ROC-AUC", f"{h.get('roc_auc',0):.3f}")
        pr = h.get("curves", {}).get("pr", {})
        roc = h.get("curves", {}).get("roc", {})
        cal = h.get("curves", {}).get("calibration", {})
        cm = h.get("confusion", {})
        g1,g2 = st.columns(2)
        if pr and roc:
            with g1:
                st.altair_chart(pr_curve_chart(pr), use_container_width=True)
            with g2:
                st.altair_chart(roc_curve_chart(roc), use_container_width=True)
        if cal:
            st.altair_chart(reliability_chart(cal), use_container_width=True)
        if cm:
            st.altair_chart(confusion_heatmap(cm), use_container_width=True)
        st.caption(f"Threshold = {h.get('threshold', 0.5):.3f} • F1 95% CI: "
                   f"[{_safe_get(h,'f1_ci','lo', default=0):.3f}, {_safe_get(h,'f1_ci','hi', default=0):.3f}] "
                   f"• AP 95% CI: [{_safe_get(h,'ap_ci','lo', default=0):.3f}, {_safe_get(h,'ap_ci','hi', default=0):.3f}]")
    if "cv" in nlp:
        st.subheader("Cross-Validation (on 70%)")
        cv = nlp["cv"]
        if "folds" in cv and cv["folds"]:
            st.dataframe(pd.DataFrame(cv["folds"]), use_container_width=True, height=240)
        if "summary" in cv:
            st.json(cv["summary"], expanded=False)

# --- Graph tab ---
with tab_graph:
    graph = sec.get("graph", {})
    if not graph:
        st.caption("Graph section not available.")
    else:
        st.subheader("Hold-out link prediction (heuristics)")
        hold = graph.get("holdout", {})
        if hold.get("available", True) is False and "reason" in hold:
            st.warning(hold.get("reason"))
        else:
            st.altair_chart(heuristics_bar_chart(hold), use_container_width=True)

        st.subheader("Structure distributions (from processed DF)")
        if _df_proc is not None and not _df_proc.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(degree_cdf_chart(_df_proc), use_container_width=True)
            with c2:
                st.altair_chart(component_cdf_chart(_df_proc), use_container_width=True)
        else:
            st.caption("Processed dataframe unavailable for CDF plots.")

        st.subheader("Graph descriptives")
        st.json(graph.get("descriptives", {}), expanded=False)

        st.subheader("CV summary")
        cv = graph.get("cv", {}).get("summary", {})
        if cv:
            st.json(cv, expanded=False)

# --- GIS tab ---
with tab_gis:
    gis = sec.get("gis", {})
    if not gis:
        st.caption("GIS section not available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        ge = gis.get("geocode", {})
        tr = gis.get("trajectories", {})
        nxh = gis.get("nextloc", {}).get("holdout", {})
        eta = gis.get("eta", {}).get("holdout", {})
        c1.metric("Geocode rate", f"{ge.get('rate', 0):.3f}")
        c2.metric("#Trajectories", f"{int(tr.get('trajectories', 0)):,}")
        c3.metric("Next-loc acc@1", f"{nxh.get('acc@1', 0):.3f}")
        c4.metric("ETA MAE (days)", f"{eta.get('mae_days', 0):.2f}")


        st.subheader("Map (heat + sample markers)")
        fmap = make_map(_df_proc)
        render_map(st, fmap)

        st.subheader("Time-gap histogram")
        st.altair_chart(gaps_histogram(_df_proc), use_container_width=True)

        st.subheader("Details")
        st.json({"geocode": ge, "clustering": gis.get("clustering", {}), "trajectories": tr}, expanded=False)

        st.subheader("CV summary (Next-loc / ETA)")
        st.json(gis.get("nextloc", {}).get("cv", {}), expanded=False)
        st.json(gis.get("eta", {}).get("cv", {}), expanded=False)

# --- Query tab ---
with tab_query:
    q = sec.get("query", {})
    if not q:
        st.caption("Query section not available.")
    else:
        st.subheader("Hold-out metrics & latency")
        hold = q.get("holdout", {})
        metrics = hold.get("metrics", {})
        if metrics:
            st.altair_chart(query_metrics_bar(metrics), use_container_width=True)
        st.altair_chart(latency_box(hold.get("latency", {})), use_container_width=True)
        st.caption(f"Queries evaluated: {int(q.get('n_queries',0))} • Indexed docs: {int(q.get('n_docs',0))}")

        st.subheader("Cross-Validation summary")
        cvs = q.get("cv", {}).get("summary", {})
        if cvs:
            st.json(cvs, expanded=False)

# ---------------- Save / Export ----------------

st.divider()
st.subheader("Save / Export")

default_name = f"Automated Eval • {', '.join(map(str, ds_ids))}"
name = st.text_input("Report name", value=default_name)
owner = st.text_input("Owner email (optional)", value="")

c1, c2 = st.columns([1,1])
with c1:
    if st.button("💾 Save Report", use_container_width=True):
        try:
            rid = save_evaluation_report(name, report, owner=(owner or None), sources=ds_ids)
            st.success(f"Saved evaluation report id: {rid}")
        except Exception as e:
            st.error(f"Save failed: {e}")
with c2:
    if st.button("📦 Prepare ZIP (JSON + CSV)", use_container_width=True):
        try:
            data = export_report_zip(report)
            st.download_button(
                "⬇️ Download evaluation_report.zip",
                data=data,
                file_name="evaluation_report.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

# ---------------- Past runs ----------------

st.divider()
st.subheader("📜 Past Evaluation Reports")

try:
    past = list_past_reports()
except Exception as e:
    past = []
    st.error(f"Could not list past reports: {e}")

if not past:
    st.caption("No saved reports yet.")
else:
    def _fmt_past(it: Dict[str, Any]) -> str:
        nm = it.get("name") or it.get("meta", {}).get("name") or "<unnamed>"
        rid = it.get("id", "?")
        ts = it.get("created_at") or it.get("meta", {}).get("created_at") or ""
        return f"{nm} • {rid} • {ts}"

    choice = st.selectbox("Select a report", options=past, format_func=_fmt_past)
    if choice:
        try:
            loaded = load_report(choice.get("id") or choice.get("rid") or choice.get("uid"))
            st.json(loaded, expanded=False)
        except Exception as e:
            st.error(f"Unable to load selected report: {e}")

st.divider()
st.caption("All metrics computed on your actual processed data with 30% fixed test split and K-fold CV on the remaining 70%.")
