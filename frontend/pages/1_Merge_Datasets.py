# frontend/pages/1_Merge_Datasets.py
from __future__ import annotations
import json
import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.dataset_merge import analyze_merge, merge_datasets, list_merges, delete_dataset

st.set_page_config(page_title="Merge Datasets", page_icon="🧩", layout="wide")
st.title("🧩 Merge Datasets (dedupe + conflict resolution + history)")

st.markdown("""
**Workflow**  
1) Select two or more **Processed** datasets and click **Analyze**  
2) Review conflicts and choose a default resolution strategy (Priority / Union Lists / Keep Last)  
3) Optionally override **per-conflict** decisions  
4) **Save** the resolved merge and download artifacts
""")

# --- select inputs ---
processed = registry.list_datasets(kind="processed")
def _fmt(e): return f"{e.get('name')} • {e.get('id')} • {e.get('created_at')}"

if not processed:
    st.info("No processed datasets available yet.")
    st.stop()

selected = st.multiselect("Processed datasets to merge (left→right = priority):", options=processed, format_func=_fmt, key="merge_inputs")

if len(selected) < 2:
    st.warning("Pick at least two datasets and click Analyze.")
    st.stop()

dataset_ids = [e["id"] for e in selected]

# --- analyze ---
if st.button("🔎 Analyze", type="primary"):
    with st.spinner("Loading datasets and detecting conflicts..."):
        info = analyze_merge(dataset_ids)
    st.session_state["merge_analysis"] = info
    st.success(f"Loaded {info['rows']:,} rows • conflicts: {info['conflict_count']}")

analysis = st.session_state.get("merge_analysis")
if not analysis:
    st.stop()

conflicts: dict = analysis["conflicts"]
conflict_count = analysis["conflict_count"]
st.info(f"Detected **{conflict_count}** conflict groups (same Victim/Order/Location but different Perpetrators/Chiefs).")

# --- default strategy ---
st.subheader("Resolution Strategy")
colA, colB = st.columns([1.2, 2])
with colA:
    strategy = st.selectbox(
        "Default strategy",
        options=["priority", "union_lists", "keep_last"],
        format_func=lambda x: {"priority":"Priority order (left→right)", "union_lists":"Union list fields", "keep_last":"Keep last-selected source"}[x],
        index=0,
        key="merge_default_strategy"
    )
with colB:
    st.markdown("**Priority order** (drag to reorder by changing multiselect order above).")
    st.caption("We use the left→right order of your selection as priority. You can still override groups below.")

# --- per-conflict overrides ---
st.subheader("Per-conflict Overrides (optional)")
st.caption("For large merges, use the default strategy. You can override specific groups below. (Showing up to 200 groups.)")

manual_decisions = st.session_state.get("merge_manual_decisions", {})

shown = 0
for k, g in conflicts.items():
    if shown >= 200:
        break
    shown += 1
    with st.expander(f"Conflict #{shown} — {k}  •  sources: {', '.join(g['_Source_Name'].unique().tolist())}"):
        # Display the differing rows (compact)
        view_cols = [c for c in [COL for COL in [ "Serialized ID","Unique ID","Location","Route_Order","Perpetrators (NLP)","Chiefs (NLP)","_Source_Name","_Source_ID"] if c in g.columns]]
        st.dataframe(g[view_cols], use_container_width=True, hide_index=True, height=180)

        dd_key = f"decision_{k}"
        cur = manual_decisions.get(k, {"mode":"auto"})
        sel = st.radio(
            "Choose override for this group",
            ["Use default strategy", "Union lists", "Choose a specific source"],
            index=0 if cur.get("mode") in ("auto", "") else (1 if cur.get("mode")=="union" else 2),
            key=dd_key
        )
        if sel == "Union lists":
            manual_decisions[k] = {"mode":"union"}
        elif sel == "Choose a specific source":
            src = st.selectbox(
                "Source to keep",
                options=g["_Source_ID"].tolist(),
                format_func=lambda sid: f"{sid}  •  {g[g['_Source_ID']==sid]['_Source_Name'].iloc[0]}",
                key=f"{dd_key}_src"
            )
            manual_decisions[k] = {"mode":"choose", "source_id": src}
        else:
            # remove if previously set
            manual_decisions.pop(k, None)

st.session_state["merge_manual_decisions"] = manual_decisions

# --- save / run merge ---
st.subheader("Save Resolved Merge")
merge_name = st.text_input("Name for the merged dataset", value="Merged dataset")
if st.button("💾 Resolve & Save", type="primary"):
    with st.spinner("Resolving conflicts and saving merged artifact..."):
        result = merge_datasets(
            dataset_ids,
            strategy=strategy,
            priority_sources=dataset_ids,             # left→right order
            manual_decisions=manual_decisions,
            name=merge_name,
        )
    st.success(f"Saved merged dataset: **{result['merged_id']}**  •  report: **{result['report_id']}**")
    st.json(result["summary"])

    # Download merged CSV
    try:
        mdf = registry.load_df(result["merged_id"])
        st.download_button(
            "⬇️ Download merged CSV",
            data=mdf.to_csv(index=False).encode("utf-8"),
            file_name=f"{result['merged_id']}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Load merged failed: {e}")

st.divider()

# --- history ---
st.subheader("📜 Merge History")
past = list_merges()
if not past:
    st.caption("No merged datasets yet.")
else:
    def _fmt_hist(e): return f"{e.get('name')} • {e.get('id')} • {e.get('created_at')}"
    choice = st.selectbox("Select a merged dataset", options=past, format_func=_fmt_hist, key="merge_history_sel")
    if choice:
        c1, c2 = st.columns([1,1])
        with c1:
            try:
                df = registry.load_df(choice["id"])
                st.metric("Rows", len(df))
                st.metric("Victims", df["Serialized ID"].nunique() if "Serialized ID" in df.columns else 0)
                st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{choice['id']}.csv", mime="text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Load failed: {e}")
        with c2:
            with st.popover("🗑️ Delete this merged dataset", use_container_width=True):
                confirm = st.checkbox("I understand this will permanently delete the file.")
                if st.button("Delete", disabled=not confirm):
                    try:
                        delete_dataset(choice["id"])
                        st.success("Deleted. Refresh to update list.")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
