# frontend/pages/6_GNN_Trafficker_Prediction.py
from __future__ import annotations
import pandas as pd
import streamlit as st
import altair as alt

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames, unique_victims
from backend.api.predict import (
    predict_next_locations, save_nextloc_run,
    predict_perpetrators, save_perp_run
)

# -------------------- page setup --------------------
st.set_page_config(page_title="Predictive Analytics", page_icon="🔮", layout="wide")
st.title("🔮 Predictive Analytics")

st.markdown("""
**Two models:**
1) **Next locations** via order‑2 n‑gram with backoff (works with GIS overlay).  
2) **Perpetrators** via a lightweight link scorer (baseline, fast).

*Tip:* victims with **more route steps** give better results; for sparse histories the models fall back to global statistics.
""")

# -------------------- dataset picker --------------------
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged
def _fmt(e): return f"{e.get('name')} • {e.get('kind')} • {e.get('id')}"

if not queryable:
    st.info("No processed/merged datasets available.")
    st.stop()

selected = st.multiselect("Datasets:", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Pick at least one dataset.")
    st.stop()
src_ids = [e["id"] for e in selected]

@st.cache_data(show_spinner=False)
def _load_concat(ids, bump):
    try:
        return concat_processed_frames(ids)
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        return pd.DataFrame()

df = _load_concat(src_ids, st.session_state.get("registry_bump", 0))

tab_loc, tab_perp = st.tabs(["📍 Next Locations (overlayable)", "🧑‍⚖️ Perpetrators (baseline)"])

# -------------------- helpers --------------------
def _primary_loc_first_token(val):
    """First token from list/array; else clean string; else None."""
    if isinstance(val, list) and val:
        v = val[0]
        return str(v).strip() if v is not None else None
    try:
        import numpy as np  # noqa
        if hasattr(val, "size") and getattr(val, "size", 0) > 0:
            v = val.tolist()[0]
            return str(v).strip() if v is not None else None
    except Exception:
        pass
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    return s or None

def _step_primary_loc(row: pd.Series):
    loc = None
    if "Locations (NLP)" in row:
        loc = _primary_loc_first_token(row["Locations (NLP)"])
    if not loc and "Location" in row:
        loc = _primary_loc_first_token(row["Location"])
    return loc

# ==================== TAB: NEXT LOCATIONS ====================
with tab_loc:
    st.subheader("Next Locations (n‑gram)")

    all_vics = unique_victims(df)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        vic = st.selectbox("Victim (Serialized ID)", options=["(Select)"] + all_vics, index=0)
    with c2:
        steps = st.slider("Predict next N", 1, 5, 3, 1)
    with c3:
        owner = st.text_input("Owner email (optional)", value="")

    disabled = (vic == "(Select)")
    if st.button("🚀 Predict path", disabled=disabled, use_container_width=True):
        preds = predict_next_locations(df, victim_sid=vic, steps=int(steps)) if not disabled else []
        if not preds:
            st.info("No prediction (insufficient history after cleaning). Try another victim with more steps.")
        else:
            rows = [{"Rank": i + 1, "Predicted Location": loc, "Score": round(float(p), 4)}
                    for i, (loc, p) in enumerate(preds)]
            t = pd.DataFrame(rows)
            st.dataframe(t, use_container_width=True, hide_index=True)
            colA, colB = st.columns(2)
            with colA:
                st.download_button(
                    "⬇️ CSV",
                    data=t.to_csv(index=False).encode("utf-8"),
                    file_name=f"nextloc_{vic}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with colB:
                if st.button("💾 Save (for GIS overlay)", use_container_width=True):
                    rid = save_nextloc_run(src_ids, victim_sid=vic, preds=preds, owner=(owner or None))
                    st.success(f"Saved run id: {rid} — overlay it on the GIS page.")

# ==================== TAB: PERPETRATORS ====================
with tab_perp:
    st.subheader("Perpetrators (baseline link predictor)")

    all_vics = unique_victims(df)
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        vics = st.multiselect("Victims (one or many)", options=all_vics, help="Pick one or multiple HTV IDs.")
    with col2:
        topk = st.slider("Top‑K per victim", 1, 10, 5, 1)
    with col3:
        owner2 = st.text_input("Owner email (optional)", key="owner_perp", value="")

    if st.button("🚀 Predict perpetrators", use_container_width=True):
        if not vics:
            st.info("Pick at least one victim.")
        else:
            preds = predict_perpetrators(df, victims=vics, top_k=int(topk)) or {}

            # --- build table safely (always include Rank/Victim columns)
            rows = []
            for v, plist in preds.items():
                if not plist:
                    continue
                for i, (p, s) in enumerate(plist, start=1):
                    rows.append({
                        "Victim": str(v),
                        "Rank": int(i),
                        "Perpetrator": str(p),
                        "Score": float(s),
                    })
            tbl = pd.DataFrame(rows)

            if tbl.empty:
                st.info("No predictions produced (model backoff couldn’t find co‑occurrences). Try another victim or add more data.")
            else:
                sort_cols = [c for c in ["Victim", "Rank"] if c in tbl.columns]
                if sort_cols:
                    tbl = tbl.sort_values(sort_cols, kind="stable")
                st.dataframe(tbl, use_container_width=True, hide_index=True, height=420)

                # -------- insights
                st.subheader("Insights")
                left, right = st.columns(2)

                with left:
                    st.markdown("**Aggregated Top Perpetrators**")
                    agg = tbl.groupby("Perpetrator")["Victim"].nunique().reset_index(name="Victim Count") \
                             .sort_values("Victim Count", ascending=False)
                    if not agg.empty:
                        chart = alt.Chart(agg.head(20)).mark_bar().encode(
                            x=alt.X("Victim Count:Q", title="Victims (unique)"),
                            y=alt.Y("Perpetrator:N", sort='-x', title="Perpetrator"),
                            tooltip=["Perpetrator:N", "Victim Count:Q"]
                        ).properties(height=360)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("No data for chart.")

                with right:
                    st.markdown("**Perpetrators per *last* location (cross‑tab)**")
                    # last primary location per victim (from Locations (NLP) → first token; fallback to Location)
                    if not df.empty and "Serialized ID" in df.columns:
                        tmp = df.copy()
                        # derive primary step location
                        tmp["_primary_loc"] = tmp.apply(_step_primary_loc, axis=1)
                        # order & grab last
                        if "Route_Order" in tmp.columns:
                            tmp["Route_Order"] = pd.to_numeric(tmp["Route_Order"], errors="coerce")
                            tmp = tmp.sort_values(["Serialized ID", "Route_Order"], kind="stable")
                        last_loc = tmp.groupby("Serialized ID")["_primary_loc"].last().reset_index()
                        last_loc.columns = ["Victim", "Last Location"]
                        join = tbl.merge(last_loc, on="Victim", how="left")
                        cross = join.groupby(["Last Location", "Perpetrator"]).size().reset_index(name="Count") \
                                    .sort_values("Count", ascending=False)
                        if not cross.empty:
                            chart = alt.Chart(cross.head(30)).mark_bar().encode(
                                x=alt.X("Count:Q", title="Count"),
                                y=alt.Y("Perpetrator:N", sort='-x', title="Perpetrator"),
                                color=alt.Color("Last Location:N", title="Last location"),
                                tooltip=["Last Location:N", "Perpetrator:N", "Count:Q"]
                            ).properties(height=360)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.caption("No data for cross‑tab.")
                    else:
                        st.caption("No victim/location data available for cross‑tab.")

                # downloads & save
                l2, r2 = st.columns(2)
                with l2:
                    st.markdown("**Victims with no predictions**")
                    missing = [v for v in vics if v not in preds or len(preds[v]) == 0]
                    st.write(", ".join(missing) if missing else "—")
                with r2:
                    st.markdown("**Download & Save**")
                    st.download_button(
                        "⬇️ CSV (all predictions)",
                        data=tbl.to_csv(index=False).encode("utf-8"),
                        file_name="perp_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    if st.button("💾 Save run (JSON)", use_container_width=True):
                        rid = save_perp_run(src_ids, predictions=preds, owner=(owner2 or None))
                        st.success(f"Saved perp run id: {rid}")
