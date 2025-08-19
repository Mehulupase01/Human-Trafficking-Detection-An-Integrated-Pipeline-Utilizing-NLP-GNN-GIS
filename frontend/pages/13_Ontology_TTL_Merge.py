# frontend/pages/13_Ontology_TTL_Merge.py
import json
import streamlit as st
import pandas as pd

from backend.core import dataset_registry as registry
from backend.api.ontology import (
    default_mapping,
    build_and_export_ttl,
    save_ttl_to_registry,
)

st.set_page_config(page_title="Ontology (Turtle) + User Mapping Merge", page_icon="🧩", layout="wide")
st.title("🧩 Ontology Export (Turtle) + User Mapping Merge")

st.markdown("""
Export the processed datasets as **RDF/Turtle** using a clean, data-driven ontology:

- Classes: **Victim**, **Trafficker**, **Chief**, **Location**, **Visit**
- Triples include **visits** with **order**, **perpetrators**, **chiefs**, and optional **geo (lat/lon)** on locations
- Override any URI with a **JSON mapping**; merge with an uploaded **user TTL** file
""")

# ---------- Data selection ----------
st.subheader("1) Choose datasets (Processed or Merged)")
processed = registry.list_datasets(kind="processed")
merged = registry.list_datasets(kind="merged")
queryable = processed + merged

def _fmt(e: dict) -> str:
    return f"{e.get('name')}  •  {e.get('kind')}  •  {e.get('id')}"

if not queryable:
    st.info("No processed or merged datasets found.")
    st.stop()

selected = st.multiselect("Select dataset(s):", options=queryable, format_func=_fmt)
if not selected:
    st.warning("Select at least one dataset to continue.")
    st.stop()

src_ids = [e["id"] for e in selected]
owner_email = st.text_input("Owner email (optional)", value="")

# ---------- Config ----------
st.subheader("2) Ontology settings")
col1, col2 = st.columns([2,1])
with col1:
    base_uri = st.text_input("Base namespace (use a stable HTTPS URI)", value="https://example.org/htn#")
with col2:
    include_geo = st.toggle("Include geo (lat/lon) on Locations", value=True)

with st.expander("🔧 Mapping override (JSON)", expanded=False):
    st.caption("Upload a JSON to override any of the default class or predicate URIs.")
    st.code(json.dumps(default_mapping(base_uri), indent=2), language="json")
    map_file = st.file_uploader("Upload mapping JSON (optional)", type=["json"])
    mapping_override = None
    if map_file is not None:
        try:
            mapping_override = json.load(map_file)
            st.success("Mapping JSON loaded.")
        except Exception as e:
            st.error(f"Invalid mapping JSON: {e}")

with st.expander("➕ Merge with your TTL (optional)", expanded=False):
    st.caption("Upload a Turtle (.ttl) file to be **unioned** with the generated graph (your triples are preserved).")
    ttl_file = st.file_uploader("Upload TTL", type=["ttl"])
    user_ttl = None
    if ttl_file is not None:
        try:
            user_ttl = ttl_file.read().decode("utf-8", errors="ignore")
            st.success("TTL loaded.")
        except Exception as e:
            st.error(f"Failed to read TTL: {e}")

# ---------- Build ----------
build = st.button("🚀 Build Ontology (TTL)", type="primary")
if build:
    try:
        with st.spinner("Building RDF graph and serializing to Turtle..."):
            ttl, stats = build_and_export_ttl(
                dataset_ids=src_ids,
                base_uri=base_uri,
                mapping_override=mapping_override,
                include_geo=include_geo,
                user_ttl_merge=user_ttl,
            )
        st.success("Ontology built.")

        st.subheader("3) Preview & download")
        st.write(f"**Triples summary** — Victims: {stats['victims']:,} | Locations: {stats['locations']:,} | Visits: {stats['visits']:,} | Traffickers: {stats['traffickers']:,} | Chiefs: {stats['chiefs']:,}")
        st.download_button(
            "⬇️ Download TTL",
            data=ttl.encode("utf-8"),
            file_name="ontology_export.ttl",
            mime="text/turtle",
            use_container_width=True,
        )
        st.text_area("TTL preview (first ~5,000 chars)", ttl[:5000], height=300)

        if st.button("💾 Save to registry", use_container_width=True):
            rid = save_ttl_to_registry(
                name="Ontology TTL",
                ttl=ttl,
                owner=(owner_email or None),
                sources=src_ids,
            )
            st.success(f"Saved ontology TTL id: {rid}")

    except Exception as e:
        st.error(f"Ontology build failed: {e}")

st.divider()

# ---------- Past exports ----------
st.subheader("📜 Past Ontology Exports")
past = registry.list_datasets(kind="ontology_ttl")
if not past:
    st.caption("No TTL exports saved yet.")
else:
    choice = st.selectbox(
        "Select a past TTL",
        options=past,
        format_func=lambda e: f"{e.get('name')} • {e.get('id')} • {e.get('created_at')}",
    )
    if choice:
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("👁️ Preview", use_container_width=True):
                try:
                    ttl_text = registry.load_text(choice["id"])
                    st.text_area("TTL preview", ttl_text[:10000], height=420)
                except Exception as e:
                    st.error(f"Failed to load TTL: {e}")
        with c2:
            try:
                ttl_text = registry.load_text(choice["id"])
                st.download_button(
                    "⬇️ Download",
                    data=ttl_text.encode("utf-8"),
                    file_name=f"{choice['name'].replace(' ','_')}_{choice['id']}.ttl",
                    mime="text/turtle",
                    use_container_width=True,
                    key=f"dl_{choice['id']}"
                )
            except Exception:
                pass
        with c3:
            with st.popover("🗑️ Delete", use_container_width=True):
                st.write(f"Delete **{choice['name']}** ({choice['id']})?")
                confirm = st.checkbox("I understand this will permanently delete the file and registry entry.")
                if st.button("Confirm delete", disabled=not confirm):
                    try:
                        registry.delete(choice["id"])
                        st.success("Deleted. Refresh to update the list.")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                        
                        