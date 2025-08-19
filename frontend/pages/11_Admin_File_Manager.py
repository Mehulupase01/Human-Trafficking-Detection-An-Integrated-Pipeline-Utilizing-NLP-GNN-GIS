# frontend/pages/11_Admin_File_Manager.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames

st.set_page_config(page_title="Admin File Manager", page_icon="🗃️", layout="wide")
st.title("🗃️ Admin File Manager")
st.caption("Preview, download, rename, and delete datasets. "
           "If the backend doesn’t support those actions, this page uses safe local overrides (soft-rename/soft-delete).")

# ---------- Sidecar overrides (works even if registry has no management API) ----------

OVERRIDE_PATH = Path("backend/.admin_overrides.json")

def _load_overrides() -> dict:
    if OVERRIDE_PATH.exists():
        try:
            return json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"renames": {}, "deleted": []}

def _save_overrides(data: dict) -> None:
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _apply_overrides(items: list[dict]) -> list[dict]:
    """Hide soft-deleted and apply soft-renames."""
    ov = _load_overrides()
    deleted = set(ov.get("deleted", []))
    renames = ov.get("renames", {})
    out = []
    for d in items:
        did = d.get("id", "")
        if did in deleted:
            continue
        if did in renames:
            d = dict(d)
            d["name"] = renames[did]
        out.append(d)
    return out

# ---------- Registry helpers with graceful fallbacks ----------

def _call_safe(fn_name: str, *args, **kwargs):
    if not hasattr(registry, fn_name):
        return False, f"registry.{fn_name}() not available"
    try:
        val = getattr(registry, fn_name)(*args, **kwargs)
        return True, val
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _list(kind: str):
    ok, val = _call_safe("list_datasets", kind=kind)
    return val if ok and isinstance(val, list) else []

def _hard_rename(dataset_id: str, new_name: str) -> tuple[bool, str]:
    for fn in ("rename_dataset",):
        ok, _ = _call_safe(fn, dataset_id, new_name)
        if ok:
            return True, fn
    ok, _ = _call_safe("update_dataset", dataset_id, {"name": new_name})
    if ok:
        return True, "update_dataset"
    ok, _ = _call_safe("set_metadata", dataset_id, name=new_name)
    if ok:
        return True, "set_metadata"
    return False, "No supported rename API found"

def _hard_delete(dataset_id: str) -> tuple[bool, str]:
    for fn in ("delete_dataset", "remove_dataset"):
        ok, _ = _call_safe(fn, dataset_id)
        if ok:
            return True, fn
    return False, "No supported delete API found"

# ---------- UI helpers ----------

def _preview_and_download(ds: dict, key_suffix: str):
    dsid = ds.get("id")
    if not dsid:
        st.warning("Missing dataset id.")
        return
    try:
        df = concat_processed_frames([dsid])
    except Exception:
        df = None
    if df is None or df.empty:
        st.warning("Could not load or dataset seems empty.")
        return
    st.dataframe(df.head(200), use_container_width=True, height=360)
    st.download_button(
        "⬇️ Download full CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ds.get('name', dsid)}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"dl_{key_suffix}_{dsid}",
    )

def _dataset_row_ui(ds: dict, kind: str):
    st.markdown("---")
    colA, colB = st.columns([3, 2])
    with colA:
        st.markdown(f"**Name:** `{ds.get('name','')}`  •  **Kind:** `{kind}`  •  **ID:** `{ds.get('id','')}`")
        st.caption(f"Owner: {ds.get('owner','—')}  •  Created: {ds.get('created_at') or ds.get('ts') or '—'}")

        # Rename form
        with st.form(key=f"rename_{kind}_{ds.get('id','')}"):
            new_name = st.text_input("New name", value=ds.get("name",""), label_visibility="collapsed")
            preferred = st.selectbox(
                "Rename mode",
                options=["Try backend rename (hard)", "Soft rename (local override)"],
                index=0,
                help="If the backend doesn’t expose a rename API, pick soft rename. "
                     "Soft rename changes only the display name in this app.",
            )
            submitted = st.form_submit_button("✏️ Apply rename")
            if submitted:
                nm = (new_name or "").strip()
                if not nm:
                    st.warning("Name cannot be empty.")
                else:
                    if preferred.startswith("Try backend"):
                        ok, how = _hard_rename(ds.get("id",""), nm)
                        if ok:
                            st.success(f"Renamed via `{how}` to: {nm}")
                            st.rerun()
                        else:
                            st.error(how)
                    else:
                        ov = _load_overrides()
                        ov.setdefault("renames", {})[ds.get("id","")] = nm
                        _save_overrides(ov)
                        st.success(f"Soft-renamed to: {nm}")
                        st.rerun()


        # Delete controls
        c1, c2 = st.columns([1, 1])
        with c1:
            confirm = st.checkbox(
                "Confirm delete",
                key=f"confirm_{kind}_{ds.get('id','')}",
                help="Tick to enable deletion for this dataset.",
            )
        with c2:
            mode = st.selectbox(
                "Delete mode",
                options=["Try backend delete (hard)", "Soft delete (hide only)"],
                index=0,
                key=f"delmode_{kind}_{ds.get('id','')}",
            )
            disabled = not confirm
            if st.button("🗑️ Delete dataset", key=f"del_{kind}_{ds.get('id','')}", disabled=disabled):
                if mode.startswith("Try backend"):
                    ok, how = _hard_delete(ds.get("id",""))
                    if ok:
                        st.success(f"Deleted via `{how}`.")
                        st.rerun()

                    else:
                        st.error(how)
                else:
                    ov = _load_overrides()
                    deleted = set(ov.get("deleted", []))
                    deleted.add(ds.get("id",""))
                    ov["deleted"] = sorted(deleted)
                    _save_overrides(ov)
                    st.success("Soft-deleted (hidden from the Admin File Manager).")
                    st.rerun()


    with colB:
        with st.expander("👁️ Preview & Download", expanded=False):
            _preview_and_download(ds, key_suffix=f"{kind}")

# ---------- Page ----------

processed = _apply_overrides(_list("processed"))
merged = _apply_overrides(_list("merged"))

if not (processed or merged):
    st.info("No uploaded files found.")
    st.stop()

tab1, tab2 = st.tabs(["Processed Datasets", "Merged Datasets"])

with tab1:
    if not processed:
        st.caption("— none —")
    else:
        try:
            processed = sorted(processed, key=lambda x: x.get("created_at") or x.get("ts") or "", reverse=True)
        except Exception:
            pass
        for ds in processed:
            _dataset_row_ui(ds, "processed")

with tab2:
    if not merged:
        st.caption("— none —")
    else:
        try:
            merged = sorted(merged, key=lambda x: x.get("created_at") or x.get("ts") or "", reverse=True)
        except Exception:
            pass
        for ds in merged:
            _dataset_row_ui(ds, "merged")
