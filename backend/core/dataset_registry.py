# backend/core/dataset_registry.py
"""
A lightweight local dataset registry that persists datasets and metadata on disk.
- DataFrames as Parquet/CSV
- JSON payloads as .json
- NEW: Arbitrary text artifacts (e.g., .ttl) via save_text/load_text
- NEW: Management APIs (rename/update/delete) and list_artifacts
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List

BASE_DIR = os.environ.get("APP_DATA_DIR") or os.path.join(
    os.path.dirname(__file__), "..", "..", "data"
)
os.makedirs(BASE_DIR, exist_ok=True)

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas must be installed to use dataset_registry") from e

DATA_ROOT = os.environ.get("APP_DATA_DIR", "data")
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")
META_PATH = os.path.join(DATASETS_DIR, "_index.json")

os.makedirs(DATASETS_DIR, exist_ok=True)

# ------------------------- meta index helpers -------------------------

def _load_meta() -> Dict[str, Any]:
    if not os.path.exists(META_PATH):
        return {"datasets": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_meta(meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def _next_id(prefix: str = "ds") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"

def _choose_format() -> str:
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except Exception:
        return "csv"

def _get_entry(meta: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    entry = meta.get("datasets", {}).get(dataset_id)
    if not entry:
        raise KeyError(f"Dataset id not found: {dataset_id}")
    return entry

# ------------------------- save operations -------------------------

def save_df(
    name: str,
    df: "pd.DataFrame",
    kind: str,
    owner: Optional[str] = None,
    source: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = _load_meta()
    did = _next_id(prefix="ds")
    fmt = _choose_format()
    fn = f"{did}.{fmt}"
    fpath = os.path.join(DATASETS_DIR, fn)
    if fmt == "parquet":
        df.to_parquet(fpath, index=False)
    else:
        df.to_csv(fpath, index=False, encoding="utf-8")
    entry = {
        "id": did,
        "name": name,
        "kind": kind,
        "path": fpath,
        "format": fmt,
        "owner": owner,
        "source": source,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        entry.update(extra_meta)
    meta.setdefault("datasets", {})[did] = entry
    _save_meta(meta)
    return did

def save_json(
    name: str,
    payload: Dict[str, Any],
    kind: str,
    owner: Optional[str] = None,
    source: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = _load_meta()
    jid = _next_id(prefix="js")
    fn = f"{jid}.json"
    fpath = os.path.join(DATASETS_DIR, fn)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    entry = {
        "id": jid,
        "name": name,
        "kind": kind,
        "path": fpath,
        "format": "json",
        "owner": owner,
        "source": source,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        entry.update(extra_meta)
    meta.setdefault("datasets", {})[jid] = entry
    _save_meta(meta)
    return jid

# NEW: save/load arbitrary text artifacts (e.g., Turtle .ttl)
def save_text(
    name: str,
    text: str,
    kind: str,
    owner: Optional[str] = None,
    source: Optional[str] = None,
    ext: str = "txt",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = _load_meta()
    tid = _next_id(prefix="tx")
    ext = ext.lstrip(".")
    fn = f"{tid}.{ext}"
    fpath = os.path.join(DATASETS_DIR, fn)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    entry = {
        "id": tid,
        "name": name,
        "kind": kind,
        "path": fpath,
        "format": ext,
        "owner": owner,
        "source": source,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        entry.update(extra_meta)
    meta.setdefault("datasets", {})[tid] = entry
    _save_meta(meta)
    return tid

# ------------------------- listing & loading -------------------------

def list_datasets(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    meta = _load_meta()
    items = list(meta.get("datasets", {}).values())
    if kind:
        items = [it for it in items if it.get("kind") == kind]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items

def find_datasets(kind: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
    items = list_datasets(kind=kind)
    for k, v in filters.items():
        items = [it for it in items if it.get(k) == v]
    return items

def get_entry(dataset_id: str) -> Dict[str, Any]:
    return _get_entry(_load_meta(), dataset_id)

def load_df(dataset_id: str) -> "pd.DataFrame":
    meta = get_entry(dataset_id)
    path = meta["path"]
    fmt = meta.get("format", "csv")
    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format for DataFrame: {fmt}")

def load_json(dataset_id: str) -> Dict[str, Any]:
    meta = get_entry(dataset_id)
    if meta.get("format") != "json":
        raise ValueError("Dataset is not a JSON payload")
    with open(meta["path"], "r", encoding="utf-8") as f:
        return json.load(f)

# NEW:
def load_text(dataset_id: str) -> str:
    meta = get_entry(dataset_id)
    with open(meta["path"], "r", encoding="utf-8") as f:
        return f.read()

# ------------------------- management APIs -------------------------

def rename_dataset(dataset_id: str, new_name: str) -> Dict[str, Any]:
    """
    Update only the 'name' field of a dataset/artifact.
    """
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("new_name cannot be empty")
    meta = _load_meta()
    entry = _get_entry(meta, dataset_id)
    entry["name"] = new_name
    _save_meta(meta)
    return entry

def update_dataset(dataset_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge the provided fields into the entry (shallow update).
    """
    if not isinstance(patch, dict):
        raise TypeError("patch must be a dict")
    meta = _load_meta()
    entry = _get_entry(meta, dataset_id)
    entry.update(patch)
    _save_meta(meta)
    return entry

def set_metadata(dataset_id: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience wrapper to update arbitrary metadata fields.
    """
    return update_dataset(dataset_id, kwargs)

def delete_dataset(dataset_id: str, *, remove_files: bool = True) -> bool:
    """
    Hard delete: remove from registry and (optionally) delete associated file.
    Returns True if entry existed and was removed.
    """
    meta = _load_meta()
    entry = meta.get("datasets", {}).pop(dataset_id, None)
    if not entry:
        _save_meta(meta)  # keep index consistent even if nothing changed
        return False

    _save_meta(meta)

    if remove_files:
        try:
            p = entry.get("path")
            if p:
                pth = Path(p)
                if pth.is_file():
                    pth.unlink(missing_ok=True)
                elif pth.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    return True

def remove_dataset(dataset_id: str) -> bool:
    """Alias for delete_dataset(dataset_id)."""
    return delete_dataset(dataset_id)

# Legacy name kept for backward compatibility with older callers
def delete(dataset_id: str) -> None:
    delete_dataset(dataset_id, remove_files=True)

# ------------------------- artifacts listing (JSON/TEXT) -------------------------

def list_artifacts(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List non-DataFrame artifacts (typically JSON or text) saved via save_json/save_text.
    If 'kind' is provided, filter by kind.
    """
    items = list_datasets(kind=None)
    arts = [it for it in items if it.get("format") in {"json", "txt", "ttl"}]
    if kind:
        arts = [a for a in arts if a.get("kind") == kind]
    return arts
