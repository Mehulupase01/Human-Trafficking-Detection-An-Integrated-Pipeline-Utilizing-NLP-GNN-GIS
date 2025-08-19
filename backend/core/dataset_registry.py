# backend/core/dataset_registry.py
"""
A lightweight local dataset registry that persists datasets and metadata on disk.
- DataFrames as Parquet/CSV
- JSON payloads as .json
- NEW: Arbitrary text artifacts (e.g., .ttl) via save_text/load_text
"""

from __future__ import annotations
import os
import json
import time
from typing import Optional, Dict, Any, List

BASE_DIR = os.environ.get("APP_DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(BASE_DIR, exist_ok=True)

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas must be installed to use dataset_registry") from e

DATA_ROOT = os.environ.get("APP_DATA_DIR", "data")
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")
META_PATH = os.path.join(DATASETS_DIR, "_index.json")

os.makedirs(DATASETS_DIR, exist_ok=True)

def _load_meta() -> Dict[str, Any]:
    if not os.path.exists(META_PATH):
        return {"datasets": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_meta(meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def _next_id(prefix: str = "ds") -> str:
    return f"{prefix}_{int(time.time()*1000)}"

def _choose_format() -> str:
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except Exception:
        return "csv"

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
    meta["datasets"][did] = entry
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
    meta["datasets"][jid] = entry
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
    meta["datasets"][tid] = entry
    _save_meta(meta)
    return tid

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
    meta = _load_meta().get("datasets", {})
    if dataset_id not in meta:
        raise KeyError(f"Dataset id not found: {dataset_id}")
    return meta[dataset_id]

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

def delete(dataset_id: str) -> None:
    meta = _load_meta()
    entry = meta.get("datasets", {}).pop(dataset_id, None)
    if not entry:
        return
    try:
        os.remove(entry["path"])
    except Exception:
        pass
    _save_meta(meta)
