import sys
import types
import numpy as np
import pandas as pd

from eval_harness.data_access import DataAccess


# ---------- Fake registries with different API shapes ----------

class NewStyleRegistry:
    """Has list_datasets(kind), load_json, get_meta (with raw link)."""
    def __init__(self):
        self._proc = pd.DataFrame([{"sid": "s1", "pid": "p1", "text": "t", "label": 1}])
        self._raw = pd.DataFrame([{"raw_field": "x"}])

    def list_datasets(self, kind=None):
        items = [
            {"id": "p1", "name": "proc-1", "kind": "processed"},
            {"id": "r1", "name": "raw-1", "kind": "raw"},
        ]
        if kind is None:
            return items
        return [x for x in items if x["kind"] == kind]

    def load_json(self, ds_id):
        if ds_id == "p1":
            return self._proc.to_dict(orient="records")
        if ds_id == "r1":
            return self._raw.to_dict(orient="records")
        raise KeyError(ds_id)

    def get_meta(self, ds_id):
        if ds_id == "p1":
            return {"id": "p1", "kind": "processed", "raw_id": "r1"}
        if ds_id == "r1":
            return {"id": "r1", "kind": "raw"}
        return {}


class OldStyleRegistry:
    """Has list_all(), read_json(); no kind filter in list."""
    def __init__(self):
        self._proc = pd.DataFrame([{"sid": "s2", "pid": "p2", "text": "t2", "label": 0}])

    def list_all(self):
        return [
            {"id": "p2", "name": "proc-2", "kind": "processed"},
        ]

    def read_json(self, ds_id):
        if ds_id == "p2":
            return self._proc.to_dict(orient="records")
        raise KeyError(ds_id)


class TextRegistry:
    """Returns JSONL text via load_text()."""
    def __init__(self):
        self._jsonl = '\n'.join([
            '{"sid":"s3","pid":"p3","text":"u","label":1}',
            '{"sid":"s3","pid":"p4","text":"v","label":0}'
        ])

    def list_datasets(self, kind=None):
        items = [{"id": "p3", "name": "proc-3", "kind": "processed"}]
        return items if kind in (None, "processed") else []

    def load_text(self, ds_id):
        if ds_id == "p3":
            return self._jsonl
        raise KeyError(ds_id)


# ---------- Tests ----------

def test_new_style_registry_processed_and_raw():
    reg = NewStyleRegistry()
    da = DataAccess(reg)
    # listing with kind filter
    items = da.list(kind="processed")
    assert isinstance(items, list) and items and items[0]["kind"] == "processed"

    dfp = da.load_processed(["p1"])
    dfr = da.load_raw(["p1"])
    assert isinstance(dfp, pd.DataFrame) and not dfp.empty
    assert isinstance(dfr, pd.DataFrame) and not dfr.empty
    # raw should come from raw_id in metadata
    assert "raw_field" in dfr.columns


def test_old_style_registry_fallbacks():
    reg = OldStyleRegistry()
    da = DataAccess(reg)
    # list(kind=...) should still work via filtering the list_all output
    items = da.list(kind="processed")
    assert items and items[0]["id"] == "p2"

    dfp = da.load_processed(["p2"])
    assert isinstance(dfp, pd.DataFrame) and not dfp.empty
    assert set(["sid", "pid", "text", "label"]).issubset(set(dfp.columns))


def test_text_registry_jsonl_parsing():
    reg = TextRegistry()
    da = DataAccess(reg)
    dfp = da.load_processed(["p3"])
    assert isinstance(dfp, pd.DataFrame) and len(dfp) == 2
    assert dfp.iloc[0]["sid"] == "s3"


def test_concat_processed_frames_short_circuit(monkeypatch):
    """
    Ensure DataAccess prefers backend.api.graph_queries.concat_processed_frames if present.
    We'll inject a fake module path into sys.modules.
    """
    # Create module tree: backend, backend.api, backend.api.graph_queries
    backend_mod = types.ModuleType("backend")
    api_mod = types.ModuleType("backend.api")
    gq_mod = types.ModuleType("backend.api.graph_queries")

    # fake concat that ignores ds_ids and returns a known frame
    def concat_processed_frames(ds_ids):
        return pd.DataFrame([{"sid": "sX", "pid": "pX", "text": "Z", "label": 1}])

    gq_mod.concat_processed_frames = concat_processed_frames
    backend_mod.api = api_mod

    sys.modules["backend"] = backend_mod
    sys.modules["backend.api"] = api_mod
    sys.modules["backend.api.graph_queries"] = gq_mod

    # Registry won't be used, but provide a minimal stub
    class DummyReg:
        def list_datasets(self, kind=None): return []

    da = DataAccess(DummyReg())
    dfp = da.load_processed(["doesnt_matter"])
    assert isinstance(dfp, pd.DataFrame) and len(dfp) == 1 and dfp.iloc[0]["sid"] == "sX"

    # cleanup
    for name in ["backend.api.graph_queries", "backend.api", "backend"]:
        sys.modules.pop(name, None)


def test_load_raw_fallback_when_no_raw_id():
    class MinimalRegistry:
        def __init__(self):
            self._df = pd.DataFrame([{"x": 1}])
        def list_datasets(self, kind=None):
            return [{"id": "p4", "name": "proc-4", "kind": "processed"}]
        def load_json(self, ds_id):
            return self._df.to_dict(orient="records")
        def get_meta(self, ds_id):
            return {"id": ds_id, "kind": "processed"}  # no raw_id

    reg = MinimalRegistry()
    da = DataAccess(reg)
    dfr = da.load_raw(["p4"])
    # Falls back to loading the same id if no raw link is present
    assert isinstance(dfr, pd.DataFrame) and not dfr.empty and "x" in dfr.columns
