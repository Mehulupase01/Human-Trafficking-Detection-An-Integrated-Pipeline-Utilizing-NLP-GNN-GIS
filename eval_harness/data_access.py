from __future__ import annotations
"""
Data Access Layer for evaluation.

Goals
-----
- Load both **processed** and **raw** dataframes for the selected dataset IDs.
- Work with your existing registry and helper functions even if APIs differ:
  * Tries: dataset_registry.list_datasets / list_all
  * Tries: dataset_registry.load_json / load_parquet / load_csv / load_text
  * Tries: backend.api.graph_queries.concat_processed_frames(ds_ids) for processed
  * Falls back to file paths inside dataset metadata (path/url)
- Unifies multiple dataset IDs into a **single DataFrame** with an outer-join concat.

Public API
----------
da = DataAccess(registry)
da.list(kind=None) -> list[dict]
da.load_processed(ds_ids: list[str]) -> pandas.DataFrame
da.load_raw(ds_ids: list[str]) -> pandas.DataFrame
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import io
import json
import os
import pandas as pd

# Registry is passed in from your code: backend.core.dataset_registry
# We never import it directly here so this layer is reusable.

class DataAccess:
    def __init__(self, registry):
        self.registry = registry

    # ---------- Listing ----------

    def list(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List datasets, optionally filtering by kind ("raw", "processed", "merged", ...).
        Returns a list of dicts with at least: {"id": ..., "name": ..., "kind": ...}
        """
        # new-style
        try:
            items = self.registry.list_datasets(kind=kind) or []
            return items
        except Exception:
            pass
        # old-style (no kind filter)
        try:
            items = self.registry.list_datasets() or []
        except Exception:
            try:
                items = self.registry.list_all() or []
            except Exception:
                items = []
        if kind:
            items = [x for x in items if (x.get("kind") or x.get("type")) == kind]
        return items

    # ---------- Loading helpers ----------

    def _load_by_id_any(self, ds_id: str) -> Optional[pd.DataFrame]:
        """
        Try several registry methods in order; return DataFrame or None.
        Supports JSON/JSONL/CSV/Parquet payloads or dict/records in memory.
        """
        # 1) load_json
        for fn in ("load_json", "read_json"):
            try:
                data = getattr(self.registry, fn)(ds_id)
                df = self._to_df(data)
                if df is not None:
                    return df
            except Exception:
                pass
        # 2) load_parquet / load_csv
        for fn in ("load_parquet", "load_csv"):
            try:
                data = getattr(self.registry, fn)(ds_id)
                if isinstance(data, str) and os.path.exists(data):
                    return self._read_path(data)
                if hasattr(data, "read"):  # file-like
                    return self._read_buffer(data, fmt=("parquet" if "parquet" in fn else "csv"))
            except Exception:
                pass
        # 3) load_text (json/jsonl as text)
        for fn in ("load_text", "read_text"):
            try:
                txt = getattr(self.registry, fn)(ds_id)
                if isinstance(txt, str):
                    return self._read_text(txt)
            except Exception:
                pass
        # 4) fetch metadata and look for a path/url
        meta = self._metadata(ds_id)
        if meta:
            for key in ("path", "file", "uri", "url"):
                p = meta.get(key)
                if isinstance(p, str):
                    try:
                        return self._read_path(p)
                    except Exception:
                        continue
        return None

    def _metadata(self, ds_id: str) -> Dict[str, Any]:
        # Try a few conventions
        for fn in ("get_meta", "metadata", "describe"):
            try:
                m = getattr(self.registry, fn)(ds_id)
                if isinstance(m, dict):
                    return m
            except Exception:
                pass
        # Sometimes list_datasets() returns full entries; find the matching id
        try:
            all_items = self.list(kind=None)
            for it in all_items:
                if str(it.get("id")) == str(ds_id):
                    return it
        except Exception:
            pass
        return {}

    @staticmethod
    def _to_df(obj: Any) -> Optional[pd.DataFrame]:
        # Accept dict, list of dicts, or {"data": [...]} shapes
        try:
            if isinstance(obj, pd.DataFrame):
                return obj
            if isinstance(obj, list):
                if obj and isinstance(obj[0], dict):
                    return pd.DataFrame(obj)
            if isinstance(obj, dict):
                if "data" in obj and isinstance(obj["data"], list):
                    return pd.DataFrame(obj["data"])
                # flat dict of columns?
                if all(isinstance(v, list) for v in obj.values()):
                    return pd.DataFrame(obj)
            return None
        except Exception:
            return None

    @staticmethod
    def _read_text(txt: str) -> Optional[pd.DataFrame]:
        s = txt.strip()
        # Try JSONL first (lines)
        if "\n" in s and s.lstrip().startswith("{"):
            rows = []
            for line in s.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # ignore bad line
                    pass
            if rows:
                return pd.DataFrame(rows)
        # Try JSON array/object
        try:
            obj = json.loads(s)
            return DataAccess._to_df(obj)
        except Exception:
            pass
        # Try CSV
        try:
            return pd.read_csv(io.StringIO(txt))
        except Exception:
            return None

    @staticmethod
    def _read_path(path: str) -> pd.DataFrame:
        p = path.lower()
        if p.endswith(".parquet") or p.endswith(".pq"):
            return pd.read_parquet(path)
        if p.endswith(".jsonl") or p.endswith(".ndjson"):
            return pd.read_json(path, lines=True)
        if p.endswith(".json"):
            return pd.DataFrame(json.load(open(path, "r", encoding="utf-8")))
        if p.endswith(".csv"):
            return pd.read_csv(path)
        # last resort: try to sniff
        try:
            return pd.read_parquet(path)
        except Exception:
            try:
                return pd.read_json(path, lines=True)
            except Exception:
                return pd.read_csv(path)

    @staticmethod
    def _read_buffer(fobj, fmt: str) -> pd.DataFrame:
        if fmt == "parquet":
            return pd.read_parquet(fobj)
        if fmt == "csv":
            return pd.read_csv(fobj)
        # default to CSV
        return pd.read_csv(fobj)

    # ---------- Public: processed & raw ----------

    def load_processed(self, ds_ids: Sequence[str]) -> pd.DataFrame:
        """
        Load 'processed' frames, preferring your graph_queries concat function.
        Concats across ids with outer join on columns (row-wise concat).
        """
        # Preferred path: your own concat function
        try:
            from backend.api.graph_queries import concat_processed_frames  # type: ignore
            df = concat_processed_frames(ds_ids)  # expected to return a DataFrame
            if isinstance(df, pd.DataFrame):
                return df.reset_index(drop=True)
        except Exception:
            pass

        # Otherwise, union of individual dataset payloads
        frames: List[pd.DataFrame] = []
        for ds_id in ds_ids:
            df = self._load_by_id_any(ds_id)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)

    def load_raw(self, ds_ids: Sequence[str]) -> pd.DataFrame:
        """
        Load 'raw' data for the same ids (useful for NER/label verification or gazetteer).
        Tries registry first; if meta has "raw_id" it will follow it.
        """
        frames: List[pd.DataFrame] = []
        for ds_id in ds_ids:
            # Follow link to raw dataset if present
            meta = self._metadata(ds_id)
            raw_id = meta.get("raw_id") or meta.get("source_id") or None
            if raw_id:
                df = self._load_by_id_any(raw_id)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    frames.append(df)
                    continue
            # Otherwise load the id itself (may already be raw)
            df = self._load_by_id_any(ds_id)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)
