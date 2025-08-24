from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .split_manager import build_splits, summarize_splits
from .column_resolver import resolve  # <- map your schema first

def _maybe_import(path: str):
    try:
        return __import__(path, fromlist=["*"])
    except Exception:
        return None

@dataclass
class RunnerConfig:
    seed: int = 42
    k: int = 5
    test_frac: float = 0.30
    graph_max_samples: int = 300
    query_top_k: int = 10
    enable_sections: Sequence[str] = ("nlp", "graph", "gis", "query")

def _nonempty(df: pd.DataFrame, min_rows: int = 10) -> bool:
    return isinstance(df, pd.DataFrame) and len(df) >= min_rows

def run_all(ds_ids: Sequence[str], registry, cfg: Optional[RunnerConfig] = None) -> Dict[str, Any]:
    cfg = cfg or RunnerConfig()

    # --- load processed data from registry ---
    # minimal, robust loader (works with your dataset_registry)
    def _load_processed(ids: Sequence[str]) -> pd.DataFrame:
        dfs = []
        for ds_id in ids:
            df = None
            for fn in ("load_parquet", "load_csv", "load_json"):
                try:
                    df = getattr(registry, fn)(ds_id)
                    break
                except Exception:
                    continue
            # text fallback
            if df is None:
                try:
                    txt = registry.load_text(ds_id)
                    if txt:
                        import io, pandas as pd
                        df = pd.read_csv(io.StringIO(txt))
                except Exception:
                    pass
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()

    df_proc = _load_processed(list(ds_ids))
    if not _nonempty(df_proc):
        return {"meta": {"seed": cfg.seed, "k": cfg.k, "ds_ids": list(ds_ids)},
                "splits": {"available": False, "reason": "no processed data"},
                "sections": {}}

    # --- resolve schema ONCE so splits use your true IDs ---
    res = resolve(df_proc, registry=registry)
    df_resolved = res["df"]
    cols = res["columns"]  # sid/route_order/location_name/doc_id/text/actors/eta_days/lat/lon

    df_for_splits = df_resolved.copy()
    # Guarantee a 'sid' column for grouping (avoid leakage)
    sid = cols.get("sid") or "_row_"
    if sid not in df_for_splits.columns:
        df_for_splits["sid"] = np.arange(len(df_for_splits))
    elif sid != "sid":
        df_for_splits["sid"] = df_for_splits[sid]

    splits = build_splits(df_for_splits, seed=cfg.seed, k=cfg.k, test_frac=cfg.test_frac)
    split_info = summarize_splits(df_for_splits, splits)

    sections: Dict[str, Any] = {}

    # --- NLP (extraction evaluator) ---
    if "nlp" in cfg.enable_sections:
        nlp_mod = _maybe_import("eval_harness.components.nlp_eval") or _maybe_import("eval_harness.nlp_eval")
        if nlp_mod:
            try:
                sections["nlp"] = nlp_mod.eval_classification(df_resolved, splits, seed=cfg.seed)
            except Exception as e:
                sections["nlp"] = {"available": False, "reason": f"NLP evaluator failed: {type(e).__name__}: {e}"}
        else:
            sections["nlp"] = {"available": False, "reason": "nlp_eval module not present"}

    # --- GRAPH (victim↔actor heuristics) ---
    if "graph" in cfg.enable_sections:
        mod = _maybe_import("eval_harness.components.graph_eval")
        if mod:
            try:
                sections["graph"] = mod.eval_all(df_proc, splits, registry=registry,
                                                 max_samples=cfg.graph_max_samples, seed=cfg.seed)
            except Exception as e:
                sections["graph"] = {"available": False, "reason": f"Graph evaluator failed: {type(e).__name__}: {e}"}
        else:
            sections["graph"] = {"available": False, "reason": "graph_eval module not present"}

    # --- GIS (geocode + next-loc + ETA) ---
    if "gis" in cfg.enable_sections:
        mod = _maybe_import("eval_harness.components.gis_eval")
        if mod:
            try:
                sections["gis"] = mod.eval_all(df_proc, splits, registry=registry)
            except Exception as e:
                sections["gis"] = {"available": False, "reason": f"GIS evaluator failed: {type(e).__name__}: {e}"}
        else:
            sections["gis"] = {"available": False, "reason": "gis_eval module not present"}

    # --- QUERY (TF-IDF / fallback) ---
    if "query" in cfg.enable_sections:
        mod = _maybe_import("eval_harness.components.query_eval")
        if mod:
            try:
                # hold-out on test
                test_df = df_proc.iloc[np.sort(splits.test_idx)]
                hold = mod.eval_queries(registry=registry, df_processed=test_df,
                                        seed=cfg.seed, top_k=cfg.query_top_k)
                # CV on query set (mod handles its own query folds)
                sections["query"] = hold
            except Exception as e:
                sections["query"] = {"available": False, "reason": f"Query evaluator failed: {type(e).__name__}: {e}"}
        else:
            sections["query"] = {"available": False, "reason": "query_eval module not present"}

    return {
        "meta": {"seed": cfg.seed, "k": cfg.k, "test_frac": cfg.test_frac,
                 "ds_ids": list(ds_ids), "query_top_k": cfg.query_top_k},
        "resolved_columns": cols,
        "splits": {"available": True, "summary": split_info, "group_col": "sid"},
        "sections": sections,
    }
