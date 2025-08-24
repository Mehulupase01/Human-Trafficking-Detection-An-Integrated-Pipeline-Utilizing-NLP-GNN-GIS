from __future__ import annotations
"""
Unified evaluation runner (hold-out + k-fold CV).

This integrates with your data via DataAccess:
- Loads processed (and raw, when needed) dataframes for selected dataset IDs.
- Builds deterministic 30% test split + GroupKFold on the remaining 70%.
- Invokes component evaluators (NLP, Graph, GIS, Query).
- Returns a single structured report dict that the Streamlit page can render.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_access import DataAccess
from .split_manager import build_splits, summarize_splits

def _maybe_import(path: str):
    try:
        mod = __import__(path, fromlist=["*"])
        return mod
    except Exception:
        return None

@dataclass
class RunnerConfig:
    seed: int = 42
    k: int = 5
    test_frac: float = 0.30
    column_overrides: Optional[Dict[str, str]] = None
    graph_max_samples: int = 300   # for link-pred benches
    enable_sections: Sequence[str] = ("nlp", "graph", "gis", "query")
    query_top_k: int = 10

def _nonempty(df: pd.DataFrame, min_rows: int = 10) -> bool:
    return isinstance(df, pd.DataFrame) and len(df) >= min_rows

def run_all(ds_ids: Sequence[str], registry, cfg: Optional[RunnerConfig] = None) -> Dict[str, Any]:
    cfg = cfg or RunnerConfig()
    da = DataAccess(registry)

    # 1) Load processed (main frame) and raw (aux) data
    df_proc = da.load_processed(ds_ids)
    df_raw = da.load_raw(ds_ids)

    if not _nonempty(df_proc):
        return {
            "meta": {"seed": cfg.seed, "k": cfg.k, "ds_ids": list(ds_ids)},
            "splits": {"available": False, "reason": "No processed data loaded"},
            "sections": {},
        }

    # 2) Build splits and summarize (on processed DF)
    splits = build_splits(df_proc, seed=cfg.seed, k=cfg.k, test_frac=cfg.test_frac)
    split_info = summarize_splits(df_proc, splits)

    sections: Dict[str, Any] = {}

    # 3) NLP
    if "nlp" in cfg.enable_sections:
        nlp_mod = _maybe_import("eval_harness.components.nlp_eval")
        if nlp_mod is not None:
            try:
                nlp_res = nlp_mod.eval_classification(
                    df_proc, splits, seed=cfg.seed, column_overrides=cfg.column_overrides
                )
                sections["nlp"] = nlp_res
            except Exception as e:
                sections["nlp"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
        else:
            sections["nlp"] = {"available": False, "reason": "nlp_eval module not present"}

    # 4) GRAPH (heuristics)
    if "graph" in cfg.enable_sections:
        graph_mod = _maybe_import("eval_harness.components.graph_eval")
        if graph_mod is not None:
            try:
                gr = graph_mod.eval_link_pred(
                    df_proc, splits,
                    heuristics=("jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"),
                    max_samples=cfg.graph_max_samples
                )
                gdesc = graph_mod.graph_descriptives(df_proc)
                sections["graph"] = {"holdout": gr.get("holdout"), "cv": gr.get("cv"), "descriptives": gdesc}
            except Exception as e:
                sections["graph"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
        else:
            sections["graph"] = {"available": False, "reason": "graph_eval module not present"}

    # 5) GIS
    if "gis" in cfg.enable_sections:
        gis_mod = _maybe_import("eval_harness.components.gis_eval")
        if gis_mod is not None:
            try:
                gres = gis_mod.eval_all(df_proc, splits)
                sections["gis"] = gres
            except Exception as e:
                sections["gis"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
        else:
            sections["gis"] = {"available": False, "reason": "gis_eval module not present"}

    # 6) QUERY
    if "query" in cfg.enable_sections:
        q_mod = _maybe_import("eval_harness.components.query_eval")
        if q_mod is not None:
            try:
                # HOLD-OUT: build index on TEST slice if using local index;
                # If app search API is available, query_eval will use it transparently.
                test_df = df_proc.iloc[np.sort(splits.test_idx)]
                q_hold = q_mod.eval_queries(
                    registry, ds_ids=list(ds_ids), df_processed=test_df,
                    seed=cfg.seed, top_k=cfg.query_top_k
                )
                # CV: build per-fold TRAIN indexes and evaluate queries on each
                cv_folds = []
                for i, (tr_idx, _va_idx) in enumerate(splits.folds, start=1):
                    tr_df = df_proc.iloc[np.sort(tr_idx)]
                    r = q_mod.eval_queries(
                        registry, ds_ids=list(ds_ids), df_processed=tr_df,
                        seed=cfg.seed, top_k=cfg.query_top_k
                    )
                    r["fold"] = i
                    cv_folds.append(r)
                # Summaries
                def _summ(metric_key: str) -> Dict[str, float]:
                    vals = []
                    for fr in cv_folds:
                        m = fr.get("holdout", {}).get("metrics", {}).get(metric_key)
                        if isinstance(m, (int, float)):
                            vals.append(float(m))
                    if not vals:
                        return {"mean": 0.0, "std": 0.0}
                    arr = np.asarray(vals, dtype=float)
                    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0)}
                q_summary = {
                    "ndcg@5": _summ("ndcg@5"),
                    "ndcg@10": _summ("ndcg@10"),
                    "map": _summ("map"),
                    "mrr": _summ("mrr"),
                    "p@10": _summ("p@10"),
                    "recall@10": _summ("recall@10"),
                }
                sections["query"] = {"holdout": q_hold, "cv": {"folds": cv_folds, "summary": q_summary}}
            except Exception as e:
                sections["query"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
        else:
            sections["query"] = {"available": False, "reason": "query_eval module not present"}

    # 7) Collate report
    report: Dict[str, Any] = {
        "meta": {
            "seed": cfg.seed, "k": cfg.k, "test_frac": cfg.test_frac,
            "ds_ids": list(ds_ids), "query_top_k": cfg.query_top_k
        },
        "splits": {"available": True, "summary": split_info, "group_col": split_info.get("group_col")},
        "sections": sections,
    }
    return report
