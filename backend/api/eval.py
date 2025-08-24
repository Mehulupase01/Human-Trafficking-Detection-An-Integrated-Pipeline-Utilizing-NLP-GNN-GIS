from __future__ import annotations
"""
backend/api/eval.py
Adapter around the evaluation harness + persistence helpers.

Public API
---------
run_evaluations(ds_ids: list[str], *, seed=42, k=5, test_frac=0.30,
                graph_max_samples=300, query_top_k=10) -> dict

save_evaluation_report(name: str, report: dict, owner: str | None = None,
                       sources: list[str] | None = None) -> str

export_report_zip(report: dict) -> bytes
list_past_reports() -> list[dict]
load_report(report_id: str) -> dict
"""

import io
import json
import zipfile
from typing import Any, Dict, List, Optional, Sequence

from eval_harness.runner import run_all, RunnerConfig
from backend.core import dataset_registry as registry


# ------------------------- evaluation runner -------------------------

def run_evaluations(
    ds_ids: Sequence[str],
    *,
    seed: int = 42,
    k: int = 5,
    test_frac: float = 0.30,
    graph_max_samples: int = 300,
    query_top_k: int = 10,
) -> Dict[str, Any]:
    """
    Thin wrapper used by frontends/CLI to execute the full evaluation.
    """
    cfg = RunnerConfig(
        seed=seed, k=k, test_frac=test_frac,
        graph_max_samples=graph_max_samples,
        query_top_k=query_top_k,
    )
    return run_all(ds_ids, registry, cfg=cfg)


# ----------------------------- persistence -----------------------------

def _best_effort_save_json(
    name: str,
    payload: dict,
    *,
    kind: str = "evaluation_report",
    owner: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> str:
    meta = {
        "name": name,
        "kind": kind,
        "owner": owner,
        "sources": list(sources or []),
    }
    doc = {"meta": meta, "report": payload}
    # Preferred (new-style)
    try:
        return registry.save_json(kind=kind, name=name, payload=doc)  # type: ignore[arg-type]
    except Exception:
        pass
    # Older signature
    try:
        return registry.save_json(name, doc)  # type: ignore[misc]
    except Exception:
        pass
    # Fallback to text
    try:
        return registry.save_text(kind, json.dumps(doc, ensure_ascii=False))
    except Exception as e:
        raise RuntimeError(f"Unable to persist evaluation report: {e}")

def save_evaluation_report(
    name: str,
    report: dict,
    owner: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> str:
    """
    Save the evaluation report to the registry with kind='evaluation_report'.
    Returns the registry id.
    """
    return _best_effort_save_json(name, report, kind="evaluation_report", owner=owner, sources=sources)


def list_past_reports() -> List[Dict[str, Any]]:
    """
    Return a list of saved evaluation reports from the registry.
    """
    # New-style with kind filter
    try:
        items = registry.list_datasets(kind="evaluation_report") or []
        return items
    except Exception:
        pass
    # Fallback: list all + filter
    try:
        items = registry.list_datasets() or []
    except Exception:
        try:
            items = registry.list_all() or []
        except Exception:
            items = []
    return [x for x in items if (x.get("kind") or x.get("type")) == "evaluation_report"]


def load_report(report_id: str) -> Dict[str, Any]:
    """
    Load a saved report (compatible with multiple registry APIs).
    Returns the full saved object (meta + report) if available,
    or just the report dict if that's what the registry stores.
    """
    for fn in ("load_json", "read_json"):
        try:
            data = getattr(registry, fn)(report_id)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    # Last resort: try text
    for fn in ("load_text", "read_text"):
        try:
            txt = getattr(registry, fn)(report_id)
            if isinstance(txt, str):
                return json.loads(txt)
        except Exception:
            continue
    raise RuntimeError(f"Could not load report id {report_id!r} from registry")


# ------------------------------- export --------------------------------

def _maybe_table(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Light helper: if `obj` looks like a list of dicts (table), return it.
    Otherwise None.
    """
    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
        return obj
    return None

def export_report_zip(report: dict) -> bytes:
    """
    Build an in-memory ZIP with:
      - report.json  (the full evaluation report)
      - splits_folds.csv
      - nlp_cv_folds.csv
      - graph_cv_folds.csv
      - query_cv_folds.csv
    Missing tables are skipped gracefully.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # Full JSON (prettified)
        z.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2))

        # Splits folds
        try:
            folds = report.get("splits", {}).get("summary", {}).get("folds", [])
            if folds:
                import csv
                out = io.StringIO()
                w = csv.DictWriter(out, fieldnames=sorted({k for r in folds for k in r.keys()}))
                w.writeheader()
                for r in folds:
                    w.writerow(r)
                z.writestr("tables/splits_folds.csv", out.getvalue())
        except Exception:
            pass

        # NLP CV folds
        try:
            nlp_folds = report.get("sections", {}).get("nlp", {}).get("cv", {}).get("folds", [])
            if nlp_folds:
                import csv
                out = io.StringIO()
                w = csv.DictWriter(out, fieldnames=sorted({k for r in nlp_folds for k in r.keys()}))
                w.writeheader()
                for r in nlp_folds:
                    w.writerow(r)
                z.writestr("tables/nlp_cv_folds.csv", out.getvalue())
        except Exception:
            pass

        # Graph CV folds (flatten a little)
        try:
            graph_folds = report.get("sections", {}).get("graph", {}).get("cv", {}).get("folds", [])
            if graph_folds:
                import csv
                out = io.StringIO()
                # Expand per-heuristic keys into columns where possible
                all_keys = set(["fold"])
                for r in graph_folds:
                    for heur, stats in r.items():
                        if heur in ("available", "columns", "reason", "n_candidates", "fold"):
                            continue
                        if isinstance(stats, dict):
                            for mk in ("hits@1","hits@3","hits@5","mrr","ap","roc_auc","n_eval"):
                                all_keys.add(f"{heur}.{mk}")
                w = csv.DictWriter(out, fieldnames=sorted(all_keys))
                w.writeheader()
                for r in graph_folds:
                    row = {"fold": r.get("fold")}
                    for heur, stats in r.items():
                        if heur in ("available", "columns", "reason", "n_candidates", "fold"):
                            continue
                        if isinstance(stats, dict):
                            for mk in ("hits@1","hits@3","hits@5","mrr","ap","roc_auc","n_eval"):
                                row[f"{heur}.{mk}"] = stats.get(mk)
                    w.writerow(row)
                z.writestr("tables/graph_cv_folds.csv", out.getvalue())
        except Exception:
            pass

        # Query CV folds (each fold is a small dict with 'holdout.metrics')
        try:
            q_folds = report.get("sections", {}).get("query", {}).get("cv", {}).get("folds", [])
            if q_folds:
                import csv
                rows: List[Dict[str, Any]] = []
                for r in q_folds:
                    met = r.get("holdout", {}).get("metrics", {})
                    rows.append({"fold": r.get("fold"), **met})
                out = io.StringIO()
                if rows:
                    w = csv.DictWriter(out, fieldnames=sorted({k for rr in rows for k in rr.keys()}))
                    w.writeheader()
                    for rr in rows:
                        w.writerow(rr)
                    z.writestr("tables/query_cv_folds.csv", out.getvalue())
        except Exception:
            pass

    buf.seek(0)
    return buf.read()
