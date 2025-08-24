import math
import numpy as np
import pandas as pd

from eval_harness.runner import run_all, RunnerConfig
from eval_harness.components.query_eval import eval_queries as query_eval


class _FakeRegistry:
    """
    Minimal registry exposing a processed dataset and a graded queries set.
    Compatible with DataAccess and query evaluator.
    """
    def __init__(self, df_processed: pd.DataFrame, queries_rows: list[dict]):
        self._processed = df_processed.reset_index(drop=True)
        self._queries = queries_rows

    # Listing
    def list_datasets(self, kind=None):
        items = [
            {"id": "proc1", "name": "processed-toy", "kind": "processed"},
            {"id": "queries1", "name": "graded-queries", "kind": "queries"},
        ]
        if kind is None:
            return items
        return [x for x in items if x["kind"] == kind]

    def list_all(self):
        return self.list_datasets(kind=None)

    # Loaders
    def load_json(self, ds_id):
        if ds_id == "proc1":
            return self._processed.to_dict(orient="records")
        if ds_id == "queries1":
            return self._queries
        raise KeyError(ds_id)


def _make_processed_df(n_docs=30, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_docs):
        label = (i % 2)
        text = ("risk danger incident " if label else "safe routine normal ") + f"doc {i}"
        rows.append({"doc_id": f"D{i:04d}", "text": text})
    # add a few extra fields to mimic integrated corpus
    df = pd.DataFrame(rows)
    df["sid"] = (np.arange(n_docs) % 10).astype(str)
    df["pid"] = (100 + (np.arange(n_docs) % 7)).astype(str)
    return df


def _make_queries(df: pd.DataFrame, n=6):
    ids = df["doc_id"].tolist()
    out = []
    for i in range(n):
        # alternate between risk-ish and safe-ish to give graded labels
        if i % 2 == 0:
            rel = [{"doc_id": ids[(i + 1) % len(ids)], "grade": 3},
                   {"doc_id": ids[(i + 3) % len(ids)], "grade": 1}]
            q = "risk incident"
        else:
            rel = [{"doc_id": ids[(i + 2) % len(ids)], "grade": 2}]
            q = "safe routine"
        out.append({"qid": f"Q{i+1:03d}", "text": q, "relevance": rel})
    return out


def _is_finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def test_query_eval_direct_local_index_or_fallback():
    df = _make_processed_df()
    queries = _make_queries(df)
    reg = _FakeRegistry(df, queries)

    # Direct call to evaluator with provided df as the corpus (hold-out style)
    res = query_eval(registry=reg, df_processed=df, seed=5, top_k=5)
    assert res.get("available", False) is True
    assert int(res.get("n_queries", 0)) == len(queries)

    hold = res.get("holdout", {})
    metrics = hold.get("metrics", {})
    # Keys exist; values finite (zeros allowed if sklearn not installed)
    for k in ("ndcg@5", "ndcg@10", "map", "mrr", "p@10", "recall@10"):
        assert _is_finite(metrics.get(k)), f"missing/NaN metric {k}"

    lat = hold.get("latency", {})
    assert _is_finite(lat.get("p50_ms", 0.0))
    assert _is_finite(lat.get("p90_ms", 0.0))


def test_query_eval_via_runner_with_cv_summary():
    df = _make_processed_df()
    queries = _make_queries(df)
    reg = _FakeRegistry(df, queries)

    cfg = RunnerConfig(seed=3, k=3, test_frac=0.30, graph_max_samples=40, query_top_k=5)
    report = run_all(["proc1"], reg, cfg=cfg)

    # Section present
    qsec = report.get("sections", {}).get("query", {})
    assert qsec, "Query section missing from runner output"

    # Hold-out metrics exist
    hold = qsec.get("holdout", {})
    metrics = hold.get("metrics", {})
    for k in ("ndcg@5", "ndcg@10", "map", "mrr", "p@10", "recall@10"):
        assert k in metrics

    # CV summary present with the same keys (zeros allowed)
    cvs = qsec.get("cv", {}).get("summary", {})
    for k in ("ndcg@5", "ndcg@10", "map", "mrr", "p@10", "recall@10"):
        assert k in cvs and "mean" in cvs[k] and "std" in cvs[k]
        assert _is_finite(cvs[k]["mean"])
        assert _is_finite(cvs[k]["std"])
