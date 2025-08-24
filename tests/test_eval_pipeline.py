import math
import numpy as np
import pandas as pd

from eval_harness.runner import run_all, RunnerConfig


class _FakeRegistry:
    """
    Minimal in-memory registry so the DataAccess layer can load both the
    processed dataset and the queries set. We only implement methods the
    harness tries to call.
    """

    def __init__(self, processed_rows: pd.DataFrame, queries_rows: list[dict]):
        self._processed = processed_rows.reset_index(drop=True)
        self._queries = queries_rows

    # ---- list datasets ----
    def list_datasets(self, kind=None):
        items = []
        # processed dataset
        items.append({"id": "ds1", "name": "toy-processed", "kind": "processed"})
        # queries
        items.append({"id": "q1", "name": "toy-queries", "kind": "queries"})
        if kind is None:
            return items
        return [x for x in items if x["kind"] == kind]

    # ---- loaders ----
    def load_json(self, ds_id: str):
        if ds_id == "ds1":
            return self._processed.to_dict(orient="records")
        if ds_id == "q1":
            return self._queries
        raise KeyError(ds_id)

    # Fallbacks that DataAccess may try (won't be used here, but exist for safety)
    def load_text(self, ds_id: str):
        raise NotImplementedError

    def list_all(self):
        return self.list_datasets(kind=None)


def _make_processed_df():
    """
    Tiny, multi-modal processed frame:
    - NLP: 'text', 'label' (token 'pos' => label 1; 'neg' => 0)
    - Graph: (sid, pid) edges with mild structure
    - GIS: lat/lon + timestamp + start/end for ETA
    - Query: doc_id for retrieval evaluator
    """
    rows = []
    ts0 = pd.Timestamp("2024-01-01 00:00:00")
    # Build 20 rows across 8 subjects and 6 partner ids
    for i in range(1, 21):
        sid = (i % 8) + 1                # 1..8
        pid = (i % 6) + 100              # 100..105
        label = 1 if (i % 2 == 0) else 0
        text = "pos example about risk" if label == 1 else "neg example about safe"
        lat = 52.0 + (i % 5) * 0.01
        lon = 4.0 + (i % 5) * 0.02
        ts = ts0 + pd.Timedelta(hours=i)
        start = ts
        end = ts + pd.Timedelta(days=(i % 3))
        doc_id = f"D{i:04d}"
        rows.append(
            {
                "sid": sid,
                "pid": pid,
                "text": text,
                "label": label,
                "lat": lat,
                "lon": lon,
                "timestamp": ts.isoformat(),
                "start_ts": start.isoformat(),
                "end_ts": end.isoformat(),
                "doc_id": doc_id,
            }
        )
    return pd.DataFrame(rows)


def _make_queries(processed_df: pd.DataFrame) -> list[dict]:
    """
    Build 5 graded queries that reference existing doc_ids. Even if scikit-learn
    isn't installed (so the local TF-IDF fallback is disabled), the evaluator
    will still return a metrics dict with zeros — which is fine for the test.
    """
    ids = processed_df["doc_id"].tolist()
    # pick a few plausible mappings
    return [
        {
            "qid": "Q001",
            "text": "risk example",
            "relevance": [{"doc_id": ids[1], "grade": 3}, {"doc_id": ids[3], "grade": 2}],
        },
        {
            "qid": "Q002",
            "text": "safe example",
            "relevance": [{"doc_id": ids[0], "grade": 2}, {"doc_id": ids[2], "grade": 1}],
        },
        {
            "qid": "Q003",
            "text": "incident route",
            "relevance": [{"doc_id": ids[5], "grade": 1}],
        },
        {
            "qid": "Q004",
            "text": "transaction over threshold",
            "relevance": [{"doc_id": ids[7], "grade": 2}],
        },
        {
            "qid": "Q005",
            "text": "warehouse river",
            "relevance": [{"doc_id": ids[9], "grade": 1}],
        },
    ]


def _is_finite_number(x):
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def test_end_to_end_runner_minimal():
    # Prepare data + registry
    df = _make_processed_df()
    queries = _make_queries(df)
    reg = _FakeRegistry(df, queries)

    # Run with small K to keep it light
    cfg = RunnerConfig(seed=7, k=3, test_frac=0.30, graph_max_samples=50, query_top_k=5)
    report = run_all(["ds1"], reg, cfg=cfg)

    # Basic structure
    assert "meta" in report and "sections" in report and "splits" in report
    assert report["splits"]["available"] is True
    assert set(report["meta"]["ds_ids"]) == {"ds1"}

    sections = report["sections"]
    for key in ("nlp", "graph", "gis", "query"):
        assert key in sections, f"missing section {key}"

    # ---- NLP checks ----
    nlp = sections["nlp"]
    if nlp.get("available", True) is False and "reason" in nlp:
        # If unavailable (bad columns), fail explicitly to surface schema drift
        raise AssertionError(f"NLP unavailable: {nlp.get('reason')}")
    h = nlp.get("holdout", {})
    for k in ("precision", "recall", "f1", "ap", "roc_auc", "brier"):
        assert _is_finite_number(h.get(k)), f"NLP metric {k} missing/NaN"

    # ---- Graph checks ----
    graph = sections["graph"]
    hold = graph.get("holdout", {})
    # We don't assume a particular heuristic is great; just ensure keys exist and are numeric
    for heur in ("jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"):
        stats = hold.get(heur, {})
        # Skip if the graph section reported unavailability
        if hold.get("available") is False and "reason" in hold:
            break
        for k in ("hits@1", "hits@3", "hits@5", "mrr"):
            assert _is_finite_number(stats.get(k)), f"Graph {heur} metric {k} missing/NaN"

    # ---- GIS checks ----
    gis = sections["gis"]
    # geocode rate present
    ge = gis.get("geocode", {})
    assert ge.get("available", True) is True
    assert _is_finite_number(ge.get("rate", 0.0))
    # next-loc + ETA holdout present
    nxh = gis.get("nextloc", {}).get("holdout", {})
    eta = gis.get("eta", {}).get("holdout", {})
    if nxh.get("available", True):
        for k in ("acc@1", "acc@3"):
            assert _is_finite_number(nxh.get(k)), f"GIS next-loc {k} missing/NaN"
    if eta.get("available", True):
        assert _is_finite_number(eta.get("mae_days", 0.0))

    # ---- Query checks ----
    q = sections["query"]
    holdq = q.get("holdout", {})
    # Even without sklearn (local TF-IDF), metrics keys should exist (zeros allowed)
    met = holdq.get("metrics", {})
    for k in ("ndcg@5", "ndcg@10", "map", "mrr", "p@10", "recall@10"):
        assert _is_finite_number(met.get(k)), f"Query metric {k} missing/NaN"
