import io
import json
import zipfile

from backend.api.eval import export_report_zip


def _minimal_report():
    # Minimal but representative structure
    return {
        "meta": {"seed": 42, "k": 3, "test_frac": 0.30, "ds_ids": ["demo"]},
        "splits": {
            "available": True,
            "summary": {
                "rows": 100,
                "test_size": 30,
                "train_pool_size": 70,
                "folds": [
                    {"fold": 1, "train_size": 50, "val_size": 20},
                    {"fold": 2, "train_size": 50, "val_size": 20},
                    {"fold": 3, "train_size": 50, "val_size": 20},
                ],
            },
        },
        "sections": {
            "nlp": {
                "available": True,
                "holdout": {"f1": 0.8, "ap": 0.85, "roc_auc": 0.9, "n": 30},
                "cv": {
                    "folds": [
                        {"fold": 1, "precision": 0.8, "recall": 0.8, "f1": 0.8, "ap": 0.82, "roc_auc": 0.88, "brier": 0.15},
                        {"fold": 2, "precision": 0.79, "recall": 0.81, "f1": 0.80, "ap": 0.83, "roc_auc": 0.89, "brier": 0.16},
                    ],
                    "summary": {"f1": {"mean": 0.80, "std": 0.01}},
                },
            },
            "graph": {
                "holdout": {"jaccard": {"hits@1": 0.2, "hits@3": 0.4, "hits@5": 0.5, "mrr": 0.3, "n_eval": 50}},
                "cv": {
                    "folds": [
                        {"fold": 1, "jaccard": {"hits@1": 0.25, "hits@3": 0.45, "hits@5": 0.55, "mrr": 0.35, "n_eval": 40}},
                        {"fold": 2, "jaccard": {"hits@1": 0.22, "hits@3": 0.43, "hits@5": 0.53, "mrr": 0.33, "n_eval": 42}},
                    ],
                    "summary": {"jaccard": {"hits@1": {"mean": 0.235, "std": 0.02}}},
                },
            },
            "query": {
                "holdout": {"metrics": {"ndcg@10": 0.44, "map": 0.25}, "latency": {"p50_ms": 12.3, "p90_ms": 34.5}},
                "cv": {
                    "folds": [
                        {"fold": 1, "holdout": {"metrics": {"ndcg@10": 0.4, "map": 0.22}}},
                        {"fold": 2, "holdout": {"metrics": {"ndcg@10": 0.5, "map": 0.28}}},
                    ],
                    "summary": {"ndcg@10": {"mean": 0.45, "std": 0.07}},
                },
            },
        },
    }


def test_export_report_zip_contains_expected_files_and_valid_json():
    report = _minimal_report()
    blob = export_report_zip(report)
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0

    with zipfile.ZipFile(io.BytesIO(blob), "r") as z:
        names = set(z.namelist())
        # Always present
        assert "report.json" in names
        # Optional tables (present given our minimal report)
        assert "tables/splits_folds.csv" in names
        assert "tables/nlp_cv_folds.csv" in names
        assert "tables/graph_cv_folds.csv" in names
        assert "tables/query_cv_folds.csv" in names

        # JSON round-trip
        data = json.loads(z.read("report.json").decode("utf-8"))
        assert data["meta"]["k"] == 3
        assert "sections" in data

        # CSV headers exist (basic smoke check)
        for csv_name in [
            "tables/splits_folds.csv",
            "tables/nlp_cv_folds.csv",
            "tables/graph_cv_folds.csv",
            "tables/query_cv_folds.csv",
        ]:
            txt = z.read(csv_name).decode("utf-8")
            # should have at least a header and one newline
            assert "\n" in txt and "," in txt.splitlines()[0]
