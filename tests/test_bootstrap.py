import numpy as np

from eval_harness.bootstrap import bootstrap_ci
from eval_harness.metrics import precision_recall_f1


def test_bootstrap_ci_basic_properties():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])  # decent F1

    def f1_metric(yt, yp):
        return precision_recall_f1(yt, yp)[2]

    res = bootstrap_ci(f1_metric, (y_true, y_pred), n_boot=500, alpha=0.10, seed=123)
    assert "mean" in res and "lo" in res and "hi" in res and "n_boot" in res
    assert res["n_boot"] == 500
    # bounds ordered and mean inside
    assert res["lo"] <= res["mean"] <= res["hi"]

    # determinism for same seed
    res2 = bootstrap_ci(f1_metric, (y_true, y_pred), n_boot=500, alpha=0.10, seed=123)
    assert res["mean"] == res2["mean"] and res["lo"] == res2["lo"] and res["hi"] == res2["hi"]


def test_bootstrap_mismatched_lengths_raises():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])  # shorter
    def f1_metric(yt, yp):
        return precision_recall_f1(yt, yp)[2]
    try:
        bootstrap_ci(f1_metric, (y_true, y_pred), n_boot=10, seed=1)
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        assert True
