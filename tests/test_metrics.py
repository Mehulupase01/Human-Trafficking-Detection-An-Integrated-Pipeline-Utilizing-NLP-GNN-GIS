import math
import numpy as np

from eval_harness.metrics import (
    precision_recall_f1,
    ap_pr,
    roc_auc,
    pr_curve_points,
    roc_curve_points,
    reliability_bins,
    expected_calibration_error,
)


def test_precision_recall_f1_basic_and_bounds():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])

    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0

    # simple sanity: if predictions match exactly -> perfect
    p2, r2, f12 = precision_recall_f1(y_true, y_true)
    assert p2 == 1.0 and r2 == 1.0 and f12 == 1.0


def test_ap_pr_and_roc_auc_monotonic_curves():
    # Construct a small ranking where positives have higher scores
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.4])

    # Metrics in [0,1]
    ap = ap_pr(y, s)
    auc = roc_auc(y, s)
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= auc <= 1.0

    # PR curve recall should be non-decreasing; precision within [0,1]
    prec, rec = pr_curve_points(y, s)
    assert len(prec) == len(rec) and len(prec) >= 2
    assert np.all((prec >= 0.0) & (prec <= 1.0))
    assert np.all(np.diff(rec) >= -1e-12)  # allow tiny numerical jitter

    # ROC curve FPR and TPR in [0,1] and FPR non-decreasing
    fpr, tpr = roc_curve_points(y, s)
    assert len(fpr) == len(tpr) and len(fpr) >= 2
    assert np.all((fpr >= 0.0) & (fpr <= 1.0))
    assert np.all((tpr >= 0.0) & (tpr <= 1.0))
    assert np.all(np.diff(fpr) >= -1e-12)


def test_reliability_bins_and_ece_properties():
    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, size=200)
    # Slightly calibrated-ish scores
    s = 0.1 + 0.8 * rng.random(size=200)
    rb = reliability_bins(y, s, n_bins=10)
    # keys present and lengths match
    for k in ("bin_centers", "confidence", "accuracy", "sizes"):
        assert k in rb
    n = len(rb["bin_centers"])
    assert n == len(rb["confidence"]) == len(rb["accuracy"]) == len(rb["sizes"])
    # Values in range
    assert np.all((np.asarray(rb["confidence"]) >= 0.0) & (np.asarray(rb["confidence"]) <= 1.0))
    assert np.all((np.asarray(rb["accuracy"]) >= 0.0) & (np.asarray(rb["accuracy"]) <= 1.0))
    # ECE finite and >= 0
    ece = expected_calibration_error(y, s, n_bins=10)
    assert math.isfinite(float(ece)) and ece >= 0.0
