import math
import numpy as np
import pandas as pd

from eval_harness.split_manager import build_splits
from eval_harness.components.nlp_eval import eval_classification


def _toy_df(n_pos=60, n_neg=60, seed=7):
    """
    Create a clearly separable text dataset so both the TF-IDF+LR baseline
    (if sklearn is installed) and the fallback Naive Bayes can learn.
    We also include a 'sid' column for grouping (no leakage).
    """
    rng = np.random.default_rng(seed)
    rows = []
    # positives
    for i in range(n_pos):
        rows.append(
            {
                "sid": f"s{(i % 30):02d}",
                "text": "pos token risk danger urgent " + ("extra " * (i % 3)),
                "label": 1,
            }
        )
    # negatives
    for i in range(n_neg):
        rows.append(
            {
                "sid": f"s{(i % 30):02d}",
                "text": "neg token safe normal routine " + ("filler " * (i % 2)),
                "label": 0,
            }
        )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _is_prob(x):
    try:
        return isinstance(x, (int, float)) and (0.0 - 1e-9) <= float(x) <= (1.0 + 1e-9)
    except Exception:
        return False


def test_nlp_eval_returns_metrics_and_curves():
    df = _toy_df()
    splits = build_splits(df, seed=13, k=4, test_frac=0.30)

    res = eval_classification(df, splits, seed=13)
    assert res.get("available", True) is True, f"unavailable: {res}"

    # Hold-out scalar metrics exist and are finite / in-range
    h = res["holdout"]
    for key in ("precision", "recall", "f1", "ap", "roc_auc"):
        assert _is_prob(h.get(key)), f"metric {key} missing/out of range"

    # Curves should have points
    pr = h.get("curves", {}).get("pr", {})
    roc = h.get("curves", {}).get("roc", {})
    cal = h.get("curves", {}).get("calibration", {})
    assert len(pr.get("precision", [])) > 0 and len(pr.get("recall", [])) > 0
    assert len(roc.get("tpr", [])) > 0 and len(roc.get("fpr", [])) > 0
    assert len(cal.get("bin_centers", [])) > 0

    # Confusion counts present; sums should equal n
    cm = h.get("confusion", {})
    total = cm.get("tp", 0) + cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0)
    assert total == h.get("n", 0)

    # CV summary exists; mean values are finite
    cv = res.get("cv", {}).get("summary", {})
    for k in ("precision", "recall", "f1", "ap", "roc_auc", "brier"):
        v = cv.get(k, {}).get("mean", None)
        assert v is not None and math.isfinite(float(v))
