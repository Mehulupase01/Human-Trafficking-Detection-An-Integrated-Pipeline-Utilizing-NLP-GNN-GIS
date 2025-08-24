import math
import numpy as np
import pandas as pd

from eval_harness.components import graph_eval as ge


def _bipartite_df():
    """
    Construct a tiny bipartite edge list with duplicate observations so that
    some evaluation edges **also exist** in the training graph (required by
    the evaluator which removes the edge and ranks it back).
    """
    rows = []
    # Base edges
    base = [
        ("s1", "p1"),
        ("s1", "p2"),
        ("s2", "p1"),
        ("s2", "p2"),  # the edge we will also place in eval
        ("s3", "p3"),
        ("s4", "p3"),
    ]
    # Duplicate a few edges (these duplicates can land in eval)
    dup = [
        ("s2", "p2"),
        ("s1", "p1"),
        ("s4", "p3"),
    ]
    for sid, pid in base + dup:
        rows.append({"sid": sid, "pid": pid})
    df = pd.DataFrame(rows)
    return df


def _metrics_in_range(d: dict, keys):
    for k in keys:
        v = d.get(k, None)
        if v is None:
            return False
        try:
            x = float(v)
        except Exception:
            return False
        if not (0.0 - 1e-9) <= x <= (1.0 + 1e-9):
            return False
    return True


def test_graph_eval_on_manual_split_produces_metrics():
    df = _bipartite_df()

    # Build a manual split:
    # - TRAIN: all base edges (first 6 rows) + one duplicate (so train has the (s2,p2) edge)
    # - EVAL: remaining duplicates (which also exist in train)
    train_idx = np.arange(0, 7)   # rows 0..6 (includes the first duplicate of s2-p2)
    eval_idx = np.arange(7, len(df))  # rows 7..8 (duplicates present in train)

    rng = np.random.default_rng(0)
    res = ge._eval_on_split(
        df,
        train_idx=train_idx,
        eval_idx=eval_idx,
        heuristics=("jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"),
        max_samples=50,
        negatives_per_pos=10,
        rng=rng,
    )

    assert res.get("available", True) is True, f"graph eval unavailable: {res}"
    # We expect at least one evaluated edge
    # (skipped happens if nodes are missing; with our construction this should be >0)
    checked = False
    for heur in ("jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"):
        if heur in res:
            stats = res[heur]
            # Metrics in [0,1]
            assert _metrics_in_range(stats, ("hits@1", "hits@3", "hits@5", "mrr"))
            # n_eval present and positive
            assert int(stats.get("n_eval", 0)) >= 1
            checked = True
    assert checked, "no heuristic results returned"

    # Descriptives are separately covered by graph_descriptives, but quick smoke check:
    desc = ge.graph_descriptives(df)
    assert desc.get("available", True) is True
    assert int(desc.get("nodes", 0)) >= 2
    assert int(desc.get("edges", 0)) >= 2
