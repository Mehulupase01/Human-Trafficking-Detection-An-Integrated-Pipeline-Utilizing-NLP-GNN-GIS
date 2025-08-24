import numpy as np
import pandas as pd

from eval_harness.split_manager import build_splits, summarize_splits


def make_grouped_df(n_groups=10, reps=3, seed=7):
    """
    Build a tiny DataFrame with repeated groups under 'sid' so we can
    validate the 30% hold-out + GroupKFold behavior deterministically.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(1, n_groups + 1):
        for _ in range(reps):
            rows.append(
                {
                    "sid": g,
                    "text": f"sample {g}",
                    "label": 1 if (g % 2 == 1) else 0,
                }
            )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def test_split_sizes_and_leakage():
    df = make_grouped_df(n_groups=10, reps=3, seed=11)  # 30 rows
    splits = build_splits(df, seed=42, k=5, test_frac=0.30)
    info = summarize_splits(df, splits)

    # size sanity (allow a little wiggle room due to group-based selection)
    test_frac = info["test_size"] / info["rows"]
    assert 0.25 <= test_frac <= 0.35, f"expected ~30% test; got {test_frac:.3f}"

    # no index overlap between train pool and test
    assert info["leak_train_test_overlap"] == 0

    # no val indices should overlap with test
    for f in info["folds"]:
        assert f["leak_val_test_overlap"] == 0

    # fold sizes are non-zero and sum up roughly to the train pool across folds
    fold_val_sizes = [f["val_size"] for f in info["folds"]]
    assert all(s > 0 for s in fold_val_sizes), "each fold should have validation rows"


def test_determinism():
    df = make_grouped_df(n_groups=12, reps=2, seed=13)
    a = build_splits(df, seed=123, k=4, test_frac=0.30)
    b = build_splits(df, seed=123, k=4, test_frac=0.30)

    # exact identity for same seed/params
    assert np.array_equal(a.test_idx, b.test_idx)
    assert np.array_equal(a.train_idx, b.train_idx)
    for (atr, ava), (btr, bva) in zip(a.folds, b.folds):
        assert np.array_equal(atr, btr)
        assert np.array_equal(ava, bva)
