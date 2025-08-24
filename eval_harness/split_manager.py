from __future__ import annotations
"""
Deterministic splits with leakage control.

- Fixed hold-out: TEST = 30% of groups (by victim/subject/case id).
- CV on the remaining 70%: GroupKFold (default k=5), same grouping key.
- Works even if scikit-learn is missing (has a fallback GroupKFold).
- Persists split manifests back to the dataset registry (best-effort).

Public API
----------
build_splits(df, *, seed=42, k=5, test_frac=0.30, group_cols=None) -> Splits
summarize_splits(df, splits) -> dict
persist_splits(splits, name, registry) -> str
apply_indices(df, idx) -> DataFrame
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd

# ---- candidate column names for grouping (avoid leakage across these ids)
GROUP_COL_CANDIDATES = [
    "sid", "subject_id", "victim_id", "case_id", "trajectory_id"
]
TEXT_COL_CANDIDATES = ["text", "content", "body", "desc", "description"]

@dataclass
class Splits:
    """Index-based splits over the original DataFrame (0..n-1)."""
    test_idx: np.ndarray
    train_idx: np.ndarray            # 70% remainder (train+cv pool)
    folds: List[Tuple[np.ndarray, np.ndarray]]  # list of (train_idx, val_idx) for CV
    seed: int
    k: int
    group_col: str

# ------------------ helpers ------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _pick_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def _resolve_group_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override and override in df.columns:
        return override
    col = _pick_first(df.columns, GROUP_COL_CANDIDATES)
    if col:
        return col
    # last resort: hash of text (keeps sentences from same doc together)
    tcol = _pick_first(df.columns, TEXT_COL_CANDIDATES)
    if tcol:
        return tcol  # we'll hash its values below
    # absolute fallback: a single group (prevents leakage controls but stays deterministic)
    return "__row__"

def _group_labels(df: pd.DataFrame, group_col: str) -> np.ndarray:
    """Return an array of group ids for each row (string labels)."""
    if group_col == "__row__":
        return np.array([f"row:{i}" for i in range(len(df))], dtype=object)
    s = df[group_col].astype(str).fillna("NA")
    # If it's a free text column, reduce to a stable hash prefix to group by document
    if group_col in TEXT_COL_CANDIDATES:
        return s.apply(lambda x: f"tx:{abs(hash(x)) % (10**10)}").to_numpy()
    return s.to_numpy()

def _index_split_by_groups(n: int, groups: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) as integer arrays using group-wise split."""
    uniq = pd.Index(groups).unique().to_numpy()
    g_rng = _rng(seed)
    g_rng.shuffle(uniq)

    # Greedy fill until we reach target fraction in row counts (not just group counts).
    target = int(round(test_frac * n))
    test_groups = []
    seen = set()
    counts = pd.Series(groups).value_counts().to_dict()
    acc = 0
    for g in uniq:
        if g in seen:
            continue
        test_groups.append(g)
        seen.add(g)
        acc += counts.get(g, 0)
        if acc >= target and len(seen) < len(uniq):
            break
    test_mask = np.isin(groups, np.array(test_groups, dtype=object))
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    # Safety: ensure both sides non-empty
    if len(test_idx) == 0 or len(train_idx) == 0:
        # fallback: split by rows if grouping is degenerate
        all_idx = np.arange(n)
        g_rng.shuffle(all_idx)
        cutoff = max(1, int(round(test_frac * n)))
        test_idx = np.sort(all_idx[:cutoff])
        train_idx = np.sort(all_idx[cutoff:])
    else:
        test_idx = np.sort(test_idx)
        train_idx = np.sort(train_idx)
    return train_idx, test_idx

def _group_kfold_indices(train_idx: np.ndarray, groups_all: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return GroupKFold splits (train_subidx, val_subidx), indices refer to original df."""
    # Try sklearn if available
    try:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=k)
        groups = groups_all[train_idx]
        folds = []
        for tr_sub, va_sub in gkf.split(train_idx, groups=groups):
            folds.append((np.sort(train_idx[tr_sub]), np.sort(train_idx[va_sub])))
        return folds
    except Exception:
        pass

    # Fallback: simple group-wise round-robin bucketization
    groups = groups_all[train_idx]
    uniq = pd.Index(groups).unique().to_numpy()
    rng = _rng(seed)
    rng.shuffle(uniq)
    buckets: List[List[Any]] = [[] for _ in range(k)]
    for i, g in enumerate(uniq):
        buckets[i % k].append(g)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_groups = set(buckets[i])
        val_mask = np.isin(groups, np.array(list(val_groups), dtype=object))
        va_idx = train_idx[np.where(val_mask)[0]]
        tr_idx = train_idx[np.where(~val_mask)[0]]
        folds.append((np.sort(tr_idx), np.sort(va_idx)))
    return folds

# ------------------ public API ------------------

def build_splits(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    k: int = 5,
    test_frac: float = 0.30,
    group_cols: Optional[Sequence[str]] = None,
) -> Splits:
    """
    Create 30% test hold-out + GroupKFold on the remaining 70%.
    Grouping column is auto-detected unless group_cols provided.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("build_splits: input DataFrame is empty")

    group_col = _resolve_group_col(df, override=(group_cols or [None])[0])
    groups = _group_labels(df, group_col)
    n = len(df)

    train_idx, test_idx = _index_split_by_groups(n, groups, test_frac=test_frac, seed=seed)
    folds = _group_kfold_indices(train_idx, groups, k=k, seed=seed)

    return Splits(
        test_idx=test_idx,
        train_idx=train_idx,
        folds=folds,
        seed=seed,
        k=k,
        group_col=group_col,
    )

def summarize_splits(df: pd.DataFrame, splits: Splits) -> Dict[str, Any]:
    """Basic invariants and sizes; useful for UI and tests."""
    def _overlap(a: np.ndarray, b: np.ndarray) -> int:
        return len(np.intersect1d(a, b))

    info: Dict[str, Any] = {
        "rows": int(len(df)),
        "group_col": splits.group_col,
        "test_size": int(len(splits.test_idx)),
        "train_pool_size": int(len(splits.train_idx)),
        "k": splits.k,
        "seed": splits.seed,
        "leak_train_test_overlap": _overlap(splits.train_idx, splits.test_idx),
        "folds": [],
    }
    for i, (tr, va) in enumerate(splits.folds, start=1):
        info["folds"].append({
            "fold": i,
            "train_size": int(len(tr)),
            "val_size": int(len(va)),
            "leak_val_test_overlap": _overlap(va, splits.test_idx),
            "leak_val_train_overlap": _overlap(va, splits.train_idx),
        })
    return info

def apply_indices(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Return a view by integer indices (sorted)."""
    return df.iloc[np.sort(idx)]

def persist_splits(splits: Splits, name: str, registry) -> str:
    """
    Persist split membership (test/train/folds) to the dataset registry.
    Tries multiple registry method signatures; returns a registry id.
    """
    payload = {
        "name": name,
        "kind": "eval_splits",
        "seed": splits.seed,
        "k": splits.k,
        "group_col": splits.group_col,
        "test_idx": splits.test_idx.tolist(),
        "train_idx": splits.train_idx.tolist(),
        "folds": [{"train_idx": tr.tolist(), "val_idx": va.tolist()} for tr, va in splits.folds],
    }
    # Try new-style API
    try:
        return registry.save_json(kind="eval_splits", name=name, payload=payload)  # type: ignore[arg-type]
    except Exception:
        pass
    # Try simple API
    try:
        return registry.save_json(name, payload)  # type: ignore[misc]
    except Exception:
        pass
    # Fallback: text
    try:
        import json
        return registry.save_text("eval_splits", json.dumps(payload))
    except Exception as e:
        raise RuntimeError(f"Unable to persist splits: {e}")
