from __future__ import annotations
"""
NLP evaluation (binary text classification) with:
- Hold-out (30% test) + K-fold CV on the 70% pool
- Threshold tuning by maximizing F1 on validation data
- Metrics: Precision, Recall, F1 (micro), AP (PR AUC), ROC-AUC, Brier, ECE
- Curves: PR, ROC, Calibration (reliability)
- Confusion counts

This module prefers your existing inference if provided via `model_fn`.
If no model_fn is given, it uses a TF-IDF + LogisticRegression baseline
(when scikit-learn is available) or a tiny Multinomial Naive Bayes fallback.

Public entry points
-------------------
detect_columns(df) -> dict
eval_classification(df, splits, *, model_fn=None, seed=42) -> dict
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Sequence, Tuple, List
import numpy as np
import pandas as pd

from ..metrics import (
    precision_recall_f1,
    ap_pr,
    roc_auc,
    brier_score,
    expected_calibration_error,
    pr_curve_points,
    roc_curve_points,
    confusion_counts,
)
from ..bootstrap import bootstrap_ci

# ---------------- Column detection ----------------

TEXT_COL_CANDIDATES = ["text", "content", "body", "desc", "description"]
LABEL_COL_CANDIDATES = ["label", "labels", "y", "target", "class"]

def _pick_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def detect_columns(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """Auto-detect text and label columns; allow overrides via dict."""
    overrides = overrides or {}
    text = overrides.get("text") or _pick_first(df.columns, TEXT_COL_CANDIDATES)
    label = overrides.get("label") or _pick_first(df.columns, LABEL_COL_CANDIDATES)
    return {"text": text, "label": label}

# ---------------- Simple model baselines ----------------

class _SklearnTFIDFLogReg:
    """TF-IDF + Logistic Regression using scikit-learn (preferred)."""
    def __init__(self, seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        self.vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), lowercase=True)
        self.clf = LogisticRegression(
            solver="liblinear", random_state=seed, max_iter=200, n_jobs=None
        )

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "._SklearnTFIDFLogReg":
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, np.asarray(y).astype(int))
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vec.transform(texts)
        # returns prob of positive class
        p = self.clf.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)

class _MiniMultinomialNB:
    """Tiny dependency-free fallback (bag of words with Laplace smoothing)."""
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.cw_pos = 0
        self.cw_neg = 0
        self.counts_pos: Dict[int, int] = {}
        self.counts_neg: Dict[int, int] = {}
        self.p_pos = 0.5

    @staticmethod
    def _tok(s: str) -> List[str]:
        return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split() if t]

    def _id(self, w: str) -> int:
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) + 1
        return self.vocab[w]

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "_MiniMultinomialNB":
        y = np.asarray(y).astype(int)
        self.p_pos = float(np.mean(y))
        for s, yi in zip(texts, y):
            ids = [self._id(w) for w in self._tok(s)]
            if yi == 1:
                for i in ids:
                    self.counts_pos[i] = self.counts_pos.get(i, 0) + 1
                    self.cw_pos += 1
            else:
                for i in ids:
                    self.counts_neg[i] = self.counts_neg.get(i, 0) + 1
                    self.cw_neg += 1
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        V = max(1, len(self.vocab))
        out = []
        for s in texts:
            ids = [self._id(w) for w in self._tok(s)]
            lp_pos = np.log(self.p_pos + 1e-8)
            lp_neg = np.log(1.0 - self.p_pos + 1e-8)
            for i in ids:
                cpos = self.counts_pos.get(i, 0)
                cneg = self.counts_neg.get(i, 0)
                lp_pos += np.log((cpos + 1) / (self.cw_pos + V))
                lp_neg += np.log((cneg + 1) / (self.cw_neg + V))
            # convert two-logits to prob
            m = max(lp_pos, lp_neg)
            p1 = np.exp(lp_pos - m)
            p0 = np.exp(lp_neg - m)
            out.append(float(p1 / (p1 + p0)))
        return np.asarray(out, dtype=float)

def _default_model(seed: int = 42):
    """Return the best available baseline model."""
    try:
        # Try sklearn first
        import sklearn  # noqa: F401
        return _SklearnTFIDFLogReg(seed=seed)
    except Exception:
        return _MiniMultinomialNB()

# ---------------- Threshold selection ----------------

def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose a threshold that maximizes F1 on given labels/scores."""
    order = np.argsort(-scores)
    yt = y_true[order].astype(int)
    ys = scores[order]
    # Candidate thresholds: unique score values (plus 1.0)
    candidates = np.r_[ys, [1.0]]
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        ypred = (scores >= t).astype(int)
        _, _, f1 = precision_recall_f1(y_true, ypred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t

# ---------------- Core evaluation ----------------

def _prepare_xy(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    x = df[text_col].astype(str).to_numpy()
    y = df[label_col].astype(int).to_numpy()
    return x, y

def _pack_curves(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    prec, rec = pr_curve_points(y_true, scores)
    fpr, tpr = roc_curve_points(y_true, scores)
    bins = 15
    rel = expected_calibration_error(y_true, scores, n_bins=bins)  # returns scalar
    # For plotting, we also need the raw bin stats:
    b = _reliability_bins_data(y_true, scores, n_bins=bins)
    return {
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "calibration": b,
        "ece": rel,
    }

def _reliability_bins_data(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 15) -> Dict[str, List[float]]:
    from ..metrics import reliability_bins
    rb = reliability_bins(y_true, scores, n_bins=n_bins)
    return {
        "bin_centers": rb["bin_centers"].tolist(),
        "confidence": rb["confidence"].tolist(),
        "accuracy": rb["accuracy"].tolist(),
        "sizes": [int(x) for x in rb["sizes"]],
    }

def eval_classification(
    df: pd.DataFrame,
    splits,
    *,
    model_fn: Optional[Callable[[], Any]] = None,
    seed: int = 42,
    column_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate binary text classification with a hold-out test set and K-fold CV.

    Parameters
    ----------
    df : DataFrame
        Full dataset (before splitting).
    splits : Splits
        From eval_harness.split_manager.build_splits.
    model_fn : callable returning model with .fit(texts, y) and .predict_proba(texts)
    column_overrides : dict, optional
        e.g., {"text": "my_text_col", "label": "is_positive"}.
    """
    cols = detect_columns(df, overrides=column_overrides)
    text_col, label_col = cols.get("text"), cols.get("label")
    if text_col is None or label_col is None:
        return {"available": False, "reason": "text/label columns not found", "columns": cols}

    # ---- CV on train pool ----
    cv_records = []
    for i, (tr_idx, va_idx) in enumerate(splits.folds, start=1):
        xtr, ytr = _prepare_xy(df.iloc[tr_idx], text_col, label_col)
        xva, yva = _prepare_xy(df.iloc[va_idx], text_col, label_col)

        model = (model_fn or (lambda: _default_model(seed=seed)))()
        model.fit(xtr, ytr)
        va_scores = model.predict_proba(xva)
        thr = _best_f1_threshold(yva, va_scores)

        ypred = (va_scores >= thr).astype(int)
        p, r, f1 = precision_recall_f1(yva, ypred)
        ap = ap_pr(yva, va_scores)
        auc = roc_auc(yva, va_scores)
        br = brier_score(yva, va_scores)

        cv_records.append({
            "fold": i, "threshold": float(thr),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "ap": float(ap), "roc_auc": float(auc), "brier": float(br),
            "n": int(len(yva)),
        })

    # aggregate CV
    def _agg(key: str) -> Dict[str, float]:
        arr = np.asarray([r[key] for r in cv_records], dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}
    cv_summary = {
        "k": int(len(cv_records)),
        "precision": _agg("precision"),
        "recall": _agg("recall"),
        "f1": _agg("f1"),
        "ap": _agg("ap"),
        "roc_auc": _agg("roc_auc"),
        "brier": _agg("brier"),
    }

    # Choose a stable test threshold as the mean of per-fold optima (or refit on full train with inner tuning — mean works well here)
    test_thr = float(np.mean([r["threshold"] for r in cv_records])) if cv_records else 0.5

    # ---- Hold-out test ----
    train_pool = df.iloc[splits.train_idx]
    test_df = df.iloc[splits.test_idx]
    xtr, ytr = _prepare_xy(train_pool, text_col, label_col)
    xts, yts = _prepare_xy(test_df, text_col, label_col)

    model = (model_fn or (lambda: _default_model(seed=seed)))()
    model.fit(xtr, ytr)
    ts_scores = model.predict_proba(xts)
    ts_pred = (ts_scores >= test_thr).astype(int)

    p, r, f1 = precision_recall_f1(yts, ts_pred)
    ap = ap_pr(yts, ts_scores)
    auc = roc_auc(yts, ts_scores)
    br = brier_score(yts, ts_scores)

    # bootstrap CI for F1 and AP on test set
    f1_ci = bootstrap_ci(lambda yt, yp: precision_recall_f1(yt, yp)[2], (yts, ts_pred), seed=seed)
    ap_ci = bootstrap_ci(lambda yt, ys: ap_pr(yt, ys), (yts, ts_scores), seed=seed)

    curves = _pack_curves(yts, ts_scores)
    cm = confusion_counts(yts, ts_pred)

    holdout = {
        "n": int(len(yts)),
        "threshold": float(test_thr),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "ap": float(ap), "roc_auc": float(auc), "brier": float(br),
        "f1_ci": f1_ci, "ap_ci": ap_ci,
        "curves": curves,
        "confusion": cm,
    }

    return {
        "available": True,
        "columns": cols,
        "holdout": holdout,
        "cv": {"folds": cv_records, "summary": cv_summary},
    }
