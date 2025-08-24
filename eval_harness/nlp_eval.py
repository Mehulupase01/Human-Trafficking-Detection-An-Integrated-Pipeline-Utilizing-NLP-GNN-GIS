from __future__ import annotations
"""
NLP Extraction Evaluation (no supervised label needed)

We score your extracted actors (from "Perpetrators (NLP)"/"Chiefs (NLP)")
against the *reference* actor mentions embedded in raw text fields like:
- "Name of the Perpetrators involved"
- "Human traffickers/ Chief of places"
- "Hierarchy of Perpetrators"

Method
------
- For each row, we build:
    truth = set of reference names parsed from the raw fields
    pred  = set of names from the NLP list columns
- We match pred↔truth with greedy best-match (string similarity).
- Similarity = RapidFuzz token_set_ratio if available; else difflib ratio.
- We sweep thresholds T ∈ {0.60, 0.70, 0.80, 0.90, 0.95}
  and compute micro P/R/F1; pick the T with max F1 for the "operating point".
- We compute:
    * Hold-out (30% test) metrics
    * CV (K-fold) metrics (on the 70%) for stability

Output structure (compatible with the page):
{
  "available": True,
  "mode": "extraction",
  "columns": {...},
  "holdout": {
      "precision": float,
      "recall": float,
      "f1": float,
      "ap": float,           # area under PR across thresholds (approx)
      "threshold": float,    # best-F1 threshold
      "confusion": {"tp": int, "fp": int, "fn": int, "tn": 0, "n": int},
      "curves": {"pr": {"precision": [...], "recall": [...], "thresholds": [...]}},
      "n_rows": int,
      "n_pred_mentions": int,
      "n_true_mentions": int,
  },
  "cv": {
      "folds": [{"fold": i, "precision":..., "recall":..., "f1":..., "ap":...}, ...],
      "summary": {"precision":{"mean":...,"std":...}, ...}
  }
}
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import re

import numpy as np
import pandas as pd

from eval_harness.column_resolver import resolve, parse_listish


# ---------------------- similarity ----------------------

def _sim(a: str, b: str) -> float:
    """
    Similarity in [0,1]. Prefer RapidFuzz if present; else difflib.
    """
    try:
        from rapidfuzz.fuzz import token_set_ratio  # type: ignore
        return float(token_set_ratio(a, b)) / 100.0
    except Exception:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------- reference parsing ----------------------

# Trivial stopwords/titles likely present in raw text fields
_REF_STOP = {
    "mr", "mrs", "ms", "dr", "chief", "trafficker", "traffickers",
    "human", "of", "and", "the", "unknown", "none", "n/a", "na"
}

_SPLIT_RE = re.compile(r"[;,/|]| and | & |\n|\r", flags=re.IGNORECASE)
_CAPSEQ_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

def _clean_name(s: str) -> str:
    s = s.strip().strip("'\"")
    s = re.sub(r"\s+", " ", s)
    return s

def _is_bad(token: str) -> bool:
    t = token.lower().strip(" .")
    return (not t) or (t in _REF_STOP) or (len(t) <= 1)

def _names_from_text(cell: Any) -> List[str]:
    """
    Extract name-like strings from a raw text cell:
    - If cell looks like a Python list string -> parse_listish
    - Else: take quoted tokens, otherwise split on delimiters,
            plus capture Capitalized Name sequences as a fallback.
    """
    if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []

    # If bracketed list -> parse like listish
    if s.startswith("[") and ("]" in s):
        return parse_listish(s)

    out: List[str] = []

    # Quoted tokens first
    quoted = re.findall(r"""['"]([^'"]{2,})['"]""", s)
    out += quoted

    # Delimiter split
    parts = _SPLIT_RE.split(s)
    out += [p for p in parts if p and len(p.strip()) >= 2]

    # Capitalized sequences (e.g., "Walid Medhanie")
    caps = _CAPSEQ_RE.findall(s)
    out += caps

    # Clean + filter
    cleaned = []
    seen = set()
    for cand in out:
        cc = _clean_name(cand)
        if _is_bad(cc):
            continue
        key = cc.lower()
        if key not in seen:
            cleaned.append(cc)
            seen.add(key)
    return cleaned


def _truth_from_row(row: pd.Series, ref_cols: Sequence[str]) -> List[str]:
    names: List[str] = []
    for c in ref_cols:
        if c in row and pd.notna(row[c]):
            names += _names_from_text(row[c])
    # uniquify
    seen = set()
    out = []
    for n in names:
        k = n.lower()
        if k not in seen:
            out.append(n)
            seen.add(k)
    return out


# ---------------------- matching + metrics ----------------------

def _greedy_match(preds: List[str], truths: List[str], thr: float) -> Tuple[int, int, int, List[Tuple[str,str,float]]]:
    """
    Greedy one-to-one match of preds to truths by best similarity >= thr.
    Returns (tp, fp, fn, matches).
    """
    tp = 0
    matches: List[Tuple[str,str,float]] = []
    used_truth = set()
    # sort preds by best-possible similarity to any truth (desc) to make greedy more stable
    order = sorted(
        [(p, max((_sim(p, t) for t in truths), default=0.0)) for p in preds],
        key=lambda x: x[1], reverse=True,
    )
    for p, _ in order:
        best_t = None
        best_s = 0.0
        for j, t in enumerate(truths):
            if j in used_truth:
                continue
            s = _sim(p, t)
            if s > best_s:
                best_s, best_t = s, (j, t)
        if best_t is not None and best_s >= thr:
            used_truth.add(best_t[0])
            tp += 1
            matches.append((p, best_t[1], float(best_s)))
    fp = len(preds) - tp
    fn = len(truths) - tp
    return tp, fp, fn, matches


def _micro_scores(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def _evaluate_block(df: pd.DataFrame, pred_col: str, ref_cols: Sequence[str], thresholds: Sequence[float]) -> Dict[str, Any]:
    """
    Evaluate over all rows in df for all thresholds; return best operating point + PR curve + AP.
    """
    totals = {thr: {"tp":0, "fp":0, "fn":0} for thr in thresholds}
    n_rows = 0
    n_pred_mentions = 0
    n_true_mentions = 0

    for _, row in df.iterrows():
        preds = row.get(pred_col) or []
        truths = _truth_from_row(row, ref_cols)
        if not isinstance(preds, (list, tuple, set)):
            preds = []
        preds = [str(x).strip() for x in preds if str(x).strip()]
        n_rows += 1
        n_pred_mentions += len(preds)
        n_true_mentions += len(truths)
        for thr in thresholds:
            tp, fp, fn, _ = _greedy_match(preds, truths, thr)
            totals[thr]["tp"] += tp
            totals[thr]["fp"] += fp
            totals[thr]["fn"] += fn

    # PR curve, find best threshold by F1, approximate AP (area under PR vs recall)
    curve = {"precision": [], "recall": [], "thresholds": []}
    best = {"f1": -1.0, "thr": None, "precision": 0.0, "recall": 0.0, "tp":0, "fp":0, "fn":0}
    for thr in thresholds:
        tp, fp, fn = totals[thr]["tp"], totals[thr]["fp"], totals[thr]["fn"]
        p, r, f1 = _micro_scores(tp, fp, fn)
        curve["precision"].append(p)
        curve["recall"].append(r)
        curve["thresholds"].append(thr)
        if f1 > best["f1"]:
            best.update({"f1": f1, "thr": thr, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn})

    # approximate AP with trapezoidal integration over recall
    if len(curve["recall"]) >= 2:
        # sort by recall
        idx = np.argsort(curve["recall"])
        rec = np.asarray(curve["recall"])[idx]
        prec = np.asarray(curve["precision"])[idx]
        ap = float(np.trapz(prec, rec))  # area
    else:
        ap = 0.0

    return {
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "ap": float(ap),
        "threshold": float(best["thr"] or 0.80),
        "confusion": {"tp": int(best["tp"]), "fp": int(best["fp"]), "fn": int(best["fn"]), "tn": 0, "n": int(best["tp"]+best["fp"]+best["fn"])},
        "curves": {"pr": curve},
        "n_rows": int(n_rows),
        "n_pred_mentions": int(n_pred_mentions),
        "n_true_mentions": int(n_true_mentions),
    }


def eval_classification(df: pd.DataFrame, splits, seed: int = 42) -> Dict[str, Any]:
    """
    We keep the name `eval_classification` so the runner doesn't need changes.
    If no supervised label is present, we run the **extraction** evaluation.
    """
    # Resolve to get predicted actors list and see what raw fields we have
    res = resolve(df, registry=None)  # df here is already resolved by the runner, but resolve() is idempotent
    dfr = res["df"]
    C = res["columns"]

    pred_col = C.get("actors")  # list[str]
    if not pred_col or pred_col not in dfr.columns:
        return {"available": False, "reason": "no NLP-extracted actors found (predictions list unavailable)"}

    # Reference/raw text columns present in your processed CSV
    candidate_refs = [
        "Name of the Perpetrators involved",
        "Human traffickers/ Chief of places",
        "Hierarchy of Perpetrators",
    ]
    ref_cols = [c for c in candidate_refs if c in dfr.columns]
    if not ref_cols:
        return {"available": False, "reason": "no reference perpetrator text columns found (cannot score extraction)"}

    # Build hold-out / CV frames using provided indices
    test_df = dfr.iloc[splits.test_idx]
    train_pool = dfr.iloc[splits.train_idx]  # not used for training, only for reporting CV

    THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]

    # Hold-out
    hold = _evaluate_block(test_df, pred_col, ref_cols, THRESHOLDS)

    # CV over folds
    folds: List[Dict[str, Any]] = []
    for i, (tr_idx, va_idx) in enumerate(splits.folds, start=1):
        va_df = dfr.iloc[va_idx]
        met = _evaluate_block(va_df, pred_col, ref_cols, THRESHOLDS)
        folds.append({"fold": i, **{k: met[k] for k in ("precision","recall","f1","ap","threshold","n_rows","n_pred_mentions","n_true_mentions")}})

    # Summaries
    if folds:
        dfm = pd.DataFrame(folds)
        summary = {
            "precision": {"mean": float(dfm["precision"].mean()), "std": float(dfm["precision"].std(ddof=1) if len(dfm)>1 else 0.0)},
            "recall":    {"mean": float(dfm["recall"].mean()),    "std": float(dfm["recall"].std(ddof=1) if len(dfm)>1 else 0.0)},
            "f1":        {"mean": float(dfm["f1"].mean()),        "std": float(dfm["f1"].std(ddof=1) if len(dfm)>1 else 0.0)},
            "ap":        {"mean": float(dfm["ap"].mean()),        "std": float(dfm["ap"].std(ddof=1) if len(dfm)>1 else 0.0)},
        }
    else:
        summary = {}

    return {
        "available": True,
        "mode": "extraction",
        "columns": {"pred": pred_col, "ref": ref_cols},
        "holdout": hold,
        "cv": {"folds": folds, "summary": summary},
    }
