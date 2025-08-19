# backend/api/predict.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import pandas as pd

from backend.core import dataset_registry as registry
from backend.models.sequence_predictor import (
    NgramSequenceModel,
    build_sequences_from_df,
    last_context_for_victim,
)
from backend.models.link_predictor import LinkPredictor

# Canonical columns (used by insights)
COL_SID   = "Serialized ID"
COL_LOC   = "Location"
COL_ROUTE = "Route_Order"

# ---------------------------------------------------------------------
# Models: training / single-victim predictions / global insights
# ---------------------------------------------------------------------

def train_models(df: pd.DataFrame) -> Dict[str, object]:
    """
    Train both models once so the UI can reuse them for multiple queries.
    Returns a dict with keys: {"sequence": NgramSequenceModel, "links": LinkPredictor}.
    """
    # Sequence model
    seq_model = NgramSequenceModel(alpha=0.05)
    seq_model.fit(build_sequences_from_df(df))

    # Link predictor
    lp = LinkPredictor()
    lp.fit(df)

    return {"sequence": seq_model, "links": lp}


# -------- Next locations (sequence model) --------

def predict_next_locations(
    df: pd.DataFrame,
    victim_sid: str,
    steps: int = 3,
    model: Optional[NgramSequenceModel] = None,
) -> List[Tuple[str, float]]:
    """
    Predict the next N locations for a given victim.
    If 'model' is None, we fit a fresh NgramSequenceModel.
    Returns list of (location, score) in rank order.
    """
    m = model
    if m is None:
        m = NgramSequenceModel(alpha=0.05)
        m.fit(build_sequences_from_df(df))

    history = last_context_for_victim(df, victim_sid)
    if not history:
        return []
    return m.predict_path(history, steps=max(1, int(steps)))


def global_next_location_insights(
    df: pd.DataFrame,
    model: Optional[NgramSequenceModel] = None,
    steps: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each victim, predict the immediate next location (steps=1 recommended) and aggregate:
      - Top-10 next locations overall
      - Mapping: Next Location -> #Victims predicted next (descending)
    Returns (top10_df, mapping_df).
    """
    if model is None:
        model = NgramSequenceModel(alpha=0.05)
        model.fit(build_sequences_from_df(df))

    counts = Counter()
    loc_to_victims: Dict[str, set] = defaultdict(set)

    sids = sorted(df.get(COL_SID, pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    for sid in sids:
        pred = predict_next_locations(df, sid, steps=max(1, steps), model=model)
        if not pred:
            continue
        loc, _ = pred[0]
        counts[loc] += 1
        loc_to_victims[loc].add(sid)

    top10 = pd.DataFrame(counts.most_common(10), columns=["Next Location", "Victim Count"])
    mapping = (
        pd.DataFrame([(loc, len(vs)) for loc, vs in loc_to_victims.items()],
                     columns=["Next Location", "Victim Count"])
        .sort_values("Victim Count", ascending=False)
        .reset_index(drop=True)
    )
    return top10, mapping


# -------- Perpetrators (link predictor) --------

def predict_perpetrators_for_victim(
    df: pd.DataFrame,
    victim_sid: str,
    top_k: int = 3,
    model: Optional[LinkPredictor] = None,
) -> List[Tuple[str, float]]:
    """
    Predict top-K perpetrators for one victim.
    If 'model' is None, fits a fresh LinkPredictor.
    Returns list of (perpetrator, score).
    """
    lp = model or LinkPredictor()
    if model is None:
        lp.fit(df)
    return lp.predict_for_victim(str(victim_sid), top_k=max(1, int(top_k)))


def predict_perpetrators(
    df: pd.DataFrame,
    victims: List[str],
    top_k: int = 5,
    model: Optional[LinkPredictor] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Batch version: returns {victim_sid: [(perp, score), ...], ...}
    """
    lp = model or LinkPredictor()
    if model is None:
        lp.fit(df)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for v in victims:
        out[str(v)] = lp.predict_for_victim(str(v), top_k=max(1, int(top_k)))
    return out


def global_next_perp_insights(
    df: pd.DataFrame,
    model: Optional[LinkPredictor] = None,
    top_k: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each victim, take the top-1 (or top_k) predicted perpetrator and aggregate:
      - Top-10 perpetrators overall
      - Mapping: Perpetrator -> #Victims predicted
    Returns (top10_df, mapping_df).
    """
    lp = model or LinkPredictor()
    if model is None:
        lp.fit(df)

    counts = Counter()
    perp_to_victims: Dict[str, set] = defaultdict(set)

    sids = sorted(df.get(COL_SID, pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    for sid in sids:
        preds = lp.predict_for_victim(str(sid), top_k=max(1, int(top_k)))
        if not preds:
            continue
        first = preds[0][0]
        counts[first] += 1
        perp_to_victims[first].add(str(sid))

    top10 = pd.DataFrame(counts.most_common(10), columns=["Perpetrator", "Victim Count"])
    mapping = (
        pd.DataFrame([(p, len(vs)) for p, vs in perp_to_victims.items()],
                     columns=["Perpetrator", "Victim Count"])
        .sort_values("Victim Count", ascending=False)
        .reset_index(drop=True)
    )
    return top10, mapping


# ---------------------------------------------------------------------
# Persistence helpers (JSON artifacts in registry)
# ---------------------------------------------------------------------

def save_nextloc_run(
    sources: List[str],
    victim_sid: str,
    preds: List[Tuple[str, float]],
    owner: Optional[str] = None,
) -> str:
    payload = {
        "victim": str(victim_sid),
        "predicted_next_locations": [{"location": loc, "score": float(p)} for (loc, p) in preds],
    }
    rid = registry.save_json(
        name=f"NextLoc run for {victim_sid}",
        payload=payload,
        kind="prediction_run",
        owner=owner,
        source=",".join(sources),
    )
    return rid


def save_perp_run(
    sources: List[str],
    predictions: Dict[str, List[Tuple[str, float]]],
    owner: Optional[str] = None,
) -> str:
    payload = {
        "predictions": {
            v: [{"perpetrator": p, "score": float(s)} for (p, s) in rows]
            for v, rows in predictions.items()
        }
    }
    rid = registry.save_json(
        name="Perpetrator predictions",
        payload=payload,
        kind="perp_prediction_run",
        owner=owner,
        source=",".join(sources),
    )
    return rid


def save_prediction_run(
    sources: List[str],
    owner: Optional[str],
    victim_sid: str,
    next_locations: List[Tuple[str, float]],
    next_perps: List[Tuple[str, float]],
    steps: int,
    topk_perps: int,
) -> str:
    """
    Combined artifact (compatible with your earlier predictive.py).
    Saves under kind="prediction_run".
    """
    payload = {
        "sources": sources,
        "victim": str(victim_sid),
        "steps": int(steps),
        "topk_perps": int(topk_perps),
        "predicted_next_locations": [{"location": l, "confidence": float(c)} for (l, c) in next_locations],
        "predicted_next_perpetrators": [{"perpetrator": p, "score": float(s)} for (p, s) in next_perps],
    }
    pid = registry.save_json(
        name=f"Predictions for {victim_sid}",
        payload=payload,
        kind="prediction_run",
        owner=owner,
        source=",".join(sources),
    )
    return pid
