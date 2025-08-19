# backend/api/eta.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from datetime import date

import pandas as pd

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.models.eta_model import build_duration_stats, estimate_path_durations, cumulative_arrival_dates
from backend.models.sequence_predictor import (
    NgramSequenceModel,
    build_sequences_from_df,
    last_context_for_victim,
)

COL_SID = "Serialized ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"


def train_sequence(df: pd.DataFrame) -> NgramSequenceModel:
    seqs = build_sequences_from_df(df)
    model = NgramSequenceModel(alpha=0.05)
    model.fit(seqs)
    return model


def predict_eta_for_victim(
    df: pd.DataFrame,
    victim_sid: str,
    steps: int = 3,
    default_days: int = 7,
    start_date: Optional[date] = None,
) -> Dict[str, object]:
    """
    Predict next locations via order-2 ngram, then assign ETAs (days) using learned duration stats.
    """
    if steps < 1:
        steps = 1

    # Train once (small, fast)
    seq_model = train_sequence(df)

    # History
    history = last_context_for_victim(df, victim_sid)
    if not history:
        return {"victim": victim_sid, "history": [], "predicted": [], "eta_days": [], "arrival_dates": []}

    # Predict next locations
    preds = seq_model.predict_path(history, steps=steps)
    next_locs = [l for (l, _) in preds]

    # Duration stats from the corpus
    stats = build_duration_stats(df)
    eta_days = estimate_path_durations(history, next_locs, stats=stats, default_days=default_days)

    # Arrival dates (optional)
    arr_dates: List[str] = []
    if start_date is not None:
        arr = cumulative_arrival_dates(start_date, eta_days)
        arr_dates = [d.isoformat() for d in arr]

    return {
        "victim": victim_sid,
        "history": history,
        "predicted": next_locs,
        "eta_days": eta_days,
        "arrival_dates": arr_dates,
        "stats_summary": {
            "pair_rules": len(stats.get("pair_median", {})),
            "loc_rules": len(stats.get("loc_median", {})),
            "global_median": stats.get("global_median", None),
        }
    }


def save_eta_run(
    sources: List[str],
    owner: Optional[str],
    victim_sid: str,
    next_locs: List[str],
    eta_days: List[int],
    start_date_iso: Optional[str],
) -> str:
    payload = {
        "sources": sources,
        "victim": victim_sid,
        "steps": len(next_locs),
        "start_date": start_date_iso,
        "steps_detail": [
            {"to": loc, "eta_days": int(days), "eta_weeks": round(int(days) / 7.0, 2)}
            for loc, days in zip(next_locs, eta_days)
        ],
    }
    rid = registry.save_json(
        name=f"ETA for {victim_sid}",
        payload=payload,
        kind="eta_run",
        owner=owner,
        source=",".join(sources),
    )
    return rid
