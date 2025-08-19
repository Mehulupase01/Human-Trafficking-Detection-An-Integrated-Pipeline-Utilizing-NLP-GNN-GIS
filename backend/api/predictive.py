# backend/api/predictive.py
# Temporary shim to keep legacy imports working. Re-exports from predict.py.
from __future__ import annotations
import warnings

from .predict import (
    train_models,
    predict_next_locations,
    global_next_location_insights,
    predict_perpetrators_for_victim,
    predict_perpetrators,
    global_next_perp_insights,
    save_nextloc_run,
    save_perp_run,
    save_prediction_run,
)

warnings.warn(
    "backend.api.predictive is deprecated; use backend.api.predict instead.",
    DeprecationWarning,
    stacklevel=2,
)
