# backend/api/nlp.py
"""
NLP & preprocessing pipeline (strict) + caching for identical inputs.

Outputs:
- Processed long-format DataFrame (one row per victim-location step)
- Serialized ID (HTV1, HTV2, ...) propagated to all rows for that victim
- Split & standardized Perpetrators/Chiefs lists at victim level
- Clean exploded Location with Route_Order
- Standardized Gender/Nationality; static fields propagated
- Blanks normalized
- Artifacts persisted to registry with a fingerprint for cache hits
"""

from __future__ import annotations
import os
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np

from backend.core.schema_check import normalize_headers, validate_schema
from backend.core.standardize import (
    split_and_standardize_people,
    split_and_standardize_locations,
    standardize_gender,
    standardize_nationality,
    standardize_location,
    smart_blank,
    dedupe_preserve_order,
)
from backend.core.fingerprint import compute_df_fingerprint
from backend.core import dataset_registry as registry


# Column keys (after normalize_headers)
COL_UNIQUE_ID = "Unique ID"
COL_INTERVIEWER = "Interviewer Name"
COL_DATE = "Date of Interview"
COL_GENDER = "Gender of Victim"
COL_NATIONALITY = "Nationality of Victim"
COL_LEFT_YEAR = "Left Home Country Year"
COL_BORDERS = "Borders Crossed"
COL_LOCATIONS_RAW = "City / Locations Crossed"
COL_FINAL_LOCATION = "Final Location"
COL_PERPS = "Name of the Perpetrators involved"
COL_HIERARCHY = "Hierarchy of Perpetrators"
COL_CHIEFS = "Human traffickers/ Chief of places"
COL_TIME_SPENT = "Time Spent in Location / Cities / Places"

# Output columns
COL_SERIAL_ID = "Serialized ID"
COL_LOC_NLP_LIST = "Locations (NLP)"
COL_PERPS_LIST = "Perpetrators (NLP)"
COL_CHIEFS_LIST = "Chiefs (NLP)"
COL_LOCATION = "Location"  # exploded single location
COL_ROUTE_ORDER = "Route_Order"


STATIC_FIELDS = [
    COL_INTERVIEWER, COL_DATE, COL_GENDER, COL_NATIONALITY, COL_LEFT_YEAR,
    COL_FINAL_LOCATION, COL_BORDERS, COL_TIME_SPENT,
]


def _assign_serial_ids(df: pd.DataFrame) -> Dict[str, str]:
    id_map: Dict[str, str] = {}
    counter = 1
    for uid in df[COL_UNIQUE_ID].astype(str).tolist():
        if uid not in id_map:
            id_map[uid] = f"HTV{counter}"
            counter += 1
    return id_map


def _append_final_location(route: List[str], final_loc: str) -> List[str]:
    fin = standardize_location(final_loc) if final_loc else ''
    if fin and (not route or route[-1].lower() != fin.lower()):
        route = list(route) + [fin]
    return route


def _explode_routes(df: pd.DataFrame) -> pd.DataFrame:
    loc_lists = df[COL_LOCATIONS_RAW].apply(split_and_standardize_locations)
    fin_locs = df[COL_FINAL_LOCATION].apply(standardize_location)
    combined = []
    for base_list, fin in zip(loc_lists, fin_locs):
        deduped = dedupe_preserve_order(base_list)
        deduped = _append_final_location(deduped, fin)
        combined.append(deduped)
    df[COL_LOC_NLP_LIST] = combined

    exploded = df.explode(COL_LOC_NLP_LIST, ignore_index=True)
    exploded[COL_LOCATION] = exploded[COL_LOC_NLP_LIST]
    exploded[COL_ROUTE_ORDER] = exploded.groupby(COL_SERIAL_ID).cumcount() + 1
    return exploded


def _propagate_static_fields(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_GENDER] = df[COL_GENDER].apply(standardize_gender)
    df[COL_NATIONALITY] = df[COL_NATIONALITY].apply(standardize_nationality)
    df = df.sort_values([COL_SERIAL_ID, COL_ROUTE_ORDER], kind="stable")
    for col in STATIC_FIELDS:
        df[col] = df.groupby(COL_SERIAL_ID)[col].ffill().bfill()
    return df


def _finalize_blanks(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(smart_blank)
    return df


def _load_cached_if_available(fingerprint: str, owner_email: Optional[str]) -> Optional[Tuple[pd.DataFrame, Dict[str, str]]]:
    # Find a processed dataset with same fingerprint (and owner, if provided)
    filters = {"fingerprint": fingerprint}
    if owner_email:
        filters["owner"] = owner_email
    candidates = registry.find_datasets(kind="processed", **filters)
    if not candidates:
        # fallback: ignore owner filter to maximize hit chance
        candidates = registry.find_datasets(kind="processed", fingerprint=fingerprint)

    if not candidates:
        return None

    # Use the newest match
    entry = candidates[0]
    processed_df = registry.load_df(entry["id"])

    # Find the corresponding id_map (json with kind processed_id_map and source=processed_ds_id)
    id_map_candidates = registry.find_datasets(kind="processed_id_map", source=entry["id"])
    id_map: Dict[str, str] = {}
    if id_map_candidates:
        id_map = registry.load_json(id_map_candidates[0]["id"])
    return processed_df, id_map


def process(
    raw_df: pd.DataFrame,
    owner_email: Optional[str] = None,
    source_name: Optional[str] = None,
    enable_cache: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Main entry to generate the processed long-format dataset and id map.
    Also persists the processed dataset & id_map to the local registry,
    and returns cached results if `enable_cache` and inputs match a prior run.

    Returns:
        processed_df, id_map
    """
    if raw_df is None or len(raw_df) == 0:
        raise ValueError("Empty DataFrame provided")

    # Normalize headers & validate schema
    df = normalize_headers(raw_df)
    validate_schema(df)

    # Compute a deterministic fingerprint for caching
    fingerprint = compute_df_fingerprint(df, index=False)

    # Try cache
    if enable_cache:
        cached = _load_cached_if_available(fingerprint, owner_email)
        if cached is not None:
            return cached

    # Fresh processing path
    id_map = _assign_serial_ids(df)
    df[COL_SERIAL_ID] = df[COL_UNIQUE_ID].astype(str).map(id_map)

    df[COL_PERPS_LIST] = df[COL_PERPS].apply(split_and_standardize_people)
    df[COL_CHIEFS_LIST] = df[COL_CHIEFS].apply(split_and_standardize_people)

    exploded = _explode_routes(df)
    exploded = _propagate_static_fields(exploded)
    exploded = _finalize_blanks(exploded)

    # Persist artifacts with fingerprint in metadata
    name = source_name or "Processed Dataset"
    ds_id = registry.save_df(
        name=name,
        df=exploded,
        kind="processed",
        owner=owner_email,
        source=source_name,
        extra_meta={"fingerprint": fingerprint},
    )
    _ = registry.save_json(
        name=f"{name} (id_map)",
        payload=id_map,
        kind="processed_id_map",
        owner=owner_email,
        source=ds_id,
        extra_meta={"fingerprint": fingerprint},
    )

    return exploded, id_map
