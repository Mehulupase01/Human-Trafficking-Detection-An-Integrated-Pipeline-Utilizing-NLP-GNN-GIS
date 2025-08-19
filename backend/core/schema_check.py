# backend/core/schema_check.py
"""
Simple schema validation for uploaded/master datasets.
"""

from __future__ import annotations
from typing import List
import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "Unique ID",
    "Interviewer Name",
    "Date of Interview",
    "Gender of Victim",
    "Nationality of Victim",
    "Left Home Country Year",
    "Borders Crossed",
    "City / Locations Crossed",
    "Final Location",
    "Name of the Perpetrators involved",
    "Hierarchy of Perpetrators",
    "Human traffickers/ Chief of places",
    "Time Spent in Location / Cities / Places",
]

# Accept common aliases for robustness
ALIASES = {
    "Unique ID (Victim)": "Unique ID",
    "UniqueID": "Unique ID",
    "Victim ID": "Unique ID",
    "Interviewer": "Interviewer Name",
    "Gender": "Gender of Victim",
    "Nationality": "Nationality of Victim",
    "Left Home Country (Year Range)": "Left Home Country Year",
    "Borders/Countries Crossed": "Borders Crossed",
    "Locations Crossed": "City / Locations Crossed",
    "City/Locations Crossed": "City / Locations Crossed",
    "Name of the Perpetrators": "Name of the Perpetrators involved",
    "Human traffickers / Chief of places": "Human traffickers/ Chief of places",
    "Time Spent in Location/Cities/Places": "Time Spent in Location / Cities / Places",
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # strip, collapse whitespace
    df = df.copy()
    new_cols = []
    for c in df.columns:
        cc = " ".join(str(c).strip().split())
        cc = ALIASES.get(cc, cc)
        new_cols.append(cc)
    df.columns = new_cols
    return df

def validate_schema(df: pd.DataFrame) -> None:
    cols = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
