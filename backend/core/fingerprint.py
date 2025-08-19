# backend/core/fingerprint.py
"""
Stable fingerprinting utilities for pandas DataFrames.
Used to cache processed artifacts for identical inputs.
"""

from __future__ import annotations
import hashlib
from typing import Optional
import pandas as pd


def compute_df_fingerprint(df: pd.DataFrame, index: bool = False) -> str:
    """
    Compute a stable SHA-256 fingerprint for a DataFrame that includes:
      - column names (order matters)
      - dtypes (order matters)
      - values (order matters)
    """
    # Ensure consistent column order and dtype capture
    cols_part = "|".join([str(c) for c in df.columns])
    dtypes_part = "|".join([str(t) for t in df.dtypes])

    # Hash the values deterministically using pandas' hashing
    # (uint64 array -> bytes)
    values_hash = pd.util.hash_pandas_object(df, index=index).values.tobytes()

    h = hashlib.sha256()
    h.update(cols_part.encode("utf-8"))
    h.update(b"||")
    h.update(dtypes_part.encode("utf-8"))
    h.update(b"||")
    h.update(values_hash)
    return h.hexdigest()
