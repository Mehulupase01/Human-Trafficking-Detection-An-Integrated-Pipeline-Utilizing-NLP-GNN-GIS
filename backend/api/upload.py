# backend/api/upload.py
from __future__ import annotations
from typing import Optional, List, Dict
import pandas as pd
from backend.core import dataset_registry as registry

# Tolerant import: prefer backend.nlp, fall back to root-level nlp
try:
    from backend.nlp.entity_extraction import standardize_to_processed, save_processed  # if you moved NLP under backend/
except ModuleNotFoundError:
    from nlp.entity_extraction import standardize_to_processed, save_processed          # root-level nlp/

SUPPORTED_TYPES = (".csv", ".xlsx", ".xls", ".json")

def _read_single_file(file) -> pd.DataFrame:
    name = getattr(file, "name", "uploaded")
    lname = name.lower()
    if lname.endswith(".csv"):
        return pd.read_csv(file)
    if lname.endswith(".xlsx") or lname.endswith(".xls"):
        return pd.read_excel(file)
    if lname.endswith(".json"):
        try:
            return pd.read_json(file, lines=True)
        except Exception:
            file.seek(0)
            return pd.read_json(file)
    raise ValueError(f"Unsupported file type for '{name}'. Supported: {', '.join(SUPPORTED_TYPES)}")

def read_files(files: List) -> pd.DataFrame:
    frames = []
    for f in files:
        frames.append(_read_single_file(f))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

def process_dataframe(
    df: pd.DataFrame,
    *,
    already_processed: bool = False,
    extract_from_text: bool = True,
    overwrite_entities: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if already_processed:
        return standardize_to_processed(df, extract_from_text=False)
    return standardize_to_processed(
        df,
        extract_from_text=bool(extract_from_text),
        overwrite_entities=bool(overwrite_entities),
    )

def process_and_save(
    files: List,
    *,
    dataset_name: str,
    owner: Optional[str] = None,
    already_processed: bool = False,
    extract_from_text: bool = True,
    overwrite_entities: bool = True,
) -> Dict[str, object]:
    raw = read_files(files)
    proc = process_dataframe(
        raw,
        already_processed=already_processed,
        extract_from_text=extract_from_text,
        overwrite_entities=overwrite_entities,
    )
    did = save_processed(proc, name=dataset_name, owner=owner, source="upload")
    return {
        "dataset_id": did,
        "rows": len(proc),
        "victims": int(proc["Serialized ID"].nunique()) if "Serialized ID" in proc.columns else 0,
        "preview": proc.head(200),
    }
