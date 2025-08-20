# backend/utils/normalization.py
import pandas as pd

def normalize_gender(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")
    m = (
        s.astype(str).str.strip().str.lower()
         .replace({
             "m":"male","f":"female","man":"male","woman":"female",
             "female":"female","male":"male",
             "unknown":"unknown","unk":"unknown","na":"unknown",
             "none":"unknown","nan":"unknown"
         })
    )
    return m.where(m.isin(["male","female","unknown"]), "unknown")

def normalize_nationality(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")
    return (
        s.astype(str).str.strip()
         .replace({"Not specified":"Unknown","not specified":"Unknown","nan":"Unknown","NaN":"Unknown","None":"Unknown"})
         .replace(r"^\s*$", "Unknown", regex=True)
)
