# nlp/topic_modeling.py
# Ultra-light topic "modeling" that extracts top keywords per victim/location
# without extra dependencies (TF style). Outputs a tidy table you can join.

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re
from collections import Counter

import pandas as pd

COL_SID   = "Serialized ID"
COL_LOC   = "Location"
COL_ROUTE = "Route_Order"

_WORD_RE = re.compile(r"[A-Za-z]{3,}")

def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]

def topics_per_victim(df: pd.DataFrame, text_columns: Optional[List[str]] = None, top_k: int = 8) -> pd.DataFrame:
    if text_columns is None:
        text_columns = [c for c in df.columns if "text" in c.lower() or "narrative" in c.lower()]
    rows = []
    for sid, grp in df.groupby(COL_SID):
        toks: List[str] = []
        for c in text_columns:
            if c in grp.columns:
                toks.extend([t for x in grp[c].tolist() for t in _tokenize(str(x))])
        if not toks:
            rows.append({"Serialized ID": str(sid), "Topics": []})
            continue
        ctr = Counter(toks)
        topics = [w for (w, _) in ctr.most_common(top_k)]
        rows.append({"Serialized ID": str(sid), "Topics": topics})
    return pd.DataFrame(rows)
