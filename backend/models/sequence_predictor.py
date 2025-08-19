# backend/models/sequence_predictor.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Iterable, Optional
from collections import defaultdict, Counter
import math


if __name__ == "__main__":
    

    def _clean_loc(v):
        if isinstance(v, list) and v:
            return str(v[0])
        return "" if v is None else str(v)

    sample_preds = [("Tripoli", 0.7), ("Libya", 0.3)]
    rows = [
        {"Rank": i + 1, "Predicted Location": _clean_loc(loc), "Score": round(float(p), 4)}
        for i, (loc, p) in enumerate(sample_preds)
    ]
    print(pd.DataFrame(rows).to_string(index=False))



def _first_token(val) -> Optional[str]:
    """
    Return the first token from a list/array-like value; otherwise a clean string; else None.
    """
    if isinstance(val, list) and val:
        v = val[0]
        return str(v).strip() if v is not None else None
    try:
        import numpy as np  # noqa
        if hasattr(val, "size") and getattr(val, "size", 0) > 0:
            v = val.tolist()[0]
            return str(v).strip() if v is not None else None
    except Exception:
        pass
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    return s or None


def _primary_loc_from_row(row: pd.Series) -> Optional[str]:
    """
    Pick the step's primary location:
    - Prefer first token from 'Locations (NLP)'
    - Fallback to raw 'Location'
    Returns clean string or None.
    """
    loc = None
    if "Locations (NLP)" in row:
        loc = _first_token(row["Locations (NLP)"])
    if not loc and "Location" in row:
        loc = _first_token(row["Location"])
    return loc


def _sequence_from_group(grp: pd.DataFrame) -> List[str]:
    """
    Build a single victim's ordered location sequence:
    - sort by Route_Order (numeric; coerced)
    - derive primary_loc per row
    - collapse consecutive duplicates
    """
    g = grp.copy()
    if "Route_Order" in g.columns:
        g["Route_Order"] = pd.to_numeric(g["Route_Order"], errors="coerce")
        g = g.sort_values("Route_Order", kind="stable")
    seq: List[str] = []
    last = None
    for _, row in g.iterrows():
        loc = _primary_loc_from_row(row)
        if not loc:
            continue
        if loc == last:
            continue
        seq.append(loc)
        last = loc
    return seq


def build_sequences_from_df(df: pd.DataFrame) -> List[List[str]]:
    """
    Build sequences for ALL victims in the dataframe.
    Uses 'Serialized ID' to group and the logic above to form paths.
    """
    if df is None or df.empty:
        return []
    d = df.copy()
    if "Serialized ID" not in d.columns:
        # try to tolerate older name
        if "Victim ID" in d.columns:
            d = d.rename(columns={"Victim ID": "Serialized ID"})
        else:
            return []
    d["Serialized ID"] = d["Serialized ID"].astype(str)
    sequences: List[List[str]] = []
    for _, grp in d.groupby("Serialized ID"):
        seq = _sequence_from_group(grp)
        if seq:
            sequences.append(seq)
    return sequences


def last_context_for_victim(df: pd.DataFrame, victim_sid: str, order: int = 2) -> List[str]:
    """
    Return the last `order` locations for a given victim (order=2 by default).
    """
    if df is None or df.empty:
        return []
    d = df[df["Serialized ID"].astype(str) == str(victim_sid)]
    if d.empty:
        return []
    seq = _sequence_from_group(d)
    if not seq:
        return []
    k = max(1, int(order))
    return seq[-k:]


# --------------------------- n-gram predictor ---------------------------

class NgramSequenceModel:
    """
    Order‑2 n-gram model with **backoff** to order‑1 and **unigram/global**,
    plus simple additive (Laplace) smoothing controlled by `alpha`.

    API:
        .fit(list_of_sequences)
        .predict_next(history, topk=3) -> List[(location, prob)]
        .predict_path(history, steps=3) -> List[(location, prob)]  # iterative top1 rollout
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        # bigram[(prev2, prev1)] -> Counter(next)
        self.bigram: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        # unigram[prev1] -> Counter(next)
        self.unigram: Dict[str, Counter] = defaultdict(Counter)
        # global next counts (marginal)
        self.global_counts: Counter = Counter()
        self.vocab: set[str] = set()
        self.fitted: bool = False

    # ------------------------ training ------------------------

    def fit(self, sequences: Iterable[Iterable[str]]) -> None:
        """
        Train on a list of sequences (each is a list of location strings).
        """
        for seq in sequences:
            prev1 = None
            prev2 = None
            for loc in seq:
                if not isinstance(loc, str) or not loc.strip():
                    continue
                loc = loc.strip()
                self.vocab.add(loc)
                self.global_counts[loc] += 1

                if prev1 is not None:
                    self.unigram[prev1][loc] += 1
                if prev1 is not None and prev2 is not None:
                    self.bigram[(prev2, prev1)][loc] += 1

                prev2, prev1 = prev1, loc

        self.fitted = True

    # ------------------------ inference ------------------------

    def predict_next(self, history: List[str], topk: int = 3) -> List[Tuple[str, float]]:
        """
        Return a ranked list of next locations for the given history.
        Backoff chain: P(x|h2,h1) → P(x|h1) → P(x) (global).
        Laplace smoothing ensures probabilities exist even if sparse.
        """
        if not self.fitted or not self.vocab:
            return []

        # sanitize history
        hist = [h.strip() for h in history if isinstance(h, str) and h.strip()]
        if not hist:
            return self._from_global(topk)

        # try order-2
        if len(hist) >= 2:
            ctx = (hist[-2], hist[-1])
            if ctx in self.bigram and self.bigram[ctx]:
                return self._ranked(self.bigram[ctx], topk)

        # try order-1
        ctx1 = hist[-1]
        if ctx1 in self.unigram and self.unigram[ctx1]:
            return self._ranked(self.unigram[ctx1], topk)

        # global
        return self._from_global(topk)

    def predict_path(self, history: List[str], steps: int = 3) -> List[Tuple[str, float]]:
        """
        Roll out the model for `steps` moves, each time feeding back the top1 guess.
        Returns a list of (next_location, probability_for_that_step).
        """
        results: List[Tuple[str, float]] = []
        hist = list(history)[:] if history else []
        for _ in range(max(1, int(steps))):
            cand = self.predict_next(hist, topk=1)
            if not cand:
                break
            loc, p = cand[0]
            results.append((loc, float(p)))
            hist.append(loc)
        return results

    # ------------------------ helpers ------------------------

    def _ranked(self, counter: Counter, topk: int) -> List[Tuple[str, float]]:
        alpha = self.alpha
        V = max(1, len(self.vocab))
        total = sum(counter.values())
        denom = total + alpha * V
        items = []
        for loc, cnt in counter.most_common(topk):
            num = cnt + alpha
            items.append((loc, num / denom))
        return items


    def _from_global(self, topk: int) -> List[Tuple[str, float]]:
        total = sum(self.global_counts.values())
        if total == 0:
            return []
        items = [(loc, cnt / total) for loc, cnt in self.global_counts.most_common(topk)]
        # already normalized over all; if we cut to topk, renormalize:
        s = sum(p for _, p in items) or 1.0
        return [(loc, p / s) for loc, p in items]


# ----------------------- convenience wrappers -----------------------

def build_sequences_from_df_safe(df: pd.DataFrame) -> List[List[str]]:
    """
    (Deprecated shim) kept for compatibility if older imports refer to this name.
    """
    return build_sequences_from_df(df)
