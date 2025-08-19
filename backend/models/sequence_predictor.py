# backend/models/sequence_predictor.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import pandas as pd

COL_SID = "Serialized ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"


class NgramSequenceModel:
    """
    Order-2 (trigram) backoff to order-1 (bigram) and unigram (global) with additive smoothing.
    Deterministic, fast, and robust for small datasets.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.next_counts_1: Dict[str, Counter] = defaultdict(Counter)          # prev1 -> next
        self.next_counts_2: Dict[Tuple[str, str], Counter] = defaultdict(Counter)  # (prev2, prev1) -> next
        self.unigram_counts: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, sequences: List[List[str]]):
        for seq in sequences:
            clean = [s for s in seq if s]
            self.vocab.update(clean)
            for i in range(len(clean)):
                self.unigram_counts[clean[i]] += 1
                if i >= 1:
                    self.next_counts_1[clean[i-1]][clean[i]] += 1
                if i >= 2:
                    ctx = (clean[i-2], clean[i-1])
                    self.next_counts_2[ctx][clean[i]] += 1

    def _normalize(self, counter: Counter) -> Dict[str, float]:
        # Additive smoothing over observed keys
        total = sum(counter.values()) + self.alpha * max(len(counter), 1)
        if total == 0:
            return {}
        return {k: (v + self.alpha) / total for k, v in counter.items()}

    def predict_next_dist(self, prev2: Optional[str], prev1: Optional[str]) -> Dict[str, float]:
        # Try trigram context
        if prev2 and prev1 and (prev2, prev1) in self.next_counts_2:
            return self._normalize(self.next_counts_2[(prev2, prev1)])
        # Backoff: bigram
        if prev1 and prev1 in self.next_counts_1:
            return self._normalize(self.next_counts_1[prev1])
        # Backoff: unigram (most frequent overall next)
        if self.unigram_counts:
            # normalize top counters (limit to top 100 for stability)
            top_items = self.unigram_counts.most_common(100)
            total = sum(c for _, c in top_items)
            return {loc: c / total for loc, c in top_items if total > 0}
        return {}

    def predict_path(self, history: List[str], steps: int = 3) -> List[Tuple[str, float]]:
        """
        Iteratively pick the argmax at each step, updating context.
        Returns list of (next_location, confidence).
        """
        path: List[Tuple[str, float]] = []
        prev1 = history[-1] if len(history) >= 1 else None
        prev2 = history[-2] if len(history) >= 2 else None

        for _ in range(max(1, steps)):
            dist = self.predict_next_dist(prev2, prev1)
            if not dist:
                break
            # argmax
            next_loc = max(dist.items(), key=lambda kv: kv[1])[0]
            conf = dist[next_loc]
            path.append((next_loc, float(conf)))
            # shift context
            prev2, prev1 = prev1, next_loc
        return path


def build_sequences_from_df(df: pd.DataFrame) -> List[List[str]]:
    if not {COL_SID, COL_LOC, COL_ROUTE}.issubset(df.columns):
        raise ValueError("DataFrame missing required columns for sequences.")
    seqs: List[List[str]] = []
    for sid, grp in df.groupby(COL_SID):
        order = grp.sort_values(COL_ROUTE, kind="stable")
        seq = order[COL_LOC].astype(str).tolist()
        seqs.append(seq)
    return seqs


def last_context_for_victim(df: pd.DataFrame, victim_sid: str) -> List[str]:
    sub = df[df[COL_SID] == victim_sid].sort_values(COL_ROUTE, kind="stable")
    return sub[COL_LOC].astype(str).tolist()
