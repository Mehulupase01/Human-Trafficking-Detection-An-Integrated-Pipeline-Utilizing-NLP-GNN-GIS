from __future__ import annotations

"""
backend/models/sequence_predictor.py

Lightweight order-3 n-gram sequence model with backoff and Laplace smoothing.
Designed for next-location prediction on trajectories (strings), with a
backward-compatible shim `predict_next_dist(prev2, prev1)` used by older code.

APIs
----
- NgramSequenceModel(alpha=0.05, weights=(0.6, 0.3, 0.1))
    * fit(sequences)
    * predict_next(history, topk=10) -> List[Tuple[str, float]]
    * predict_path(history, steps=3) -> List[Tuple[str, float]]
    * predict_next_dist(prev2, prev1, topk=50) -> Dict[str, float]   # legacy shim
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


class NgramSequenceModel:
    """
    Order-3 (trigram) model with backoff to bigram and unigram.
    Each conditional distribution uses Laplace smoothing with parameter `alpha`.
    Final prediction is a convex combination of the three conditionals using
    `weights = (w3, w2, w1)` for (tri, bi, uni).

    Notes
    -----
    - Inputs are sequences of hashable items; we cast to `str` for consistency.
    - The model is intentionally small and dependency-free for easy embedding.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be a 3-tuple of nonnegative numbers with a positive sum")

        self.alpha: float = float(alpha)
        self.weights: Tuple[float, float, float] = weights

        # counts
        self.unigram: Counter[str] = Counter()
        self.bigram: Dict[str, Counter[str]] = defaultdict(Counter)          # prev1 -> next counts
        self.trigram: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)  # (prev2, prev1) -> next counts

        # cached totals
        self.total_tokens: int = 0
        self.bigram_totals: Dict[str, int] = defaultdict(int)
        self.trigram_totals: Dict[Tuple[str, str], int] = defaultdict(int)

        # vocab
        self._vocab: set[str] = set()
        self._fitted: bool = False

    # ------------------------- Training -------------------------

    def fit(self, sequences: Iterable[Iterable[str]]) -> "NgramSequenceModel":
        """
        Fit counts from sequences.
        Parameters
        ----------
        sequences : Iterable of sequences (iterables of tokens)
        """
        # reset
        self.unigram.clear()
        self.bigram.clear()
        self.trigram.clear()
        self.bigram_totals.clear()
        self.trigram_totals.clear()
        self.total_tokens = 0
        self._vocab.clear()

        for seq in sequences:
            # normalize tokens to non-empty strings
            toks = [str(t).strip() for t in seq if str(t).strip()]
            n = len(toks)
            if n == 0:
                continue

            # unigrams
            self.unigram.update(toks)
            self.total_tokens += n
            self._vocab.update(toks)

            # bigrams
            for i in range(1, n):
                p1, nx = toks[i - 1], toks[i]
                self.bigram[p1][nx] += 1
                self.bigram_totals[p1] += 1

            # trigrams
            for i in range(2, n):
                p2, p1, nx = toks[i - 2], toks[i - 1], toks[i]
                ctx = (p2, p1)
                self.trigram[ctx][nx] += 1
                self.trigram_totals[ctx] += 1

        self._fitted = True
        return self

    # ------------------------- Helpers -------------------------

    @property
    def vocab(self) -> List[str]:
        return sorted(self._vocab)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(sequences) first.")

    def _laplace_prob(self, count: int, denom: int, V: int) -> float:
        # Laplace / add-alpha smoothing
        return (count + self.alpha) / (denom + self.alpha * max(1, V))

    # ------------------------- Prediction -------------------------

    def predict_next(self, history: List[str] | Tuple[str, ...] | None, topk: int = 10) -> List[Tuple[str, float]]:
        """
        Predict next token distribution given a (possibly long) history.
        We use only the last two items for trigram context.

        Returns
        -------
        List of (token, probability) sorted descending, top-k truncated.
        """
        self._ensure_fitted()
        if not self._vocab:
            return []

        hist = [str(h).strip() for h in (history or []) if str(h).strip()]
        prev1 = hist[-1] if len(hist) >= 1 else None
        prev2 = hist[-2] if len(hist) >= 2 else None
        V = len(self._vocab)

        # Gather candidate set from contexts; fall back to whole vocab if tiny.
        candidates: set[str] = set()
        if prev1 is not None and prev1 in self.bigram:
            candidates.update(self.bigram[prev1].keys())
        if prev2 is not None and (prev2, prev1) in self.trigram:
            candidates.update(self.trigram[(prev2, prev1)].keys())
        if not candidates:
            # fall back to frequent unigrams (still deterministic)
            candidates.update(self.unigram.keys())

        # Precompute denominators
        denom3 = self.trigram_totals.get((prev2, prev1), 0) if (prev2 is not None and prev1 is not None) else 0
        denom2 = self.bigram_totals.get(prev1, 0) if prev1 is not None else 0
        denom1 = self.total_tokens

        # Select mixture weights only for available contexts
        w3, w2, w1 = self.weights
        active_weights = []
        if denom3 > 0:
            active_weights.append(("tri", w3))
        if denom2 > 0:
            active_weights.append(("bi", w2))
        active_weights.append(("uni", w1))  # always available

        # Normalize active weights to sum to 1
        ws = sum(w for _, w in active_weights) or 1.0
        weight_map = {name: (w / ws) for name, w in active_weights}

        # Build distribution
        probs: Dict[str, float] = {}
        for tok in candidates:
            p = 0.0
            # trigram component
            if denom3 > 0:
                c3 = self.trigram[(prev2, prev1)].get(tok, 0)
                p += weight_map.get("tri", 0.0) * self._laplace_prob(c3, denom3, V)
            # bigram component
            if denom2 > 0:
                c2 = self.bigram[prev1].get(tok, 0)
                p += weight_map.get("bi", 0.0) * self._laplace_prob(c2, denom2, V)
            # unigram component (always)
            c1 = self.unigram.get(tok, 0)
            p += weight_map.get("uni", 0.0) * self._laplace_prob(c1, denom1, V)

            probs[tok] = float(p)

        # Normalize to sum to 1 across candidates (for neatness / determinism)
        Z = sum(probs.values()) or 1.0
        for k in list(probs.keys()):
            probs[k] = probs[k] / Z

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if topk is not None:
            topk = max(1, int(topk))
            ranked = ranked[:topk]
        return ranked

    def predict_path(self, history: List[str] | Tuple[str, ...] | None, steps: int = 3) -> List[Tuple[str, float]]:
        """
        Greedy roll-out of length `steps`, returning [(token, prob), ...].
        """
        self._ensure_fitted()
        hist: List[str] = [str(h).strip() for h in (history or []) if str(h).strip()]
        out: List[Tuple[str, float]] = []
        for _ in range(max(0, int(steps))):
            dist = self.predict_next(hist, topk=1)
            if not dist:
                break
            tok, p = dist[0]
            out.append((tok, p))
            hist.append(tok)
        return out

    # ------------------------- Legacy Shim -------------------------

    def predict_next_dist(
        self,
        prev2: Optional[str] = None,
        prev1: Optional[str] = None,
        topk: int = 50,
    ) -> Dict[str, float]:
        """
        Backward-compatible method used by older evaluation code:

            model.predict_next_dist(prev2, prev1) -> {token: prob}

        Internally forwards to `predict_next([prev2, prev1], topk)` and returns a dict.
        """
        hist = [x for x in [prev2, prev1] if isinstance(x, str) and x.strip()]
        ranked = self.predict_next(hist, topk=max(1, int(topk)))
        return {tok: float(p) for tok, p in ranked}
