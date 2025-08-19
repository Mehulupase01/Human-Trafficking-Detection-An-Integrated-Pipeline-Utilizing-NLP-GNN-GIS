# backend/models/link_predictor.py
from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math

import pandas as pd

COL_SID = "Serialized ID"
COL_LOC = "Location"
COL_PERPS = "Perpetrators (NLP)"
COL_CHIEFS = "Chiefs (NLP)"  # (kept for future extension)


class LinkPredictor:
    """
    Heuristic link prediction for Victim → Perpetrator edges.
    Score is a weighted sum of:
      - Location Jaccard between victim's locations and perpetrator's locations
      - Co-victim overlap ratio (victim shares locations with other victims tied to the perpetrator)
      - Popularity (log-degree) of perpetrator

    All signals are min-max normalized per victim for comparability, then combined.
    """

    def __init__(self, w_jaccard: float = 0.5, w_covictim: float = 0.3, w_popularity: float = 0.2):
        self.w_j = w_jaccard
        self.w_c = w_covictim
        self.w_p = w_popularity
        # Indexes
        self.victim_locs: Dict[str, set[str]] = defaultdict(set)
        self.perp_locs: Dict[str, set[str]] = defaultdict(set)
        self.perp_victims: Dict[str, set[str]] = defaultdict(set)
        self.victim_known_perps: Dict[str, set[str]] = defaultdict(set)
        self.perp_popularity: Counter = Counter()
        self.all_perps: set[str] = set()

    def fit(self, df: pd.DataFrame):
        # Victim locations
        for sid, grp in df.groupby(COL_SID):
            locs = set(grp[COL_LOC].dropna().astype(str).tolist())
            self.victim_locs[str(sid)] |= locs

        # Perp occurrence per row -> perpetrator-locations and perpetrator-victims
        if COL_PERPS not in df.columns:
            return
        for _, row in df.iterrows():
            sid = str(row[COL_SID])
            loc = str(row[COL_LOC]) if pd.notna(row[COL_LOC]) else None
            perps = row[COL_PERPS] if isinstance(row[COL_PERPS], list) else []
            for p in perps:
                p = str(p).strip()
                if not p:
                    continue
                self.all_perps.add(p)
                self.perp_victims[p].add(sid)
                self.perp_popularity[p] += 1
                if loc:
                    self.perp_locs[p].add(loc)
                self.victim_known_perps[sid].add(p)

    def _minmax(self, values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        vmin = min(values.values())
        vmax = max(values.values())
        if vmax == vmin:
            return {k: 0.0 for k in values.keys()}
        return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

    def predict_for_victim(self, victim_sid: str, top_k: int = 3) -> List[Tuple[str, float]]:
        sid = str(victim_sid)
        victim_lset = self.victim_locs.get(sid, set())
        known = self.victim_known_perps.get(sid, set())

        # Candidates: all perps not already known
        candidates = [p for p in self.all_perps if p not in known]
        if not candidates:
            return []

        # Jaccard on locations
        jaccard = {}
        for p in candidates:
            plocs = self.perp_locs.get(p, set())
            inter = len(victim_lset & plocs)
            union = len(victim_lset | plocs) or 1
            jaccard[p] = inter / union

        # Co-victim overlap: fraction of perpetrator's victims who share any location with this victim
        cov = {}
        for p in candidates:
            pvics = self.perp_victims.get(p, set())
            if not pvics:
                cov[p] = 0.0
                continue
            share = 0
            for other_v in pvics:
                if other_v == sid:
                    continue
                if self.victim_locs.get(other_v, set()) & victim_lset:
                    share += 1
            cov[p] = share / max(len(pvics) - (1 if sid in pvics else 0), 1)

        # Popularity (log-degree)
        pop = {}
        for p in candidates:
            pop[p] = math.log1p(self.perp_popularity.get(p, 0))

        # Normalize each signal per victim
        jn = self._minmax(jaccard)
        cn = self._minmax(cov)
        pn = self._minmax(pop)

        # Weighted sum
        scores = {p: (self.w_j * jn.get(p, 0.0) + self.w_c * cn.get(p, 0.0) + self.w_p * pn.get(p, 0.0)) for p in candidates}

        # Normalize final scores to sum=1
        total = sum(scores.values())
        if total > 0:
            scores = {p: s / total for p, s in scores.items()}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_k)]
        return [(p, float(s)) for p, s in ranked]
