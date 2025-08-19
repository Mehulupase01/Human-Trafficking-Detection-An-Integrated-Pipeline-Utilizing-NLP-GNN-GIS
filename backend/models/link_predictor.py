# backend/models/link_predictor.py
from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import pandas as pd

class LinkPredictor:
    """
    Baseline link scorer:
      1) Per victim V, gather all perps that co-occur with V's locations; score by frequency.
      2) Backoff to globally frequent perps at V's last location.
      3) Backoff to global top perps.
    """

    def __init__(self):
        self.perp_global = Counter()             # perp -> count
        self.perp_by_loc = defaultdict(Counter)  # loc -> perp -> count
        self.locs_by_victim = defaultdict(list)  # victim -> [locs in order]
        self.vocab_perps = set()
        self.fitted = False

    @staticmethod
    def _first_token(x):
        if isinstance(x, list) and x:
            return str(x[0]).strip()
        if pd.isna(x) or x is None:
            return None
        return str(x).strip() or None

    def fit(self, df: pd.DataFrame) -> None:
        if df is None or df.empty or "Serialized ID" not in df.columns:
            self.fitted = True
            return

        d = df.copy()
        # step location (primary)
        if "Locations (NLP)" in d.columns:
            d["_loc"] = d["Locations (NLP)"].apply(self._first_token)
        else:
            d["_loc"] = None
        if d["_loc"].isna().any() and "Location" in d.columns:
            mask = d["_loc"].isna()
            d.loc[mask, "_loc"] = d.loc[mask, "Location"].apply(self._first_token)

        # route order
        if "Route_Order" in d.columns:
            d["Route_Order"] = pd.to_numeric(d["Route_Order"], errors="coerce")
            d = d.sort_values(["Serialized ID", "Route_Order"], kind="stable")

        # collect perps and stats
        for vid, g in d.groupby("Serialized ID"):
            # victim's ordered unique locs
            last = None
            v_locs = []
            for loc in g["_loc"].tolist():
                if not isinstance(loc, str) or not loc:
                    continue
                if loc == last:
                    continue
                v_locs.append(loc); last = loc
            self.locs_by_victim[str(vid)] = v_locs

            # perps per row (list) → attach to that row's loc
            if "Perpetrators (NLP)" in g.columns:
                for _, row in g.iterrows():
                    loc = row["_loc"]
                    perps = row["Perpetrators (NLP)"]
                    if not isinstance(perps, list) or not loc:
                        continue
                    for p in perps:
                        if not p:
                            continue
                        p = str(p).strip()
                        self.vocab_perps.add(p)
                        self.perp_global[p] += 1
                        self.perp_by_loc[loc][p] += 1

        self.fitted = True

    def predict_for_victim(self, victim_sid: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.fitted or not self.vocab_perps:
            return []

        v = str(victim_sid)
        v_locs = self.locs_by_victim.get(v, [])
        scores = Counter()

        # 1) perps seen at this victim's locations
        for loc in v_locs:
            scores.update(self.perp_by_loc.get(loc, {}))

        # 2) if nothing, try last location
        if not scores and v_locs:
            last = v_locs[-1]
            scores.update(self.perp_by_loc.get(last, {}))

        # 3) if still nothing, use global popularity
        if not scores:
            scores.update(self.perp_global)

        if not scores:
            return []

        total = sum(scores.values())
        ranked = [(p, cnt / total) for p, cnt in scores.most_common(top_k)]
        return ranked
