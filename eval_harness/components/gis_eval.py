from __future__ import annotations
"""
GIS evaluator using the column resolver + gazetteer.

Computes:
- Geocode resolution rate (after joining gazetteer).
- Trajectory stats per sid sorted by route_order (hops, per-hop gap hours if timestamps exist, basic clustering hooks).
- Next-location baseline (Markov / bigram majority) with acc@1/3/5 on the 30% hold-out and K-fold CV.
- ETA baseline MAE (days): train medians per (current_loc -> next_loc) and global median fallback.

This file is robust to missing columns; it returns {"available": False, "reason": "..."} when a piece is not evaluable.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from eval_harness.column_resolver import resolve


def _acc_at_k(truth: str, preds: List[str], k: int) -> float:
    return 1.0 if truth in preds[:k] else 0.0


def _majority_next_loc(train_df: pd.DataFrame, sid_col: str, route_col: str, loc_col: str) -> Dict[str, List[str]]:
    """
    Train a simple bigram model: for each current location L, rank next locations by frequency.
    Returns mapping: current_loc -> ranked list of next locs.
    """
    order = train_df[[sid_col, route_col, loc_col]].dropna().copy()
    order[route_col] = pd.to_numeric(order[route_col], errors="coerce")
    order = order.dropna(subset=[route_col])
    order = order.sort_values([sid_col, route_col])
    # compute bigrams per sid
    pairs = []
    for _, sub in order.groupby(sid_col):
        locs = sub[loc_col].tolist()
        if len(locs) < 2:
            continue
        pairs += list(zip(locs[:-1], locs[1:]))
    if not pairs:
        return {}
    dfp = pd.DataFrame(pairs, columns=["cur", "nxt"])
    counts = dfp.value_counts().reset_index(name="n")
    # build ranking per current loc
    ranks: Dict[str, List[str]] = {}
    for cur, grp in counts.groupby("cur"):
        ranks[cur] = grp.sort_values("n", ascending=False)["nxt"].tolist()
    return ranks


def _predict_next_loc(ranks: Dict[str, List[str]], cur_loc: str, kmax: int = 5) -> List[str]:
    if cur_loc in ranks:
        return ranks[cur_loc][:kmax]
    return []


def _eta_baseline(train_df: pd.DataFrame, head_col: str, tail_col: str, eta_col: str) -> Dict[Tuple[str, str], float]:
    """
    Median ETA per (current_loc, next_loc) pair; returns mapping (cur, nxt) -> days.
    """
    sub = train_df[[head_col, tail_col, eta_col]].dropna()
    if sub.empty:
        return {}
    med = sub.groupby([head_col, tail_col])[eta_col].median().reset_index()
    return {(r[head_col], r[tail_col]): float(r[eta_col]) for _, r in med.iterrows()}


def _eta_predict(med_map: Dict[Tuple[str, str], float], cur: str, nxt: str, global_med: float) -> float:
    return float(med_map.get((cur, nxt), global_med))


# ---------------- public API ----------------

def eval_all(df_raw: pd.DataFrame, splits, registry=None) -> Dict[str, Any]:
    """
    df_raw: processed frame (your CSV).
    splits: SplitBundle with attributes train_idx, test_idx, folds (list of (train_idx, val_idx)).
    registry: for gazetteer loading.

    Returns a dict:
      {
        "available": True,
        "geocode": {...},
        "trajectories": {...},
        "nextloc": {"holdout": {...}, "cv": {...}},
        "eta": {"holdout": {...}, "cv": {...}},
      }
    """
    if df_raw is None or df_raw.empty:
        return {"available": False, "reason": "empty dataframe"}

    # Resolve + attach gazetteer coords and derived fields
    res = resolve(df_raw, registry=registry)
    df = res["df"]
    C = res["columns"]

    out: Dict[str, Any] = {"available": True}

    # Geocode stats
    out["geocode"] = res.get("geocode", {"resolved": 0, "total": 0, "rate": 0.0})

    sid = C.get("sid")
    route = C.get("route_order")
    loc = C.get("location_name")
    eta = C.get("eta_days")

    if not sid or not route or not loc:
        out["trajectories"] = {"available": False, "reason": "missing sid/route_order/location_name"}
    else:
        # Trajectory stats
        sub = df[[sid, route, loc, eta]].copy()
        sub[route] = pd.to_numeric(sub[route], errors="coerce")
        sub = sub.dropna(subset=[sid, route, loc]).sort_values([sid, route])

        # hops per trajectory
        traj_sizes = sub.groupby(sid)[loc].size().to_numpy()
        n_traj = int(len(traj_sizes))
        median_hops = float(np.median(traj_sizes)) if n_traj else 0.0

        # per-hop time gaps (in hours) if eta_days exists
        gaps_h = []
        if eta and eta in sub.columns:
            # interpret eta_days as per-node dwell; per-hop gap approximated by the next node's eta_days
            eta_vals = sub[eta].to_numpy()
            # collecte finite values
            eta_finite = eta_vals[np.isfinite(eta_vals)]
            # summary
            gap_hours_median = float(np.nanmedian(eta_finite) * 24) if eta_finite.size else 0.0
            gap_hours_p90 = float(np.nanpercentile(eta_finite, 90) * 24) if eta_finite.size else 0.0
        else:
            gap_hours_median = 0.0
            gap_hours_p90 = 0.0

        out["trajectories"] = {
            "available": True,
            "trajectories": n_traj,
            "median_hops": median_hops,
            "gap_hours_median": gap_hours_median,
            "gap_hours_p90": gap_hours_p90,
        }

    # ---- Next-location (hold-out + CV) ----
    nextloc = {"holdout": {"available": False}, "cv": {}}
    if sid and route and loc:
        # build bigram model on train, eval on test
        train = df.iloc[splits.train_idx]
        test = df.iloc[splits.test_idx]
        ranks = _majority_next_loc(train, sid, route, loc)

        # Build test pairs (current -> next) per sid
        te = test[[sid, route, loc]].dropna().copy()
        te[route] = pd.to_numeric(te[route], errors="coerce")
        te = te.dropna(subset=[route]).sort_values([sid, route])
        pairs = []
        for _, g in te.groupby(sid):
            L = g[loc].tolist()
            if len(L) >= 2:
                pairs += list(zip(L[:-1], L[1:]))
        # Evaluate
        if pairs:
            a1 = a3 = a5 = 0.0
            for cur, truth in pairs:
                preds = _predict_next_loc(ranks, cur, kmax=5)
                a1 += _acc_at_k(truth, preds, 1)
                a3 += _acc_at_k(truth, preds, 3)
                a5 += _acc_at_k(truth, preds, 5)
            n = float(len(pairs))
            nextloc["holdout"] = {"available": True, "acc@1": a1 / n, "acc@3": a3 / n, "acc@5": a5 / n, "n_pairs": int(n)}
        else:
            nextloc["holdout"] = {"available": False, "reason": "not enough transitions in test split"}

        # CV
        folds_stats: List[Dict[str, Any]] = []
        for i, (tr_idx, va_idx) in enumerate(splits.folds, start=1):
            tr = df.iloc[tr_idx]
            va = df.iloc[va_idx]
            ranks_cv = _majority_next_loc(tr, sid, route, loc)
            va_pairs = []
            sub_va = va[[sid, route, loc]].dropna().copy()
            sub_va[route] = pd.to_numeric(sub_va[route], errors="coerce")
            sub_va = sub_va.dropna(subset=[route]).sort_values([sid, route])
            for _, g in sub_va.groupby(sid):
                L = g[loc].tolist()
                if len(L) >= 2:
                    va_pairs += list(zip(L[:-1], L[1:]))
            if not va_pairs:
                folds_stats.append({"fold": i, "acc@1": 0.0, "acc@3": 0.0, "acc@5": 0.0, "n_pairs": 0})
                continue
            a1 = a3 = a5 = 0.0
            for cur, truth in va_pairs:
                preds = _predict_next_loc(ranks_cv, cur, kmax=5)
                a1 += _acc_at_k(truth, preds, 1)
                a3 += _acc_at_k(truth, preds, 3)
                a5 += _acc_at_k(truth, preds, 5)
            n = float(len(va_pairs))
            folds_stats.append({"fold": i, "acc@1": a1 / n, "acc@3": a3 / n, "acc@5": a5 / n, "n_pairs": int(n)})

        if folds_stats:
            dfcv = pd.DataFrame(folds_stats)
            summary = {k: {"mean": float(dfcv[k].mean()), "std": float(dfcv[k].std(ddof=1) if len(dfcv)>1 else 0.0)}
                       for k in ["acc@1","acc@3","acc@5"]}
            nextloc["cv"] = {"folds": folds_stats, "summary": summary}
    else:
        nextloc["holdout"] = {"available": False, "reason": "missing sid/route_order/location_name"}
    out["nextloc"] = nextloc

    # ---- ETA (hold-out + CV) ----
    eta_block = {"holdout": {"available": False}, "cv": {}}
    if sid and route and loc and eta and eta in df.columns:
        # build (cur -> nxt) model of median eta_days on train, evaluate MAE on test pairs
        # We need pairs aligned to have eta at the "edge". We'll use the next node's eta_days as edge label.
        work = df[[sid, route, loc, eta]].dropna().copy()
        work[route] = pd.to_numeric(work[route], errors="coerce")
        work = work.dropna(subset=[route]).sort_values([sid, route])

        def pairs_with_eta(subdf: pd.DataFrame) -> List[Tuple[str, str, float]]:
            res = []
            for _, g in subdf.groupby(sid):
                L = g[loc].tolist()
                E = g[eta].tolist()
                if len(L) >= 2:
                    # edge eta: use eta of the destination node
                    for a, b, e in zip(L[:-1], L[1:], E[1:]):
                        if np.isfinite(e):
                            res.append((a, b, float(e)))
            return res

        tr_pairs = pairs_with_eta(df.iloc[splits.train_idx])
        te_pairs = pairs_with_eta(df.iloc[splits.test_idx])

        if tr_pairs and te_pairs:
            tr_df = pd.DataFrame(tr_pairs, columns=["cur","nxt","eta"])
            med_map = _eta_baseline(tr_df, "cur", "nxt", "eta")
            global_med = float(tr_df["eta"].median())

            # evaluate
            errs = []
            for cur, nxt, true_eta in te_pairs:
                pred = _eta_predict(med_map, cur, nxt, global_med)
                errs.append(abs(true_eta - pred))
            mae = float(np.mean(errs)) if errs else 0.0
            eta_block["holdout"] = {"available": True, "mae_days": mae, "n_edges": int(len(errs))}
        else:
            eta_block["holdout"] = {"available": False, "reason": "insufficient edges with eta_days in hold-out"}

        # CV
        folds_stats: List[Dict[str, Any]] = []
        for i, (tr_idx, va_idx) in enumerate(splits.folds, start=1):
            tr_pairs = pairs_with_eta(df.iloc[tr_idx])
            va_pairs = pairs_with_eta(df.iloc[va_idx])
            if not tr_pairs or not va_pairs:
                folds_stats.append({"fold": i, "mae_days": 0.0, "n_edges": 0})
                continue
            tr_df = pd.DataFrame(tr_pairs, columns=["cur","nxt","eta"])
            med_map = _eta_baseline(tr_df, "cur", "nxt", "eta")
            global_med = float(tr_df["eta"].median())
            errs = []
            for cur, nxt, true_eta in va_pairs:
                pred = _eta_predict(med_map, cur, nxt, global_med)
                errs.append(abs(true_eta - pred))
            mae = float(np.mean(errs)) if errs else 0.0
            folds_stats.append({"fold": i, "mae_days": mae, "n_edges": int(len(errs))})

        if folds_stats:
            dfcv = pd.DataFrame(folds_stats)
            eta_block["cv"] = {
                "folds": folds_stats,
                "summary": {"mae_days": {"mean": float(dfcv["mae_days"].mean()),
                                         "std": float(dfcv["mae_days"].std(ddof=1) if len(dfcv)>1 else 0.0)}}
            }
    else:
        eta_block["holdout"] = {"available": False, "reason": "missing sid/route_order/location_name/eta_days"}
    out["eta"] = eta_block

    return out
