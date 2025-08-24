from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random

import numpy as np
import pandas as pd
import networkx as nx

from eval_harness.column_resolver import resolve


HEURISTICS = ("jaccard", "adamic_adar", "resource_allocation", "preferential_attachment")


def _edge_list_from_rows(df: pd.DataFrame, sid_col: str, actors_col: str) -> List[Tuple[str, str]]:
    """
    Build bipartite edges (S:<sid>) -- (A:<actor>) for each actor listed on the row.
    Deduplicate within a row. Empty actor lists produce no edges.
    """
    edges: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        sid = row.get(sid_col)
        acts = row.get(actors_col, []) or []
        if pd.isna(sid) or not isinstance(acts, (list, tuple, set)):
            continue
        u = f"S:{str(sid).strip()}"
        seen = set()
        for a in acts:
            if not a and a != 0:
                continue
            a_str = str(a).strip()
            if not a_str:
                continue
            key = a_str.lower()
            if key in seen:
                continue
            seen.add(key)
            v = f"A:{a_str}"
            edges.append((u, v))
    # de-dup all edges
    return list(set(edges))


def _nx_from_edges(edges: Sequence[Tuple[str, str]]) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def _negatives_for_u(g: nx.Graph, u: str, k: int, rng: random.Random) -> List[str]:
    """
    Sample k candidate actor-nodes not connected to u.
    """
    # actor nodes start with 'A:'
    candidates = [n for n in g.nodes if isinstance(n, str) and n.startswith("A:") and not g.has_edge(u, n)]
    if not candidates:
        return []
    if k >= len(candidates):
        return candidates
    idxs = rng.sample(range(len(candidates)), k)
    return [candidates[i] for i in idxs]


def _score_pairs(g: nx.Graph, pairs: List[Tuple[str, str]], kind: str) -> Dict[Tuple[str, str], float]:
    """
    Compute heuristic scores for (u,v) pairs using NetworkX generators.
    """
    if kind == "jaccard":
        gen = nx.jaccard_coefficient(g, pairs)
    elif kind == "adamic_adar":
        gen = nx.adamic_adar_index(g, pairs)
    elif kind == "resource_allocation":
        gen = nx.resource_allocation_index(g, pairs)
    elif kind == "preferential_attachment":
        gen = nx.preferential_attachment(g, pairs)
    else:
        raise ValueError(f"unknown heuristic {kind}")
    out = {}
    for u, v, s in gen:
        try:
            out[(u, v)] = float(s)
        except Exception:
            out[(u, v)] = 0.0
    # prefer_attachment sometimes returns int; ensure float
    return out


def _rank_metrics(score_true: float, scores_neg: List[float]) -> Tuple[int, float, float, float]:
    """
    Given a true pair score and a list of negative scores, compute:
    - rank (1 = best), hits@1/3/5 (0/1), and reciprocal rank.
    Tie-breaking: place the true score after equal negatives (pessimistic).
    """
    arr = np.asarray(scores_neg, dtype=float)
    higher = int(np.sum(arr > score_true))
    equal  = int(np.sum(arr == score_true))
    # pessimistic rank: 1 + higher + equal
    rank = 1 + higher + equal
    hits1 = 1.0 if rank <= 1 else 0.0
    hits3 = 1.0 if rank <= 3 else 0.0
    hits5 = 1.0 if rank <= 5 else 0.0
    rr = 1.0 / float(rank)
    return rank, hits1, hits3, hits5, rr


def _eval_on_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    sid_col: str,
    actors_col: str,
    *,
    heuristics: Sequence[str] = HEURISTICS,
    max_samples: int = 300,
    negatives_per_pos: int = 30,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Evaluate link prediction heuristics by ranking the true actor among sampled negatives
    for edges that are present in both train and eval (so the nodes exist & are meaningful).
    """
    # Build train/test edge sets
    edges_train = _edge_list_from_rows(df.iloc[train_idx], sid_col, actors_col)
    edges_eval  = _edge_list_from_rows(df.iloc[eval_idx],  sid_col, actors_col)
    set_train = set(edges_train)
    set_eval  = set(edges_eval)

    # Only evaluate edges that appear in both (intersection)
    evalable = list(set_eval.intersection(set_train))
    if not evalable:
        return {"available": False, "reason": "no overlapping edges between train and eval (no duplicates)"}

    # Cap samples for speed
    rng = rng or random.Random(0)
    rng.shuffle(evalable)
    evalable = evalable[:max_samples]

    # Train graph
    g = _nx_from_edges(edges_train)

    # per-heuristic accumulators
    out: Dict[str, Any] = {"available": True, "n_candidates": len(evalable), "columns": {"sid": sid_col, "actors": actors_col}}
    for kind in heuristics:
        hits1 = hits3 = hits5 = 0.0
        mrr = 0.0
        n_eval = 0

        # Precompute negative samples per u
        neg_cache: Dict[str, List[str]] = {}

        for (u, v) in evalable:
            if u not in g or v not in g:
                continue
            # true score
            true_score = _score_pairs(g, [(u, v)], kind).get((u, v), 0.0)
            # negatives
            if u not in neg_cache:
                neg_cache[u] = _negatives_for_u(g, u, negatives_per_pos, rng)
            negs = neg_cache[u]
            if not negs:
                continue
            pairs = [(u, vn) for vn in negs]
            neg_scores_map = _score_pairs(g, pairs, kind)
            neg_scores = [neg_scores_map.get((u, vn), 0.0) for vn in negs]

            rank, h1, h3, h5, rr = _rank_metrics(true_score, neg_scores)
            hits1 += h1
            hits3 += h3
            hits5 += h5
            mrr += rr
            n_eval += 1

        if n_eval == 0:
            out[kind] = {"hits@1": 0.0, "hits@3": 0.0, "hits@5": 0.0, "mrr": 0.0, "n_eval": 0}
        else:
            out[kind] = {
                "hits@1": hits1 / n_eval,
                "hits@3": hits3 / n_eval,
                "hits@5": hits5 / n_eval,
                "mrr": mrr / n_eval,
                "n_eval": n_eval,
            }

    return out


def graph_descriptives(df: pd.DataFrame, sid_col: str, actors_col: str) -> Dict[str, Any]:
    edges = _edge_list_from_rows(df, sid_col, actors_col)
    if not edges:
        return {"available": False, "reason": "no edges"}
    g = _nx_from_edges(edges)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    comps = list(nx.connected_components(g))
    comp_sizes = [len(c) for c in comps]
    degs = [d for _, d in g.degree()]
    return {
        "available": True,
        "nodes": int(n),
        "edges": int(m),
        "components": int(len(comps)),
        "largest_component": int(max(comp_sizes)) if comp_sizes else 0,
        "degree_median": float(np.median(degs)) if degs else 0.0,
        "degree_p90": float(np.percentile(degs, 90)) if degs else 0.0,
    }


def eval_all(
    df_raw: pd.DataFrame,
    splits,
    registry=None,
    *,
    max_samples: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Public entry point for the Graph section.
    - Resolves schema (to get sid + actors list)
    - Evaluates heuristics on hold-out + CV
    - Returns descriptives as well
    """
    if df_raw is None or df_raw.empty:
        return {"available": False, "reason": "empty dataframe"}

    # Resolve schema to get actors list & sid
    res = resolve(df_raw, registry=registry)
    df = res["df"]
    C = res["columns"]
    sid = C.get("sid")
    actors = C.get("actors")
    if not sid or not actors or sid not in df.columns or actors not in df.columns:
        return {"available": False, "reason": "sid/actors columns unavailable after resolution"}

    rng = random.Random(seed)

    # Hold-out
    hold = _eval_on_split(
        df, train_idx=splits.train_idx, eval_idx=splits.test_idx,
        sid_col=sid, actors_col=actors,
        heuristics=HEURISTICS, max_samples=max_samples, negatives_per_pos=30, rng=rng,
    )

    # CV
    folds: List[Dict[str, Any]] = []
    for i, (tr_idx, va_idx) in enumerate(splits.folds, start=1):
        res_i = _eval_on_split(
            df, train_idx=tr_idx, eval_idx=va_idx,
            sid_col=sid, actors_col=actors,
            heuristics=HEURISTICS, max_samples=max_samples, negatives_per_pos=30, rng=rng,
        )
        folds.append({"fold": i, **{k: v for k, v in res_i.items() if isinstance(v, dict)}})

    # Summaries (mean/std where applicable)
    def _summ_stat(folds: List[Dict[str, Any]], metric: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for kind in HEURISTICS:
            vals = []
            for f in folds:
                if kind in f and metric in f[kind]:
                    vals.append(float(f[kind][metric]))
            if vals:
                arr = np.asarray(vals, dtype=float)
                out.setdefault(kind, {})
                out[kind][metric] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0)}
        return out

    summary = {}
    for m in ("hits@1", "hits@3", "hits@5", "mrr"):
        part = _summ_stat(folds, m)
        if part:
            summary[m] = part

    desc = graph_descriptives(df, sid, actors)

    return {"available": True, "holdout": hold, "cv": {"folds": folds, "summary": summary}, "descriptives": desc}
