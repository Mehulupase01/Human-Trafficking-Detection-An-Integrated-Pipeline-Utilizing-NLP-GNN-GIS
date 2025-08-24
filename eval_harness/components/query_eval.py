from __future__ import annotations
"""
Query / Retrieval evaluation

What it does
------------
- Resolves your processed corpus (uses column_resolver) to get `doc_id` and `text`
- Loads graded queries from:
    1) registry kind="queries" (preferred), or
    2) local "data/queries.jsonl" (fallback)
  Each query: {"qid": "...", "text": "...", "relevance": [{"doc_id":"...", "grade": 0-3}, ...]}
- Builds a TF-IDF index (scikit-learn if available; else a pure-Python token-overlap + IDF fallback)
- Scores & ranks docs for each query
- Computes metrics: nDCG@5/10, MAP, MRR, P@10, Recall@10 + latency (p50/p90 ms)
- Produces a simple K-fold (K=5 or fewer) CV over queries (to give mean/std stability on small sets)

Returns a dict with:
{
  "available": True/False,
  "reason": "...",              # if False
  "n_docs": int,
  "n_queries": int,
  "holdout": { "metrics": {...}, "latency": {...} },
  "cv": { "folds": [...], "summary": {...} }
}
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eval_harness.column_resolver import resolve


# -------------------- Queries loading --------------------

def _load_queries_from_registry(registry) -> Optional[List[Dict[str, Any]]]:
    if registry is None:
        return None
    items = []
    try:
        items = registry.list_datasets(kind="queries") or []
    except Exception:
        try:
            # some registries expose list_all()
            items = [x for x in (registry.list_all() or []) if (x.get("kind") or x.get("type")) == "queries"]
        except Exception:
            items = []
    if not items:
        return None
    qid = items[0].get("id")
    # Try json readers first
    for fn in ("load_json", "read_json"):
        try:
            rows = getattr(registry, fn)(qid)
            if isinstance(rows, list) and rows:
                return rows
        except Exception:
            pass
    # Try text/jsonl
    for fn in ("load_text", "read_text"):
        try:
            txt = getattr(registry, fn)(qid)
            if isinstance(txt, str) and txt.strip():
                out: List[Dict[str, Any]] = []
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                return out
        except Exception:
            pass
    return None


def _load_queries_from_file(path="data/queries.jsonl") -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out or None


def _normalize_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        qid = str(r.get("qid") or r.get("id") or r.get("query_id") or "").strip()
        text = str(r.get("text") or r.get("query") or "").strip()
        rel = r.get("relevance") or r.get("labels") or []
        rel_norm: List[Dict[str, Any]] = []
        if isinstance(rel, list):
            for it in rel:
                try:
                    did = str(it.get("doc_id") or it.get("id") or it.get("uid") or "").strip()
                    grade = int(it.get("grade", 0))
                    rel_norm.append({"doc_id": did, "grade": grade})
                except Exception:
                    continue
        if qid and text:
            out.append({"qid": qid, "text": text, "relevance": rel_norm})
    return out


# -------------------- Tokenize (fallback) --------------------

def _tokenize(s: str) -> List[str]:
    # very simple word tokenizer
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if len(t) >= 2]


# -------------------- Metrics --------------------

def _dcg_at_k(grades: List[int], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(grades[:k]):
        gain = (2 ** g - 1)
        denom = math.log2(i + 2)  # i=0 -> log2(2)=1
        dcg += gain / denom
    return dcg


def _ndcg_at_k(ranked_ids: List[str], rel_map: Dict[str, int], k: int) -> float:
    grades = [rel_map.get(did, 0) for did in ranked_ids]
    dcg = _dcg_at_k(grades, k)
    ideal = sorted(rel_map.values(), reverse=True)
    idcg = _dcg_at_k(ideal, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _map_mrr_p_r_at_k(ranked_ids: List[str], rel_map: Dict[str, int], k: int) -> Tuple[float, float, float, float]:
    # binary relevance: grade > 0
    hits = 0
    ap_sum = 0.0
    rr = 0.0
    for i, did in enumerate(ranked_ids[:k], start=1):
        rel = 1 if rel_map.get(did, 0) > 0 else 0
        if rel:
            hits += 1
            ap_sum += hits / i
            if rr == 0.0:
                rr = 1.0 / i
    p_at_k = hits / float(k) if k > 0 else 0.0
    # Recall@k: relevant found / total relevant
    total_rel = sum(1 for v in rel_map.values() if v > 0)
    recall_at_k = (hits / float(total_rel)) if total_rel > 0 else 0.0
    # MAP: average precision across all positives in the ranking window (k) — we approximate using top-k
    map_k = ap_sum / float(total_rel) if total_rel > 0 else 0.0
    return map_k, rr, p_at_k, recall_at_k


# -------------------- Scorers --------------------

class _TfIdfIndex:
    def __init__(self, docs: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        self.v = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.X = self.v.fit_transform(docs)  # (n_docs, vocab)
    def score(self, q: str) -> np.ndarray:
        qv = self.v.transform([q])
        # cosine ~ dot product when vectors are L2-normalized (default)
        scores = (qv @ self.X.T).toarray()[0]
        return scores


class _FallbackIndex:
    def __init__(self, docs: List[str]):
        # simple IDF weighted token overlap
        self.doc_tokens: List[set] = []
        df_counts: Dict[str, int] = {}
        for text in docs:
            toks = set(_tokenize(text))
            self.doc_tokens.append(toks)
            for t in toks:
                df_counts[t] = df_counts.get(t, 0) + 1
        N = max(1, len(docs))
        self.idf = {t: math.log((N + 1) / (df + 1)) + 1.0 for t, df in df_counts.items()}

    def score(self, q: str) -> np.ndarray:
        qtoks = set(_tokenize(q))
        scores: List[float] = []
        for toks in self.doc_tokens:
            overlap = qtoks.intersection(toks)
            s = sum(self.idf.get(t, 0.0) for t in overlap)
            scores.append(s)
        return np.asarray(scores, dtype=float)


def _build_index(docs: List[str]):
    try:
        # Try sklearn TF-IDF
        return _TfIdfIndex(docs)
    except Exception:
        # Fallback to pure Python scorer
        return _FallbackIndex(docs)


# -------------------- Evaluation --------------------

def eval_queries(
    *,
    registry,
    df_processed: pd.DataFrame,
    seed: int = 42,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate retrieval with graded queries.
    """
    if df_processed is None or df_processed.empty:
        return {"available": False, "reason": "empty processed dataframe"}

    # Resolve to get doc_id & text
    res = resolve(df_processed, registry=registry)
    df = res["df"]
    C = res["columns"]
    did_col = C.get("doc_id")
    txt_col = C.get("text")

    if not did_col or did_col not in df.columns or not txt_col or txt_col not in df.columns:
        return {"available": False, "reason": "doc_id/text columns unavailable after resolution"}

    corpus = df[[did_col, txt_col]].dropna()
    corpus[txt_col] = corpus[txt_col].astype(str).str.strip()
    corpus = corpus[corpus[txt_col].str.len() > 0].drop_duplicates(subset=[did_col])
    if corpus.empty:
        return {"available": False, "reason": "no non-empty text to index"}

    # Load queries
    qrows = _load_queries_from_registry(registry) or _load_queries_from_file()
    if not qrows:
        return {"available": False, "reason": "no graded queries found (kind='queries' or data/queries.jsonl)"}
    queries = _normalize_queries(qrows)
    if not queries:
        return {"available": False, "reason": "queries present but malformed/empty"}

    # Build index
    docs = corpus[txt_col].tolist()
    doc_ids = corpus[did_col].astype(str).tolist()
    index = _build_index(docs)

    # Score all queries
    lat_ms: List[float] = []
    per_q_metrics: List[Dict[str, float]] = []
    for q in queries:
        t0 = time.perf_counter()
        scores = index.score(q["text"])
        # top-k doc ids
        order = np.argsort(-scores)  # descending
        ranked = [doc_ids[i] for i in order[:max(top_k, 10)]]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt_ms)

        rel_map = {str(d["doc_id"]): int(d.get("grade", 0)) for d in q["relevance"] or []}

        ndcg5 = _ndcg_at_k(ranked, rel_map, 5)
        ndcg10 = _ndcg_at_k(ranked, rel_map, 10)
        mapk, mrr, p10, r10 = _map_mrr_p_r_at_k(ranked, rel_map, 10)
        per_q_metrics.append(
            {"ndcg@5": ndcg5, "ndcg@10": ndcg10, "map": mapk, "mrr": mrr, "p@10": p10, "recall@10": r10}
        )

    # Aggregate
    dfm = pd.DataFrame(per_q_metrics)
    metrics_hold = {k: float(dfm[k].mean()) for k in dfm.columns}
    lat_summary = {
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p90_ms": float(np.percentile(lat_ms, 90)),
        "n": int(len(lat_ms)),
    }

    # CV over queries (for stability plots): K=min(5, n_queries)
    K = max(2, min(5, len(queries)))
    folds: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(queries))
    rng.shuffle(idx)
    chunks = np.array_split(idx, K)
    for i, ch in enumerate(chunks, start=1):
        # evaluate on this fold subset
        sub_metrics = dfm.iloc[ch]
        folds.append({
            "fold": i,
            "holdout": {"metrics": {k: float(sub_metrics[k].mean()) for k in sub_metrics.columns}},
        })
    # summary
    summary = {k: {"mean": float(dfm[k].mean()), "std": float(dfm[k].std(ddof=1) if len(dfm) > 1 else 0.0)} for k in dfm.columns}

    return {
        "available": True,
        "n_docs": int(len(corpus)),
        "n_queries": int(len(queries)),
        "holdout": {"metrics": metrics_hold, "latency": lat_summary},
        "cv": {"folds": folds, "summary": summary},
    }
