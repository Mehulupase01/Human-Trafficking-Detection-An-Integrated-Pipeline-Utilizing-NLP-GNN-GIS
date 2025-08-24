from __future__ import annotations
import pandas as pd
import altair as alt

def query_metrics_bar(metrics: dict) -> alt.Chart:
    """
    Bar chart for retrieval metrics dict with keys like:
    ndcg@5, ndcg@10, map, mrr, p@10, recall@10
    """
    if not isinstance(metrics, dict) or not metrics:
        return alt.Chart(pd.DataFrame({"x":[0], "y":[0]})).mark_bar().encode(x="x", y="y").properties(height=200)
    df = pd.DataFrame([metrics]).melt(var_name="metric", value_name="value")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", sort="-y", title="Metric"),
            y=alt.Y("value:Q", title="Score"),
            tooltip=["metric:N", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(title="Retrieval metrics (hold-out)", height=260)
    )

def latency_box(lat_summary: dict) -> alt.Chart:
    """
    Simple box-like render from p50/p90 summary (no raw points).
    """
    if not isinstance(lat_summary, dict) or not lat_summary:
        return alt.Chart(pd.DataFrame({"x":[0], "y":[0]})).mark_bar().encode(x="x", y="y").properties(height=120)
    p50 = float(lat_summary.get("p50_ms", 0.0))
    p90 = float(lat_summary.get("p90_ms", 0.0))
    df = pd.DataFrame({"stat": ["p50", "p90"], "ms": [p50, p90]})
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("stat:N", title="Latency"),
            y=alt.Y("ms:Q", title="Milliseconds"),
            tooltip=["stat:N", alt.Tooltip("ms:Q", format=".1f")],
        )
        .properties(title="Per-query latency (hold-out)", height=220)
    )
