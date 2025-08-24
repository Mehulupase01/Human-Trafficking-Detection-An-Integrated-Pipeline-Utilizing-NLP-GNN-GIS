from __future__ import annotations
import pandas as pd
import altair as alt

def pr_curve_chart(pr_dict: dict) -> alt.Chart:
    """Precision–Recall curve."""
    df = pd.DataFrame({
        "recall": pr_dict.get("recall", []),
        "precision": pr_dict.get("precision", []),
    })
    df = df[(df["recall"] >= 0) & (df["recall"] <= 1)]
    return alt.Chart(df).mark_line().encode(
        x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0,1])),
        tooltip=[alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")]
    ).properties(title="Precision–Recall", height=260)

def roc_curve_chart(roc_dict: dict) -> alt.Chart:
    """ROC curve."""
    df = pd.DataFrame({
        "fpr": roc_dict.get("fpr", []),
        "tpr": roc_dict.get("tpr", []),
    })
    base = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line(strokeDash=[4,4], opacity=0.4).encode(x="x", y="y")
    curve = alt.Chart(df).mark_line().encode(
        x=alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=[0,1])),
        tooltip=[alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")]
    )
    return (base + curve).properties(title="ROC", height=260)

def reliability_chart(cal_dict: dict) -> alt.Chart:
    """Reliability diagram (calibration)."""
    df = pd.DataFrame({
        "bin_center": cal_dict.get("bin_centers", []),
        "confidence": cal_dict.get("confidence", []),
        "accuracy": cal_dict.get("accuracy", []),
        "size": cal_dict.get("sizes", []),
    })
    ideal = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line(strokeDash=[4,4], opacity=0.4).encode(x="x", y="y")
    pts = alt.Chart(df).mark_point(filled=True).encode(
        x=alt.X("confidence:Q", title="Predicted probability", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("accuracy:Q", title="Empirical accuracy", scale=alt.Scale(domain=[0,1])),
        size=alt.Size("size:Q", title="Bin size", legend=None),
        tooltip=[alt.Tooltip("confidence:Q", format=".2f"),
                 alt.Tooltip("accuracy:Q", format=".2f"),
                 alt.Tooltip("size:Q")]
    )
    return (ideal + pts).properties(title="Calibration (Reliability)", height=260)

def confusion_heatmap(cm_dict: dict) -> alt.Chart:
    """2x2 confusion matrix heatmap from counts {tp, tn, fp, fn}."""
    vals = [
        {"pred":"Positive", "true":"Positive", "count": cm_dict.get("tp", 0)},
        {"pred":"Negative", "true":"Positive", "count": cm_dict.get("fn", 0)},
        {"pred":"Positive", "true":"Negative", "count": cm_dict.get("fp", 0)},
        {"pred":"Negative", "true":"Negative", "count": cm_dict.get("tn", 0)},
    ]
    df = pd.DataFrame(vals)
    return alt.Chart(df).mark_rect().encode(
        x=alt.X("pred:N", title="Predicted"),
        y=alt.Y("true:N", title="True"),
        color=alt.Color("count:Q", title="Count"),
        tooltip=["true:N","pred:N","count:Q"]
    ).properties(title="Confusion Matrix", height=220)
