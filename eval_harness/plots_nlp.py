from __future__ import annotations
import pandas as pd
import altair as alt

# These helpers are intentionally tolerant: if given empty/missing data,
# they render a tiny placeholder rather than erroring.

def pr_curve_chart(pr: dict):
    if not isinstance(pr, dict):
        pr = {}
    precision = pr.get("precision") or []
    recall = pr.get("recall") or []
    if not precision or not recall or len(precision) != len(recall):
        df = pd.DataFrame({"recall": [0.0, 1.0], "precision": [0.0, 0.0]})
    else:
        df = pd.DataFrame({"recall": recall, "precision": precision})
    return alt.Chart(df).mark_line().encode(
        x=alt.X("recall:Q", title="Recall"),
        y=alt.Y("precision:Q", title="Precision")
    ).properties(height=220)

def roc_curve_chart(roc: dict):
    if not isinstance(roc, dict):
        roc = {}
    fpr = roc.get("fpr") or []
    tpr = roc.get("tpr") or []
    if not fpr or not tpr or len(fpr) != len(tpr):
        df = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    else:
        df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    return alt.Chart(df).mark_line().encode(
        x=alt.X("fpr:Q", title="False Positive Rate"),
        y=alt.Y("tpr:Q", title="True Positive Rate")
    ).properties(height=220)

def reliability_chart(cal: dict):
    # Expected: {"bin_centers":[...], "confidence":[...], "accuracy":[...]}
    if not isinstance(cal, dict):
        cal = {}
    x = cal.get("bin_centers") or []
    conf = cal.get("confidence") or []
    acc = cal.get("accuracy") or []
    if not x or not conf or not acc or not (len(x)==len(conf)==len(acc)):
        df = pd.DataFrame({"x":[0.0,1.0], "y":[0.0,1.0]})
        return alt.Chart(df).mark_line().encode(x=alt.X("x:Q", title="Confidence"), y=alt.Y("y:Q", title="Accuracy")).properties(height=220)
    df = pd.DataFrame({"x": x, "conf": conf, "acc": acc})
    line1 = alt.Chart(df).mark_line().encode(x="x:Q", y=alt.Y("conf:Q", title="Confidence"))
    line2 = alt.Chart(df).mark_line(color="red").encode(x="x:Q", y=alt.Y("acc:Q", title="Accuracy"))
    return (line1 + line2).properties(height=220)

def confusion_heatmap(cm: dict):
    # Expect keys: tp, tn, fp, fn
    tp = int(cm.get("tp", 0)); tn = int(cm.get("tn", 0)); fp = int(cm.get("fp", 0)); fn = int(cm.get("fn", 0))
    df = pd.DataFrame([
        {"pred": "Positive", "true": "Positive", "n": tp},
        {"pred": "Negative", "true": "Positive", "n": fn},
        {"pred": "Positive", "true": "Negative", "n": fp},
        {"pred": "Negative", "true": "Negative", "n": tn},
    ])
    return alt.Chart(df).mark_rect().encode(
        x=alt.X("pred:N", title="Predicted"),
        y=alt.Y("true:N", title="True"),
        tooltip=["n:Q"],
        color="n:Q"
    ).properties(height=220)
