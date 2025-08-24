import pandas as pd
from backend.core import dataset_registry as registry
from backend.api.eval import run_evaluations

def test_run_evaluations_end_to_end():
    df = pd.DataFrame({
        "Serialized ID":["v1","v1","v2","v2"],
        "Unique ID":["u1","u1","u2","u2"],
        "Location":["A","B","A","C"],
        "Route_Order":[1,2,1,2],
        "Perpetrators (NLP)":[["p1"],[],["p2"],[]],
        "Chiefs (NLP)":[[],[],[],[]],
        "Gender of Victim":["F","F","M","M"],
        "Nationality of Victim":["N1","N1","N2","N2"],
        "Time Spent in Location / Cities / Places":[None,"3 days",None,"1 week"]
    })
    did = registry.save_df("processed_synth", df, kind="processed")
    report = run_evaluations([did], link_max_samples=50)
    assert "summary" in report and "tables" in report and "details" in report
    assert report["summary"]["victims"] == 2

def test_ngram_shim_compatibility():
    from backend.models.sequence_predictor import NgramSequenceModel
    m = NgramSequenceModel(alpha=0.05)
    m.fit([["A","B","C"], ["A","B","D"]])
    dist = m.predict_next_dist("A", "B")   # legacy call
    assert isinstance(dist, dict) and len(dist) >= 1
