import pandas as pd
from backend.models.sequence_predictor import NgramSequenceModel, build_sequences_from_df
from backend.models.link_predictor import LinkPredictor
from backend.api.graph_build import build_network_from_processed

def sample_df():
    return pd.DataFrame({
        "Serialized ID":["v1","v1","v1","v2","v2","v3","v3"],
        "Location":["A","B","C","A","B","B","C"],
        "Route_Order":[1,2,3,1,2,1,2],
        "Perpetrators (NLP)":[["p1"],["p1","p2"],[],[],["p2"],["p3"],[]],
        "Chiefs (NLP)":[[],[],[],[],[],[],[]]
    })

def test_ngram_predictor_and_graph():
    df = sample_df()
    seqs = build_sequences_from_df(df)
    m = NgramSequenceModel(alpha=0.05)
    m.fit(seqs)
    dist = m.predict_next_dist("A","B")
    assert isinstance(dist, dict) and len(dist) >= 1
    G = build_network_from_processed(df)
    assert G.number_of_nodes() >= 5 and G.number_of_edges() >= 5

def test_link_predictor():
    df = sample_df()
    lp = LinkPredictor()
    lp.fit(df)
    preds = lp.predict_for_victim("v2", top_k=3)
    assert isinstance(preds, list) and len(preds) >= 1 and isinstance(preds[0][0], str)
