import pandas as pd
from backend.api.ontology import build_graph_from_processed, serialize_ttl
from backend.geo.gazetteer import ingest_custom_csv, set_active_gazetteer

def test_ontology_ttl_export():
    gdf = pd.DataFrame([{"name":"A","lat":1.0,"lon":2.0}])
    gid = ingest_custom_csv(gdf.to_csv(index=False), name="g")
    set_active_gazetteer(gid)
    df = pd.DataFrame({
        "Serialized ID":["v1","v1"],
        "Unique ID":["u1","u1"],
        "Location":["A","A"],
        "Route_Order":[1,2],
        "Perpetrators (NLP)":[["p1"],["p2"]],
        "Chiefs (NLP)":[[],[]],
        "Gender of Victim":["F","F"],
        "Nationality of Victim":["N","N"]
    })
    g, stats = build_graph_from_processed(df, base_uri="https://ex.org/htn#")
    ttl = serialize_ttl(g)
    assert "Victim" in ttl and "Location" in ttl and "geo" in ttl
    assert stats["victims"] == 1 and stats["locations"] >= 1
