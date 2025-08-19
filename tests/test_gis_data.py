import pandas as pd
from backend.api.gis_data import compute_location_stats, build_timestamped_geojson
from backend.geo.gazetteer import ingest_custom_csv, set_active_gazetteer

def test_location_stats_and_timestamped_geojson():
    # Gazetteer for coords
    gdf = pd.DataFrame([
        {"name":"A", "lat":1.0, "lon":2.0},
        {"name":"B", "lat":3.0, "lon":4.0},
        {"name":"C", "lat":5.0, "lon":6.0},
    ])
    gid = ingest_custom_csv(gdf.to_csv(index=False), name="g")
    set_active_gazetteer(gid)

    # Processed-like df
    df = pd.DataFrame({
        "Serialized ID":["v1","v1","v1","v2","v2"],
        "Location":["A","B","C","A","B"],
        "Route_Order":[1,2,3,1,2],
        "Perpetrators (NLP)":[[],["p1"],[],[],["p2"]],
        "Chiefs (NLP)":[[],["c1"],[],[],[]],
        "Time Spent in Location / Cities / Places":[None,"3 days","1 week",None,"2 weeks"]
    })
    nodes, edges, locmap = compute_location_stats(df)
    assert {"location","lat","lon","incoming","outgoing","count"}.issubset(nodes.columns)
    assert any(edges["weight"] >= 1)
    tj = build_timestamped_geojson(df, default_days_per_hop=7, base_date="2020-01-01")
    assert tj["type"] == "FeatureCollection"
    assert len(tj["features"]) >= 2
