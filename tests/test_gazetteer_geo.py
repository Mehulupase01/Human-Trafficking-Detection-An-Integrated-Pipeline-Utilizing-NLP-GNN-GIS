import pandas as pd
from backend.geo.gazetteer import ingest_custom_csv, set_active_gazetteer
from backend.geo.gazetteer import resolve_with_gazetteer
from backend.core import dataset_registry as registry
from backend.geo.geo_utils import resolve_locations_to_coords

def test_custom_gazetteer_resolution():
    df = pd.DataFrame([
        {"name":"Alpha", "lat":10.0, "lon":20.0, "country":"X"},
        {"name":"Beta City", "lat":11.0, "lon":21.0, "country":"X"},
    ])
    gid = ingest_custom_csv(df.to_csv(index=False), name="Custom", owner=None)
    set_active_gazetteer(gid)
    res = resolve_with_gazetteer(["Alpha", "Beta City"])
    assert res["Alpha"] == (10.0, 20.0)
    # geo_utils should delegate to gazetteer
    res2 = resolve_locations_to_coords(["Alpha"])
    assert res2["Alpha"] == (10.0, 20.0)
