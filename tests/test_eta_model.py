import pandas as pd
from backend.models.eta_model import build_duration_stats, estimate_path_durations

def test_eta_stats_and_estimation():
    df = pd.DataFrame({
        "Serialized ID":["v1","v1","v1","v2","v2"],
        "Location":["A","B","C","A","B"],
        "Route_Order":[1,2,3,1,2],
        "Time Spent in Location / Cities / Places":[None,"3 days","2 weeks",None,"7"]
    })
    stats = build_duration_stats(df)
    assert stats["global_median"] >= 5
    hist = ["A","B"]
    nxt = ["C"]
    days = estimate_path_durations(hist, nxt, stats, default_days=7)
    assert isinstance(days, list) and len(days) == 1 and days[0] > 0
