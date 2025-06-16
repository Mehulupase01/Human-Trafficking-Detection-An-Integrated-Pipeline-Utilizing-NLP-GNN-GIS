from backend.api.graph import run_graph_pipeline
import os

def test_graph_export():
    structured_data = [
        {"Victim ID": "1105", "Locations": ["Tripoli", "Sabha"], "Names": ["Ahmed"]}
    ]
    path = run_graph_pipeline(structured_data)
    assert os.path.exists(path)
