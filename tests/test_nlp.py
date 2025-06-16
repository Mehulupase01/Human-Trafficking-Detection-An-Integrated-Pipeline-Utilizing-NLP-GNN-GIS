import pytest
from backend.api.nlp import run_nlp_pipeline
import pandas as pd

def test_run_nlp_pipeline():
    df = pd.DataFrame({
        "Unique ID (Victim)": ["123"],
        "City / Locations Crossed": ["Tripoli, Misrata, Sabha"],
        "Name of the Perpetrators involved": ["Ahmed, Yusef"]
    })
    results = run_nlp_pipeline(df)
    assert isinstance(results, list)
    assert "Locations" in results[0]