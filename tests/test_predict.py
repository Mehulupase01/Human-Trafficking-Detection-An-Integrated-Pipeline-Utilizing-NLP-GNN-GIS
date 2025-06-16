from backend.api.predict import run_prediction_pipeline

def test_prediction_structure():
    structured_data = [
        {"Victim ID": "1105", "Locations": ["Asmara", "Khartoum", "Tripoli", "Misrata"]},
        {"Victim ID": "1106", "Locations": ["Addis Ababa", "Khartoum", "Tripoli"]}
    ]
    result = run_prediction_pipeline(structured_data)
    assert isinstance(result, list) or isinstance(result, str)
