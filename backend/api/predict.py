from backend.predict.lstm_predictor import train_and_predict

def run_prediction_pipeline(structured_data):
    sequences = [entry['Locations'] for entry in structured_data if len(entry['Locations']) > 3]
    return train_and_predict(sequences)