import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequence_data(victim_sequences, seq_length=3):
    le = LabelEncoder()
    flat_locations = list(set([loc for seq in victim_sequences for loc in seq]))
    le.fit(flat_locations)

    encoded = [le.transform(seq).tolist() for seq in victim_sequences if len(seq) > seq_length]
    X, y = [], []
    for seq in encoded:
        for i in range(len(seq) - seq_length):
            X.append(seq[i:i+seq_length])
            y.append(seq[i+seq_length])

    return torch.tensor(X).unsqueeze(-1).float(), torch.tensor(y), le

def train_and_predict(victim_sequences):
    X, y, le = prepare_sequence_data(victim_sequences)
    if len(X) == 0:
        return "Insufficient sequence data to train."

    model = LSTMModel(1, 32, len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Predict next location for each victim's last trajectory
    predictions = []
    for seq in victim_sequences:
        if len(seq) < 3:
            continue
        input_seq = torch.tensor(le.transform(seq[-3:])).unsqueeze(0).unsqueeze(-1).float()
        pred_idx = torch.argmax(model(input_seq), dim=1).item()
        predictions.append({"Victim": seq, "Next Location": le.inverse_transform([pred_idx])[0]})

    return predictions
