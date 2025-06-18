import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from torch.nn import Linear, ReLU, Module
from torch_geometric.nn import GCNConv

class GNNModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = ReLU()
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def prepare_gnn_graph(df: pd.DataFrame):
    victims = df["Unique ID"].astype(str)
    traffickers = df["Name of the Perpetrators involved"].fillna("")
    nodes = list(victims.unique())
    edges = []

    # Create edges: victim ↔ trafficker
    for i, row in df.iterrows():
        vid = str(row["Unique ID"])
        if isinstance(row["Name of the Perpetrators involved"], str):
            for t in row["Name of the Perpetrators involved"].split("and"):
                t = t.strip()
                if t:
                    nodes.append(t)
                    edges.append((vid, t))

    # Encode node labels
    unique_nodes = list(set(nodes))
    le = LabelEncoder()
    node_idx = {n: i for i, n in enumerate(unique_nodes)}
    y = torch.tensor([1 if ' ' in n else 0 for n in unique_nodes])  # crude: names w/ space = person

    # Features: simple one-hot
    x = torch.eye(len(unique_nodes))

    edge_index = torch.tensor([[node_idx[a], node_idx[b]] for a, b in edges] + [[node_idx[b], node_idx[a]] for a, b in edges], dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.node_names = unique_nodes
    data.label_encoder = le

    return data

def run_gnn_prediction(data):
    model = GNNModel(data.num_node_features, 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    predictions = {data.node_names[i]: int(pred[i]) for i in range(len(pred))}
    return predictions
