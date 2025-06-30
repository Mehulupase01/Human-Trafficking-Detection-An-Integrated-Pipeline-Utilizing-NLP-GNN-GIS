import torch
from torch.nn import Linear, ReLU, Module
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class GNNModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = ReLU()
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def prepare_gnn_graph(df: pd.DataFrame) -> Data:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    victims = df["Unique ID"].astype(str)
    trafficker_col = "Human traffickers/ Chief of places"

    nodes = set(victims.unique())
    edges = []

    for _, row in df.iterrows():
        victim_id = str(row["Unique ID"])
        raw_traffickers = row.get(trafficker_col, [])

        # Handle if string instead of list
        if isinstance(raw_traffickers, str):
            if raw_traffickers.strip().lower() not in ["", "no", "not available"]:
                raw_traffickers = [t.strip() for t in raw_traffickers.replace(" and ", ",").split(",") if t.strip()]
            else:
                raw_traffickers = []

        for t in raw_traffickers:
            t_node = f"Trafficker_{t}"
            nodes.add(t_node)
            edges.append((victim_id, t_node))
            edges.append((t_node, victim_id))  # bi-directional

    node_list = sorted(list(nodes))
    node_idx = {name: i for i, name in enumerate(node_list)}

    # Basic features: 1-hot vectors
    x = torch.eye(len(node_list))

    # Labels (for training): 1 = predicted trafficker, 0 = others
    y = torch.tensor([
        1 if name.startswith("Trafficker_") else 0
        for name in node_list
    ], dtype=torch.long)

    edge_index = torch.tensor(
        [[node_idx[a], node_idx[b]] for a, b in edges],
        dtype=torch.long
    ).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.node_names = node_list
    return data

def run_gnn_prediction(data: Data):
    model = GNNModel(data.num_node_features, 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    predictions = {
        data.node_names[i]: int(pred[i]) for i in range(len(pred))
    }
    return predictions
