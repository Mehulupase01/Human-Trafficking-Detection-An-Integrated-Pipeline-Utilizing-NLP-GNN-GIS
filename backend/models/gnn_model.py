import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def build_gnn_data(records):
    G = nx.Graph()
    for rec in records:
        victim = f"Victim_{rec['Unique ID']}"
        locations = rec.get("City / Locations Crossed", [])
        traffickers = rec.get("Human traffickers/ Chief of places", [])

        G.add_node(victim, type="victim")

        for loc in locations:
            loc_node = f"Location_{loc}"
            G.add_node(loc_node, type="location")
            G.add_edge(victim, loc_node)

        for perp in traffickers:
            perp_node = f"Trafficker_{perp}"
            G.add_node(perp_node, type="trafficker")
            G.add_edge(victim, perp_node)

    label_map = {"victim": 0, "location": 1, "trafficker": 2}
    features = []
    labels = []
    node_list = list(G.nodes)

    for node in node_list:
        node_type = G.nodes[node]["type"]
        labels.append(label_map[node_type])
        features.append([label_map[node_type]])

    edge_index = torch.tensor(
        [[node_list.index(src), node_list.index(dst)] for src, dst in G.edges],
        dtype=torch.long
    ).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, LabelEncoder().fit(list(label_map.keys()))

def train_gnn(data, num_classes):
    model = SimpleGCN(in_channels=1, hidden_channels=16, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(data)

    return model, output
