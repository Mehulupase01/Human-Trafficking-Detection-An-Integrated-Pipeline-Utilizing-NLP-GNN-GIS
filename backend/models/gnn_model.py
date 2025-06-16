import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def build_gnn_data(structured_data):
    G = nx.Graph()
    labels = {}
    all_nodes = set()
    label_encoder = LabelEncoder()

    for entry in structured_data:
        victim = f"V_{entry['Victim ID']}"
        all_nodes.add(victim)
        labels[victim] = "victim"

        for loc in entry["Locations"]:
            loc_id = loc.replace(" ", "_")
            G.add_edge(victim, loc_id)
            all_nodes.add(loc_id)
            labels[loc_id] = "location"

        for perp in entry["Names"]:
            p_id = perp.replace(" ", "_")
            G.add_edge(victim, p_id)
            all_nodes.add(p_id)
            labels[p_id] = "perpetrator"

    node_list = list(all_nodes)
    node_index = {n: i for i, n in enumerate(node_list)}

    edge_index = torch.tensor([[node_index[u], node_index[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

    label_encoder.fit(list(set(labels.values())))
    y = torch.tensor([label_encoder.transform([labels[n]])[0] for n in node_list], dtype=torch.long)

    x = torch.eye(len(node_list), dtype=torch.float)  # identity features for now
    data = Data(x=x, edge_index=edge_index, y=y)

    return data, label_encoder


def train_gnn(data, num_classes):
    model = GCN(data.num_node_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    return model, out
