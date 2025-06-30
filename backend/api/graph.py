import os
import networkx as nx
import plotly.graph_objects as go
from backend.utils.helpers import save_html

def run_graph_pipeline(structured_data):
    G = nx.DiGraph()

    for record in structured_data:
        victim = record.get("Unique ID", "")
        locations = record.get("City / Locations Crossed", "")
        traffickers = record.get("Name of the Perpetrators involved", "")
        chiefs = record.get("Human traffickers/ Chief of places", "")

        location_nodes = [loc.strip() for loc in str(locations).split(",") if loc.strip()]
        trafficker_nodes = [t.strip() for t in str(traffickers).split(",") if t.strip()]
        chief_nodes = [c.strip() for c in str(chiefs).split(",") if c.strip()]

        # Add nodes
        G.add_node(victim, type="victim")
        for loc in location_nodes:
            G.add_node(loc, type="location")
            G.add_edge(victim, loc)

        for t in trafficker_nodes:
            G.add_node(t, type="trafficker")
            G.add_edge(victim, t)

        for c in chief_nodes:
            G.add_node(c, type="chief")
            G.add_edge(c, victim)

    # Positioning
    pos = nx.spring_layout(G, seed=42)
    node_x, node_y, labels, colors = [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        ntype = G.nodes[node].get("type", "")
        colors.append(
            "red" if ntype == "victim" else
            "blue" if ntype == "location" else
            "green" if ntype == "trafficker" else
            "purple"
        )

    edge_x, edge_y = [], []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'), mode='lines'))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=12, color=colors),
        text=labels, textposition="bottom center"
    ))
    fig.update_layout(title="Victim-Trafficker-Location Graph", showlegend=False)

    # Save to HTML
    path = save_html(fig, "graph_network.html")
    return path
