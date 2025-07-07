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

        G.add_node(victim, type="victim")

        for loc in str(locations).split(","):
            loc = loc.strip()
            if loc:
                G.add_node(loc, type="location")
                G.add_edge(victim, loc)

        for t in str(traffickers).split(","):
            t = t.strip()
            if t:
                G.add_node(t, type="trafficker")
                G.add_edge(t, victim)

        for c in str(chiefs).split(","):
            c = c.strip()
            if c:
                G.add_node(c, type="chief")
                G.add_edge(c, victim)

    pos = nx.spring_layout(G, seed=42)

    node_x, node_y, node_color, node_text = [], [], [], []

    type_colors = {
        "victim": "red",
        "location": "blue",
        "trafficker": "green",
        "chief": "purple"
    }

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = G.nodes[node].get("type", "unknown")
        node_color.append(type_colors.get(ntype, "gray"))
        node_text.append(f"{node} ({ntype})")

    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n if G.nodes[n]["type"] == "victim" else "" for n in G.nodes()],
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=10,
            line_width=1
        ),
        textposition="bottom center",
        textfont=dict(size=8)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
    layout=go.Layout(
        title=dict(
            text="Victim-Trafficker-Location Graph",
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
         )
        )

    output_path = "frontend/graphs/graph_output.html"
    save_html(fig, output_path)
    return output_path
