import networkx as nx
from pyvis.network import Network
import os

def build_graph(data):
    G = nx.DiGraph()
    for entry in data:
        victim = f"Victim_{entry['Victim ID']}"
        G.add_node(victim, label="Victim", color="#ff6961", shape="ellipse")

        for loc in entry["Locations"]:
            loc_node = loc.replace(" ", "_")
            G.add_node(loc_node, label="Location", color="#77dd77", shape="box")
            G.add_edge(victim, loc_node, label="passed through")

        for name in entry["Names"]:
            p_node = name.replace(" ", "_")
            G.add_node(p_node, label="Perpetrator", color="#779ecb", shape="diamond")
            G.add_edge(victim, p_node, label="interacted with")

    return G

def export_pyvis_graph(G, filename="graph.html"):
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.show(filename)
    return os.path.abspath(filename)
