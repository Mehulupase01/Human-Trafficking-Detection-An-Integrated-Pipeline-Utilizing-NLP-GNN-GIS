import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

def build_trafficker_hierarchy(df: pd.DataFrame, output_path="/mnt/data/hierarchy_graph.png"):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        hierarchy_raw = row.get("Hierarchy of Perpetrators")
        if pd.notna(hierarchy_raw):
            levels = [level.strip() for level in hierarchy_raw.split("→") if level.strip()]
            for i in range(len(levels)-1):
                G.add_edge(levels[i], levels[i+1])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.8)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10, arrows=True)
    plt.title("Trafficker Hierarchy Graph")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path
