from backend.graph.graph_builder import build_graph, export_pyvis_graph

def run_graph_pipeline(structured_data):
    G = build_graph(structured_data)
    html_path = export_pyvis_graph(G)
    return html_path


