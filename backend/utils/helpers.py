import os

def save_html(fig, filename="graph_output.html"):
    output_dir = "frontend/graphs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    fig.write_html(output_path)
    return output_path
