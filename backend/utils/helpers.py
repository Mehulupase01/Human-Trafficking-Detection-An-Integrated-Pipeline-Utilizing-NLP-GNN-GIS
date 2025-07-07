import os
from pathlib import Path

def save_html(fig, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    return output_path
