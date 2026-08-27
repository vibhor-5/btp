import json
import os

path = "notebooks/04_rnn_attention_models.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            if "history_len: int = 28" in line:
                line = line.replace("history_len: int = 28", "history_len: int = 14")
            if "PROJECT_ROOT = Path.cwd()" in line:
                line = line.replace("Path.cwd()", "Path('/Users/anurag/Library/CloudStorage/OneDrive-iiitm.ac.in/BTP/btp-repo')")
            new_source.append(line)
        cell["source"] = new_source

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
