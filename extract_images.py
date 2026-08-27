import json
import base64
import os

path = "notebooks/04_rnn_attention_models.ipynb"
artifact_dir = "docs/images"

with open(path, "r") as f:
    nb = json.load(f)

img_count = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code' and cell.get('outputs'):
        for output in cell['outputs']:
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                img_bytes = base64.b64decode(img_data)
                
                img_name = ""
                if img_count == 0:
                    img_name = "attention_heatmap.png"
                elif img_count == 1:
                    img_name = "ig_heatmap.png"
                elif img_count == 2:
                    img_name = "ig_barplot.png"
                elif img_count == 3:
                    img_name = "shap_summary.png"
                else:
                    img_name = f"extra_plot_{img_count}.png"
                    
                img_path = os.path.join(artifact_dir, img_name)
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)
                print(f"Saved {img_name}")
                img_count += 1
