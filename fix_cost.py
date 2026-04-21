import io
import json

with io.open('Clustering_Logistica.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'inertia = model.computeCost(predictions)' in line:
                new_source.append(line.replace('model.computeCost(predictions)', 'model.summary.trainingCost'))
            elif 'inertia_kmeans_final = kmeans_model.computeCost(df_kmeans)' in line:
                new_source.append(line.replace('kmeans_model.computeCost(df_kmeans)', 'kmeans_model.summary.trainingCost'))
            else:
                new_source.append(line)
        cell['source'] = new_source

with io.open('Clustering_Logistica.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
