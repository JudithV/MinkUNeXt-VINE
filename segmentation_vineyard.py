import pandas as pd
import numpy as np
import utm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from collections import Counter


def distancia_punto_a_segmento(px, py, x1, y1, x2, y2):
    """Calcula la distancia entre un punto y un segmento"""
    A = np.array([px, py])
    B = np.array([x1, y1])
    C = np.array([x2, y2])
    AB = C - B
    AP = A - B
    t = np.clip(np.dot(AP, AB) / np.dot(AB, AB), 0, 1)
    closest = B + t * AB
    return np.linalg.norm(A - closest)

# Cargar CSV y convertir coordenadas
df = pd.read_csv('vmd/vineyard/run3_09_v/gps.csv')
utm_coords = np.array([utm.from_latlon(lat, lon)[:2] for lat, lon in zip(df['latitude'], df['longitude'])])
x_coords, y_coords = utm_coords[:, 0], utm_coords[:, 1]

# Etiquetado automático mendiante algoritmos de agrupación
# PCA para alinear el viñedo
pca = PCA(n_components=2)
coords_pca = pca.fit_transform(utm_coords)
main_axis = coords_pca[:, 0]  # eje longitudinal del viñedo (a lo largo de las filas)
perp_axis = coords_pca[:, 1]  # eje transversal (a través de las filas)

# Definir extremos en el eje principal (no en y_coords)
main_min, main_max = np.percentile(main_axis, [2, 98])
margin = 5.0  # metros (ajustable dependiendo del viñedo)

extremo_inferior = main_axis < (main_min + margin)
extremo_superior = main_axis > (main_max - margin)
interior = ~(extremo_inferior | extremo_superior)

# Clustering de filas solo en puntos interiores (usando el eje transversal)
k = 3 # Vineyard: run1 10, run2 8, run3 3 / Pergola (all): 4

# Using Kmeans...
fila_axis = perp_axis[interior].reshape(-1, 1)
kmeans = KMeans(n_clusters=k).fit(fila_axis)
fila_ids = kmeans.labels_

# Ordenar etiquetas de fila según posición en eje perpendicular
centroids = kmeans.cluster_centers_.flatten()
sorted_idx = np.argsort(centroids)
label_map = {old: new+1 for new, old in enumerate(sorted_idx)}  # +1 para dejar 0 a extremos

# Aplicar el mapeo
fila_ids_mapeados = np.array([label_map[label] if label in label_map else 0 for label in fila_ids])

# Asignar etiquetas a todos los puntos"""
segment_labels = np.zeros(len(x_coords), dtype=int)
segment_labels[extremo_inferior] = 0
segment_labels[extremo_superior] = max(label_map.values()) + 1  # último número +1
segment_labels[interior] = fila_ids_mapeados

# Manual segmentation (vineyards)

# Manual definition of the begin and end coordinates per vineyard row
row_1 = ((405155.2190205742, 5025153.456242999), (405129.3731774929, 5025049.3743693605))
row_2 = ((405157.4321226374, 5025152.425544577), (405131.5971746107, 5025048.628570092))
row_3 = ((405159.6183889506, 5025150.998329473), (405133.90704598336, 5025047.799673114))
row_4 = ((405161.6454257898, 5025149.608967498), (405136.10399282177, 5025046.8771700645))
row_5 = ((405163.924661071, 5025148.1458705915), (405138.7723665842, 5025047.226359841))
row_6 = ((405166.0998263047, 5025146.162998042), (405141.31138470565, 5025046.407845422))
row_7 = ((405168.1540389298, 5025144.708553424), (405143.7429185019, 5025046.328255839))
row_8 = ((405170.3656885423, 5025143.809923135), (405146.2338889275, 5025046.047070159))
row_9 = ((405172.57488153444, 5025143.034990951), (405148.50564491365, 5025045.692208357))
row_10 = ((405174.9257648668, 5025142.054617021), (405151.1129026281, 5025046.025697024))

rows_run_1 = [row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10]
rows_run_2 = [row_1, row_2, row_3, row_4, row_6, row_7, row_8, row_9]

rows_run_1_pca = [(pca.transform([[x1, y1]])[0], pca.transform([[x2, y2]])[0]) for (x1, y1), (x2, y2) in rows_run_1]
rows_run_2_pca = [(pca.transform([[x1, y1]])[0], pca.transform([[x2, y2]])[0]) for (x1, y1), (x2, y2) in rows_run_2]

interior_indices = np.where(interior)[0]

for i, (px, py) in enumerate(zip(main_axis[interior], perp_axis[interior])):
    distancias = [distancia_punto_a_segmento(px, py, x1, y1, x2, y2) for (x1, y1), (x2, y2) in rows_run_2_pca]
    label_mas_cercano = np.argmin(distancias) + 1  # etiquetas 1..N
    segment_labels[interior_indices[i]] = label_mas_cercano


df['segment'], df['type'] = segment_labels, "V"
df.to_csv('vmd/vineyard/run3_09_v/gps.csv', index=False)
