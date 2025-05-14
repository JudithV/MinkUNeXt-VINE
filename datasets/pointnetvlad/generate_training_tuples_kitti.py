import os
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import pandas as pd
import pickle
from config import PARAMS
import utm
from tqdm import tqdm
from sklearn.neighbors import KDTree

from datasets.base_datasets import TrainingTuple
from datasets.pointnetvlad.generate_test_sets import P29, P30, check_in_test_set

# Define zonas de test (puedes personalizarlas si quieres)
P = [P29, P30]

SECUENCIAS_FOLDER = "KittiDataset/sequences/"
FILENAME = "pointcloud_locations_kitti.csv"

def load_positions(sequence_path):
    gps_folder = os.path.join(sequence_path, 'oxts', 'data')
    lidar_folder = os.path.join(sequence_path, 'velodyne_points', 'data')

    files = sorted(os.listdir(gps_folder))
    data = []

    for file in files:
        gps_path = os.path.join(gps_folder, file)
        with open(gps_path, 'r') as f:
            lat, lon, *_ = map(float, f.read().strip().split())
            utm_coord = utm.from_latlon(lat, lon)
            northing, easting = utm_coord[0], utm_coord[1]

            bin_filename = file.replace('.txt', '.bin')
            rel_path = os.path.join(sequence_path, 'velodyne_points', 'data', bin_filename)

            data.append({'file': rel_path, 'northing': northing, 'easting': easting})

    return pd.DataFrame(data)

def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)

    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        timestamp = int(os.path.splitext(os.path.basename(query))[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]
        positives = positives[positives != anchor_ndx]

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=np.sort(positives),
            non_negatives=np.sort(non_negatives),
            position=anchor_pos
        )

    with open(os.path.join(base_path, filename), 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {filename} with {len(queries)} queries")

if __name__ == '__main__':
    base_path = PARAMS.dataset_folder
    sequence_folders = sorted(os.listdir(SECUENCIAS_FOLDER))
    
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    sequence = "2011_09_26_drive_0051_sync"

    sequence_path = os.path.join(base_path,SECUENCIAS_FOLDER, sequence)
    df = load_positions(sequence_path)
    
    for i, row in df.iterrows():
        """if check_in_test_set(row['northing'], row['easting'], P):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)"""
        if i < (df.shape[0] / 2):
            df_train = df_train.append(row, ignore_index=True)
        else:
            df_test = df_test.append(row, ignore_index=True)

    print(f"Entrenamiento: {len(df_train)} muestras")
    print(f"Test: {len(df_test)} muestras")

    construct_query_dict(df_train, base_path, "training_queries_kitti.pickle", ind_nn_r=10)
    construct_query_dict(df_test, base_path, "test_queries_kitti.pickle", ind_nn_r=10)

