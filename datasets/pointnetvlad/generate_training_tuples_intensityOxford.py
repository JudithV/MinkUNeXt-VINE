# Modified PointNetVLAD code: https://github.com/mikacuy/pointnetvlad
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS
from datasets.base_datasets import TrainingTuple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = PARAMS.dataset_folder

runs_folder = "IntensityOxford/runs/"
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/pointcloud_20m_10overlap/"

all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
folders = []

index_list = range(len(all_folders)-1)
print("Number of runs: " + str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124, 620084.402381]
p2 = [5735611.299219, 620540.270327]
p3 = [5735237.358209, 620543.094379]
p4 = [5734749.303802, 619932.693364]
p = [p1, p2, p3, p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing and northing < point[0] + x_width and point[
            1] - y_width < easting and easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set


##########################################


def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=10)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=50)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = ind_nn[i]
        non_negatives = ind_r[i]
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])
        anchor_pos = np.array(df_centroids.iloc[i][['northing', 'easting']])
        positives = positives[positives != i]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)
        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[i] = TrainingTuple(id=i, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


####Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder in folders:
    df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
    df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        if (check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)

print("Number of training submaps: " + str(len(df_train['file'])))
print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))

construct_query_dict(df_train, "intensityOxford_training_queries.pickle")
construct_query_dict(df_test, "intensityOxford_test_queries.pickle")
