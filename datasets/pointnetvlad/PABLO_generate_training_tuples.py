import numpy as np
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS
import pandas as pd
import utm
from sklearn.neighbors import KDTree
import pickle
import tqdm

from datasets.base_datasets import TrainingTuple
from datasets.pointnetvlad.generate_test_sets import check_in_test_set

# For BLT
n1,n2,zone_number,zone_letter=utm.from_latlon(40.45022395412607, 22.92418714031768)
P1 = [n1,n2]
n1,n2,zone_number,zone_letter=utm.from_latlon(40.45003414581829, 22.9241992901574)
P2 = [n1,n2]

# Test set boundaries
P = [P1, P2]
RUNS_FOLDER = "blt/ktima/"
RUNS_GPS = "robot0/gps0/"
FILENAME = "data.csv"
POINTCLOUD_FOLS = "/PuntosLidar/"

def convert_gps_to_utm(input_file, output_file):
    df = pd.read_csv(input_file)
    new_df = df.copy()
    
    def convert_to_utm(lat, lon):
        easting, northing, _, _ = utm.from_latlon(lat, lon)
        return northing, easting
    
    new_df[['northing', 'easting']] = df.apply(lambda row: convert_to_utm(row.iloc[1], row.iloc[2]), axis=1, result_type='expand')
    new_df = new_df[["timestamp", "northing", "easting"]]
    new_df.columns = ["timestamp", "northing", "easting"]
    new_df.to_csv(output_file, index=False)

def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=5):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.csv', f"Expected .csv file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])
        positives = np.sort(ind_nn[anchor_ndx][ind_nn[anchor_ndx] != anchor_ndx])
        non_negatives = np.sort(ind_r[anchor_ndx])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)
    with open(os.path.join(base_path, filename), 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder
    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    folders = all_folders[:-1]
    print("Number of runs: " + str(len(folders)))
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    for folder in tqdm.tqdm(folders):
        input_csv = os.path.join(base_path, RUNS_FOLDER, folder,RUNS_GPS, FILENAME)
        output_csv = os.path.join(base_path, RUNS_FOLDER, folder, "pablo_" + FILENAME)
        print(f"Convirtiendo archivo: {output_csv}")
        convert_gps_to_utm(input_csv, output_csv)
        if not os.path.exists(output_csv):
            print(f"ERROR: El archivo {output_csv} no se generó.")
        else:
            print(f"El archivo convertido se generó correctamente: {output_csv}")
        df_locations = pd.read_csv(output_csv, sep=',')
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.csv'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for _, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)
    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    construct_query_dict(df_train, base_path, "training_queries_pablo.pickle", ind_nn_r=1)
    construct_query_dict(df_test, base_path, "test_queries_pablo.pickle", ind_nn_r=1)