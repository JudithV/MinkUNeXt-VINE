# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import sys
import bisect
import utm
import csv
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import tqdm


from datasets.base_datasets import TrainingTuple
# Import test set boundaries
from datasets.pointnetvlad.generate_test_sets import P21, P22, P23, P24, P25, check_in_test_set

# Test set boundaries
P = [P21, P22, P23, P24]

RUNS_FOLDER = "nclt/"
GPS_FOLS ="gps/"
FILENAME = "data.csv"
POINTCLOUD_FOLS = "/velodyne_sync/"


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def gps2utm(df_gps):
    latitude = df_gps['latitude']
    longitude = df_gps['longitude']

    lat = np.array(latitude)
    lon = np.array(longitude)

    UTMx, UTMy, _, _ = utm.from_latlon(lat, lon)

    return UTMx, UTMy

def sample_gps(deltaxy, UTMx, UTMy, timestamp):
    gps_times = []
    utm_positions = []
    gpsi = gpsi1 = gps = []
    for ind in timestamp.index:
        gps = np.array([UTMx[ind], UTMy[ind]])
        current_time = timestamp[ind]
        if ind == 0: 
            gps_times.append(current_time)
            utm_positions.append([UTMx[ind], UTMy[ind]])
            gpsi = gps

        gpsi1 = gps
        dxy = np.linalg.norm(gpsi1[0:2] - gpsi[0:2])
        if dxy > deltaxy:
            gps_times.append(current_time)
            utm_positions.append([UTMx[ind], UTMy[ind]])
            gpsi = gpsi1
    return np.array(gps_times), np.array(utm_positions)

def get_closest_times(master_sensor_times, sensor_times, warning_max_time_dif_s=0.5*1e9):
    """
    For each time in master_sensor_times, find the closest time in sensor_times
    """
    output_times = []
    # for each master_sensor_times, find the closest time in sensor_times
    for timestamp in master_sensor_times:
        d = np.abs(sensor_times-timestamp)
        index = np.argmin(d)
        time_diff_s = d[index]
        output_times.append(sensor_times[index])
        if time_diff_s > warning_max_time_dif_s:
            print('CAUTION!!! Found time difference (s): ', time_diff_s/1e9)
            print('CAUTION!!! Should we associate data??')
    output_times = np.array(output_times)
    return output_times


def get_closest_data(df_data, time_list, gps_mode='utm', scan_data = False):
    positions = []
    orientations = []
    corresp_time_list = []

    for i in range(len(time_list)):
        try:
            ind = df_data['#timestamp [ns]'].sub(time_list[i]).abs().idxmin()
        except Exception as e:
            print(f"Error: {e}")
        try:
            position = [df_data['x'][ind], df_data['y'][ind], df_data['z'][ind]]
            positions.append(position)
        except Exception as e:
            pass
        try:
            latitude = df_data['latitude'][ind]
            longitude = df_data['longitude'][ind]
            altitude = df_data['altitude'][ind]

            lat = np.array(latitude)
            lon = np.array(longitude)
            alt = np.array(altitude)


            if gps_mode == 'utm':
                x, y, _, _ = utm.from_latlon(lat, lon)
                position = [x, y] #, 0]
            else:
                position = [lat, lon, alt]
            positions.append(position)
        except Exception as e:
            pass
        if scan_data:
            corresp_time = df_data[ind]
        else:
            corresp_time = df_data['#timestamp [ns]'][ind]
        corresp_time_list.append(corresp_time)
    
    return np.array(corresp_time_list), np.array(positions), np.array(orientations)

if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders))
    index_list = [0]
    print("Number of runs: " + str(len(index_list)))
    for index in range(len(all_folders)):
        folders.append(all_folders[index])
    print(folders)
    folders = ["2012-03-17", "2012-05-26", "2012-06-15", "2012-08-20", "2012-09-28", "2012-10-28"]

    for folder in tqdm.tqdm(folders):
        files = os.listdir(RUNS_FOLDER+folder+POINTCLOUD_FOLS)
        scantimes_pcds = []

        for f in files:
            timestamp = int(f.split('.')[0])
            scantimes_pcds.append(timestamp)
        scantimes_pcds = list(set(scantimes_pcds))
        scan_data = pd.DataFrame({"#timestamp [ns]": scantimes_pcds})
        scan_data.to_csv(RUNS_FOLDER+folder+"/scan_times.csv", index = False)
    
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    
    for folder in tqdm.tqdm(folders):
        with open(RUNS_FOLDER+folder+"/"+FILENAME, 'w', newline='') as file:
            # Crear un objeto escritor CSV
            escritor_csv = csv.writer(file)
            escritor_csv.writerow(['timestamp','northing','easting'])
            scan_data = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, "scan_times.csv"), sep=',')
            gps_data = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, GPS_FOLS, FILENAME), sep=',')
            gps_ts = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, "gps_rtk.csv"), sep=',', header=None)
            # Reemplazar la columna deseada (ejemplo: columna index 1)
            gps_data.iloc[:, 0] = gps_ts.iloc[:, 0]  # Sustituyendo la segunda columna de df1 con la primera de df2

            # Guardar el resultado en un nuevo archivo
            gps_data.to_csv(os.path.join(base_path, RUNS_FOLDER, folder, GPS_FOLS, FILENAME), index=False)
            UTMx, UTMy = gps2utm(gps_data)
            gps_times = gps_data["#timestamp [ns]"]
            ref_times, _ = sample_gps(deltaxy=5.0,UTMx=UTMx, UTMy=UTMy, timestamp=gps_times)
            
            scan_times, _, _ = get_closest_data(scan_data, ref_times)
            _, utm_pos, _ = get_closest_data(gps_data, scan_times, gps_mode='utm')
            ind = 0
            for ts in scan_times:
                escritor_csv.writerow([ts,utm_pos[ind][0],utm_pos[ind][1]])
                ind += 1
    
    for folder in tqdm.tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, FILENAME), sep=',')
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P): # iter == (len(all_folders) - 1)
                df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "training_queries_nclt_pruebas.pickle", ind_nn_r=10)
    construct_query_dict(df_test, base_path, "test_queries_nclt_pruebas.pickle", ind_nn_r=10)
