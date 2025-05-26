# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import csv
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
from sklearn.neighbors import KDTree
import pickle
import utm
import random
import tqdm


from datasets.base_datasets import TrainingTuple
# Import test set boundaries
from datasets.pointnetvlad.generate_test_sets import P26, P27, P28, check_in_test_set

# Test set boundaries
P = [P26, P27, P28]

RUNS_FOLDER = "vmd/"
FILENAME_GPS = "gps.csv"
FILENAME = "data.csv"
POINTCLOUD_FOLS = "pointcloud/lidar3d_0/"

def plot_split_for_anchor(df_centroids, queries, filename, anchor_ndx=0,
                          delta_pos_north=8, delta_pos_east=2.5, 
                          delta_neg_north=10, delta_neg_east=3):
    # Obtén la posición del ancla y la matriz de coordenadas
    anchor = queries[anchor_ndx]
    anchor_pos = anchor.position  # [northing, easting]
    coords = df_centroids[['northing', 'easting']].values

    # Configura la figura y el eje
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Graficar todos los scans
    ax.scatter(coords[:, 1], coords[:, 0], color='gray', alpha=0.6, label='Scans')
    
    # Graficar el ancla
    ax.scatter(anchor_pos[1], anchor_pos[0], color='blue', s=100, label='Anchor', zorder=5)
    
    # Graficar los puntos positivos
    pos_idx = anchor.positives
    ax.scatter(coords[pos_idx, 1], coords[pos_idx, 0],
               color='green', s=80, label='Positivos', zorder=4)
    
    # Graficar los puntos no-negativos
    nonneg_idx = anchor.non_negatives
    ax.scatter(coords[nonneg_idx, 1], coords[nonneg_idx, 0],
               color='red', marker='x', s=80, label='No-negativos', zorder=4)
    
    # Dibujar rectángulo para la región positiva
    pos_rect = patches.Rectangle(
        (anchor_pos[1] - delta_pos_east, anchor_pos[0] - delta_pos_north),
        2 * delta_pos_east, 2 * delta_pos_north,
        linewidth=2, edgecolor='green', facecolor='none', label='Région positiva'
    )
    ax.add_patch(pos_rect)
    
    # Dibujar rectángulo para la región no-negativa
    nonneg_rect = patches.Rectangle(
        (anchor_pos[1] - delta_neg_east, anchor_pos[0] - delta_neg_north),
        2 * delta_neg_east, 2 * delta_neg_north,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
        label='Région no-negativa'
    )
    ax.add_patch(nonneg_rect)
    
    # Configurar etiquetas y leyenda
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.legend()
    ax.set_title(f"Visualización del Split para Anchor {anchor_ndx}")
    plt.savefig(filename, dpi=300)

def construct_query_dict(df_centroids, base_path, filename, 
                                     delta_pos_north=8, delta_pos_east=2.5, 
                                     delta_neg_north=10, delta_neg_east=3):

    queries = {}
    coords = df_centroids[['northing', 'easting']].values

    for anchor_ndx in range(len(df_centroids)):
        anchor_pos = coords[anchor_ndx]
        query = df_centroids.iloc[anchor_ndx]["file"]

        # Comparar distancias relativas para encontrar positivos y no-negativos
        diffs = coords - anchor_pos  # vector restado a cada punto
        abs_diffs = np.abs(diffs)

        # Positivos: dentro de un rectángulo pequeño (excepto el mismo punto)
        positive_mask = (
            (abs_diffs[:, 0] <= delta_pos_north) &
            (abs_diffs[:, 1] <= delta_pos_east) &
            (np.arange(len(coords)) != anchor_ndx)
        )
        positives = np.where(positive_mask)[0]

        # No-negativos: dentro de un rectángulo grande
        non_negative_mask = (
            (abs_diffs[:, 0] <= delta_neg_north) &
            (abs_diffs[:, 1] <= delta_neg_east)
        )
        non_negatives = np.where(non_negative_mask)[0]

        # Extract timestamp from filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.csv', f"Expected .csv file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        # Sort the results
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=positives,
            non_negatives=non_negatives,
            position=anchor_pos
        )

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done", filename)
    return queries

def construct_query_dict_pnv(df_centroids, base_path, filename):
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=5)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=12)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":query,
                      "positives":positives,"negatives":negatives}
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def get_df_at_times(file_name, df_data, time_list, time_pcs):
    """
    Build a pandas df from exaclty the times specified
    """
    # now find odo corresponding to closest times
    data = []
    index_pcs = 0
    for timestamp in time_list:
        # find the exact timestamp in df
        ind = df_data['#timestamp [ns]'] == timestamp
        row = df_data.loc[ind]
        UTMx, UTMy, _, _ = utm.from_latlon(row['latitude'], row['longitude'])
        file_name.writerow([time_pcs[index_pcs], UTMx[0], UTMy[0]])
        #data.append([time_pcs[index_pcs], UTMx[0], UTMy[0]])
        index_pcs += 1

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

def get_closest_data(df_data, time_list, gps_mode='utm'):
    positions = []
    orientations = []
    corresp_time_list = []

    for timestamp in time_list:
        ind = df_data['timestamp'].sub(timestamp).abs().idxmin()
        try:
            position = [df_data['x'][ind], df_data['y'][ind], df_data['z'][ind]]
            positions.append(position)
        except:
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
        except:
            pass

        corresp_time = df_data['timestamp'][ind]
        corresp_time_list.append(corresp_time)
    return np.array(corresp_time_list), np.array(positions), np.array(orientations)

if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER + "pergola/"))) + sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER + "vineyard/")))

    #all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER + "pergola/")))

    folders = []
    lidar_i = 0
    lidar_folders = [0, 1]  # Alternamos entre 0 y 1

    # All runs are used for training (both full and partial)
    index_list = len(all_folders)
    print("Number of runs: " + str(index_list))
    for index in range(index_list):
        folders.append(all_folders[index])
    print(folders)

    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    run_path = ""
    for folder in tqdm.tqdm(folders):
        files, scantimes_pcds, ref_times, scan_times, utm_pos = [], [], [], [], []
        if os.path.exists(RUNS_FOLDER+"pergola/"+folder):
            run_path = os.path.join(RUNS_FOLDER,"pergola", folder)
            files = os.listdir(os.path.join(run_path,POINTCLOUD_FOLS))

            for f in files:
                timestamp = int(f.split('.')[0])
                scantimes_pcds.append(timestamp)
            scantimes_pcds = list(set(scantimes_pcds))
            scan_data = pd.DataFrame({"timestamp": scantimes_pcds})
            scan_data.to_csv(os.path.join(run_path,"scan_times.csv"), index = False)
            with open(run_path+"/"+FILENAME, 'w', newline='') as file:
                escritor_csv = csv.writer(file)
                escritor_csv.writerow(['timestamp','northing','easting'])
                scan_data = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, "pergola/", folder, "scan_times.csv"), sep=',')
                gps_data = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, "pergola/", folder, FILENAME_GPS), sep=',')
                UTMx, UTMy = gps2utm(gps_data)
                gps_times = gps_data['timestamp']
                ref_times, _ = sample_gps(deltaxy=1.0,UTMx=UTMx, UTMy=UTMy, timestamp=gps_times)
                scan_times, _, _ = get_closest_data(scan_data, ref_times)
                _, utm_pos, _ = get_closest_data(gps_data, scan_times, gps_mode='utm')
                ind = 0
                for ts in scan_times:
                    escritor_csv.writerow([ts,utm_pos[ind][0],utm_pos[ind][1]])
                    ind += 1
                # Delete unused scans that were excluded by the sampling in order to free disk space
                used_scans = [str(scan).strip() + ".csv" for scan in scan_times]
                unused_scans = list(set(files) - set(used_scans))
                print(f"Len used: {len(used_scans)}")
                print(f"Len unused: {len(unused_scans)}")
                print(f"Total files: {len(files)}")

                """for f in unused_scans:
                    try:
                        os.remove(os.path.join(run_path, POINTCLOUD_FOLS, f))
                    except Exception as e:
                        print(f"Error deleting {f}: {e}")"""
        elif os.path.exists(RUNS_FOLDER+"vineyard/"+folder):
            run_path = os.path.join(RUNS_FOLDER,"vineyard",folder)
            files = os.listdir(os.path.join(run_path, POINTCLOUD_FOLS))

            for f in files:
                timestamp = int(f.split('.')[0])
                scantimes_pcds.append(timestamp)
            scantimes_pcds = list(set(scantimes_pcds))
            scan_data = pd.DataFrame({"timestamp": scantimes_pcds})
            scan_data.to_csv(os.path.join(run_path, "scan_times.csv"), index = False)
            with open(run_path + "/" +FILENAME, 'w', newline='') as file:
                escritor_csv = csv.writer(file)
                escritor_csv.writerow(['timestamp','northing','easting'])
                scan_data = pd.read_csv(os.path.join(base_path, run_path, "scan_times.csv"), sep=',')
                gps_data = pd.read_csv(os.path.join(base_path, run_path, FILENAME_GPS), sep=',')
                UTMx, UTMy = gps2utm(gps_data)
                gps_times = gps_data['timestamp']
                ref_times, _ = sample_gps(deltaxy=1.0,UTMx=UTMx, UTMy=UTMy, timestamp=gps_times)
                scan_times, _, _ = get_closest_data(scan_data, ref_times)
                _, utm_pos, _ = get_closest_data(gps_data, scan_times, gps_mode='utm')
                ind = 0
                for ts in scan_times:
                    escritor_csv.writerow([ts,utm_pos[ind][0],utm_pos[ind][1]])
                    ind += 1
                # Delete unused scans that were excluded by the sampling in order to free disk space
                used_scans = [str(scan).strip() + ".csv" for scan in scan_times]
                unused_scans = list(set(files) - set(used_scans))
                print(f"Len used: {len(used_scans)}")
                print(f"Len unused: {len(unused_scans)}")
                print(f"Total files: {len(files)}")

                """for f in unused_scans:
                    try:
                        os.remove(os.path.join(run_path, POINTCLOUD_FOLS, f))
                    except Exception as e:
                        print(f"Error deleting {f}: {e}")"""
        if lidar_i == 0:
            lidar_i = 1
        else:
            lidar_i = 0
            
    iter = 0
    for folder in tqdm.tqdm(folders):
        if os.path.exists(RUNS_FOLDER + "pergola/" + folder):
            df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, "pergola/", folder, FILENAME), sep=',')
            #print(df_locations)
            #df_locations['timestamp'] = RUNS_FOLDER + "pergola/" + folder + POINTCLOUD_FOLS + (df_locations.index % 2).astype(str) + "/" + df_locations['timestamp'].astype(str) + '.csv'
            df_locations['timestamp'] = RUNS_FOLDER + "pergola/" + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.csv'
            df_locations = df_locations.rename(columns={'timestamp': 'file'})
            #print((df_locations.index % 2).astype(str))
            for index, row in df_locations.iterrows():
                """if check_in_test_set(row['northing'], row['easting'], P): # iter == (len(all_folders) - 1)
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)"""
                if "run1" in row['file']:
                    df_train = df_train.append(row, ignore_index=True)
                else:
                    df_test = df_test.append(row, ignore_index=True)
        elif os.path.exists(RUNS_FOLDER + "vineyard/" + folder):
            df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, "vineyard/", folder, FILENAME), sep=',')
            #print(df_locations)
            #df_locations['timestamp'] = RUNS_FOLDER + "vineyard/" + folder + POINTCLOUD_FOLS + (df_locations.index % 2).astype(str) + "/" + df_locations['timestamp'].astype(str) + '.csv'
            df_locations['timestamp'] = RUNS_FOLDER + "vineyard/" + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.csv'
            df_locations = df_locations.rename(columns={'timestamp': 'file'})
            #print((df_locations.index % 2).astype(str))
            for index, row in df_locations.iterrows():
                """if check_in_test_set(row['northing'], row['easting'], P): # iter == (len(all_folders) - 1)
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)"""
                if "run1" in row['file']:
                    df_train = df_train.append(row, ignore_index=True)
                else:
                    df_test = df_test.append(row, ignore_index=True)
        if lidar_i == 0:
            lidar_i = 1
        else:
            lidar_i = 0
        iter += 1

    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    print("Pergola count in train: ", df_train["file"].str.count("pergola").sum())
    print("Vineyard count in train: ", df_train["file"].str.count("vineyard").sum())
    print("Pergola count in test: ", df_test["file"].str.count("pergola").sum())
    print("Vineyard count in test: ", df_test["file"].str.count("vineyard").sum())

    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    train_queries = construct_query_dict(df_train, base_path, "training_queries_vmd_feb-may.pickle")
    plot_split_for_anchors(df_train, train_queries, "scans_train_set.png")
    test_queries = construct_query_dict(df_test, base_path, "test_queries_vmd_feb-may.pickle")
    plot_split_for_anchors(df_test, test_queries, "scans_test_set.png")
    #construct_query_dict_pnv(df_train, base_path, "PNV_training_queries_vmd.pickle")
    #construct_query_dict_pnv(df_test, base_path, "PNV_test_queries_vmd.pickle")

