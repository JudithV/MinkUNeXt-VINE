# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS

# For training and test data splits, ARVC = 10, BLT = 13, VMD = 23, NCLT = 100, Oxford = 150
X_WIDTH = 23
Y_WIDTH = 23

"""X_WIDTH = 150 
Y_WIDTH = 150"""

# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]

# ARVC
P11 = [702451.781643, 4238859.741135]
P12 = [702431.0067860653, 4238938.581805546]
P13 = [702388.3360408501, 4238945.463840115]
P14 = [702413.545255068, 4238913.822592095] # 38.275482714486884, -0.6860005510446198
P15 = [702437.4984199596, 4238914.70657325] # 38.275485275281, -0.685726671171021

# BLT
#KTIMA
P16 = [663182.204586871, 4479512.96605084]
P17 = [663141.686467221, 4479464.873340834]
P18 = [663116.3699656122, 4479498.777999125]

#RISEHOLME
P19 = [665082.1563399071, 5904963.1919269]
P20 = [665100.1582773875, 5904972.71202409]

# NCLT
P21 = [276617.85459597525, 4685889.601471452]
P22 = [276385.4404570175, 4685737.3182648895]
P23 = [275855.947086381, 4685987.118303991]
P24 = [275830.45268349943, 4685272.447089739]
P25 = [275834.0886877172, 4685718.746155128]

# VMD
P26 = [405156.7520412804, 5025041.790304738]
#P27 = [405107.0985735883, 5025050.372015398]
P28 = [405154.7637688267, 5025158.401943873]

P27 = [405104.76748520625, 5025096.011784253]
P29 = [405179.49875482655, 5025142.53828079]
#P28 = [405178.6821447582, 5025143.267948912]

# KITTI
P30 = [457190.21529351856, 5428835.785329838]
P31 = [457323.8517238996, 5428045.37636426]

P_DICT = {"oxford": [P1, P2, P3, P4], "university": [P5, P6, P7], "residential": [P8, P9, P10], "business": [],
          "arvc": [P12,P13, P14, P15], "blt-ktima": [P16, P17, P18], "blt-riseholme": [P19, P20], 
          "nclt": [P21, P22, P23, P24], "vmd": [P26, P27, P28, P29]}


def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set

def filter_first_visits(df, radius=10.0):
    coords = df[['northing', 'easting']].values
    tree = KDTree(coords)

    mask = np.ones(len(df), dtype=bool)
    visited = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if not visited[i]:
            # Encuentra todos los puntos dentro del radio
            idxs = tree.query_radius([coords[i]], r=radius)[0]
            visited[idxs] = True  # Marca como visitados
        else:
            mask[i] = False  # Este ya se ha visitado antes

    return df[mask].reset_index(drop=True)


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    database_trees = []
    test_trees = []
    ind = 0
    start_row_6 = (405165.94364507403, 5025149.177896624)
    end_row_9 = (405147.7747196499, 5025045.434330138)
    min_easting = min(start_row_6[0], end_row_9[0])
    max_easting = max(start_row_6[0], end_row_9[0])
    min_northing = min(start_row_6[1], end_row_9[1])
    max_northing = max(start_row_6[1], end_row_9[1])

    for folder in folders:
        #print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        """mask_first_rows_wp = (((df_locations['segment'] == 0) | ((df_locations['segment'] == 11) & ("run1" in folder)) |
                            ((df_locations['segment'] == 9) & ("run2" in folder)) | ((df_locations['segment'] == 4) & ("run3" in folder)))
                            & (df_locations['northing'] < 405160.9000598852))
        df_locations = df_locations[((df_locations['segment'] >= 1) & (df_locations['segment'] <= 3)) | mask_first_rows_wp]"""

        #df_locations = filter_first_visits(df_locations, radius=2.0)
        #df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        #df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test.append(row, ignore_index=True)
            elif output_name == "vmd":
                if check_in_test_set(row['easting'], row['northing'], p):
                    df_test = df_test.append(row, ignore_index=True)
            else:
                if check_in_test_set(row['northing'], row['easting'], p):
                    df_test = df_test.append(row, ignore_index=True)
            """if "run2" in folder:
                df_test = df_test.append(row, ignore_index=True)"""
            """if ind == (len(all_folders) - 1):
                df_test = df_test.append(row, ignore_index=True)"""
            df_database = df_database.append(row, ignore_index=True)
            #df_database = pd.concat([df_database, pd.DataFrame([row])], ignore_index=True)
        database_tree = KDTree(df_database[['northing', 'easting']])
        database_trees.append(database_tree)
        #if ind == (len(all_folders) - 1):
        if not df_test.empty:
            test_tree = KDTree(df_test[['northing', 'easting']])
            test_trees.append(test_tree)
        ind += 1

    test_sets = []
    database_sets = []
    ind = 0
    lidar_i = 0
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        if output_name == 'blt' or output_name == 'arvc' or output_name == 'vmd':
            df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.csv' 
        else:
            df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin' 
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        total_rows = len(df_locations)
        for index, row in df_locations.iterrows():
            """if index >= (total_rows / 2):
                break""" #consider half of the vineyard rows (seeking BLT similarity)
            # entire business district is in the test set
            if output_name == "business":
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            elif output_name == "vmd":
                if check_in_test_set(row['easting'], row['northing'], p):
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            else:
                if check_in_test_set(row['northing'], row['easting'], p):
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            """if "run2" in row['file']:
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}"""
            """if ind == (len(all_folders) - 1):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}"""
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        #if ind == (len(all_folders) - 1):
        if len(test) > 0:
            test_sets.append(test)
        ind += 1

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=5) # ARVC, VMD or BLT -> r = 5, NCLT or Baseline/Refined -> r = 25
                # indices of the positive matches in database i of each query (key) in test set j
                #test_sets[j][key][i] = index[0].tolist()
                index_list = [int(k) for k in index[0]]  # asegura que son ints y no arrays
                intersection = list(set(index_list) & set(test_sets[j].keys()))
                test_sets[j][key][i] = intersection

    output_to_file(database_sets, base_path+"train_test_sets/"+PARAMS.protocol, "minkloc_" + output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, base_path+"train_test_sets/"+PARAMS.protocol, "minkloc_" + output_name + '_evaluation_query.pickle')


if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    # For Oxford
    """folders = []
    runs_folder = "oxford/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_20m/",
                                      "pointcloud_locations_20m.csv", P_DICT["oxford"], "oxford")

    # For University Sector
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    uni_index = range(10, 15)
    for index in uni_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["university"], "university")

    # For Residential Area
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    res_index = range(5, 10)
    for index in res_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["residential"], "residential")

    # For Business District
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    bus_index = range(5)
    for index in bus_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["business"], "business")"""
    
    # For ARVC
    """folders = []
    runs_folder = "arvc/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    for ind in range(1, len(all_folders) -1):
        folders.append(all_folders[ind])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/robot0/lidar/data/",
                                      "data.csv", P_DICT["arvc"], "arvc")"""
                                      
    # For BLT
    """folders = []
    runs_folder = "blt/"
    
    # Process the two different scenarios within the dataset
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder + "ktima/"))) #+ sorted(os.listdir(os.path.join(base_path, runs_folder + "riseholme/")))
    for folder in all_folders:
        if os.path.exists(os.path.join(base_path, runs_folder + "ktima/" + folder)):
            folders.append("ktima/"+folder)
    
    construct_query_and_database_sets(base_path, runs_folder, folders, "/robot0/lidar_3d/data/",
                                      "data.csv", P_DICT["blt-ktima"], "blt")"""
    """folders = []
    for folder in all_folders:
            if folder != "session0" and folder != "session1" and os.path.exists(os.path.join(base_path, runs_folder + "riseholme/" + folder)):
                folders.append("riseholme/"+folder)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/robot0/lidar/data/",
                                      "data.csv", P_DICT["blt-riseholme"], "blt")"""
    
    # For NCLT
    """folders = []
    runs_folder = "nclt/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    for idx in range(0, 3):
        folders.append(all_folders[idx])
    folders = ["2012-03-17", "2012-05-26", "2012-06-15", "2012-08-20", "2012-09-28", "2012-10-28"]
    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/velodyne_sync/",
                                      "data.csv", P_DICT["nclt"], "nclt")"""

    # For VMD (PIC4SeR)
    folders = []
    runs_folder = "vmd/"
    
    # Process the two different scenarios within the dataset
    #all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder + "pergola/"))) + sorted(os.listdir(os.path.join(base_path, runs_folder + "vineyard/")))
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder + "vineyard/")))
    for folder in all_folders:
        if "run2_03_p" in folder or "run3" not in folder:
            continue
        if os.path.exists(os.path.join(base_path, runs_folder + "pergola/" + folder)):
            folders.append("pergola/"+folder)
        else:
            folders.append("vineyard/"+folder)
    
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud/lidar3d_0/",
                                      "data.csv", P_DICT["vmd"], "vmd")
