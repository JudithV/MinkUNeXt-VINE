import numpy as np
import os
import pickle
import torch
import MinkowskiEngine as ME
import pandas as pd
import ast
from sklearn.cluster import MeanShift, AffinityPropagation, DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from model.minkunext import model
from config import PARAMS
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
import random
import gc
import tqdm
import csv
import re

def string2array(desc):
    data = desc.strip("[]")
    numbers = re.split(r'\s+', data)
    numbers = [num for num in numbers if num.strip()]
    array = np.array(numbers, dtype=float)
    return array

def cl_mean_shift(X):

    mean_shift = MeanShift(bandwidth=0.1)
    mean_shift.fit(X)

    centers = mean_shift.cluster_centers_
    labels = mean_shift.labels_

    return centers, labels


def cl_debscan(X):
    dbscan = DBSCAN(eps=0.7, min_samples=2)  # eps -> distancia máxima entre descriptores
                                             # para ser considerados del mismo cluster
    dbscan.fit(X)
    labels = dbscan.labels_
    return labels


def cl_kmeans(X, n_clusters=40):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100)
    kmeans.fit(X)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centers, labels


def cl_ahc(X):  # Agglomerative Hierarchical Clustering

    agglomerative = AgglomerativeClustering(n_clusters=5)

    labels = agglomerative.fit_predict(X)

    return labels


def cl_gmm(X):  # Gaussian Mixture Models (GMM)
    # Crear el modelo GMM
    gmm = GaussianMixture(n_components=5)

    # Ajustar el modelo a los datos
    gmm.fit(X)

    # Obtener las probabilidades de pertenencia a cada cluster para cada punto
    probabilidades = gmm.predict_proba(X)
    labels = []
    for prob in probabilidades:
        ind = np.where(prob == 1)[0][0]
        labels.append(ind)
    return labels

def cl_ap(X): # Affinity Propagation
    af = AffinityPropagation().fit(X)

    # Obtener los centroides de los clusters y las etiquetas de cluster para cada punto
    centroides = af.cluster_centers_
    etiquetas = af.labels_
    return centroides, etiquetas


def cl_sc(X):
    sc = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', random_state=42) # affinity='nearest_neighbors' affinity='sigmoid'

    # Ajustar el modelo a los datos
    sc.fit(X)

    # Obtener las etiquetas de cluster para cada punto
    etiquetas = sc.labels_
    return etiquetas


def plot_clusters(mapa, cluster):

    plt.figure(figsize=(10, 10))
    xmap = mapa[:, 0]
    ymap = mapa[:, 1]

    color = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black', 'orange', 'brown', 'pink', 'purple',
             'gold', 'lime', 'teal', 'lavender', 'maroon', 'turquoise', 'olive', 'coral']

    colores_hex = ['#FF5733', '#33FF57', '#5733FF', '#FF33C7', '#33FFAA', '#57FF33', '#AA33FF',
                   '#FF3357', '#33AAFF', '#33FF33', '#FFAA33', '#57FFAA', '#AAFF33', '#FF33AA',
                   '#33AA57', '#5733AA', '#AA5733', '#57AA33', '#AA3357', '#3357AA', '#7FFFC7',
                   '#7FC7FF', '#FF7FC7', '#FFC77F', '#C77FFF', '#FFC733', '#FF33C7', '#33FF7F',
                   '#7FFF33', '#337FFF', '#7FFF33', '#33C7FF', '#C7FF33', '#7FFFC7', '#7FC7FF',
                   '#FF7FC7', '#FFC77F', '#C77FFF', '#FFC733', '#FF33C7', '#33FF7F']

    marker = [
        '.',  # punto
        ',',  # píxel
        'o',  # círculo
        'v',  # triángulo hacia abajo
        '^',  # triángulo hacia arriba
        '<',  # triángulo hacia la izquierda
        '>',  # triángulo hacia la derecha
        '1',  # triángulo hacia abajo (extremo)
        '2',  # triángulo hacia arriba (extremo)
        '3',  # triángulo hacia la izquierda (extremo)
        '4',  # triángulo hacia la derecha (extremo)
        's',  # cuadrado
        'p',  # pentágono
        '*',  # estrella
        'h',  # hexágono 1
        'H',  # hexágono 2
        '+',  # cruz
        'x',  # cruz diagonal
        'D',  # diamante
        'd',  # diamante pequeño
        '|',  # línea vertical
        '_',  # línea horizontal
    ]
    color = color+colores_hex

    # def rgb_to_hex(rgb):
    #     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    # color = [rgb_to_hex((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) for _ in range(1793)]

    for i, pos in enumerate(mapa):
        plt.plot(pos[1], pos[0], marker='o', markersize=5, linestyle='None', color=color[cluster[i]])
        # plt.text(pos[0], pos[1], str(cluster[i]), fontsize=12, ha='right', va='bottom')

    """plt.xlabel('X position (m)', fontsize=15)
    plt.ylabel('Y position (m)', fontsize=15)
    # plt.xticks(ticks=(-80,-60,-40,-20,0), fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # Eliminar ejes y fondo
    plt.set_frame_on(False)
    plt.set_xticks([])  # Elimina las marcas del eje X
    plt.set_yticks([])  # Elimina las marcas del eje Y
    plt.set_xticklabels([])  # Elimina etiquetas del eje X
    plt.set_yticklabels([])  # Elimina etiquetas del eje Y"""
    # plt.xlim(-80, 10)
    #plt.axis('equal')
    # plt.gca().set_aspect('equal') #, adjustable='box')
    # plt.savefig("clustering_arvc.png")
    # plt.savefig('clustering_arvc.jpg', format='jpg', dpi=300)
    plt.savefig('descriptores-kitti_0051.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    """TODO: Plot clusters in map
    lats, lons = utm.to_latlon(xmap, ymap)
    gmap = CustomGoogleMapPlotter(lats[0], lons[0], initial_zoom, map_type='satellite')

    gmap.plot_trajectories(lats, lons, directory=os.path.join(map_dir, sequence_folder + '.html'))"""

def load_descriptors(file_training, file_test, vineyard, test_mode=False):
    p = os.path.join(PARAMS.dataset_folder, file_training)
    with open(p, 'rb') as f:
        database_sets = pickle.load(f)

    p = os.path.join(PARAMS.dataset_folder, file_test)
    with open(p, 'rb') as f:
        query_sets = pickle.load(f)
    poses = []
    descriptors = []

    if torch.cuda.is_available():
        device = PARAMS.cuda_device
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    model.to(device)
    model.load_state_dict(torch.load("/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/weights/MinkUNeXt_usyd_20250218_0857_final.pth"))
    #model.cpu()
    i=0
    route = ""
    route_selected = False
    pc_loader = PNVPointCloudLoader(device)
    if not test_mode:
        for _, idx in enumerate(database_sets):
            scan_file_path = database_sets[idx].rel_scan_filepath
            #if vineyard in scan_file_path:
            if not route_selected:
                path_split = scan_file_path.split("/")
                route = path_split[2]
                route = route.split("_")[0]
                route_selected = True
            if route in scan_file_path:
                #CSV
                """df = pd.read_csv(scan_file_path)
                points = df[["x", "y", "z"]].to_numpy()
                intensity = df["intensity"].to_numpy()"""
                dtype=np.float32
                #dtype=np.float64
                #dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                pc = np.fromfile(scan_file_path, dtype=dtype).reshape([-1, 4])
                #pc = np.fromfile(scan_file_path, dtype=dtype)
                points = pc[:, :3]
                intensity = pc[:, 3]
                #points = np.stack([pc['x'] * 0.005 - 100, pc['y'] * 0.005 - 100, -(pc['z'] * 0.005 - 100)], axis=-1)

                #intensity =  pc['intensity']
                points = torch.tensor(points, dtype=torch.float)
                intensity = torch.tensor(intensity, dtype=torch.float)
                intensity = intensity.reshape([-1, 1])

                with torch.no_grad():
                    # Sparse quantization en GPU
                    coords = ME.utils.sparse_quantize(coordinates=points.contiguous(),  # ME necesita CPU
                            features=intensity, quantization_size=PARAMS.quantization_size)

                    # Convertir a tensor y pasar a GPU
                    bcoords = ME.utils.batched_coordinates([coords[0]])
                    bfeats = coords[1]

                    batch = {'coords': bcoords.to(device), 'features': bfeats.to(device)}
                    d = model(batch)
                    #print(d.keys())
                    #print(d['global'].cpu().detach().numpy()[0])
                    descriptors.append(d['global'].cpu().detach().numpy()[0])
                    #print(i)
                    i += 1
                    #print(query_sets[idx].position)
                    poses.append([database_sets[idx].position[1], database_sets[idx].position[0]])
                    del batch, bcoords, bfeats, coords, d
                    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        for _, idx in enumerate(query_sets):
            scan_file_path = query_sets[idx].rel_scan_filepath
            #if vineyard in scan_file_path and route in scan_file_path:
            if route in scan_file_path:
                #CSV
                """df = pd.read_csv(scan_file_path)
                points = df[["x", "y", "z"]].to_numpy()
                intensity = df["intensity"].to_numpy()"""
                dtype=np.float32
                #dtype=np.float64
                #dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                pc = np.fromfile(scan_file_path, dtype=dtype).reshape([-1, 4])
                #pc = np.fromfile(scan_file_path, dtype=dtype)
                points = pc[:, :3]
                intensity = pc[:, 3]
                #points = np.stack([pc['x'] * 0.005 - 100, pc['y'] * 0.005 - 100, (pc['z'] * 0.005 - 100)], axis=-1)

                #intensity =  pc['intensity']

                points = torch.tensor(points, dtype=torch.float)
                intensity = torch.tensor(intensity, dtype=torch.float)
                intensity = intensity.reshape([-1, 1])

                with torch.no_grad():
                    # Sparse quantization en GPU
                    coords = ME.utils.sparse_quantize(coordinates=points.contiguous(),  # ME necesita CPU
                            features=intensity, quantization_size=PARAMS.quantization_size)

                    # Convertir a tensor y pasar a GPU
                    bcoords = ME.utils.batched_coordinates([coords[0]])
                    bfeats = coords[1]

                    batch = {'coords': bcoords.to(device), 'features': bfeats.to(device)}
                    d = model(batch)
                    #print(d.keys())
                    #print(d['global'].cpu().detach().numpy()[0])
                
                    descriptors.append(d['global'].cpu().detach().numpy()[0])
                    #print(i)
                    i += 1
                    #print(query_sets[idx].position)
                    poses.append([query_sets[idx].position[1], query_sets[idx].position[0]])
                    del batch, bcoords, bfeats, coords, d
                    gc.collect()
                    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
    else:  
        path = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/KittiDataset/"
        poses_path = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/KittiDataset/sequences/00/poses.txt"
        poses_file = open(poses_path, "r")
        for set in tqdm.tqdm(database_sets, desc='Computing database embeddings'):
            for i, elem_ndx in enumerate(set):
                pc_file_path = os.path.join(path, set[elem_ndx]["query_velo"])
                dtype=np.float32
                #dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                pc = np.fromfile(pc_file_path, dtype=dtype).reshape([-1, 4])
                #pc = np.fromfile(scan_file_path, dtype=dtype)
                points = pc[:, :3]
                intensity = pc[:, 3]
                points = torch.tensor(points, dtype=torch.float)
                intensity = torch.tensor(intensity, dtype=torch.float)
                intensity = intensity.reshape([-1, 1])
                
                with torch.no_grad():
                        # Sparse quantization en GPU
                        coords = ME.utils.sparse_quantize(coordinates=points.contiguous(),  # ME necesita CPU
                                features=intensity, quantization_size=PARAMS.quantization_size)

                        # Convertir a tensor y pasar a GPU
                        bcoords = ME.utils.batched_coordinates([coords[0]])
                        bfeats = coords[1]

                        batch = {'coords': bcoords.to(device), 'features': bfeats.to(device)}
                        d = model(batch)
                        #print(d.keys())
                        #print(d['global'].cpu().detach().numpy()[0])
                    
                        descriptors.append(d['global'].cpu().detach().numpy()[0])
                        #print(i)
                        i += 1
                        #print(query_sets[idx].position)
                        # Get coordinates from poses.txt
                        linea_deseada = int(pc_file_path.split("/")[-1].split(".")[0])
                        print(linea_deseada)
                        m_transform = poses_file.readlines()[linea_deseada + 1]
                        x = float(m_transform.split(" ")[3])
                        y = float(m_transform.split(" ")[7])
                        
                        poses.append([x, y])
                        poses_file = open(poses_path, "r")
                        
                        del batch, bcoords, bfeats, coords, d
                        gc.collect()
                        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        for set in tqdm.tqdm(query_sets, desc='Computing query embeddings'):
            for i, elem_ndx in enumerate(set):
                pc_file_path = os.path.join(path, set[elem_ndx]["query_velo"])
                dtype=np.float32
                #dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                pc = np.fromfile(pc_file_path, dtype=dtype).reshape([-1, 4])
                #pc = np.fromfile(scan_file_path, dtype=dtype)
                points = pc[:, :3]
                intensity = pc[:, 3]
                points = torch.tensor(points, dtype=torch.float)
                intensity = torch.tensor(intensity, dtype=torch.float)
                intensity = intensity.reshape([-1, 1])
                
                with torch.no_grad():
                        # Sparse quantization en GPU
                        coords = ME.utils.sparse_quantize(coordinates=points.contiguous(),  # ME necesita CPU
                                features=intensity, quantization_size=PARAMS.quantization_size)

                        # Convertir a tensor y pasar a GPU
                        bcoords = ME.utils.batched_coordinates([coords[0]])
                        bfeats = coords[1]

                        batch = {'coords': bcoords.to(device), 'features': bfeats.to(device)}
                        d = model(batch)
                        #print(d.keys())
                        #print(d['global'].cpu().detach().numpy()[0])
                    
                        descriptors.append(d['global'].cpu().detach().numpy()[0])
                        #print(i)
                        i += 1
                        #print(query_sets[idx].position)
                        # Get coordinates from poses.txt
                        linea_deseada = int(pc_file_path.split("/")[-1].split(".")[0])
                        m_transform = poses_file.readlines()[linea_deseada + 1]
                        x = float(m_transform.split(" ")[3])
                        y = float(m_transform.split(" ")[7])
                        
                        poses.append([x, y])
                        poses_file = open(poses_path, "r")
                        
                        del batch, bcoords, bfeats, coords, d
                        gc.collect()
                        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return np.array(poses), np.array(descriptors)


if __name__ == '__main__':
    poses, descriptors = load_descriptors('training_queries_kitti.pickle', 'test_queries_kitti.pickle', 'ktima')
    #print(descriptors[0])
    N = 6
    m = 6
    clusters = N * m
    # Oxford
    clusters = 7
    cent, etiquetas = cl_kmeans(descriptors, clusters)

    # cent, etiquetas = cl_mean_shift(descriptors)
    # cent, etiquetas = cl_debscan(descriptors)
    # cent, etiquetas = cl_sc(descriptors)
    # cent, etiquetas = cl_ap(descriptors)
    # cent, etiquetas = cl_gmm(descriptors)
    # cent, etiquetas = cl_ahc(descriptors)

    plot_clusters(poses, etiquetas)
