import copy
import time
import numpy as np
import os
import open3d as o3d
import open3d.core as o3c
import pandas as pd
import cv2
#import cupy as cp
import matplotlib.pyplot as plt
from skimage import exposure
import torch
from config import PARAMS 
from datasets.spherical_coords import SphericalCoords

from datasets.base_datasets import PointCloudLoader

class PNVPointCloudLoader(PointCloudLoader):
    def global_normalize(self, pcd):
        """
        Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
        """
        pcd = copy.deepcopy(pcd)
        points = np.asarray(pcd.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)

        x = x - x_mean
        y = y - y_mean
        z = z - z_mean

        x = x / 50.0
        y = y / 50.0
        z = z / 50.0

        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z

        return points
    
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None
    
    """def mostrar_img(self,x,y,num_points,reflectivity):
        # Definir las dimensiones de la imagen
        img_width, img_height = 500, 500

        # Crear una imagen vacía con fondo negro
        image = np.zeros((img_height, img_width))

        # Escalar y trasladar los puntos al rango de la imagen
        x_img = ((x - x.min()) / (x.max() - x.min()) * (img_width - 1)).astype(int)
        y_img = ((y - y.min()) / (y.max() - y.min()) * (img_height - 1)).astype(int)

        # Asignar los valores de reflectividad a la imagen
        for i in range(num_points):
            image[y_img[i], x_img[i]] = reflectivity[i]

        # Mostrar la imagen usando Matplotlib
        plt.imshow(image, cmap='gray', origin='lower')
        plt.colorbar(label='Reflectividad')
        plt.title('Nube de Puntos Proyectada a Imagen')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    def project2image(self,Tlc,dy,dx,fx,fy, points, reflectivity):
        points_camera = []
        cx = dx/2
        cy = dy/2
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]], np.float32)
        dist_coeffs = np.zeros((5, 1), np.float32)
        tvec = np.zeros((3, 1), np.float32)
        rvec = np.zeros((3, 1), np.float32)
        Tcl = np.linalg.inv(Tlc)

        for i in range(len(points)):
            p = np.concatenate((points[i,:],np.array([1])))
            p=np.matmul(Tcl,p.T)
            points_camera.append(p)

        points_camera=np.asarray(points_camera)
        points_camera2=points_camera[:,0:3]
        prueba=np.copy(points_camera2)
        points_2d, _ = cv2.projectPoints(prueba,rvec, tvec,camera_matrix,dist_coeffs)
        image_size = (dx, dy,3)
        image = np.zeros((image_size[1], image_size[0],image_size[2]))
        for point in points_2d.squeeze().astype(int):
            cv2.circle(image, tuple(point), 0, (255, 255, 0), -1)
        
        # Escalar y trasladar los puntos al rango de la imagen
        [x, y] = points_camera[:, 0], points_camera[:, 1]
        x_img = ((x - x.min()) / (x.max() - x.min()) * (image_size[0] - 1)).astype(int)
        y_img = ((y - y.min()) / (y.max() - y.min()) * (image_size[1] - 1)).astype(int)
        for i in range(points_camera.size):
            image[y_img[i], x_img[i]] = reflectivity[i]
        return image"""

    def correct_intensity(self, points, intensity):
        """ This function corrects the intensity value, removing the bias of the distance between point and sensor. """
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        cos_phi = points[:, 2] / r
        intensity_corr = intensity / (r**2 * cos_phi)
        return intensity_corr

    def dsor_filter(self, points, base_k=10, scaling_factor=0.5, std_ratio=2.0):
        """
        DSOR: Dynamic Statistical Outlier Removal.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        distances = []
        neighbors_list = []

        for i, point in enumerate(points):
            dist = np.linalg.norm(point)
            k = int(base_k + scaling_factor * dist)  # k dinámico según distancia
            k = min(k, len(points) - 1)  # para evitar exceder el tamaño de la nube

            [_, idxs, dists] = kdtree.search_knn_vector_3d(point, k)
            if len(dists) > 0:
                avg_dist = np.mean(np.sqrt(dists[1:]))  # quitamos el primero (distancia a sí mismo)
                distances.append(avg_dist)
                neighbors_list.append(i)

        # Calcular media y desviación globales de distancias promedio
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dev = np.std(distances)

        threshold = mean_dist + std_ratio * std_dev

        # Elegimos solo los puntos cuya distancia media a vecinos es razonable
        inlier_indices = [neighbors_list[i] for i in range(len(distances)) if distances[i] < threshold]
        filtered_pcd = pcd.select_by_index(inlier_indices)

        return filtered_pcd, inlier_indices

    def read_pc(self, device, file_pathname: str) -> np.ndarray:
        if PARAMS.format_point_cloud == 'csv':
            # SE RECIBE LA PC EN CSV
            if not PARAMS.use_downsampled:
                df = pd.read_csv(file_pathname)
            else:
                if PARAMS.protocol == 'usyd' and PARAMS.spherical_coords:
                    file_pathname = file_pathname.replace("USyd/","USyd_downsample/").replace(".bin", ".csv")
                elif PARAMS.protocol == 'usyd' and not PARAMS.spherical_coords:
                    file_pathname = file_pathname.replace("USyd/","USyd_downsample/").replace(".bin", "_NO_SP.csv")
                else:
                    file_pathname = file_pathname.replace(PARAMS.protocol, PARAMS.protocol+"_downsample")
                df = pd.read_csv(file_pathname)
            df.columns = df.columns.str.lower().str.strip()
            if PARAMS.protocol == 'blt' and PARAMS.use_2D:
                points = np.column_stack([df['x'], df['y'], np.zeros(len(df))])
            else:
                df = df.query('x != 0 and y != 0 and z != 0 and intensity != 0')
                points = df[["x", "y", "z"]].to_numpy() 
            if PARAMS.normalize:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                points = self.global_normalize(pcd)
            #print(len(points)) # Number of points in cloud
            if PARAMS.protocol == 'blt' and PARAMS.use_2D:
                intensity = np.zeros(len(df))
            else:
                intensity = df["intensity"].to_numpy()
            if PARAMS.correct_intensity:
                intensity = self.correct_intensity(points, intensity)
            if not PARAMS.use_downsampled:
                if PARAMS.correct_intensity:
                    intensity = self.correct_intensity(points, intensity)
                if PARAMS.protocol == 'vmd':
                    """pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    #pcd = pcd.voxel_down_sample(voxel_size=0.01)
                    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.01)
                    pcd, ind = self.dror_filter(pcd.points)
                    points = np.asarray(pcd.points)"""

                    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                    r2 = x ** 2 + y ** 2
                    idx = np.where((r2 < PARAMS.max_distance ** 2))
                    points = points[idx]
                    intensity = intensity[idx]
                if PARAMS.protocol == 'arvc': # Eliminar ruido presente en estos datos mediante radio min y max
                    [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                    r2 = x ** 2 + y ** 2
                    idx = np.where(r2 < PARAMS.max_distance ** 2) and np.where(r2 > PARAMS.min_distance ** 2)
                    points = points[idx]
                    intensity = intensity[idx]
                if PARAMS.protocol != 'usyd' and PARAMS.spherical_coords:
                    points = np.concatenate((points, intensity[:, np.newaxis]), axis=1)
                    points = SphericalCoords.to_spherical(points, PARAMS.protocol)
                    intensity = points[:, 3]
                    points = points[:, :3]
                    #print(min(spherical_points[:, 2]))
                if PARAMS.equalize_intensity:
                    intensity = exposure.equalize_hist(intensity)
            return points, intensity
        
        else:
            # SE RECIBE LA PC COMO BINARIO:
            # Load point cloud, does not apply any transform
            # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
            start = time.time()
            file_path = os.path.join(file_pathname)
            if PARAMS.protocol == 'nclt':
                dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                data = np.fromfile(file_path, dtype=dtype)
                points = np.stack([data['x'] * 0.005 - 100, data['y'] * 0.005 - 100, (data['z'] * 0.005 - 100)], axis=-1)

                intensity =  data['intensity']
                # Parámetro: umbral de altura para eliminar el suelo
                ground_threshold = 0.5  # metros
                [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                r2 = x ** 2 + y ** 2
                idx = (r2 < 80**2) & (z < -ground_threshold) # Procesado según paper: DiSCO
                points = points[idx]
                intensity = intensity[idx]
                pc = np.column_stack((points, intensity))
            elif PARAMS.protocol == 'intensityOxford':
                pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 4])
            else:
                pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4]) #USYD, NCLT: float32, OXFORD: float64
            #pc = pc[np.linalg.norm(pc[:, :3], axis=1) < PARAMS.max_distance]
            np.random.shuffle(pc)
            if pc.shape[0] == 0:
                pc = np.zeros((1, 4), dtype=np.float32)
            else:
                if PARAMS.correct_intensity:
                    intensity = self.correct_intensity(pc[:, :3], pc[:, 3])
                    pc = np.concatenate((pc[:, :3], intensity[:, np.newaxis]), axis=1)
                if PARAMS.spherical_coords:
                    pc = SphericalCoords.to_spherical(pc, PARAMS.protocol)
                if PARAMS.equalize_intensity:
                    pc[:, 3] = exposure.equalize_hist(pc[:, 3])
            return pc[:, :3], pc[:, 3]

    def dror_filter(self, points, base_radius=0.1, scaling_factor=0.03, min_neighbors=3):
        """
        DROR: Dynamic Radius Outlier Removal.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        inlier_indices = []

        for i, point in enumerate(points):
            dist = np.linalg.norm(point)
            dynamic_radius = base_radius + scaling_factor * dist

            [_, idxs, _] = kdtree.search_radius_vector_3d(point, dynamic_radius)

            if len(idxs) > min_neighbors:
                inlier_indices.append(i)

        filtered_pcd = pcd.select_by_index(inlier_indices)
        return filtered_pcd, inlier_indices
    
    def read_reflec(self, file_pathname: str):
        # Reads the point cloud reflectivity value
        file_path = os.path.join(file_pathname)
        scan_data = pd.read_csv(file_path, sep=',')
        reflec = np.asarray(scan_data["reflectivity"].values)
        #reflec_norm = (reflec - reflec.min()) / (reflec.max() - reflec.min())
        #reflec_norm = reflec / 255.0
        #reflec_norm = (reflec + 150) / 255.0
        #reflec_norm = exposure.equalize_hist(reflec)
        reflec_norm = []
        for r in reflec:
            if r == 0:
                reflec_norm.append(0)
            else:
                reflec_norm.append(1)
        reflec_norm = np.array(reflec_norm)
        return reflec_norm
    
    def read_intensity(self, file_pathname: str):
        # Reads the point cloud reflectivity value
        file_path = os.path.join(file_pathname)
        scan_data = pd.read_csv(file_path, sep=',')
        intensity = np.asarray(scan_data["intensity"].values)
        #intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        #intensity_norm = intensity / 255.0
        #intensity_norm = (intensity + 150) / 255.0
        #intensity_norm = exposure.equalize_hist(intensity)
        intensity_norm = []
        for i in intensity:
            if i == 0:
                intensity_norm.append(0)
            else:
                intensity_norm.append(1)
        intensity_norm = np.array(intensity_norm)
        return intensity_norm
