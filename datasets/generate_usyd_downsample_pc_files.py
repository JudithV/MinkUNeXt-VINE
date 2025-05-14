import os
import pandas as pd
import numpy as np
import open3d as o3d

# Ruta al dataset original y carpeta destino
input_folder = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/USyd/weeks"
output_folder = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/USyd_downsample/weeks"

# Crear la carpeta destino si no existe
os.makedirs(output_folder, exist_ok=True)

# Parámetro para reducción de resolución
voxel_size = 0.1  # Cambia según la densidad que desees

def read_bin(file_path):
    """Lee un archivo .bin con formato de nubes de puntos."""
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # X, Y, Z, intensidad

def write_bin(file_path, pcd):
    """Escribe un archivo .bin con formato de nubes de puntos."""
    #points.astype(np.float32).tofile(file_path)
    """Para guardar la pointcloud como .pcd"""
    pcd.to_csv(file_path, index=False, float_format="%.6f")

def to_spherical(points, dataset_name):
    spherical_points = []
    for point in points:
        if (np.abs(point[:3]) < 1e-4).all():
            continue

        r = np.linalg.norm(point[:3])

        # Theta is calculated as an angle measured from the y-axis towards the x-axis
        # Shifted to range (0, 360)
        theta = np.arctan2(point[1], point[0]) * 180 / np.pi
        if theta < 0:
            theta += 360

        if dataset_name == "USyd":
            # VLP-16 has 2 deg VRes and (+15, -15 VFoV).
            # Phi calculated from the vertical axis, so (75, 105)
            # Shifted to (0, 30)
            phi = (np.arccos(point[2] / r) * 180 / np.pi) - 75

        elif dataset_name in ['IntensityOxford', 'Oxford']:
            # Oxford scans are built from a 2D scanner.
            # Phi calculated from the vertical axis, so (0, 180)
            phi = np.arccos(point[2] / r) * 180 / np.pi

        elif dataset_name == 'KITTI':
            # HDL-64 has 0.4 deg VRes and (+2, -24.8 VFoV).
            # Phi calculated from the vertical axis, so (88, 114.8)
            # Shifted to (0, 26.8)
            phi = (np.arccos(point[2] / r) * 180 / np.pi) - 88

        if point.shape[-1] == 4:
            spherical_points.append([r, theta, phi, point[3]])
        else:
            spherical_points.append([r, theta, phi])

    return np.array(spherical_points)


def reduce_resolution(points, voxel_size):
    """Reduce la resolución de una nube de puntos usando Open3D."""
    # Crear una nube de puntos Open3D
    #cloud = o3d.geometry.PointCloud()
    #cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # Solo X, Y, Z
    #pc = points[np.linalg.norm(points[:, :3], axis=1) < 100]
    #mask = points[:, 2] < -0.2
    #indices_above_ground = np.where(mask ==  False)[0]
    #pc = points[indices_above_ground]
    intensity = points[:, 3]
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    #pcd = pcd.voxel_down_sample(voxel_size=0.1)
    #pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)  # 20, 2.0
    #pc = np.asarray(pcd.points)
    pc = points[:, :3]
    min_radius = 2.0
    max_radius = 60.0
    [x, y, z] = pc[:, 0], pc[:, 1], pc[:, 2]
    r2 = x ** 2 + y ** 2
    idx = np.where(r2 < max_radius ** 2) and np.where(r2 > min_radius ** 2)
    #points = np.hstack((pc[idx], intensity[idx]))
    spherical_points = to_spherical(pc[idx], "USyd")
    #points = np.hstack((spherical_points, intensity[idx]))
    #print(pc[idx].shape)

    return spherical_points, intensity[idx]
    
    #np.random.shuffle(spherical_points)
    end = time.time()
    #print("Tiempo de procesado de pcd: ", end-start)
    #return spherical_points[:, :3], spherical_points[:, 3]
    #return spherical_points, intensity[idx]

    # Aplicar downsampling con voxel grid
    reduced_cloud = cloud.voxel_down_sample(voxel_size)

    # Convertir de vuelta a numpy
    points, intensity = np.asarray(reduced_cloud.points)
    if points.size > 0:
        df = pd.DataFrame(points, columns=["x", "y", "z"])
        df["intensity"] = intensity


        # Añadir intensidad de vuelta
                                                                                        
        # Guardar nube de puntos reducida
        write_bin(output_path, df)
        print(f"Procesado: {file_name}")

if __name__ == '__main__':
    for folder in os.listdir(input_folder):
        folder_to_process = os.path.join(input_folder+"/"+folder+"/pointclouds_with_locations_5m")
        for file_name in os.listdir(folder_to_process):
            if file_name.endswith(".bin"):
                input_path = os.path.join(input_folder+"/"+folder+"/pointclouds_with_locations_5m", file_name)
                output_path = os.path.join(output_folder+"/"+folder+"/pointclouds_with_locations_5m", file_name.replace(".bin", ".csv"))

                # Leer nube de puntos
                points = read_bin(input_path)

                # Reducir resolución
                points, intensity = reduce_resolution(points, voxel_size)
                if points.size > 0:
                    df = pd.DataFrame(points, columns=["x", "y", "z"])
                    df["intensity"] = intensity

                    # Guardar nube de puntos reducida
                    write_bin(output_path, df)
                    file_processed = file_name.replace(".bin",".csv")
                    print(f"Procesado: {file_processed}")
