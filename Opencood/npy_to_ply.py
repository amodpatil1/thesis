import os
import numpy as np
import open3d as o3d

npy_dir = "/OpenCOOD/final"          # directory containing .npy files
ply_dir = "/OpenCOOD/final/ply"      # output directory

os.makedirs(ply_dir, exist_ok=True)

for fname in sorted(os.listdir(npy_dir)):
    if not fname.endswith(".npy"):
        continue

    npy_path = os.path.join(npy_dir, fname)
    ply_path = os.path.join(
        ply_dir, fname.replace(".npy", ".ply")
    )

    pts = np.load(npy_path)          # (N,3) or (N,4)
    xyz = pts[:, :3].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)

    print(f"Saved: {ply_path}")
