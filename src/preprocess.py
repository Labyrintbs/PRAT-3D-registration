import os, glob, csv
import numpy as np
import open3d as o3d
from pathlib import Path
import copy


def parse_obj_vertices(obj_path: str):
    """Parse vertex lines 'v x y z [r g b]' from OBJ. Returns (N,3) or None."""
    verts = []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        verts.append([x, y, z])
                    except ValueError:
                        pass
    if not verts:
        return None
    return np.asarray(verts, dtype=np.float64)


def bbox_diag_from_points(V: np.ndarray) -> float:
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    return float(np.linalg.norm(mx - mn))

def voxel_down_count(V: np.ndarray, voxel_size: float) -> int:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return len(pcd_down.points)

def scan_obj_folder(obj_dir: str, modality: str, voxel_size: float, out_csv: str):
    obj_paths = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    if not obj_paths:
        raise FileNotFoundError(f"No .obj found in: {obj_dir}")

    rows = []
    for path in obj_paths:
        name = os.path.splitext(os.path.basename(path))[0]

        V = parse_obj_vertices(path)
        if V is None:
            print(f"[WARN] Cannot parse vertices: {name}")
            continue
        repr_type = "v_lines"

        n_raw = int(V.shape[0])
        diag = bbox_diag_from_points(V)

        row = {
            "modality": modality,
            "name": name,
            "repr": repr_type,
            "obj_path": path,
            "n_raw_points": n_raw,
            "bbox_diag": float(diag),
        }

        if voxel_size is not None:
            row["voxel_size"] = float(voxel_size)
            row["n_down_points"] = int(voxel_down_count(V, voxel_size))
            row["down_ratio"] = float(row["n_down_points"]) / float(n_raw) if n_raw > 0 else float("nan")

        rows.append(row)
        print(f"[OK] {modality:>6s} | {repr_type:>14s} | {name:25s} | N={n_raw:8d} | diag={diag:.3g}")

    # Write CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")
    return rows

def classify_obj(path: Path):
    has_vertex = False
    has_face = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):   # vertex
                has_vertex = True
            elif line.startswith("f "): # face
                has_face = True
                break  # no need to read further

    if has_face:
        return "mesh"
    elif has_vertex:
        return "point_cloud"
    else:
        return "unknown"

def scan_directory(obj_dir):
    results = []
    for p in sorted(Path(obj_dir).glob("*.obj")):
        kind = classify_obj(p)
        results.append((p.name, kind))
    return results

def read_as_pointcloud(path: str) -> o3d.geometry.PointCloud:
    ext = Path(path).suffix.lower()

    # 1) OBJ: parse vertices.
    if ext == ".obj":
        V = parse_obj_vertices(path)  
        if V is None or len(V) == 0:
            raise ValueError(f"OBJ has no parseable vertices: {path}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(V)
        return pcd

    # 2) Non-OBJ: normal point cloud read
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) > 0:
        return pcd

    # 3) Fallback: maybe it's a mesh-like file; take vertices 
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if len(mesh.vertices) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        return pcd

    raise ValueError(f"Cannot read as point cloud: {path}")

def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size):
    """
    Downsample + normal estimation + FPFH feature extraction.

    Returns:
        pcd_down: downsampled point cloud with normals
        fpfh:     o3d.pipelines.registration.Feature (shape: 33 x N)
    """
    # Voxel downsample 
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Remove statistical outliers
    # Considered as outlier if avg dist among neighbors >= global avg dist + std_ratio * std
    pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Compute FPFH feature
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd_down, fpfh

def prepare_clouds_scale_to_m(
    ref_path: str,
    mov_path: str,
    scale_to_m: float = 1e-3,
    voxel_size: float = 0.005,
    outlier_nb: int = 20,
    outlier_std: float = 2.0,
):
    # ref_raw = o3d.io.read_point_cloud(ref_path)
    # mov_raw = o3d.io.read_point_cloud(mov_path)
    ref_raw = read_as_pointcloud(ref_path)
    mov_raw = read_as_pointcloud(mov_path)
    
    ref_full = copy.deepcopy(ref_raw)
    mov_full = copy.deepcopy(mov_raw)

    # TRUE unit conversion: scale around origin
    ref_full.scale(scale_to_m, center=(0.0, 0.0, 0.0))
    #mov_full.scale(scale_to_m, center=(0.0, 0.0, 0.0))

    # support clouds in meters
    ref_support = ref_full.voxel_down_sample(voxel_size)
    mov_support = mov_full.voxel_down_sample(voxel_size)

    ref_support, _ = ref_support.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)
    mov_support, _ = mov_support.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)


    return ref_full, mov_full, ref_support, mov_support

