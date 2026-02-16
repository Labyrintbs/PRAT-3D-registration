import numpy as np
import json
import open3d as o3d
import time
import os


def load_T_from_json(json_path: str, key: str) -> np.ndarray:
    with open(json_path, "r") as f:
        d = json.load(f)
    T = np.asarray(d[key], dtype=np.float64)
    assert T.shape == (4, 4)
    return T



def load_landmarks(txt_path):
    """
    Load landmark coordinates exported from CloudCompare.
    Supports both CSV (x,y,z) and space-separated formats.
    Returns: (N,3) numpy array
    """
    pts = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "," in line:
                vals = line.split(",")
            else:
                vals = line.split()

            if len(vals) < 3:
                continue

            try:
                x = float(vals[0])
                y = float(vals[1])
                z = float(vals[2])
                pts.append([x, y, z])
            except ValueError:
                continue

    return np.asarray(pts, dtype=np.float64)

def compute_rigid_transform(P, Q):
    """
    Given 2 list of points:
      P: (N, 3) moving object points (mov_landmarks)
      Q: (N, 3) reference object points (ref_landmarks)
    Solve Rigid transformation T so that Q ≈ R @ P + t
    Return:
     4x4 transform matrix T
    """
    assert P.shape == Q.shape
    # Centroid
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    # Decentroid
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Covariance
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # translation
    t = centroid_Q - R @ centroid_P

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T




def global_registration_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size: float):
    """
    RANSAC-based global registration on FPFH feature matches.
    """
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=True, # ensure src/tar NN corespondance
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), # rigid transform
        ransac_n=4, # sample point pairs to solve T 
        checkers=[ # pre filter
            # Enforce similar edge length ratios to reject degenerate matches
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            # Reject matches with too large geometric distance
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000,  
            confidence=0.999
        )
    )
    return result


def refine_registration_icp(src_full, tgt_full, init_T, voxel_size, use_point_to_plane=False):
    """
    ICP refinement (point-to-plane is usually better if normals are reliable).
    """
    # Set a tighter threshold than RANSAC
    max_corr_dist = voxel_size * 1.0

    if use_point_to_plane:
        # Normals are required for point-to-plane ICP (target normals are used)
        tgt_full.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        src_full.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result_icp = o3d.pipelines.registration.registration_icp(
        src_full, tgt_full,
        max_corr_dist,
        init_T,
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    )
    return result_icp
    

def neighborhood_stats(pcd_down, radius, max_check=5000):
    pts = np.asarray(pcd_down.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_down)
    n = pts.shape[0]
    idxs = np.random.choice(n, size=min(n, max_check), replace=False)

    counts = []
    t0 = time.time()
    for i in idxs:
        _, nn, _ = kdtree.search_radius_vector_3d(pts[i], radius)
        counts.append(len(nn))
    t = time.time() - t0
    counts = np.array(counts)
    print(f"[radius={radius}] checked {len(idxs)} points in {t:.2f}s")
    print("  nn count: min / median / p90 / max =",
          counts.min(), np.median(counts), np.percentile(counts, 90), counts.max())
    return counts


def save_aligned_cloud(
    pcd: o3d.geometry.PointCloud,
    base_dir: str,
    K: int,
    seed: int,
    method: str,
):
    """
    Save aligned point cloud to:
    base_dir/K_<K>/seed_<seed>/mov_aligned_<method>.ply
    """
    save_dir = os.path.join(base_dir, f"K_{K}", f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"mov_aligned_{method}.ply"
    path = os.path.join(save_dir, filename)

    o3d.io.write_point_cloud(path, pcd)
    print(f"[Saved] {path}")

    return path

def now():
    return time.perf_counter()

def diag(pcd: o3d.geometry.PointCloud) -> float:
    return float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))

def select_keypoints_random(pcd: o3d.geometry.PointCloud, K: int, seed: int = 0):
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if K >= n:
        idx = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=K, replace=False).astype(np.int64)

    idx.sort()  # align with o3d order
    kp = pcd.select_by_index(idx.tolist())
    return kp, idx


def save_out_json(out: dict, base_dir: str, K: int, seed: int):
    """
    Save the full `out` dict to:
    base_dir/K_<K>/seed_<seed>/out.json
    """
    save_dir = os.path.join(base_dir, f"K_{K}", f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, "out.json")

    # numpy -> python float/int 
    def convert(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        return o

    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=convert)

    print(f"[Saved] {path}")
    return path



def slice_feature_by_index(feat: o3d.pipelines.registration.Feature, idx: np.ndarray):
    """
    feat.data shape: (dim, N). Returns a Feature with shape (dim, K).
    """
    f = o3d.pipelines.registration.Feature()
    f.data = feat.data[:, idx]
    return f


def assert_kp_alignment(support_pcd, kp_pcd, idx, name, tol=1e-12, n_check=20):
    sup = np.asarray(support_pcd.points)
    kp  = np.asarray(kp_pcd.points)
    idx = np.asarray(idx, dtype=np.int64)

    assert kp.shape[0] == idx.shape[0], f"{name}: kp_n != idx_n"
    m = min(n_check, kp.shape[0])
    for j in range(m):
        d = np.linalg.norm(kp[j] - sup[idx[j]])
        assert d < tol, f"{name}: order mismatch at j={j}, ||kp-sup[idx]||={d}"