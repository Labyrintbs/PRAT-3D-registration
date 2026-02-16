import torch
import torch.nn.functional as F
import sys, os
from pathlib import Path
from collections import OrderedDict
import time
import open3d as o3d
import numpy as np

# Resolve project root and SpinNet path

_THIS_FILE = Path(__file__).resolve()
CODE_ROOT = _THIS_FILE.parents[1]              # Code/
SPINNET_ROOT = CODE_ROOT / "model" / "SpinNet"

if str(SPINNET_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINNET_ROOT))

from network.SpinNet import Descriptor_Net


def load_ckpt_strip_module(ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    state_dict = ckpt

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "net" in ckpt:
            state_dict = ckpt["net"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state_dict[new_k] = v
    return new_state_dict


def build_spinnet_model(ckpt_path,
                        des_r=0.30, rad_n=9, azi_n=80, ele_n=40,
                        voxel_r=0.04, voxel_sample=30,
                        dataset="3DMatch",
                        device="cuda:0"):

    model = Descriptor_Net(des_r, rad_n, azi_n, ele_n, voxel_r, voxel_sample, dataset)

    sd = load_ckpt_strip_module(ckpt_path, map_location="cpu")

    model.load_state_dict(sd, strict=True)

    model.eval()
    model.to(device)
    return model



def spinnet_features_for_pcd_profiled(
    pcd_down: o3d.geometry.PointCloud,
    model: torch.nn.Module,
    patch_radius: float,
    N: int = 2048,
    batch_size: int = 64,
    device: str = "cuda:0",
):
    pts = np.asarray(pcd_down.points).astype(np.float32)
    num_pts = pts.shape[0]
    kdtree = o3d.geometry.KDTreeFlann(pcd_down)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    # timers
    t_kdtree = 0.0
    t_patch  = 0.0
    t_stack  = 0.0
    t_fwd    = 0.0

    desc_list = []
    patches_buf = []


    t_total0 = time.perf_counter()

    for i in range(num_pts):
        center = pts[i]

        # 1) KDTree radius search timing
        t0 = time.perf_counter()
        _, idxs, _ = kdtree.search_radius_vector_3d(center, patch_radius)
        t_kdtree += time.perf_counter() - t0

        # 2) Patch build timing (sampling/padding)
        t0 = time.perf_counter()
        if len(idxs) < 5:
            patch = np.repeat(center[None, :], N, axis=0)
        else:
            neigh = pts[np.asarray(idxs, dtype=np.int64)]
            if neigh.shape[0] >= N:
                sel = np.random.choice(neigh.shape[0], N, replace=False)
            else:
                sel = np.random.choice(neigh.shape[0], N, replace=True)
            patch = neigh[sel]
            patch[-1] = center
        t_patch += time.perf_counter() - t0

        patches_buf.append(patch)
        # 3) batch forward timing
        if len(patches_buf) == batch_size or i == num_pts - 1:
            t0 = time.perf_counter()
            batch_np = np.stack(patches_buf, axis=0)   # (B,N,3)
            t_stack += time.perf_counter() - t0
        
            batch = torch.as_tensor(batch_np, device=dev, dtype=torch.float32)
        
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(batch)
                out = out.view(out.shape[0], -1)
                out = F.normalize(out, p=2, dim=1)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_fwd += time.perf_counter() - t0
        
            desc_list.append(out.detach().cpu().numpy())
            patches_buf = []

    t_total = time.perf_counter() - t_total0

    desc = np.concatenate(desc_list, axis=0)  # (num_pts, 32)
    feat = o3d.pipelines.registration.Feature()
    feat.data = desc.T  # (32, num_pts)

    profile = {
        "num_pts": num_pts,
        "total_s": t_total,
        "per_point_ms": (t_total / max(1, num_pts)) * 1000.0,
        "kdtree_s": t_kdtree,
        "patch_build_s": t_patch,
        "stack_s": t_stack,
        "forward_s": t_fwd,
        "kdtree_ms_per_pt": (t_kdtree / max(1, num_pts)) * 1000.0,
        "patch_ms_per_pt": (t_patch / max(1, num_pts)) * 1000.0,
        "stack_ms_per_pt": (t_stack / max(1, num_pts)) * 1000.0,
        "fwd_ms_per_pt": (t_fwd / max(1, num_pts)) * 1000.0,
    }
    return feat, profile


def spinnet_features_for_keypoints_profiled(
    support_pcd: o3d.geometry.PointCloud,   # KDTree & neighbor source
    query_pcd: o3d.geometry.PointCloud,     # centers to compute descriptors (K points)
    model: torch.nn.Module,
    patch_radius: float,
    N: int = 2048,
    batch_size: int = 64,
    device: str = "cuda:0",
):
    support_pts = np.asarray(support_pcd.points).astype(np.float32)
    query_pts   = np.asarray(query_pcd.points).astype(np.float32)
    num_q = query_pts.shape[0]

    kdtree = o3d.geometry.KDTreeFlann(support_pcd)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    # timers
    t_kdtree = 0.0
    t_patch  = 0.0
    t_stack  = 0.0
    t_fwd    = 0.0

    desc_list = []
    patches_buf = []

    t_total0 = time.perf_counter()

    for i in range(num_q):
        center = query_pts[i]

        # 1) KDTree radius search on SUPPORT
        t0 = time.perf_counter()
        _, idxs, _ = kdtree.search_radius_vector_3d(center, patch_radius)
        t_kdtree += time.perf_counter() - t0

        # 2) Patch build from SUPPORT points, but force last point = QUERY center
        t0 = time.perf_counter()
        if len(idxs) < 5:
            patch = np.repeat(center[None, :], N, axis=0)
        else:
            neigh = support_pts[np.asarray(idxs, dtype=np.int64)]
            if neigh.shape[0] >= N:
                sel = np.random.choice(neigh.shape[0], N, replace=False)
            else:
                sel = np.random.choice(neigh.shape[0], N, replace=True)
            patch = neigh[sel]
            patch[-1] = center
        t_patch += time.perf_counter() - t0

        patches_buf.append(patch)

        # 3) batch forward
        if len(patches_buf) == batch_size or i == num_q - 1:
            t0 = time.perf_counter()
            batch_np = np.stack(patches_buf, axis=0)
            t_stack += time.perf_counter() - t0

            batch = torch.from_numpy(batch_np).to(device=dev, dtype=torch.float32)

            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(batch)
                out = out.view(out.shape[0], -1)
                out = F.normalize(out, p=2, dim=1)
            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t_fwd += time.perf_counter() - t0

            desc_list.append(out.detach().cpu().numpy())
            patches_buf = []

    t_total = time.perf_counter() - t_total0

    desc = np.concatenate(desc_list, axis=0)  # (K, 32)
    feat = o3d.pipelines.registration.Feature()
    feat.data = desc.T  # (32, K)

    profile = {
        "num_query": int(num_q),
        "total_s": float(t_total),
        "per_query_ms": float((t_total / max(1, num_q)) * 1000.0),
        "kdtree_s": float(t_kdtree),
        "patch_build_s": float(t_patch),
        "stack_s": float(t_stack),
        "forward_s": float(t_fwd),
        "kdtree_ms_per_q": float((t_kdtree / max(1, num_q)) * 1000.0),
        "patch_ms_per_q": float((t_patch / max(1, num_q)) * 1000.0),
        "stack_ms_per_q": float((t_stack / max(1, num_q)) * 1000.0),
        "fwd_ms_per_q": float((t_fwd / max(1, num_q)) * 1000.0),
    }
    return feat, profile



def compute_fpfh_for_support(pcd_support: o3d.geometry.PointCloud, voxel_size: float):
    """
    Compute FPFH on a SUPPORT cloud (already downsampled + outlier-removed).
    Returns:
        fpfh: o3d.pipelines.registration.Feature (33, N_support)
    """
    # Normals
    radius_normal = voxel_size * 2.0
    pcd_support.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # FPFH
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_support,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh