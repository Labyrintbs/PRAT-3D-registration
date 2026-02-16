import torch
import copy
import open3d as o3d
import numpy as np

from src.preprocess import prepare_clouds_scale_to_m
from src.utils import global_registration_ransac, refine_registration_icp, now, diag, select_keypoints_random, save_aligned_cloud, save_out_json, slice_feature_by_index, assert_kp_alignment
from src.descriptor import spinnet_features_for_keypoints_profiled, compute_fpfh_for_support
from src.metric import registration_metrics, symmetric_chamfer 

def _make_Ks_from_fracs(
    ref_support_n,
    mov_support_n,
    fracs=(0.05, 0.10, 0.20, 0.40),
    base="min_support",
    K_min=256,
    K_max=None,
):
    if base == "min_support":
        n_base = min(ref_support_n, mov_support_n)
    elif base == "ref_support":
        n_base = ref_support_n
    elif base == "mov_support":
        n_base = mov_support_n
    elif base == "mean_support":
        n_base = int(round(0.5 * (ref_support_n + mov_support_n)))
    else:
        raise ValueError(f"Unknown base={base}")

    Ks = []
    for f in fracs:
        if not (0 < f <= 1.0):
            raise ValueError(f"Each frac must be in (0,1], got {f}")
        k = int(round(f * n_base))

        # clamp
        k = max(k, K_min)
        if K_max is not None:
            k = min(k, K_max)

        Ks.append(k)

    # unique + sorted
    Ks = sorted(set(Ks))
    return Ks, n_base


def l2_normalize_feature_data(X, eps=1e-12):
    # X: (D, N)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    return X / (norms + eps)

def concat_o3d_features(feat_a, feat_b, w_a=1.0, w_b=1.0):
    # feat_a, feat_b: open3d Feature with .data shape (D, N)
    A = np.asarray(feat_a.data)  # (D1, N)
    B = np.asarray(feat_b.data)  # (D2, N)

    A = l2_normalize_feature_data(A)
    B = l2_normalize_feature_data(B)

    C = np.vstack([w_a * A, w_b * B])  # (D1+D2, N)

    feat = o3d.pipelines.registration.Feature()
    feat.data = C
    return feat
    

def run_one_K_seed_compare_fpfh_spinnet(
    ref_full: o3d.geometry.PointCloud,
    mov_full: o3d.geometry.PointCloud,
    ref_support: o3d.geometry.PointCloud,
    mov_support: o3d.geometry.PointCloud,
    model: torch.nn.Module,
    voxel_size: float,
    patch_radius: float,
    K: int,
    seed: int,
    N_patch: int = 2048,
    batch_size: int = 48,
    device: str = "cuda:0",
    use_point_to_plane: bool = False,#True,
    eval_thresholds_m=(0.005, 0.010),
    save_dir: str = "Results/Registrations",
):
    """
    Returns a dict with:
      - keypoint info
      - timing breakdown for both methods
      - RANSAC/ICP metrics and your evaluation metrics
    """
    out = {
        "K": int(K),
        "seed": int(seed),
        "voxel_size": float(voxel_size),
        "patch_radius": float(patch_radius),
    }

    # Keypoints selection
    ref_kp, idx_ref = select_keypoints_random(ref_support, K, seed=seed)
    mov_kp, idx_mov = select_keypoints_random(mov_support, K, seed=seed)

    out["ref_support_n"] = int(len(ref_support.points))
    out["mov_support_n"] = int(len(mov_support.points))
    out["ref_kp_n"] = int(len(ref_kp.points))
    out["mov_kp_n"] = int(len(mov_kp.points))


    # Method 1: FPFH (support full -> slice K)
    t0 = now()
    ref_fpfh_all = compute_fpfh_for_support(ref_support, voxel_size)
    mov_fpfh_all = compute_fpfh_for_support(mov_support, voxel_size)
    t_fpfh_feat = now() - t0

    ref_fpfh_kp = slice_feature_by_index(ref_fpfh_all, idx_ref)
    mov_fpfh_kp = slice_feature_by_index(mov_fpfh_all, idx_mov)
    
    assert_kp_alignment(ref_support, ref_kp, idx_ref, "ref")
    assert_kp_alignment(mov_support, mov_kp, idx_mov, "mov")
    # RANSAC
    t0 = now()
    ransac_fpfh = global_registration_ransac(mov_kp, ref_kp, mov_fpfh_kp, ref_fpfh_kp, voxel_size)
    t_fpfh_ransac = now() - t0

    # ICP
    t0 = now()
    icp_fpfh = refine_registration_icp(mov_full, ref_full, ransac_fpfh.transformation, voxel_size, use_point_to_plane=use_point_to_plane)
    t_fpfh_icp = now() - t0

    # Evaluate
    mov_aligned_fpfh = copy.deepcopy(mov_full)
    mov_aligned_fpfh.transform(icp_fpfh.transformation)

    eval_fpfh = registration_metrics(mov_aligned_fpfh, ref_full, thresholds=eval_thresholds_m)
    chamfer_fpfh = symmetric_chamfer(mov_aligned_fpfh, ref_full)

    out.update({
        "fpfh_feat_s": float(t_fpfh_feat),
        "fpfh_ransac_s": float(t_fpfh_ransac),
        "fpfh_icp_s": float(t_fpfh_icp),
        "fpfh_total_s": float(t_fpfh_feat + t_fpfh_ransac + t_fpfh_icp),

        "fpfh_ransac_fitness": float(ransac_fpfh.fitness),
        "fpfh_ransac_rmse": float(ransac_fpfh.inlier_rmse),
        "fpfh_icp_fitness": float(icp_fpfh.fitness),
        "fpfh_icp_rmse": float(icp_fpfh.inlier_rmse),

        "fpfh_cov_5mm": float(eval_fpfh.get(f"coverage@{eval_thresholds_m[0]}", np.nan)),
        "fpfh_cov_10mm": float(eval_fpfh.get(f"coverage@{eval_thresholds_m[1]}", np.nan)),
        "fpfh_trimrmse_5mm": float(eval_fpfh.get(f"trimmed_rmse@{eval_thresholds_m[0]}", np.nan)),
        "fpfh_trimrmse_10mm": float(eval_fpfh.get(f"trimmed_rmse@{eval_thresholds_m[1]}", np.nan)),
        "fpfh_chamfer": float(chamfer_fpfh),
    })


    # Method 2: SpinNet (support/query separated)

    # Feature extraction (profile already contains time)
    sp_ref_feat, sp_ref_prof = spinnet_features_for_keypoints_profiled(
        support_pcd=ref_support,
        query_pcd=ref_kp,
        model=model,
        patch_radius=patch_radius,
        N=N_patch,
        batch_size=batch_size,
        device=device,
    )
    sp_mov_feat, sp_mov_prof = spinnet_features_for_keypoints_profiled(
        support_pcd=mov_support,
        query_pcd=mov_kp,
        model=model,
        patch_radius=patch_radius,
        N=N_patch,
        batch_size=batch_size,
        device=device,
    )

    t_sp_feat = sp_ref_prof["total_s"] + sp_mov_prof["total_s"]

    # RANSAC
    t0 = now()
    ransac_sp = global_registration_ransac(mov_kp, ref_kp, sp_mov_feat, sp_ref_feat, voxel_size)
    t_sp_ransac = now() - t0

    # ICP
    t0 = now()
    icp_sp = refine_registration_icp(mov_full, ref_full, ransac_sp.transformation, voxel_size, use_point_to_plane=use_point_to_plane)
    t_sp_icp = now() - t0

    # Evaluate
    mov_aligned_sp = copy.deepcopy(mov_full)
    mov_aligned_sp.transform(icp_sp.transformation)

    eval_sp = registration_metrics(mov_aligned_sp, ref_full, thresholds=eval_thresholds_m)
    chamfer_sp = symmetric_chamfer(mov_aligned_sp, ref_full)

    out.update({
        "spinnet_ref_feat_s": float(sp_ref_prof["total_s"]),
        "spinnet_mov_feat_s": float(sp_mov_prof["total_s"]),
        "spinnet_feat_s": float(t_sp_feat),
        "spinnet_ransac_s": float(t_sp_ransac),
        "spinnet_icp_s": float(t_sp_icp),
        "spinnet_total_s": float(t_sp_feat + t_sp_ransac + t_sp_icp),

        "spinnet_ransac_fitness": float(ransac_sp.fitness),
        "spinnet_ransac_rmse": float(ransac_sp.inlier_rmse),
        "spinnet_icp_fitness": float(icp_sp.fitness),
        "spinnet_icp_rmse": float(icp_sp.inlier_rmse),

        "spinnet_cov_5mm": float(eval_sp.get(f"coverage@{eval_thresholds_m[0]}", np.nan)),
        "spinnet_cov_10mm": float(eval_sp.get(f"coverage@{eval_thresholds_m[1]}", np.nan)),
        "spinnet_trimrmse_5mm": float(eval_sp.get(f"trimmed_rmse@{eval_thresholds_m[0]}", np.nan)),
        "spinnet_trimrmse_10mm": float(eval_sp.get(f"trimmed_rmse@{eval_thresholds_m[1]}", np.nan)),
        "spinnet_chamfer": float(chamfer_sp),
    })

    out["fpfh_T_icp"] = icp_fpfh.transformation
    out["spinnet_T_icp"] = icp_sp.transformation

    # Method 3: Combine (naive concat baseline: FPFH + SpinNet)

    # sanity: same K
    assert np.asarray(ref_fpfh_kp.data).shape[1] == np.asarray(sp_ref_feat.data).shape[1]
    assert np.asarray(mov_fpfh_kp.data).shape[1] == np.asarray(sp_mov_feat.data).shape[1]

    w_fpfh = 1.0
    w_sp = 1.0

    ref_comb_feat = concat_o3d_features(ref_fpfh_kp, sp_ref_feat, w_a=w_fpfh, w_b=w_sp)
    mov_comb_feat = concat_o3d_features(mov_fpfh_kp, sp_mov_feat, w_a=w_fpfh, w_b=w_sp)

    # RANSAC
    t0 = now()
    ransac_comb = global_registration_ransac(mov_kp, ref_kp, mov_comb_feat, ref_comb_feat, voxel_size)
    t_comb_ransac = now() - t0

    # ICP
    t0 = now()
    icp_comb = refine_registration_icp(
        mov_full, ref_full, ransac_comb.transformation, voxel_size,
        use_point_to_plane=use_point_to_plane
    )
    t_comb_icp = now() - t0

    # Evaluate
    mov_aligned_comb = copy.deepcopy(mov_full)
    mov_aligned_comb.transform(icp_comb.transformation)

    eval_comb = registration_metrics(mov_aligned_comb, ref_full, thresholds=eval_thresholds_m)
    chamfer_comb = symmetric_chamfer(mov_aligned_comb, ref_full)

    out.update({
        "combine_w_fpfh": float(w_fpfh),
        "combine_w_spinnet": float(w_sp),
        
        "combine_feat_s": float(t_fpfh_feat + t_sp_feat),
        "combine_ransac_s": float(t_comb_ransac),
        "combine_icp_s": float(t_comb_icp),
        "combine_total_s": float((t_fpfh_feat + t_sp_feat) + t_comb_ransac + t_comb_icp),

        "combine_ransac_fitness": float(ransac_comb.fitness),
        "combine_ransac_rmse": float(ransac_comb.inlier_rmse),
        "combine_icp_fitness": float(icp_comb.fitness),
        "combine_icp_rmse": float(icp_comb.inlier_rmse),

        "combine_cov_5mm": float(eval_comb.get(f"coverage@{eval_thresholds_m[0]}", np.nan)),
        "combine_cov_10mm": float(eval_comb.get(f"coverage@{eval_thresholds_m[1]}", np.nan)),
        "combine_trimrmse_5mm": float(eval_comb.get(f"trimmed_rmse@{eval_thresholds_m[0]}", np.nan)),
        "combine_trimrmse_10mm": float(eval_comb.get(f"trimmed_rmse@{eval_thresholds_m[1]}", np.nan)),
        "combine_chamfer": float(chamfer_comb),
    })

    out["combine_T_icp"] = icp_comb.transformation

    # Save all results:

    # save_aligned_cloud(
    # pcd=mov_aligned_fpfh,
    # base_dir=save_dir,
    # K=K,
    # seed=seed,
    # method="FPFH",
    # )

    # save_aligned_cloud(
    # pcd=mov_aligned_sp,
    # base_dir=save_dir,
    # K=K,
    # seed=seed,
    # method="SpinNet",
    # )
    return out

def run_sweep(
    ref_path: str,
    mov_path: str,
    model: torch.nn.Module,
    K_fracs=(0.05, 0.10, 0.20, 0.40),
    K_base= "min_support",
    K_min=256,
    K_max=None,           
    seeds=(0, 1, 2, 3, 4),
    scale_to_m: float = 1e-3,
    voxel_size: float = 0.005,
    patch_radius: float = 0.30,
    N_patch: int = 2048,
    batch_size: int = 96,
    device: str = "cuda:0",
    use_point_to_plane: bool = False, #True,
    eval_thresholds_m=(0.005, 0.010),
    save_dir: str = "experiment",
):
    ref_full, mov_full, ref_support, mov_support = prepare_clouds_scale_to_m(
        ref_path, mov_path, scale_to_m=scale_to_m, voxel_size=voxel_size
    )

    ref_n = len(ref_support.points)
    mov_n = len(mov_support.points)

    Ks, n_base = _make_Ks_from_fracs(
        ref_support_n=ref_n,
        mov_support_n=mov_n,
        fracs=K_fracs,
        base=K_base,
        K_min=K_min,
        K_max=K_max,
    )

    print("Prepared:")
    print("  ref_full diag:", diag(ref_full), "center:", ref_full.get_center())
    print("  mov_full diag:", diag(mov_full), "center:", mov_full.get_center())
    print("  ref_support n:", ref_n)
    print("  mov_support n:", mov_n)
    print(f"  K_base={K_base}, n_base={n_base}")
    print("  Ks:", Ks, "from fracs:", K_fracs)

    results = []
    for K in Ks:
        for seed in seeds:
            print(f"\n=== Run K={K} ({K/n_base:.1%} of base), seed={seed} ===")
            out = run_one_K_seed_compare_fpfh_spinnet(
                ref_full=ref_full,
                mov_full=mov_full,
                ref_support=ref_support,
                mov_support=mov_support,
                model=model,
                voxel_size=voxel_size,
                patch_radius=patch_radius,
                K=K,
                seed=seed,
                N_patch=N_patch,
                batch_size=batch_size,
                device=device,
                use_point_to_plane=use_point_to_plane,
                eval_thresholds_m=eval_thresholds_m,
            )
            save_out_json(out=out, base_dir=f"{save_dir}", K=K, seed=seed)
            results.append(out)
            print("  FPFH total_s:", out["fpfh_total_s"], "cov@5mm:", out["fpfh_cov_5mm"], "rmse@5mm:", out["fpfh_trimrmse_5mm"])
            print("  Spin total_s:", out["spinnet_total_s"], "cov@5mm:", out["spinnet_cov_5mm"], "rmse@5mm:", out["spinnet_trimrmse_5mm"])
            print("  Combine total_s:", out["combine_total_s"], "cov@5mm:", out["combine_cov_5mm"], "rmse@5mm:", out["combine_trimrmse_5mm"])

    return results