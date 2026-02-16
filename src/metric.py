import numpy as np
import json
import open3d as o3d

def nn_distances(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Compute nearest neighbor distances from each point in 'source' to 'target'.
    Returns:
        dists: (N,) array of Euclidean distances.
    """
    target_kd = o3d.geometry.KDTreeFlann(target)
    src_pts = np.asarray(source.points)

    dists = np.empty(len(src_pts), dtype=np.float64)
    for i, p in enumerate(src_pts):
        # 1-NN search
        _, idx, dist2 = target_kd.search_knn_vector_3d(p, 1)
        dists[i] = np.sqrt(dist2[0])
    return dists


def registration_metrics(source_aligned: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud,
                         thresholds=(5.0, 10.0),
                         percentiles=(50, 90, 95)) -> dict:
    """
    Compute overlap-aware metrics for registration evaluation.

    Metrics:
      - median / P90 / P95 distances
      - coverage@tau: ratio of points with dist < tau
      - trimmed_rmse@tau: RMSE computed only on points with dist < tau
      - mean / std (for reference)
    """
    d = nn_distances(source_aligned, target)

    out = {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "median": float(np.percentile(d, 50)),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(d, p))

    for tau in thresholds:
        inliers = d < tau
        coverage = float(inliers.mean())
        if inliers.any():
            trimmed_rmse = float(np.sqrt(np.mean(d[inliers] ** 2)))
            trimmed_mean = float(d[inliers].mean())
        else:
            trimmed_rmse = float("nan")
            trimmed_mean = float("nan")
        out[f"coverage@{tau}"] = coverage
        out[f"trimmed_mean@{tau}"] = trimmed_mean
        out[f"trimmed_rmse@{tau}"] = trimmed_rmse

    return out


def symmetric_chamfer(source_aligned: o3d.geometry.PointCloud,
                      target: o3d.geometry.PointCloud) -> float:
    """
    Symmetric Chamfer distance (mean NN distance both directions).
    """
    d_st = nn_distances(source_aligned, target).mean()
    d_ts = nn_distances(target, source_aligned).mean()
    return float(d_st + d_ts)