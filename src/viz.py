import os, glob, csv
import numpy as np
import open3d as o3d
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import pandas as pd
import json
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


COLOR_REF = [0.1, 0.3, 1.0]   # blue  (laser)
COLOR_MOV = [1.0, 0.2, 0.2]   # red   (photo)
COLOR_ALIGNED = [0.2, 0.9, 0.2]  # green
GRAY = [0.7, 0.7, 0.7] # For single obj

def _as_renderable(geom):
    """
    Ensure the geometry is renderable.
    - If it's a TriangleMesh with triangles: keep as mesh
    - If it's a TriangleMesh without triangles: render as point cloud of its vertices
    - If it's a PointCloud: keep as point cloud
    Returns: (renderable_geom, kind_str) where kind_str in {"mesh","pcd"}
    """
    if isinstance(geom, o3d.geometry.TriangleMesh):
        if len(geom.vertices) == 0:
            return None, "empty"
        if len(geom.triangles) == 0:
            # mesh-like PLY that is actually a point set: render as point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(geom.vertices))
            return pcd, "pcd"
        # proper mesh
        if not geom.has_vertex_normals():
            geom.compute_vertex_normals()
        return geom, "mesh"

    if isinstance(geom, o3d.geometry.PointCloud):
        if len(geom.points) == 0:
            return None, "empty"
        return geom, "pcd"

    raise TypeError(f"Unsupported geometry type: {type(geom)}")

def _paint(geom, color):
    """Paint both mesh and point cloud uniformly."""
    geom.paint_uniform_color(color)
    return geom

def visualize_ref_mov(ref_geom, mov_geom, aligned_geom=None, out_png=None,
                      width=1200, height=800, point_size=2.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)

    # Render options (useful if you end up rendering as point clouds)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)

    def add_one(g, color, label, keep_colors=False):
        g = copy.deepcopy(g)
        g, kind = _as_renderable(g)
        if g is None:
            print(f"[WARN] {label} is empty; skipped.")
            return
        if not keep_colors:
            _paint(g, color)
        vis.add_geometry(g)

    if aligned_geom is not None:
        add_one(ref_geom, COLOR_REF, "ref")
        add_one(aligned_geom, COLOR_ALIGNED, "aligned", keep_colors=True)  
    else:
        add_one(ref_geom, COLOR_REF, "ref")
        add_one(mov_geom, COLOR_MOV, "mov")

    # Interaction first, screenshot after
    print("Adjust the view, then press Q to save screenshot and exit. (Press S to save anytime.)")
    vis.run()

    if out_png is not None:
        vis.capture_screen_image(out_png, do_render=True)
        print(f"Saved screenshot to {out_png}")

    vis.destroy_window()




def _as_renderable_single(geom):
    """Return a renderable geometry (mesh or pcd)."""
    if isinstance(geom, o3d.geometry.TriangleMesh):
        if len(geom.vertices) == 0:
            return None
        if len(geom.triangles) == 0:
            # vertex-only mesh -> point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(geom.vertices))
            return pcd
        if not geom.has_vertex_normals():
            geom.compute_vertex_normals()
        return geom

    if isinstance(geom, o3d.geometry.PointCloud):
        if len(geom.points) == 0:
            return None
        return geom

    raise TypeError(f"Unsupported geometry type: {type(geom)}")

def visualize_single_gray(geom, out_png=None,
                          width=1200, height=800,
                          point_size=2.0):
    """
    Visualize a single geometry in gray.
    Adjust view, then press Q to save screenshot and exit.
    Press S to save anytime.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)

    g = copy.deepcopy(geom)
    g = _as_renderable_single(g)
    if g is None:
        raise ValueError("Empty geometry, nothing to visualize.")

    g.paint_uniform_color(GRAY)
    vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.asarray([1.0, 1.0, 1.0])  # white background

    print("Adjust the view, then press Q to save screenshot and exit.")
    print("Press S to save a screenshot at any time.")

    vis.run()

    if out_png is not None:
        vis.capture_screen_image(out_png, do_render=True)
        print(f"Saved screenshot to {out_png}")

    vis.destroy_window()


def geom_to_pcd_no_sampling(geom):
    if isinstance(geom, o3d.geometry.PointCloud):
        return geom
    if isinstance(geom, o3d.geometry.TriangleMesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(geom.vertices)
        )
        return pcd
    raise TypeError(type(geom))


def colorize_distance_to_ref(
    src_geom,         
    ref_geom,         
    max_dist=0.01,    
    cmap_name="jet",
):
    src_pcd = geom_to_pcd_no_sampling(src_geom)
    ref_pcd = geom_to_pcd_no_sampling(ref_geom)

    dists = np.asarray(src_pcd.compute_point_cloud_distance(ref_pcd))

    dists = np.clip(dists, 0, max_dist)
    dists_norm = dists / max_dist

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(dists_norm)[:, :3]

    src_pcd.colors = o3d.utility.Vector3dVector(colors)
    return src_pcd


def annotate_heatmap_png(
    in_png: str,
    out_png: str,
    max_dist_mm: float,
    cmap_name: str = "jet",
    caption: str | None = None,
):
    # Load screenshot
    img = mpimg.imread(in_png)

    # Canvas
    fig = plt.figure(figsize=(10, 5.8), dpi=200)
    ax  = fig.add_axes([0.05, 0.18, 0.82, 0.78])
    cax = fig.add_axes([0.86, 0.25, 0.025, 0.62])
    ax.imshow(img)
    ax.axis("off")

    # Colorbar
    norm = Normalize(vmin=0.0, vmax=max_dist_mm)
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])


    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Point-to-reference distance (mm)")
    # Optional ticks aligned with your thresholds
    ticks = [0, 5, 10] if max_dist_mm >= 10 else [0, max_dist_mm/2, max_dist_mm]
    ticks = [t for t in ticks if t <= max_dist_mm]
    cb.set_ticks(ticks)

    # # Threshold annotation on the image (top-left corner)
    # thr_text = f"Clipping: d ≤ {max_dist_mm:g} mm"
    # ax.text(
    #     0.01, 0.02, thr_text,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment="bottom",
    #     horizontalalignment="left",
    #     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    # )

    # Caption
    # if caption is None:
    #     caption = (f"Distance heatmap (jet colormap). Distances are clipped to "
    #                f"{max_dist_mm:g} mm (values above are saturated).")
    # fig.text(0.05, 0.06, caption, fontsize=10)

    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved annotated figure: {out_png}")

