from pathlib import Path
import argparse

from src.descriptor import build_spinnet_model
from src.sweep import run_sweep


def parse_args():
    p = argparse.ArgumentParser("Run SpinNet sweep")

    # Paths
    p.add_argument("--ref", type=str, required=True,
                   help="Reference point cloud path")
    p.add_argument("--mov", type=str, required=True,
                   help="Moving point cloud path")
    p.add_argument("--ckpt", type=str, required=True,
                   help="SpinNet checkpoint path")

    # Device / runtime
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--save-dir", type=str, default=None)

    # Descriptor params
    p.add_argument("--des-r", type=float, default=0.30)
    p.add_argument("--rad-n", type=int, default=9)
    p.add_argument("--azi-n", type=int, default=80)
    p.add_argument("--ele-n", type=int, default=40)
    p.add_argument("--voxel-r", type=float, default=0.04)
    p.add_argument("--voxel-sample", type=int, default=30)
    p.add_argument("--dataset", type=str, default="3DMatch")

    # Sweep params
    p.add_argument(
        "--K-fracs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4],
        help="List of K fractions, e.g. --K-fracs 0.05 0.1 0.2"
    )

    p.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[0, 1, 2, 3, 4],
    help="Random seeds, e.g. --seeds 0 1 2 3 4"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths 
    ref_path = Path(args.ref)
    mov_path = Path(args.mov)
    ckpt_path = Path(args.ckpt)

    # Build descriptor model
    model = build_spinnet_model(
        ckpt_path=ckpt_path,
        des_r=args.des_r,
        rad_n=args.rad_n,
        azi_n=args.azi_n,
        ele_n=args.ele_n,
        voxel_r=args.voxel_r,
        voxel_sample=args.voxel_sample,
        dataset="3DMatch",
        device=args.device,
    )

    # Run sweep
    results = run_sweep(
        ref_path=ref_path,
        mov_path=mov_path,
        model=model,
        K_fracs=args.K_fracs,
        seeds=args.seeds,       
        device=args.device,
        batch_size=args.batch_size,
        save_dir=args.save_dir,   
    )

    print("Sweep finished.")
    print(results)
    return results


if __name__ == "__main__":
    main()
