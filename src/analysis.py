# src/analysis.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CANDIDATE_METRICS = [
    "fpfh_cov_5mm", "fpfh_cov_10mm",
    "fpfh_trimrmse_5mm", "fpfh_trimrmse_10mm",
    "fpfh_chamfer",
    "spinnet_cov_5mm", "spinnet_cov_10mm",
    "spinnet_trimrmse_5mm", "spinnet_trimrmse_10mm",
    "spinnet_chamfer",
    "fpfh_total_s", "spinnet_total_s",
    "fpfh_feat_s", "fpfh_ransac_s", "fpfh_icp_s",
    "spinnet_feat_s", "spinnet_ransac_s", "spinnet_icp_s",
    "fpfh_ransac_fitness", "fpfh_icp_fitness",
    "spinnet_ransac_fitness", "spinnet_icp_fitness",
    "combine_w_fpfh", "combine_w_spinnet",
    "combine_feat_s", "combine_ransac_s", "combine_icp_s", "combine_total_s",
    "combine_ransac_fitness", "combine_ransac_rmse",
    "combine_icp_fitness", "combine_icp_rmse",
    "combine_cov_5mm", "combine_cov_10mm",
    "combine_trimrmse_5mm", "combine_trimrmse_10mm",
    "combine_chamfer",
]


def load_raw_runs(root: str | Path) -> tuple[pd.DataFrame, list[str]]:
    root = Path(root)
    rows: list[dict] = []
    missing: list[str] = []

    for k_dir in sorted(root.glob("K_*")):
        if not k_dir.is_dir():
            continue
        try:
            K_from_dir = int(k_dir.name.split("_", 1)[1])
        except Exception:
            continue

        for seed_dir in sorted(k_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed_from_dir = int(seed_dir.name.split("_", 1)[1])
            except Exception:
                seed_from_dir = None

            json_path = seed_dir / "out.json"
            if not json_path.exists():
                alt = seed_dir / "output.json"
                if alt.exists():
                    json_path = alt
                else:
                    missing.append(str(seed_dir))
                    continue

            with open(json_path, "r") as f:
                out = json.load(f)

            out["K"] = K_from_dir
            out["seed"] = seed_from_dir
            out["_json_path"] = str(json_path)
            rows.append(out)

    if not rows:
        raise RuntimeError(
            f"No out.json found under {root.resolve()}.\n"
            "Expected structure: ROOT/K_*/seed_*/out.json"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["K", "seed"]).reset_index(drop=True)
    return df, missing


def select_metrics(df: pd.DataFrame, candidate_metrics: Iterable[str] = CANDIDATE_METRICS) -> list[str]:
    metrics = [m for m in candidate_metrics if m in df.columns]
    if not metrics:
        raise RuntimeError(
            "None of the candidate metrics are found in your JSON files.\n"
            f"Available columns include: {list(df.columns)[:30]} ..."
        )
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return metrics


def _is_success(row: pd.Series, prefix: str) -> bool:
    rf = row.get(f"{prefix}_ransac_fitness", np.nan)
    icpf = row.get(f"{prefix}_icp_fitness", np.nan)
    if pd.isna(rf) or pd.isna(icpf):
        return False
    return (rf > 0.0) and (icpf > 0.0)


def add_success_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fpfh_success"] = df.apply(lambda r: _is_success(r, "fpfh"), axis=1)
    df["spinnet_success"] = df.apply(lambda r: _is_success(r, "spinnet"), axis=1)
    df["combine_success"] = df.apply(lambda r: _is_success(r, "combine"), axis=1)
    return df


def _agg_cols(sub: pd.DataFrame, cols: list[str]) -> dict:
    out: dict = {}
    for c in cols:
        s = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        out[f"{c}_count"] = int(s.shape[0])
        out[f"{c}_mean"] = float(s.mean()) if len(s) else float("nan")
        out[f"{c}_std"] = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        out[f"{c}_var"] = float(s.var(ddof=1)) if len(s) > 1 else float("nan")
    return out


def summarize_by_K(df_raw: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpfh_quality = [
        m for m in metrics
        if m.startswith("fpfh_") and (("cov_" in m) or ("trimrmse_" in m) or m.endswith("_chamfer"))
    ]
    spinnet_quality = [
        m for m in metrics
        if m.startswith("spinnet_") and (("cov_" in m) or ("trimrmse_" in m) or m.endswith("_chamfer"))
    ]

    combine_quality = [
    m for m in metrics
    if m.startswith("combine_") and (("cov_" in m) or ("trimrmse_" in m) or m.endswith("_chamfer"))
    ]
    
    timing_metrics = [m for m in metrics if m.endswith("_s")]

    wide_rows: list[dict] = []

    for K, subK in df_raw.groupby("K"):
        n_total = int(subK.shape[0])
        f_succ = subK[subK["fpfh_success"]]
        s_succ = subK[subK["spinnet_success"]]
        c_succ = subK[subK["combine_success"]]

        row = {
            "K": int(K),
            "n_total": n_total,
            "fpfh_n_success": int(f_succ.shape[0]),
            "fpfh_n_fail": int(n_total - f_succ.shape[0]),
            "spinnet_n_success": int(s_succ.shape[0]),
            "spinnet_n_fail": int(n_total - s_succ.shape[0]),
            "combine_n_success": int(c_succ.shape[0]),
            "combine_n_fail": int(n_total - c_succ.shape[0]),
        }

        if timing_metrics:
            g = subK[timing_metrics].agg(["count", "mean", "std", "var"])
            for m in timing_metrics:
                row[f"{m}_count"] = int(g.loc["count", m])
                row[f"{m}_mean"] = float(g.loc["mean", m])
                row[f"{m}_std"] = float(g.loc["std", m]) if n_total > 1 else float("nan")
                row[f"{m}_var"] = float(g.loc["var", m]) if n_total > 1 else float("nan")

        row.update(_agg_cols(f_succ, fpfh_quality))
        row.update(_agg_cols(s_succ, spinnet_quality))
        row.update(_agg_cols(c_succ, combine_quality))

        wide_rows.append(row)

    df_wide = pd.DataFrame(wide_rows).sort_values("K").reset_index(drop=True)

    long_rows: list[dict] = []
    for _, r in df_wide.iterrows():
        K = int(r["K"])
        for m in fpfh_quality + spinnet_quality + combine_quality + timing_metrics:
            long_rows.append({
                "K": K,
                "metric": m,
                "count": r.get(f"{m}_count", np.nan),
                "mean": r.get(f"{m}_mean", np.nan),
                "std": r.get(f"{m}_std", np.nan),
                "var": r.get(f"{m}_var", np.nan),
            })

    df_long = pd.DataFrame(long_rows).sort_values(["metric", "K"]).reset_index(drop=True)
    return df_wide, df_long


def _keep_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def make_simplified_tables(df_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpfh_keep = [
        "K", "n_total", "fpfh_n_success", "fpfh_n_fail",
        "fpfh_cov_5mm_mean", "fpfh_cov_5mm_std",
        "fpfh_cov_10mm_mean", "fpfh_cov_10mm_std",
        "fpfh_trimrmse_5mm_mean", "fpfh_trimrmse_5mm_std",
        "fpfh_trimrmse_10mm_mean", "fpfh_trimrmse_10mm_std",
        "fpfh_total_s_mean", "fpfh_feat_s_mean", "fpfh_ransac_s_mean", "fpfh_icp_s_mean",
    ]
    spinnet_keep = [
        "K", "n_total", "spinnet_n_success", "spinnet_n_fail",
        "spinnet_cov_5mm_mean", "spinnet_cov_5mm_std",
        "spinnet_cov_10mm_mean", "spinnet_cov_10mm_std",
        "spinnet_trimrmse_5mm_mean", "spinnet_trimrmse_5mm_std",
        "spinnet_trimrmse_10mm_mean", "spinnet_trimrmse_10mm_std",
        "spinnet_total_s_mean", "spinnet_feat_s_mean", "spinnet_ransac_s_mean", "spinnet_icp_s_mean",
    ]

    combine_keep = [
        "K", "n_total", "combine_n_success", "combine_n_fail",
        "combine_cov_5mm_mean", "combine_cov_5mm_std",
        "combine_cov_10mm_mean", "combine_cov_10mm_std",
        "combine_trimrmse_5mm_mean", "combine_trimrmse_5mm_std",
        "combine_trimrmse_10mm_mean", "combine_trimrmse_10mm_std",
        "combine_total_s_mean", "combine_feat_s_mean", "combine_ransac_s_mean", "combine_icp_s_mean",
        "combine_w_fpfh_mean", "combine_w_spinnet_mean",
    ]

    df_simpl_fpfh = df_wide[_keep_existing_cols(df_wide, fpfh_keep)].copy()
    df_simpl_spin = df_wide[_keep_existing_cols(df_wide, spinnet_keep)].copy()
    df_simpl_comb = df_wide[_keep_existing_cols(df_wide, combine_keep)].copy()
    return df_simpl_fpfh, df_simpl_spin, df_simpl_comb


def run_analysis(
    root: str | Path,
    stat_root: str | Path,
    *,
    candidate_metrics: list[str] = CANDIDATE_METRICS,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = Path(root)
    stat_root = Path(stat_root)
    stat_root.mkdir(parents=True, exist_ok=True)

    out_csv_wide = stat_root / "summary_by_K_wide.csv"
    out_csv_long = stat_root / "summary_by_K_long.csv"
    out_csv_raw = stat_root / "raw_runs.csv"
    out_csv_simpl_fpfh = stat_root / "summary_by_K_simplified_fpfh.csv"
    out_csv_simpl_spin = stat_root / "summary_by_K_simplified_spinnet.csv"
    out_csv_simpl_comb = stat_root / "summary_by_K_simplified_combine.csv"

    df_raw, missing = load_raw_runs(root)
    metrics = select_metrics(df_raw, candidate_metrics)
    df_raw = add_success_flags(df_raw)

    df_wide, df_long = summarize_by_K(df_raw, metrics)
    df_simpl_fpfh, df_simpl_spin, df_simpl_comb = make_simplified_tables(df_wide)

    paths = {
        "raw": out_csv_raw,
        "wide": out_csv_wide,
        "long": out_csv_long,
        "simpl_fpfh": out_csv_simpl_fpfh,
        "simpl_spinnet": out_csv_simpl_spin,
        "simpl_combine": out_csv_simpl_comb,
    }

    if save:
        df_raw.to_csv(out_csv_raw, index=False)
        df_wide.to_csv(out_csv_wide, index=False)
        df_long.to_csv(out_csv_long, index=False)
        df_simpl_fpfh.to_csv(out_csv_simpl_fpfh, index=False)
        df_simpl_spin.to_csv(out_csv_simpl_spin, index=False)
        df_simpl_comb.to_csv(out_csv_simpl_comb, index=False)


    info = {
        "loaded_runs": int(df_raw.shape[0]),
        "Ks": sorted(df_raw["K"].unique().tolist()),
        "missing": missing,
        "metrics_used": metrics,
        "paths": {k: str(v.resolve()) for k, v in paths.items()},
    }

    return df_wide, df_long, df_raw, df_simpl_fpfh, df_simpl_spin, df_simpl_comb, info


def _load_csv_with_name(csv_path: Path, data_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["data_name"] = data_name
    return df


def _remap_K_to_labels(df: pd.DataFrame, k_labels: list[str]) -> pd.DataFrame:
    """
    Replace numeric K values in column 'K' with provided percentage labels
    based on the sorted order of unique numeric K values.
    """
    if "K" not in df.columns:
        raise KeyError("Column 'K' not found.")

    k_vals = pd.to_numeric(df["K"], errors="raise")
    uniq_sorted = sorted(k_vals.unique())

    if len(uniq_sorted) != len(k_labels):
        raise ValueError(
            f"Expected {len(k_labels)} unique K values, got {len(uniq_sorted)}: {uniq_sorted}"
        )

    k_map = dict(zip(uniq_sorted, k_labels))
    out = df.copy()
    out["K"] = pd.to_numeric(out["K"], errors="raise").map(k_map)
    return out


def _sort_by_data_then_Klabel(df: pd.DataFrame, k_labels: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    order_map = {lab: i for i, lab in enumerate(k_labels)}
    return (
        df.sort_values(
            by=["data_name", "K"],
            key=lambda s: s.map(order_map) if s.name == "K" else s,
        )
        .reset_index(drop=True)
    )


def aggregate_simplified_tables(
    stat_root: str | Path,
    *,
    fpfh_name: str = "summary_by_K_simplified_fpfh.csv",
    spinnet_name: str = "summary_by_K_simplified_spinnet.csv",
    combine_name: str = "summary_by_K_simplified_combine.csv",
    k_labels: list[str] = ["5%", "10%", "20%", "40%"],
    remap_K: bool = True,
    strict_K: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Aggregate per-object simplified summary tables into three global tables:
    - fpfh_all: concatenation of all found FPFH simplified CSVs
    - spinnet_all: concatenation of all found SpinNet simplified CSVs
    - combine_all: concatenation of all found FPFH+SpinNet simplified CSVs

    Args:
        stat_root: directory containing subfolders (one per data_name).
        fpfh_name/spinnet_name/combin_name: expected simplified filenames inside each subfolder.
        k_labels: labels used to remap numeric K (sorted order) -> percent labels.
        remap_K: if True, remap column 'K' to k_labels.
        strict_K: if True, raise if a dataset has unexpected #unique K values
                  (otherwise skip remap for that dataset and keep numeric K).

    Returns:
        (fpfh_all, spinnet_all, info)
        info contains lists of missing files and datasets processed.
    """
    stat_root = Path(stat_root)

    fpfh_rows: list[pd.DataFrame] = []
    spinnet_rows: list[pd.DataFrame] = []
    combine_rows: list[pd.DataFrame] = []

    missing_fpfh: list[str] = []
    missing_spinnet: list[str] = []
    missing_combine: list[str] = []

    processed: list[str] = []
    remap_fail: list[tuple[str, str]] = []  # (data_name, method)

    for data_dir in sorted(stat_root.iterdir()):
        if not data_dir.is_dir():
            continue

        data_name = data_dir.name
        processed.append(data_name)

        fpfh_csv = data_dir / fpfh_name
        spinnet_csv = data_dir / spinnet_name
        combine_csv = data_dir / combine_name

        # FPFH
        if fpfh_csv.exists():
            df = _load_csv_with_name(fpfh_csv, data_name)
            if remap_K:
                try:
                    df = _remap_K_to_labels(df, k_labels)
                except Exception:
                    if strict_K:
                        raise
                    remap_fail.append((data_name, "fpfh"))
            fpfh_rows.append(df)
        else:
            missing_fpfh.append(data_name)

        # SpinNet
        if spinnet_csv.exists():
            df = _load_csv_with_name(spinnet_csv, data_name)
            if remap_K:
                try:
                    df = _remap_K_to_labels(df, k_labels)
                except Exception:
                    if strict_K:
                        raise
                    remap_fail.append((data_name, "spinnet"))
            spinnet_rows.append(df)

        # Combine
        if combine_csv.exists():
            df = _load_csv_with_name(combine_csv, data_name)
            if remap_K:
                try:
                    df = _remap_K_to_labels(df, k_labels)
                except Exception:
                    if strict_K:
                        raise
                    remap_fail.append((data_name, "combine"))
            combine_rows.append(df)
        else:
            missing_spinnet.append(data_name)

    fpfh_all = pd.concat(fpfh_rows, ignore_index=True) if fpfh_rows else pd.DataFrame()
    spinnet_all = pd.concat(spinnet_rows, ignore_index=True) if spinnet_rows else pd.DataFrame()
    combine_all = pd.concat(combine_rows, ignore_index=True) if combine_rows else pd.DataFrame()

    # Nice ordering when K has been remapped to labels
    if remap_K:
        fpfh_all = _sort_by_data_then_Klabel(fpfh_all, k_labels)
        spinnet_all = _sort_by_data_then_Klabel(spinnet_all, k_labels)
        combine_all = _sort_by_data_then_Klabel(combine_all, k_labels)

    info = {
        "stat_root": str(stat_root.resolve()),
        "processed": processed,
        "missing_fpfh": missing_fpfh,
        "missing_spinnet": missing_spinnet,
        "missing_combine": missing_combine,
        "remap_fail": remap_fail,
        "n_fpfh_rows": int(fpfh_all.shape[0]) if not fpfh_all.empty else 0,
        "n_spinnet_rows": int(spinnet_all.shape[0]) if not spinnet_all.empty else 0,
        "n_combine_rows": int(combine_all.shape[0]) if not combine_all.empty else 0,
    }
    return fpfh_all, spinnet_all, combine_all, info

