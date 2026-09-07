"""
evaluate_pose.py

Evaluation metrics used in the CSSGNet paper.

Pose format:
    [tx, ty, tz, rx, ry, rz]

where:
    tx, ty, tz : translation in meters
    rx, ry, rz : angle-axis rotation vector in radians

The script reports:
1. Axis-wise absolute residual mean ± std for Tables 1-4.
2. SO(3) geodesic rotation error mean ± std for Tables 1-4.
3. Translation MAE/RMSE and rotation MAE/RMSE
   according to Eqs. (15)-(18) for Table 5.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


POSE_COLUMNS = ["tx", "ty", "tz", "rx", "ry", "rz"]


def skew(v):
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    x, y, z = v

    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0]
    ], dtype=np.float64)


def angle_axis_to_matrix(r):
    """
    Convert an angle-axis vector to a 3x3 rotation matrix
    using Rodrigues' formula.

    Parameters
    ----------
    r : array-like, shape (3,)
        Angle-axis vector in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    r = np.asarray(r, dtype=np.float64)

    theta = np.linalg.norm(r)

    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = r / theta
    K = skew(axis)

    R = (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )

    return R


def so3_geodesic_error(pred_rotation, gt_rotation):
    """
    Calculate the SO(3) geodesic error defined in Eq. (14):

        e_R = acos(
            clip(
                (trace(R_pred^T R_gt) - 1) / 2,
                -1, 1
            )
        )

    Parameters
    ----------
    pred_rotation : ndarray, shape (N, 3)
        Predicted angle-axis vectors.

    gt_rotation : ndarray, shape (N, 3)
        Ground-truth angle-axis vectors.

    Returns
    -------
    errors : ndarray, shape (N,)
        SO(3) geodesic errors in radians.
    """
    if pred_rotation.shape != gt_rotation.shape:
        raise ValueError(
            "Prediction and ground-truth rotation arrays "
            "must have the same shape."
        )

    errors = []

    for pred_r, gt_r in zip(pred_rotation, gt_rotation):

        R_pred = angle_axis_to_matrix(pred_r)
        R_gt = angle_axis_to_matrix(gt_r)

        relative_R = R_pred.T @ R_gt

        cos_theta = (np.trace(relative_R) - 1.0) / 2.0

        # Numerical protection required by Eq. (14)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        error = np.arccos(cos_theta)

        errors.append(error)

    return np.asarray(errors, dtype=np.float64)


def calculate_metrics(pred_pose, gt_pose, ddof=0):
    """
    Calculate all evaluation metrics used in the manuscript.

    Parameters
    ----------
    pred_pose : ndarray, shape (N, 6)
        Predicted relative poses.

    gt_pose : ndarray, shape (N, 6)
        Ground-truth relative poses.

    ddof : int
        Degrees of freedom used for standard deviation.
        Default: 0.

    Returns
    -------
    results : dict
        Dictionary containing all metrics.
    """

    pred_pose = np.asarray(pred_pose, dtype=np.float64)
    gt_pose = np.asarray(gt_pose, dtype=np.float64)

    if pred_pose.ndim != 2 or pred_pose.shape[1] != 6:
        raise ValueError(
            "pred_pose must have shape (N, 6)."
        )

    if gt_pose.ndim != 2 or gt_pose.shape[1] != 6:
        raise ValueError(
            "gt_pose must have shape (N, 6)."
        )

    if pred_pose.shape != gt_pose.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred_pose.shape}, "
            f"gt={gt_pose.shape}"
        )

    n = pred_pose.shape[0]

    if n == 0:
        raise ValueError("No samples were provided.")

    # ---------------------------------------------------------
    # Translation
    # ---------------------------------------------------------

    pred_t = pred_pose[:, 0:3]
    gt_t = gt_pose[:, 0:3]

    # Eq. (13)
    # Input translation is in meters.
    # Convert prediction residuals to millimetres.
    delta_t_mm = 1000.0 * (pred_t - gt_t)

    abs_t_mm = np.abs(delta_t_mm)

    # Axis-wise statistics used in Tables 1-4
    axis_t_mean = np.mean(abs_t_mm, axis=0)
    axis_t_std = np.std(abs_t_mm, axis=0, ddof=ddof)

    # Eq. (15)
    # Average across all samples AND all three translation axes.
    mae_t = np.mean(abs_t_mm)

    # Eq. (16)
    # Averaging is performed before the square root.
    rmse_t = np.sqrt(np.mean(delta_t_mm ** 2))

    # ---------------------------------------------------------
    # Rotation
    # ---------------------------------------------------------

    pred_r = pred_pose[:, 3:6]
    gt_r = gt_pose[:, 3:6]

    # Diagnostic angle-axis component residuals
    # used as alpha, beta and gamma in Tables 1-4.
    delta_r = pred_r - gt_r
    abs_r = np.abs(delta_r)

    axis_r_mean = np.mean(abs_r, axis=0)
    axis_r_std = np.std(abs_r, axis=0, ddof=ddof)

    # Eq. (14): SO(3) geodesic error
    e_R = so3_geodesic_error(pred_r, gt_r)

    # Tables 1-4
    geodesic_mean = np.mean(e_R)
    geodesic_std = np.std(e_R, ddof=ddof)

    # Eq. (17)
    mae_R = np.mean(e_R)

    # Eq. (18)
    rmse_R = np.sqrt(np.mean(e_R ** 2))

    results = {
        "num_samples": int(n),

        "axis_translation_mm": {
            "x": {
                "mean": float(axis_t_mean[0]),
                "std": float(axis_t_std[0]),
            },
            "y": {
                "mean": float(axis_t_mean[1]),
                "std": float(axis_t_std[1]),
            },
            "z": {
                "mean": float(axis_t_mean[2]),
                "std": float(axis_t_std[2]),
            },
        },

        "angle_axis_component_rad": {
            "alpha": {
                "mean": float(axis_r_mean[0]),
                "std": float(axis_r_std[0]),
            },
            "beta": {
                "mean": float(axis_r_mean[1]),
                "std": float(axis_r_std[1]),
            },
            "gamma": {
                "mean": float(axis_r_mean[2]),
                "std": float(axis_r_std[2]),
            },
        },

        "geodesic_rotation_rad": {
            "mean": float(geodesic_mean),
            "std": float(geodesic_std),
        },

        "aggregate_metrics": {
            "MAE_t_mm": float(mae_t),
            "RMSE_t_mm": float(rmse_t),
            "MAE_R_rad": float(mae_R),
            "RMSE_R_rad": float(rmse_R),
        },
    }

    return results, delta_t_mm, abs_r, e_R


def load_pose_file(path):
    """
    Load Nx6 pose data.

    Supported formats:
        .csv
        .txt
        .npy

    Expected pose order:
        tx, ty, tz, rx, ry, rz
    """

    extension = os.path.splitext(path)[1].lower()

    if extension == ".npy":
        data = np.load(path)

    elif extension == ".csv":
        df = pd.read_csv(path)

        # Preferred format:
        # tx,ty,tz,rx,ry,rz
        if all(col in df.columns for col in POSE_COLUMNS):
            data = df[POSE_COLUMNS].to_numpy(dtype=np.float64)
        else:
            # Fallback: use the first six numeric columns.
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.shape[1] < 6:
                raise ValueError(
                    f"{path} does not contain six numeric pose columns."
                )

            data = numeric_df.iloc[:, :6].to_numpy(dtype=np.float64)

    elif extension == ".txt":
        data = np.loadtxt(path, dtype=np.float64)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Use .csv, .txt, or .npy."
        )

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 6:
        raise ValueError(
            f"{path}: expected 6 pose values per sample, "
            f"but got {data.shape[1]}."
        )

    return data


def save_sample_errors(
    path,
    pred_pose,
    gt_pose,
    delta_t_mm,
    abs_r,
    e_R
):
    """
    Save per-image-pair residuals for reproducibility.
    """

    df = pd.DataFrame({
        "pred_tx": pred_pose[:, 0],
        "pred_ty": pred_pose[:, 1],
        "pred_tz": pred_pose[:, 2],
        "pred_rx": pred_pose[:, 3],
        "pred_ry": pred_pose[:, 4],
        "pred_rz": pred_pose[:, 5],

        "gt_tx": gt_pose[:, 0],
        "gt_ty": gt_pose[:, 1],
        "gt_tz": gt_pose[:, 2],
        "gt_rx": gt_pose[:, 3],
        "gt_ry": gt_pose[:, 4],
        "gt_rz": gt_pose[:, 5],

        "abs_x_mm": np.abs(delta_t_mm[:, 0]),
        "abs_y_mm": np.abs(delta_t_mm[:, 1]),
        "abs_z_mm": np.abs(delta_t_mm[:, 2]),

        "abs_alpha_rad": abs_r[:, 0],
        "abs_beta_rad": abs_r[:, 1],
        "abs_gamma_rad": abs_r[:, 2],

        "eR_rad": e_R,
    })

    df.to_csv(path, index=False)


def print_results(results):
    """
    Print results in a form corresponding to the manuscript tables.
    """

    t = results["axis_translation_mm"]
    r = results["angle_axis_component_rad"]
    geo = results["geodesic_rotation_rad"]
    agg = results["aggregate_metrics"]

    print("\n==============================================")
    print("CSSGNet Relative-Pose Evaluation")
    print("==============================================")
    print(f"Number of image pairs: {results['num_samples']}")

    print("\nTables 1-4 style:")
    print("----------------------------------------------")

    print(
        f"x      : {t['x']['mean']:.4f} "
        f"± {t['x']['std']:.4f} mm"
    )

    print(
        f"y      : {t['y']['mean']:.4f} "
        f"± {t['y']['std']:.4f} mm"
    )

    print(
        f"z      : {t['z']['mean']:.4f} "
        f"± {t['z']['std']:.4f} mm"
    )

    print(
        f"alpha  : {r['alpha']['mean']:.4f} "
        f"± {r['alpha']['std']:.4f} rad"
    )

    print(
        f"beta   : {r['beta']['mean']:.4f} "
        f"± {r['beta']['std']:.4f} rad"
    )

    print(
        f"gamma  : {r['gamma']['mean']:.4f} "
        f"± {r['gamma']['std']:.4f} rad"
    )

    print(
        f"e_R    : {geo['mean']:.4f} "
        f"± {geo['std']:.4f} rad"
    )

    print("\nTable 5 style (Eqs. 15-18):")
    print("----------------------------------------------")

    print(f"MAE_t  : {agg['MAE_t_mm']:.4f} mm")
    print(f"RMSE_t : {agg['RMSE_t_mm']:.4f} mm")
    print(f"MAE_R  : {agg['MAE_R_rad']:.4f} rad")
    print(f"RMSE_R : {agg['RMSE_R_rad']:.4f} rad")

    print("==============================================\n")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate the relative-pose evaluation metrics "
            "reported in the CSSGNet paper."
        )
    )

    parser.add_argument(
        "--pred",
        required=True,
        type=str,
        help=(
            "Prediction file (.csv/.txt/.npy), "
            "pose order: tx ty tz rx ry rz."
        )
    )

    parser.add_argument(
        "--gt",
        required=True,
        type=str,
        help=(
            "Ground-truth file (.csv/.txt/.npy), "
            "pose order: tx ty tz rx ry rz."
        )
    )

    parser.add_argument(
        "--output_json",
        default="evaluation_results.json",
        type=str,
        help="Path used to save aggregate evaluation results."
    )

    parser.add_argument(
        "--output_samples",
        default="sample_errors.csv",
        type=str,
        help="Path used to save per-sample errors."
    )

    parser.add_argument(
        "--ddof",
        default=0,
        type=int,
        choices=[0, 1],
        help=(
            "Standard-deviation convention. "
            "0 = population std; 1 = sample std."
        )
    )

    args = parser.parse_args()

    pred_pose = load_pose_file(args.pred)
    gt_pose = load_pose_file(args.gt)

    results, delta_t_mm, abs_r, e_R = calculate_metrics(
        pred_pose,
        gt_pose,
        ddof=args.ddof
    )

    print_results(results)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    save_sample_errors(
        args.output_samples,
        pred_pose,
        gt_pose,
        delta_t_mm,
        abs_r,
        e_R
    )

    print(f"Aggregate results saved to: {args.output_json}")
    print(f"Per-sample errors saved to: {args.output_samples}")


if __name__ == "__main__":
    main()