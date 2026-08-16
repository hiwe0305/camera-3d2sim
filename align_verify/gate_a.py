# align_verify/gate_a.py
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_thresholds(gates_yaml_path: Path) -> dict:
    return yaml.safe_load(gates_yaml_path.read_text())["gate_a"]


def check_scale(measurements: list[dict], max_err_pct: float, min_measurements: int) -> dict:
    if len(measurements) < min_measurements:
        return {"pass": False, "reason": f"need >= {min_measurements} measurements, got {len(measurements)}"}
    errs = []
    for m in measurements:
        err_pct = abs(m["cloud_m"] - m["tape_m"]) / m["tape_m"] * 100.0
        m["err_pct"] = err_pct
        errs.append(err_pct)
    worst = max(errs)
    return {"pass": worst <= max_err_pct, "worst_err_pct": worst, "measurements": measurements}


def check_gravity(residual_deg: float, floor_inlier_ratio: float, max_residual_deg: float,
                   min_inlier_ratio: float) -> dict:
    ok = residual_deg < max_residual_deg and floor_inlier_ratio >= min_inlier_ratio
    return {"pass": ok, "residual_deg": residual_deg, "floor_inlier_ratio": floor_inlier_ratio}


def _angular_velocities_deg_s(poses):
    vels = []
    for a, b in zip(poses, poses[1:]):
        dt = b["timestamp"] - a["timestamp"]
        if dt <= 0:
            continue
        Ra = Rotation.from_quat(a["q"])
        Rb = Rotation.from_quat(b["q"])
        rel = Ra.inv() * Rb
        angle_deg = np.degrees(rel.magnitude())
        vels.append(angle_deg / dt)
    return np.array(vels)


def check_trajectory(poses: list[dict], max_gap_s: float, max_p95_angular_vel_deg_s: float,
                      max_blur_ratio: float, blur_flags: list[bool]) -> dict:
    ts = np.array([p["timestamp"] for p in poses])
    if np.isnan(ts).any():
        return {"pass": False, "reason": "NaN timestamp"}
    for p in poses:
        if np.isnan(p["t"]).any() or np.isnan(p["q"]).any():
            return {"pass": False, "reason": "NaN in pose"}

    gaps = np.diff(ts)
    max_gap = float(gaps.max()) if len(gaps) else 0.0
    angular_vels = _angular_velocities_deg_s(poses)
    p95_angular_vel = float(np.percentile(angular_vels, 95)) if len(angular_vels) else 0.0
    blur_ratio = sum(blur_flags) / len(blur_flags) if blur_flags else 0.0

    ok = (max_gap <= max_gap_s and p95_angular_vel <= max_p95_angular_vel_deg_s
          and blur_ratio <= max_blur_ratio)
    return {"pass": ok, "max_gap_s": max_gap, "p95_angular_vel_deg_s": p95_angular_vel,
            "blur_ratio": blur_ratio}


def _floor_headings_deg(poses):
    headings = []
    for p in poses:
        fwd = Rotation.from_quat(p["q"]).apply([0, 0, 1])
        headings.append(np.degrees(np.arctan2(fwd[1], fwd[0])) % 360)
    return np.array(headings)


def _max_pairwise_angular_spread_deg(headings_deg):
    diffs = []
    for i in range(len(headings_deg)):
        for j in range(i + 1, len(headings_deg)):
            d = abs(headings_deg[i] - headings_deg[j]) % 360
            diffs.append(min(d, 360 - d))
    return max(diffs) if diffs else 0.0


def check_coverage(poses: list[dict], floor_points_xy: np.ndarray, cell_size_m: float,
                    min_angle_spread_deg: float) -> dict:
    if floor_points_xy.size == 0 or not poses:
        return {"pass": False, "reason": "no floor points or poses"}

    mins = floor_points_xy.min(axis=0)
    cells = set(map(tuple, np.floor((floor_points_xy - mins) / cell_size_m).astype(int)))
    positions = np.array([p["t"][:2] for p in poses])
    headings = _floor_headings_deg(poses)
    radius = 1.5 * cell_size_m

    uncovered = []
    for cell in cells:
        center = mins + (np.array(cell) + 0.5) * cell_size_m
        dists = np.linalg.norm(positions - center, axis=1)
        nearby_idx = np.where(dists <= radius)[0]
        if len(nearby_idx) < 2:
            uncovered.append(cell)
            continue
        spread = _max_pairwise_angular_spread_deg(headings[nearby_idx])
        if spread < min_angle_spread_deg:
            uncovered.append(cell)

    coverage_ratio = 1 - len(uncovered) / len(cells)
    return {"pass": len(uncovered) == 0, "coverage_ratio": coverage_ratio,
            "uncovered_cells": len(uncovered), "total_cells": len(cells)}


def run_gate_a(manifest: dict, poses: list[dict], blur_flags: list[bool],
                floor_points_xy: np.ndarray, thresholds: dict) -> dict:
    scale = check_scale(manifest["ground_truth"]["measurements"],
                         thresholds["scale_max_err_pct"], thresholds["scale_min_measurements"])
    gravity = check_gravity(manifest["align"]["gravity_residual_deg"],
                             manifest["align"]["floor_inlier_ratio"],
                             thresholds["gravity_max_residual_deg"],
                             thresholds["gravity_min_floor_inlier_ratio"])
    trajectory = check_trajectory(poses, thresholds["trajectory_max_gap_s"],
                                   thresholds["trajectory_max_p95_angular_vel_deg_s"],
                                   thresholds["trajectory_max_blur_ratio"], blur_flags)
    coverage = check_coverage(poses, floor_points_xy, thresholds["coverage_cell_size_m"],
                               thresholds["coverage_min_angle_spread_deg"])

    checks = {"scale": scale, "gravity": gravity, "trajectory": trajectory, "coverage": coverage}
    verdict = "APPROVED" if all(c["pass"] for c in checks.values()) else "REJECTED"
    scorecard = {"verdict": verdict, "checks": checks}
    manifest["gate"]["gateA"] = scorecard
    return scorecard
