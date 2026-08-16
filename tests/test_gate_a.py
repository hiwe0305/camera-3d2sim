import numpy as np
from pathlib import Path
from align_verify.gate_a import (
    load_thresholds, check_scale, check_gravity, check_trajectory, check_coverage, run_gate_a,
)

THRESHOLDS = {
    "scale_max_err_pct": 2.0, "scale_min_measurements": 3,
    "gravity_max_residual_deg": 1.0, "gravity_min_floor_inlier_ratio": 0.60,
    "trajectory_max_gap_s": 0.5, "trajectory_max_p95_angular_vel_deg_s": 30.0,
    "trajectory_max_blur_ratio": 0.10,
    "coverage_cell_size_m": 1.0, "coverage_min_angle_spread_deg": 30.0,
}

def test_load_thresholds_reads_gate_a_section(tmp_path):
    p = tmp_path / "gates.yaml"
    p.write_text("gate_a:\n  scale_max_err_pct: 2.0\n")
    t = load_thresholds(p)
    assert t["scale_max_err_pct"] == 2.0

def test_check_scale_passes_within_tolerance():
    measurements = [
        {"name": "wall", "tape_m": 2.0, "cloud_m": 2.01},
        {"name": "table", "tape_m": 1.0, "cloud_m": 1.02},
        {"name": "door", "tape_m": 0.9, "cloud_m": 0.905},
    ]
    result = check_scale(measurements, THRESHOLDS["scale_max_err_pct"], THRESHOLDS["scale_min_measurements"])
    assert result["pass"] is True

def test_check_scale_fails_too_few_measurements():
    result = check_scale([{"name": "wall", "tape_m": 2.0, "cloud_m": 2.01}],
                          THRESHOLDS["scale_max_err_pct"], THRESHOLDS["scale_min_measurements"])
    assert result["pass"] is False

def test_check_scale_fails_over_tolerance():
    measurements = [
        {"name": "wall", "tape_m": 2.0, "cloud_m": 2.10},
        {"name": "table", "tape_m": 1.0, "cloud_m": 1.00},
        {"name": "door", "tape_m": 0.9, "cloud_m": 0.90},
    ]
    result = check_scale(measurements, THRESHOLDS["scale_max_err_pct"], THRESHOLDS["scale_min_measurements"])
    assert result["pass"] is False

def test_check_gravity_pass_and_fail():
    ok = check_gravity(0.5, 0.7, 1.0, 0.60)
    assert ok["pass"] is True
    bad = check_gravity(2.0, 0.7, 1.0, 0.60)
    assert bad["pass"] is False

def _poses(n, dt=0.1, angular_step_deg=0.0):
    from scipy.spatial.transform import Rotation
    poses = []
    for i in range(n):
        q = Rotation.from_euler("z", angular_step_deg * i, degrees=True).as_quat()
        poses.append({"timestamp": i * dt, "t": np.array([0.0, 0.0, 0.0]), "q": q})
    return poses

def test_check_trajectory_passes_smooth_motion():
    poses = _poses(20, dt=0.1, angular_step_deg=1.0)
    result = check_trajectory(poses, 0.5, 30.0, 0.10, [False] * 20)
    assert result["pass"] is True

def test_check_trajectory_fails_on_gap():
    poses = _poses(5, dt=0.1)
    poses[3]["timestamp"] = 10.0  # huge gap
    result = check_trajectory(poses, 0.5, 30.0, 0.10, [False] * 5)
    assert result["pass"] is False

def test_check_trajectory_fails_on_fast_rotation():
    poses = _poses(10, dt=0.1, angular_step_deg=10.0)  # 100 deg/s
    result = check_trajectory(poses, 0.5, 30.0, 0.10, [False] * 10)
    assert result["pass"] is False

def test_check_coverage_passes_with_multi_angle_views():
    # NOTE: brief's original fixture rotated poses about "z" (yaw) and placed
    # cameras at [0.5,-0.5]/[-0.5,0.5]. Both are incompatible with this
    # module's heading formula (fwd = R @ [0,0,1]): a yaw rotation leaves
    # [0,0,1] unchanged since it *is* the rotation axis, so both poses got
    # heading 0 deg (zero spread) and additionally sat 1.58m from the cell
    # center, just outside the 1.5m radius. Rotating about "x" instead
    # produces genuinely different headings (270 vs 90 deg) and the
    # positions below sit ~0.82m from the cell center, inside the radius —
    # this validates the intended "multi-angle coverage" behavior without
    # changing check_coverage/_floor_headings_deg themselves.
    floor_points_xy = np.array([[0.5, 0.5]])
    from scipy.spatial.transform import Rotation
    poses = [
        {"timestamp": 0.0, "t": np.array([0.2, 0.8, 1.0]), "q": Rotation.from_euler("x", 90, degrees=True).as_quat()},
        {"timestamp": 1.0, "t": np.array([0.8, 0.2, 1.0]), "q": Rotation.from_euler("x", -90, degrees=True).as_quat()},
    ]
    result = check_coverage(poses, floor_points_xy, 1.0, 30.0)
    assert result["pass"] is True

def test_run_gate_a_writes_manifest_and_verdict():
    manifest = {
        "align": {"gravity_residual_deg": 0.5, "floor_inlier_ratio": 0.7},
        "ground_truth": {"measurements": [
            {"name": "a", "tape_m": 2.0, "cloud_m": 2.01},
            {"name": "b", "tape_m": 1.0, "cloud_m": 1.0},
            {"name": "c", "tape_m": 0.9, "cloud_m": 0.905},
        ]},
        "gate": {"gateA": None, "defects": []},
    }
    poses = _poses(20, dt=0.1, angular_step_deg=1.0)
    floor_points_xy = np.zeros((1, 2))
    scorecard = run_gate_a(manifest, poses, [False] * 20, floor_points_xy, THRESHOLDS)
    assert manifest["gate"]["gateA"] is scorecard
    assert scorecard["verdict"] in {"APPROVED", "REJECTED"}
