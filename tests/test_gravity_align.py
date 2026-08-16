import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from align_verify.gravity_align import (
    fit_floor_plane, gravity_alignment_transform, apply_transform, apply_transform_to_poses,
)

def _tilted_floor_cloud(tilt_deg=5.0, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-2, 2, size=(n, 2))
    z = np.zeros(n)
    pts = np.column_stack([xy, z])
    R = Rotation.from_euler("x", tilt_deg, degrees=True).as_matrix()
    pts = pts @ R.T
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    return cloud

def test_fit_floor_plane_finds_high_inlier_ratio():
    cloud = _tilted_floor_cloud()
    plane_model, inlier_ratio = fit_floor_plane(cloud)
    assert inlier_ratio > 0.95

def test_gravity_alignment_transform_recovers_tilt_angle():
    cloud = _tilted_floor_cloud(tilt_deg=5.0)
    plane_model, _ = fit_floor_plane(cloud)
    T, residual_deg = gravity_alignment_transform(plane_model)
    assert abs(residual_deg - 5.0) < 0.5

def test_apply_transform_flattens_floor_to_z_zero():
    cloud = _tilted_floor_cloud(tilt_deg=5.0)
    plane_model, _ = fit_floor_plane(cloud)
    T, _ = gravity_alignment_transform(plane_model)
    pts = np.asarray(cloud.points)
    transformed = apply_transform(T, pts)
    assert np.abs(transformed[:, 2]).max() < 0.01

def test_apply_transform_to_poses_translates_and_rotates():
    R_T = Rotation.from_euler("z", 90.0, degrees=True)
    T = np.eye(4)
    T[:3, :3] = R_T.as_matrix()
    T[:3, 3] = [1.0, 0.0, 0.0]

    R_pose = Rotation.from_euler("x", 90.0, degrees=True)
    poses = [{
        "timestamp": 0.0,
        "t": np.array([2.0, 0.0, 0.0]),
        "q": R_pose.as_quat(),
    }]
    out = apply_transform_to_poses(T, poses)

    expected_t = R_T.apply([2.0, 0.0, 0.0]) + [1.0, 0.0, 0.0]
    expected_R = R_T * R_pose  # composed independently of the implementation

    np.testing.assert_allclose(out[0]["t"], expected_t, atol=1e-9)
    # Quaternions may differ by sign (q and -q represent the same rotation).
    q_out = out[0]["q"]
    q_expected = expected_R.as_quat()
    assert (
        np.allclose(q_out, q_expected, atol=1e-9)
        or np.allclose(q_out, -q_expected, atol=1e-9)
    )
