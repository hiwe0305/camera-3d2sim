import numpy as np
from scipy.spatial.transform import Rotation


def fit_floor_plane(cloud, distance_threshold=0.02, ransac_n=3, num_iterations=2000):
    plane_model, inliers = cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_ratio = len(inliers) / max(len(cloud.points), 1)
    return np.array(plane_model), inlier_ratio


def gravity_alignment_transform(plane_model: np.ndarray) -> tuple[np.ndarray, float]:
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
        d = -d

    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
    residual_deg = float(np.degrees(np.arccos(dot)))

    axis = np.cross(normal, z_axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(dot)
        R = Rotation.from_rotvec(axis * angle).as_matrix()

    p0 = -d * normal  # foot of perpendicular from origin to the plane
    t = -R @ p0

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, residual_deg


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def apply_transform_to_poses(T: np.ndarray, poses: list[dict]) -> list[dict]:
    R_t = T[:3, :3]
    t_t = T[:3, 3]
    out = []
    for p in poses:
        new_t = R_t @ p["t"] + t_t
        new_R = R_t @ Rotation.from_quat(p["q"]).as_matrix()
        out.append({
            "timestamp": p["timestamp"],
            "t": new_t,
            "q": Rotation.from_matrix(new_R).as_quat(),
        })
    return out
