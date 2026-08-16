import numpy as np
from scipy.spatial.transform import Rotation

ANCHOR_TAG_ID = 0


def detections_to_world(detections: list[dict], T_world_from_camera_by_frame: dict) -> list[np.ndarray]:
    results = []
    for det in detections:
        if det["tag_id"] != ANCHOR_TAG_ID:
            continue
        fi = det["frame_index"]
        if fi not in T_world_from_camera_by_frame:
            continue
        T_wc = T_world_from_camera_by_frame[fi]
        T_ct = np.eye(4)
        T_ct[:3, :3] = det["R_cam_tag"]
        T_ct[:3, 3] = det["t_cam_tag"]
        results.append(T_wc @ T_ct)
    return results


def average_transform(T_list: list[np.ndarray]) -> np.ndarray:
    if not T_list:
        raise ValueError("no transforms to average")
    ts = np.array([T[:3, 3] for T in T_list])
    quats = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_list])
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(ref, quats[i]) < 0:
            quats[i] = -quats[i]
    mean_q = quats.mean(axis=0)
    mean_q = mean_q / np.linalg.norm(mean_q)

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(mean_q).as_matrix()
    T[:3, 3] = ts.mean(axis=0)
    return T


def compute_anchor_from_world(detections: list[dict], T_world_from_camera_by_frame: dict) -> np.ndarray | None:
    samples = detections_to_world(detections, T_world_from_camera_by_frame)
    if not samples:
        return None
    T_world_from_tag = average_transform(samples)
    return np.linalg.inv(T_world_from_tag)


def detect_tags_in_frame(gray_image: np.ndarray, intrinsics: dict, tag_size_m: float) -> list[dict]:
    from pupil_apriltags import Detector

    detector = Detector(families="tag36h11")
    camera_params = (intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"])
    results = detector.detect(gray_image, estimate_tag_pose=True, camera_params=camera_params,
                               tag_size=tag_size_m)
    return [
        {"frame_index": None, "tag_id": r.tag_id, "R_cam_tag": r.pose_R, "t_cam_tag": r.pose_t.flatten()}
        for r in results
    ]
