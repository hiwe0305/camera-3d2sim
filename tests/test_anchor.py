import numpy as np
from scipy.spatial.transform import Rotation
from align_verify.anchor import (
    ANCHOR_TAG_ID, detections_to_world, average_transform, compute_anchor_from_world,
)

def test_detections_to_world_filters_tag_id_and_known_frames():
    T_wc = {0: np.eye(4), 1: np.eye(4)}
    detections = [
        {"frame_index": 0, "tag_id": ANCHOR_TAG_ID, "R_cam_tag": np.eye(3), "t_cam_tag": np.array([1.0, 0, 0])},
        {"frame_index": 1, "tag_id": 99, "R_cam_tag": np.eye(3), "t_cam_tag": np.array([0, 0, 0])},
        {"frame_index": 5, "tag_id": ANCHOR_TAG_ID, "R_cam_tag": np.eye(3), "t_cam_tag": np.array([0, 0, 0])},
    ]
    result = detections_to_world(detections, T_wc)
    assert len(result) == 1
    np.testing.assert_allclose(result[0][:3, 3], [1.0, 0, 0])

def test_average_transform_averages_translation_and_rotation():
    T1 = np.eye(4); T1[:3, 3] = [1.0, 0.0, 0.0]
    T2 = np.eye(4); T2[:3, 3] = [1.2, 0.0, 0.0]
    avg = average_transform([T1, T2])
    np.testing.assert_allclose(avg[:3, 3], [1.1, 0.0, 0.0])
    np.testing.assert_allclose(avg[:3, :3], np.eye(3), atol=1e-9)

def test_average_transform_raises_on_empty_list():
    import pytest
    with pytest.raises(ValueError):
        average_transform([])

def test_compute_anchor_from_world_is_inverse_of_tag_pose():
    T_wc = {0: np.eye(4)}
    detections = [{"frame_index": 0, "tag_id": ANCHOR_TAG_ID, "R_cam_tag": np.eye(3),
                   "t_cam_tag": np.array([2.0, 0.0, 0.0])}]
    T_anchor_from_world = compute_anchor_from_world(detections, T_wc)
    # tag is at world (2,0,0); anchor frame origin is the tag, so a world point at (2,0,0)
    # must map to the anchor origin (0,0,0)
    p_world = np.array([2.0, 0.0, 0.0])
    p_anchor = T_anchor_from_world[:3, :3] @ p_world + T_anchor_from_world[:3, 3]
    np.testing.assert_allclose(p_anchor, [0, 0, 0], atol=1e-9)

def test_compute_anchor_from_world_returns_none_when_tag_never_seen():
    result = compute_anchor_from_world([], {0: np.eye(4)})
    assert result is None
