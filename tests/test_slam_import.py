import numpy as np
import pytest
from pathlib import Path
from capture.session import create_capture_session
from capture.slam_import import parse_tum_poses, import_slam_output

TUM_SAMPLE = """# comment line
0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.1 1.0 0.0 0.0 0.0 0.0 0.0 1.0
"""

def test_parse_tum_poses(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text(TUM_SAMPLE)
    poses = parse_tum_poses(p)
    assert len(poses) == 2
    assert poses[0]["timestamp"] == 0.0
    np.testing.assert_array_equal(poses[1]["t"], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(poses[1]["q"], [0.0, 0.0, 0.0, 1.0])

def test_parse_tum_poses_rejects_bad_row_length(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text("0.0 1.0 2.0\n")
    with pytest.raises(ValueError):
        parse_tum_poses(p)

def test_import_slam_output_copies_and_returns_poses(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    poses_src = tmp_path / "rtabmap_poses.txt"
    poses_src.write_text(TUM_SAMPLE)
    cloud_src = tmp_path / "rtabmap_cloud.ply"
    cloud_src.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")

    result = import_slam_output(poses_src, cloud_src, session)

    assert result["n_poses"] == 2
    assert len(result["poses"]) == 2
    assert (session.slam_dir / "trajectory.tum").exists()
    assert (session.slam_dir / "cloud.ply").exists()
