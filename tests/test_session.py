from pathlib import Path
from capture.session import create_capture_session, next_capture_version

def test_next_capture_version_starts_at_1(tmp_path):
    assert next_capture_version(tmp_path / "lab_room_a", "2026-08-12") == 1

def test_next_capture_version_increments(tmp_path):
    env_dir = tmp_path / "lab_room_a"
    (env_dir / "captures" / "2026-08-12_v01").mkdir(parents=True)
    (env_dir / "captures" / "2026-08-12_v02").mkdir(parents=True)
    assert next_capture_version(env_dir, "2026-08-12") == 3

def test_next_capture_version_is_per_day(tmp_path):
    env_dir = tmp_path / "lab_room_a"
    (env_dir / "captures" / "2026-08-11_v01").mkdir(parents=True)
    assert next_capture_version(env_dir, "2026-08-12") == 1

def test_create_capture_session_layout(tmp_path):
    paths = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    assert paths.root == tmp_path / "lab_room_a" / "captures" / "2026-08-12_v01"
    for d in [paths.raw_rgb, paths.raw_depth, paths.raw_calib, paths.slam_dir,
              paths.align_dir, paths.eval_dir]:
        assert d.is_dir()
    assert paths.manifest_path == paths.root / "manifest.json"
    assert paths.capture_log_path == paths.root / "capture_log.md"

def test_create_capture_session_increments_version(tmp_path):
    create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    paths2 = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    assert paths2.root.name == "2026-08-12_v02"
