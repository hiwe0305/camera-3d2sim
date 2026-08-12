import numpy as np
from pathlib import Path
from capture.session import create_capture_session
from capture.record import Frame, record_session
from capture.manifest import new_manifest

def _fake_frames(n):
    for i in range(n):
        color = np.full((4, 4, 3), i, dtype=np.uint8)
        depth = np.full((4, 4), i * 100, dtype=np.uint16)
        yield Frame(index=i, timestamp=i * 0.1, color=color, depth=depth)

def test_record_session_writes_files_and_updates_manifest(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    manifest = new_manifest(session.root.name, "lab_room_a", 1, "sop-v1", "p01")
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}

    result = record_session(_fake_frames(3), intrinsics, session, manifest)

    assert result is manifest
    assert manifest["hardware"]["intrinsics"] == intrinsics
    assert manifest["capture_stats"]["n_frames"] == 3
    assert manifest["capture_stats"]["duration_s"] == 0.2
    for i in range(3):
        assert (session.raw_rgb / f"{i:06d}.png").exists()
        assert (session.raw_depth / f"{i:06d}.png").exists()

def test_record_session_respects_max_frames(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    manifest = new_manifest(session.root.name, "lab_room_a", 1, "sop-v1", "p01")
    intrinsics = {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 4, "height": 4}

    record_session(_fake_frames(10), intrinsics, session, manifest, max_frames=3)

    assert manifest["capture_stats"]["n_frames"] == 3
    assert not (session.raw_rgb / "000003.png").exists()
