import numpy as np
import imageio.v3 as iio
from capture.session import create_capture_session
from capture.record import Frame, record_session
from capture.manifest import new_manifest

def _fake_frames(n):
    for i in range(n):
        color = np.full((4, 4, 3), i, dtype=np.uint8)
        depth = np.full((4, 4), i * 1000, dtype=np.uint16)
        yield Frame(index=i, timestamp=i * 0.1, color=color, depth=depth)

def test_record_session_writes_files_and_updates_manifest(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    manifest = new_manifest(session.root.name, "lab_room_a", 1, "sop-v1", "p01")
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}

    # Create frames with known depth values for verification
    frames = list(_fake_frames(3))
    result = record_session(iter(frames), intrinsics, session, manifest)

    assert result is manifest
    assert manifest["hardware"]["intrinsics"] == intrinsics
    assert manifest["capture_stats"]["n_frames"] == 3
    assert manifest["capture_stats"]["duration_s"] == 0.2

    for i in range(3):
        rgb_path = session.raw_rgb / f"{i:06d}.png"
        depth_path = session.raw_depth / f"{i:06d}.png"

        assert rgb_path.exists()
        assert depth_path.exists()

        # Verify depth PNG preserves 16-bit depth
        depth_read = iio.imread(depth_path)
        assert depth_read.dtype == np.uint16, f"Expected uint16 depth, got {depth_read.dtype}"

        expected_depth = frames[i].depth
        assert np.array_equal(depth_read, expected_depth), \
            f"Depth frame {i}: read values don't match written values"

def test_record_session_respects_max_frames(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    manifest = new_manifest(session.root.name, "lab_room_a", 1, "sop-v1", "p01")
    intrinsics = {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 4, "height": 4}

    # Create frames to verify depth values
    frames = list(_fake_frames(10))
    record_session(iter(frames), intrinsics, session, manifest, max_frames=3)

    assert manifest["capture_stats"]["n_frames"] == 3
    assert not (session.raw_rgb / "000003.png").exists()

    # Verify written frames have correct 16-bit depth
    for i in range(3):
        depth_path = session.raw_depth / f"{i:06d}.png"
        depth_read = iio.imread(depth_path)
        assert depth_read.dtype == np.uint16, f"Expected uint16 depth, got {depth_read.dtype}"
        assert np.array_equal(depth_read, frames[i].depth)
