import shutil
import numpy as np
from pathlib import Path


def parse_tum_poses(path: Path) -> list[dict]:
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"expected 8 columns (TUM format), got {len(parts)}: {line!r}")
        ts, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in parts)
        poses.append({
            "timestamp": ts,
            "t": np.array([tx, ty, tz]),
            "q": np.array([qx, qy, qz, qw]),
        })
    return poses


def import_slam_output(rtabmap_poses_path: Path, rtabmap_cloud_path: Path, session) -> dict:
    poses = parse_tum_poses(rtabmap_poses_path)
    shutil.copy(rtabmap_poses_path, session.slam_dir / "trajectory.tum")
    shutil.copy(rtabmap_cloud_path, session.slam_dir / "cloud.ply")
    return {"poses": poses, "n_poses": len(poses)}
