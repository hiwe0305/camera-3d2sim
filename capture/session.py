from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path


@dataclass
class SessionPaths:
    root: Path
    raw_rgb: Path
    raw_depth: Path
    raw_calib: Path
    slam_dir: Path
    align_dir: Path
    eval_dir: Path
    manifest_path: Path
    capture_log_path: Path


def next_capture_version(env_dir: Path, day: str) -> int:
    captures_dir = env_dir / "captures"
    if not captures_dir.exists():
        return 1
    prefix = f"{day}_v"
    versions = []
    for p in captures_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix):]
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=0) + 1


def create_capture_session(scenes_root: Path, env_id: str, day: str | None = None) -> SessionPaths:
    day = day or date_cls.today().isoformat()
    env_dir = scenes_root / env_id
    version = next_capture_version(env_dir, day)
    root = env_dir / "captures" / f"{day}_v{version:02d}"
    raw_rgb = root / "raw" / "rgb"
    raw_depth = root / "raw" / "depth"
    raw_calib = root / "raw" / "calib"
    slam_dir = root / "slam"
    align_dir = root / "align"
    eval_dir = root / "eval"
    for d in [raw_rgb, raw_depth, raw_calib, slam_dir, align_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return SessionPaths(
        root=root,
        raw_rgb=raw_rgb,
        raw_depth=raw_depth,
        raw_calib=raw_calib,
        slam_dir=slam_dir,
        align_dir=align_dir,
        eval_dir=eval_dir,
        manifest_path=root / "manifest.json",
        capture_log_path=root / "capture_log.md",
    )
