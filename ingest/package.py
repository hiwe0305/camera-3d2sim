import hashlib
import json
import stat
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(session_root: Path) -> list[dict]:
    artifacts = []
    for p in sorted(session_root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            artifacts.append({
                "path": str(p.relative_to(session_root)),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            })
    return artifacts


def finalize_session(session_root: Path, manifest: dict) -> dict:
    manifest["artifacts"] = collect_artifacts(session_root)
    manifest_path = session_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for p in session_root.rglob("*"):
        if p.is_file():
            mode = p.stat().st_mode
            p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    return manifest
