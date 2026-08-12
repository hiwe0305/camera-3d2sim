import json
from pathlib import Path

SCHEMA_VERSION = 1


def new_manifest(capture_id: str, env_id: str, capture_version: int, sop_version: str,
                  capture_profile_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "env_id": env_id,
        "capture_version": capture_version,
        "started_at": None,
        "ended_at": None,
        "sop_version": sop_version,
        "capture_profile_id": capture_profile_id,
        "hardware": {},
        "conditions": {},
        "capture_stats": {},
        "align": {},
        "ground_truth": {"measurements": [], "holdout_frames": []},
        "artifacts": [],
        "gate": {"gateA": None, "defects": []},
    }


def save_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())
