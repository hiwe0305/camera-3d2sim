import json
import os
import stat
import pytest
from pathlib import Path
from ingest.package import sha256_file, collect_artifacts, finalize_session

def test_sha256_file_is_deterministic(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world")
    assert sha256_file(f) == sha256_file(f)
    assert len(sha256_file(f)) == 64

def test_collect_artifacts_excludes_manifest(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "frame.png").write_bytes(b"x")
    (tmp_path / "manifest.json").write_text("{}")
    artifacts = collect_artifacts(tmp_path)
    paths = [a["path"] for a in artifacts]
    assert "raw/frame.png" in paths
    assert "manifest.json" not in paths

def test_finalize_session_writes_manifest_and_makes_readonly(tmp_path):
    (tmp_path / "raw").mkdir()
    target = tmp_path / "raw" / "frame.png"
    target.write_bytes(b"x")
    manifest = {"capture_id": "2026-08-12_v01", "artifacts": []}

    result = finalize_session(tmp_path, manifest)

    assert result["artifacts"]
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["capture_id"] == "2026-08-12_v01"

    mode = target.stat().st_mode
    assert not (mode & stat.S_IWUSR)
    with pytest.raises(PermissionError):
        target.write_bytes(b"y")
