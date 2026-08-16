import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_meta" / "scripts"))
from reindex import scan_manifests, build_catalog, render_markdown, reindex


def _write_manifest(scenes_root, env_id, capture_id, verdict, scale_err_pct):
    d = scenes_root / env_id / "captures" / capture_id
    d.mkdir(parents=True)
    manifest = {
        "env_id": env_id, "capture_id": capture_id, "started_at": "2026-08-12T09:00:00",
        "gate": {"gateA": {"verdict": verdict, "checks": {"scale": {"worst_err_pct": scale_err_pct}}}},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))


def test_scan_manifests_finds_all_sessions(tmp_path):
    _write_manifest(tmp_path, "lab_room_a", "2026-08-12_v01", "APPROVED", 1.2)
    _write_manifest(tmp_path, "lab_room_b", "2026-08-12_v01", "REJECTED", 5.0)
    manifests = scan_manifests(tmp_path)
    assert len(manifests) == 2


def test_build_catalog_extracts_gate_a_summary(tmp_path):
    _write_manifest(tmp_path, "lab_room_a", "2026-08-12_v01", "APPROVED", 1.2)
    catalog = build_catalog(tmp_path)
    row = catalog["captures"][0]
    assert row["env_id"] == "lab_room_a"
    assert row["verdict"] == "APPROVED"
    assert row["scale_err_pct"] == 1.2


def test_render_markdown_includes_header_and_rows(tmp_path):
    _write_manifest(tmp_path, "lab_room_a", "2026-08-12_v01", "APPROVED", 1.2)
    catalog = build_catalog(tmp_path)
    md = render_markdown(catalog)
    assert "lab_room_a" in md
    assert "APPROVED" in md


def test_reindex_writes_catalog_files(tmp_path):
    _write_manifest(tmp_path, "lab_room_a", "2026-08-12_v01", "APPROVED", 1.2)
    (tmp_path / "_meta").mkdir()
    reindex(tmp_path)
    assert (tmp_path / "_meta" / "catalog.json").exists()
    assert (tmp_path / "_meta" / "catalog.md").exists()
