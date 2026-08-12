import json
from capture.manifest import new_manifest, save_manifest, load_manifest, SCHEMA_VERSION

def test_new_manifest_has_required_top_level_keys():
    m = new_manifest("2026-08-12_v01", "lab_room_a", 1, "sop-v1", "p01")
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["capture_id"] == "2026-08-12_v01"
    assert m["env_id"] == "lab_room_a"
    assert m["capture_version"] == 1
    assert m["sop_version"] == "sop-v1"
    assert m["capture_profile_id"] == "p01"
    for key in ["hardware", "conditions", "capture_stats", "align"]:
        assert m[key] == {}
    assert m["ground_truth"] == {"measurements": [], "holdout_frames": []}
    assert m["artifacts"] == []
    assert m["gate"] == {"gateA": None, "defects": []}

def test_save_and_load_manifest_roundtrip(tmp_path):
    m = new_manifest("2026-08-12_v01", "lab_room_a", 1, "sop-v1", "p01")
    m["hardware"]["serial"] = "12345"
    path = tmp_path / "manifest.json"
    save_manifest(m, path)
    loaded = load_manifest(path)
    assert loaded == m
    assert json.loads(path.read_text())["hardware"]["serial"] == "12345"
