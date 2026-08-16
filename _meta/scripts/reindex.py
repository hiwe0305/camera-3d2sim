import json
from pathlib import Path


def scan_manifests(scenes_root: Path) -> list[dict]:
    results = []
    for p in scenes_root.glob("*/captures/*/manifest.json"):
        try:
            results.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return results


def build_catalog(scenes_root: Path) -> dict:
    rows = []
    for m in scan_manifests(scenes_root):
        gate_a = (m.get("gate") or {}).get("gateA") or {}
        scale = (gate_a.get("checks") or {}).get("scale") or {}
        rows.append({
            "env_id": m.get("env_id"),
            "capture_id": m.get("capture_id"),
            "started_at": m.get("started_at"),
            "verdict": gate_a.get("verdict", "PENDING"),
            "scale_err_pct": scale.get("worst_err_pct"),
        })
    return {"captures": rows}


def render_markdown(catalog: dict) -> str:
    lines = ["| env_id | capture_id | started_at | verdict | scale_err_pct |",
              "|---|---|---|---|---|"]
    for r in catalog["captures"]:
        lines.append(f"| {r['env_id']} | {r['capture_id']} | {r['started_at']} | "
                      f"{r['verdict']} | {r['scale_err_pct']} |")
    return "\n".join(lines)


def reindex(scenes_root: Path) -> None:
    catalog = build_catalog(scenes_root)
    (scenes_root / "_meta").mkdir(parents=True, exist_ok=True)
    (scenes_root / "_meta" / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (scenes_root / "_meta" / "catalog.md").write_text(render_markdown(catalog))
