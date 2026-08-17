# Phase 4: `isaacsim_import/` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine a trained GSplat (Phase 2) and a collision mesh (Phase 3) into one USD stage at a shared origin, and automate the Gate C physics drop-test headlessly.

**Architecture:** Reuse Phase 0's validated `ply_to_usd` conversion (with the `export_cameras=False` fix already documented in `spikes/phase0_isaacsim_import/README.md`) to get the GSplat as USDZ, then use 3DGRUT's own `add_mesh_to_usdz.py` script to merge in the Phase 3 collision mesh with collision enabled and visibility off — this is a purpose-built tool for exactly this merge (found during Phase 0's research of `threedgrut/export/README.md`), not something to reimplement. The drop-test uses Isaac Sim's current (non-deprecated) `isaacsim.core.experimental` API, confirmed present in this install and demonstrated by a real existing test fixture in the Isaac Sim source tree.

**Tech Stack:** `~/tools/3dgrut/.venv`'s Python for the USD merge step (has the export tooling). Isaac Sim's own Python (`/home/ubuntu/isaacsim`) for the drop-test.

**Spec:** [docs/design.md](../design.md) §3 step [6], §11 Phase 4

## Global Constraints

- USD merge command (from `threedgrut/export/README.md`, read during Phase 0):
  ```bash
  PATH="/home/ubuntu/tools/3dgrut/.venv/bin:$PATH" /home/ubuntu/tools/3dgrut/.venv/bin/python \
      -m threedgrut.export.scripts.add_mesh_to_usdz \
      --input_usdz <gsplat.usdz> --output_usdz <env.usd.usdz> \
      --mesh_usd <collision.usd> --set_collision --set_invisible
  ```
- Drop-test API (confirmed against this exact Isaac Sim 6.0.1 install, not older-version docs — see
  `/home/ubuntu/isaacsim/standalone_examples/tutorials/getting_started/getting_started_robot.py` and
  `/home/ubuntu/isaacsim/exts/isaacsim.core.experimental.prims/isaacsim/core/experimental/prims/tests/test_rigid_prim.py`):
  - Headless launch: `from isaacsim import SimulationApp; simulation_app = SimulationApp({"headless": True})` — **must** precede any other `omni.*`/`isaacsim.*` import.
  - Spawn: `from isaacsim.core.experimental.objects import Cube; from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim`.
  - Step physics: `from isaacsim.core.simulation_manager import SimulationManager; SimulationManager.step(steps=N)`.
  - Read back: `RigidPrim(paths).get_world_poses()` → `(positions, orientations)`.
  - Do NOT use `isaacsim.core.api`/`omni.isaac.core` — confirmed deprecated in this install (lives under `extsDeprecated/`).
- Gate C thresholds from `_meta/gates.yaml`: `drop_test_points: 20`, `drop_test_max_height_err_m: 0.03`.
- Run all `pytest` invocations with `unset PYTHONPATH`. Drop-test code itself runs under Isaac Sim's own Python (`/home/ubuntu/isaacsim/python.sh`), not this project's `.venv` — it cannot be unit-tested by plain `pytest` (needs a GPU + the full Isaac Sim runtime to even import `isaacsim`). Test only the pure-Python parts (threshold-checking logic) in `.venv`; keep those separate from the Isaac Sim-dependent spawn/step/readback code so they're testable at all.

---

## File Structure

- `isaacsim_import/assemble_stage.py` — creates: thin wrapper around the `add_mesh_to_usdz` subprocess call.
- `isaacsim_import/drop_test_check.py` — creates: **pure logic only** — given expected floor height and a list of final resting heights, decides pass/fail per point and overall. No Isaac Sim import.
- `tests/test_drop_test_check.py` — creates: tests for the pure logic.
- `isaacsim_import/drop_test_isaacsim.py` — creates: the actual Isaac Sim script (spawn cubes, step, read back, call `drop_test_check`). Runs under `/home/ubuntu/isaacsim/python.sh`, not pytest.

## Interfaces

- `isaacsim_import.assemble_stage.assemble_env_usd(gsplat_usdz: Path, collision_usd: Path, output_path: Path) -> subprocess.CompletedProcess`
- `isaacsim_import.drop_test_check.check_drop_test(final_heights: list[float], expected_floor_height: float, max_height_err_m: float) -> dict` — the only unit-testable piece of the drop-test; consumed by `drop_test_isaacsim.py` (Isaac Sim runtime, not pytest) after it spawns/steps/reads back real physics.

---

### Task 1: `isaacsim_import/assemble_stage.py`

**Files:**
- Create: `isaacsim_import/__init__.py` (empty), `isaacsim_import/assemble_stage.py`
- Test: `tests/test_assemble_stage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_stage.py`:

```python
from pathlib import Path
from unittest.mock import patch

from isaacsim_import.assemble_stage import assemble_env_usd


def test_assemble_env_usd_calls_add_mesh_to_usdz_with_expected_args(tmp_path):
    gsplat = tmp_path / "gsplat.usdz"
    collision = tmp_path / "collision.usd"
    output = tmp_path / "env.usdz"
    gsplat.touch()
    collision.touch()

    with patch("isaacsim_import.assemble_stage.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assemble_env_usd(gsplat, collision, output)

    args = mock_run.call_args[0][0]
    assert "--input_usdz" in args and str(gsplat) in args
    assert "--output_usdz" in args and str(output) in args
    assert "--mesh_usd" in args and str(collision) in args
    assert "--set_collision" in args
    assert "--set_invisible" in args
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_assemble_stage.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'isaacsim_import'`.

- [ ] **Step 3: Write the implementation**

Create `isaacsim_import/assemble_stage.py`:

```python
"""Merges a Phase 0/2-produced GSplat USDZ with a Phase 3 collision mesh USD
using 3DGRUT's own add_mesh_to_usdz tool (docs/plans/2026-08-17-phase4-isaacsim-import.md Task 1)."""
import subprocess
from pathlib import Path

THREEDGRUT_VENV_PYTHON = "/home/ubuntu/tools/3dgrut/.venv/bin/python"
THREEDGRUT_VENV_BIN = "/home/ubuntu/tools/3dgrut/.venv/bin"


def assemble_env_usd(gsplat_usdz: Path, collision_usd: Path, output_path: Path) -> subprocess.CompletedProcess:
    import os
    env = os.environ.copy()
    env["PATH"] = f"{THREEDGRUT_VENV_BIN}:{env.get('PATH', '')}"
    return subprocess.run(
        [THREEDGRUT_VENV_PYTHON, "-m", "threedgrut.export.scripts.add_mesh_to_usdz",
         "--input_usdz", str(gsplat_usdz), "--output_usdz", str(output_path),
         "--mesh_usd", str(collision_usd), "--set_collision", "--set_invisible"],
        check=True, capture_output=True, text=True, env=env,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_assemble_stage.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add isaacsim_import/__init__.py isaacsim_import/assemble_stage.py tests/test_assemble_stage.py
git commit -m "feat: add GSplat+collision-mesh USD stage assembly wrapper"
```

---

### Task 2: `isaacsim_import/drop_test_check.py` (pure logic, unit-testable)

**Files:**
- Create: `isaacsim_import/drop_test_check.py`
- Test: `tests/test_drop_test_check.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_drop_test_check.py`:

```python
from isaacsim_import.drop_test_check import check_drop_test


def test_check_drop_test_all_within_tolerance_passes():
    heights = [0.001, -0.002, 0.0, 0.0299, -0.0299]
    result = check_drop_test(heights, expected_floor_height=0.0, max_height_err_m=0.03)

    assert result["pass"] is True
    assert result["n_points"] == 5
    assert result["n_failed"] == 0


def test_check_drop_test_one_point_falls_through_floor_fails():
    heights = [0.0] * 19 + [-0.5]  # one point fell way below the floor
    result = check_drop_test(heights, expected_floor_height=0.0, max_height_err_m=0.03)

    assert result["pass"] is False
    assert result["n_failed"] == 1
    assert -0.5 in result["failed_heights"]


def test_check_drop_test_reports_max_deviation():
    heights = [0.01, 0.02, -0.025]
    result = check_drop_test(heights, expected_floor_height=0.0, max_height_err_m=0.03)

    assert abs(result["max_deviation_m"] - 0.025) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_drop_test_check.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'isaacsim_import.drop_test_check'`.

- [ ] **Step 3: Write the implementation**

Create `isaacsim_import/drop_test_check.py`:

```python
"""Pure pass/fail logic for the Gate C physics drop-test -- deliberately has
no Isaac Sim import so it's testable under plain pytest
(docs/plans/2026-08-17-phase4-isaacsim-import.md Task 2). The Isaac Sim-side
script that spawns rigid bodies and calls this after reading back their
resting heights is drop_test_isaacsim.py (Task 3, not pytest-testable)."""


def check_drop_test(final_heights: list[float], expected_floor_height: float,
                     max_height_err_m: float) -> dict:
    deviations = [abs(h - expected_floor_height) for h in final_heights]
    failed = [h for h, d in zip(final_heights, deviations) if d > max_height_err_m]
    return {
        "pass": len(failed) == 0,
        "n_points": len(final_heights),
        "n_failed": len(failed),
        "failed_heights": failed,
        "max_deviation_m": max(deviations) if deviations else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_drop_test_check.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add isaacsim_import/drop_test_check.py tests/test_drop_test_check.py
git commit -m "feat: add Gate C drop-test pass/fail logic"
```

---

### Task 3: `isaacsim_import/drop_test_isaacsim.py` (Isaac Sim runtime script — not pytest-testable, manual verification)

**Files:**
- Create: `isaacsim_import/drop_test_isaacsim.py`

This script cannot be exercised by `pytest` in this project's `.venv` — it needs `isaacsim`, which
only exists inside the Isaac Sim install's own Python environment. Write it carefully against the API
confirmed in Global Constraints, then verify it manually once Phase 4 Task 1 produces a real `env.usdz`.

- [ ] **Step 1: Write the script**

Create `isaacsim_import/drop_test_isaacsim.py`:

```python
"""Headless Gate C physics drop-test: loads an assembled env USD, drops 20
rigid cubes across the floor, checks they land within tolerance
(docs/plans/2026-08-17-phase4-isaacsim-import.md Task 3).

Run with Isaac Sim's own Python, NOT this project's .venv:
    /home/ubuntu/isaacsim/python.sh isaacsim_import/drop_test_isaacsim.py \
        --usd-path <env.usdz> --floor-height 0.0 --grid-size 4.0 --n-points 20
"""
import argparse
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--usd-path", required=True)
parser.add_argument("--floor-height", type=float, default=0.0)
parser.add_argument("--grid-size", type=float, default=4.0, help="spread drop points across a grid_size x grid_size area centered at the origin")
parser.add_argument("--n-points", type=int, default=20)
parser.add_argument("--drop-height", type=float, default=1.0)
parser.add_argument("--max-height-err-m", type=float, default=0.03)
parser.add_argument("--physics-steps", type=int, default=180)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager

from isaacsim_import.drop_test_check import check_drop_test

omni.usd.get_context().open_stage(args.usd_path)

side = int(np.ceil(np.sqrt(args.n_points)))
xs = np.linspace(-args.grid_size / 2, args.grid_size / 2, side)
ys = np.linspace(-args.grid_size / 2, args.grid_size / 2, side)
grid = np.array([[x, y] for x in xs for y in ys])[: args.n_points]

paths = [f"/World/DropTest/cube_{i}" for i in range(len(grid))]
Cube(paths, sizes=[0.05] * len(paths))
translations = [[x, y, args.floor_height + args.drop_height] for x, y in grid]
XformPrim(paths).set_local_poses(translations=translations)
GeomPrim(paths, apply_collision_apis=True)
RigidPrim(paths, masses=[0.1] * len(paths))

SimulationManager.step(steps=args.physics_steps)

positions, _orientations = RigidPrim(paths).get_world_poses()
final_heights = [float(p[2]) for p in positions]

result = check_drop_test(final_heights, args.floor_height, args.max_height_err_m)
print(f"Gate C drop-test: {'PASS' if result['pass'] else 'FAIL'}")
print(f"  {result['n_points']} points, {result['n_failed']} failed, max deviation {result['max_deviation_m']:.4f}m")

simulation_app.close()
sys.exit(0 if result["pass"] else 1)
```

- [ ] **Step 2: Verify the script's imports/CLI against this exact Isaac Sim install before relying on it**

```bash
/home/ubuntu/isaacsim/python.sh -c "
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
print('all imports OK')
"
```

If any import fails, the API surface has drifted from what Phase 4's research found — fix the script
against the actual current API before Step 3, don't guess a patch.

- [ ] **Step 3: Manually run against a real assembled env USD once Phase 4 Task 1 + Phase 2/3 produce one**

```bash
/home/ubuntu/isaacsim/python.sh isaacsim_import/drop_test_isaacsim.py \
    --usd-path <env.usdz> --floor-height 0.0 --grid-size 4.0 --n-points 20
```

- [ ] **Step 4: Commit**

```bash
git add isaacsim_import/drop_test_isaacsim.py
git commit -m "feat: add headless Isaac Sim Gate C drop-test script"
```

---

### Task 4: Record outcome + catalog integration

**Files:**
- Modify: `docs/design.md` (§7, §11 Phase 4)

- [ ] **Step 1: Once Tasks 1-3 have run against a real build, update `docs/design.md`** with the Gate C
  result and mark `isaacsim_import/` as implemented in §7's status table.

- [ ] **Step 2: Commit**

```bash
git add docs/design.md
git commit -m "docs: record Phase 4 isaacsim_import implementation status"
```
