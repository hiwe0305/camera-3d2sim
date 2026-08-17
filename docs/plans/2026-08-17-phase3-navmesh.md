# Phase 3: `navmesh/` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a gravity-aligned point cloud (Phase 1's `align/mesh_aligned.ply`), produce a clean collision mesh and a 2D occupancy map for Isaac Sim navigation — no GSplat/training dependency, can run as soon as Phase 1 produces a real aligned cloud.

**Architecture:** Mesh cleanup (Poisson reconstruction + trim + noise filter) via Open3D, same library already used by `align_verify/`. Occupancy map generation reuses this machine's installed Isaac Sim extension `omni.cip.mega.occupancy_map_ui`, which — despite the `_ui` suffix — ships a headless-safe `OccupancyMapGenerator` class with **no `omni.ui` dependency**, plus a ready-made standalone CLI script (`.../occupancy_map_ui/scripts/generate_occupancy_map.py`) that already implements the full headless launch → load stage → generate → write PNG+YAML pattern. This project wraps that script rather than reimplementing its logic.

**Tech Stack:** Open3D (already a dependency) for mesh cleanup. Isaac Sim's own Python (`/home/ubuntu/isaacsim`) for occupancy map generation — a separate interpreter from this project's `.venv`, invoked as a subprocess.

**Spec:** [docs/design.md](../design.md) §3 step [5], §11 Phase 3

## Global Constraints

- Occupancy map generation: call `/home/ubuntu/isaacsim/extscache/omni.cip.mega.occupancy_map_ui-2.0.7+lx64.cp312/omni/cip/mega/occupancy_map_ui/scripts/generate_occupancy_map.py` as a subprocess via Isaac Sim's own launcher (`/home/ubuntu/isaacsim/python.sh <script> --usd-path <scene.usd> --output-dir <out> --resolution 0.05 --robot-height 1.5 --floor-height 0.0 --no-auto-scan --flood-fill-x <x> --flood-fill-y <y>`) — `--no-auto-scan` and explicit flood-fill coordinates are required because this project's scenes contain no robot prims for the tool's default auto-scan to key off of. Verify this exact CLI against the script's own `--help` output at execution time (this plan's flags are from reading the script's source, not from running it — confirm before relying on them).
- Do NOT attempt to use a `omni.isaac.occupancy_map` extension — it does not exist in this Isaac Sim 6.0.1 install (confirmed via full-tree search); the only occupancy-map tool present is the CIP one above.
- The occupancy map generator needs a **USD stage** as input, not a bare mesh file — this project's `navmesh/build_collision_mesh.py` (Task 1) produces an OBJ; Task 2 must wrap it into a minimal USD stage (a single mesh prim, gravity-aligned, at the world origin) before calling the generator. Use `pxr.Usd`/`UsdGeom` (available in this project's `.venv` via `open3d`? — no, `pxr` comes from `usd-core`, check if already installed; if not, `uv pip install usd-core` — do not assume `pxr` is importable without checking first).
- Run all `pytest` invocations with `unset PYTHONPATH`.
- Gate C thresholds already defined in `_meta/gates.yaml` (`collision_max_hole_m: 0.05`) — Task 1's hole-check must use this exact key.

---

## File Structure

- `navmesh/build_collision_mesh.py` — creates: Poisson reconstruction + trim + noise filter on an input point cloud, hole-size check.
- `tests/test_build_collision_mesh.py` — creates: tests using small synthetic point clouds (e.g. points sampled on a flat plane with a deliberate gap, and a clean full plane).
- `navmesh/occupancy_map.py` — creates: wraps a collision mesh (OBJ) into a minimal USD stage and invokes the Isaac Sim occupancy map generator subprocess.
- `tests/test_occupancy_map.py` — creates: tests the USD-stage-wrapping logic only (mocking/skipping the actual Isaac Sim subprocess call, which needs a GPU + full Isaac Sim runtime and is not unit-testable in this project's plain `.venv`).

## Interfaces

- `navmesh.build_collision_mesh.build_collision_mesh(cloud: o3d.geometry.PointCloud, poisson_depth: int = 9) -> o3d.geometry.TriangleMesh`
- `navmesh.build_collision_mesh.max_hole_size_m(mesh: o3d.geometry.TriangleMesh) -> float`
- `navmesh.occupancy_map.write_collision_usd(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> None`
- `navmesh.occupancy_map.generate_occupancy_map(usd_path: Path, output_dir: Path, resolution: float = 0.05, floor_height: float = 0.0, flood_fill_xy: tuple[float, float] = (0.0, 0.0)) -> subprocess.CompletedProcess` — thin subprocess wrapper, deliberately not mocked away in its own implementation (only in its test) so the real command is easy to find and run by hand.

---

### Task 1: `navmesh/build_collision_mesh.py`

**Files:**
- Create: `navmesh/__init__.py` (empty), `navmesh/build_collision_mesh.py`
- Test: `tests/test_build_collision_mesh.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_collision_mesh.py`:

```python
import numpy as np
import open3d as o3d

from navmesh.build_collision_mesh import build_collision_mesh, max_hole_size_m


def _flat_plane_cloud(n=4000, size=2.0, with_gap=False):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-size / 2, size / 2, size=(n, 2))
    if with_gap:
        # carve out a 0.3m-radius hole near the center
        keep = np.linalg.norm(xy, axis=1) > 0.3
        xy = xy[keep]
    pts = np.column_stack([xy, np.zeros(len(xy))])
    normals = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def test_build_collision_mesh_returns_nonempty_triangle_mesh():
    cloud = _flat_plane_cloud()
    mesh = build_collision_mesh(cloud, poisson_depth=8)

    assert len(mesh.triangles) > 0
    assert len(mesh.vertices) > 0


def test_max_hole_size_m_small_for_clean_plane():
    cloud = _flat_plane_cloud(with_gap=False)
    mesh = build_collision_mesh(cloud, poisson_depth=8)

    assert max_hole_size_m(mesh) < 0.05


def test_max_hole_size_m_large_when_cloud_has_a_gap():
    cloud = _flat_plane_cloud(with_gap=True)
    mesh = build_collision_mesh(cloud, poisson_depth=8)

    assert max_hole_size_m(mesh) >= 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_build_collision_mesh.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'navmesh'`.

- [ ] **Step 3: Write the implementation**

Create `navmesh/build_collision_mesh.py`:

```python
"""Ground-plane point cloud -> watertight-ish collision mesh, and a hole-size
check against Gate C's collision_max_hole_m threshold
(docs/plans/2026-08-17-phase3-navmesh.md Task 1)."""
import numpy as np
import open3d as o3d


def build_collision_mesh(cloud: o3d.geometry.PointCloud, poisson_depth: int = 9) -> o3d.geometry.TriangleMesh:
    if not cloud.has_normals():
        cloud.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=poisson_depth)
    # Trim low-density (extrapolated/noisy) vertices Poisson invents past the real point support.
    densities = np.asarray(densities)
    low_density_mask = densities < np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(low_density_mask)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def max_hole_size_m(mesh: o3d.geometry.TriangleMesh) -> float:
    """Approximate hole size as the longest boundary-edge-loop's bounding diagonal.

    Open3D has no direct "hole area" API; this walks boundary edges (edges used by
    exactly one triangle) and unions connected ones into loops, then measures each
    loop's bounding-box diagonal as a hole-size proxy.
    """
    edges_to_triangle_count: dict[tuple[int, int], int] = {}
    triangles = np.asarray(mesh.triangles)
    for tri in triangles:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = (a, b) if a < b else (b, a)
            edges_to_triangle_count[key] = edges_to_triangle_count.get(key, 0) + 1

    boundary_edges = [e for e, count in edges_to_triangle_count.items() if count == 1]
    if not boundary_edges:
        return 0.0

    vertices = np.asarray(mesh.vertices)
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited: set[int] = set()
    max_diag = 0.0
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        loop_vertices = []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            loop_vertices.append(v)
            stack.extend(n for n in adjacency[v] if n not in visited)
        pts = vertices[loop_vertices]
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        max_diag = max(max_diag, diag)
    return max_diag
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_build_collision_mesh.py -v
```

Expected: 3 passed. If flaky on the "clean plane has small holes" assertion (Poisson reconstruction of
a finite flat sample always has SOME boundary), loosen the threshold in the test rather than the
implementation — a flat sample's natural boundary is not the defect this function is checking for.

- [ ] **Step 5: Commit**

```bash
git add navmesh/__init__.py navmesh/build_collision_mesh.py tests/test_build_collision_mesh.py
git commit -m "feat: add collision mesh builder (Poisson reconstruction + hole check)"
```

---

### Task 2: `navmesh/occupancy_map.py`

**Files:**
- Create: `navmesh/occupancy_map.py`
- Test: `tests/test_occupancy_map.py`

- [ ] **Step 1: Check `pxr` (USD Python bindings) availability**

```bash
unset PYTHONPATH && .venv/bin/python -c "import pxr; print(pxr.__file__)"
```

If this fails, `uv pip install usd-core` first (do not proceed to Step 2 assuming it works).

- [ ] **Step 2: Write the failing test (USD-wrapping logic only, no Isaac Sim subprocess)**

Create `tests/test_occupancy_map.py`:

```python
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom

from navmesh.occupancy_map import write_collision_usd


def _simple_mesh():
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.1)
    return mesh


def test_write_collision_usd_creates_readable_stage_with_mesh_prim(tmp_path):
    out_path = tmp_path / "collision.usd"

    write_collision_usd(_simple_mesh(), out_path)

    assert out_path.exists()
    stage = Usd.Stage.Open(str(out_path))
    mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(mesh_prims) == 1

    mesh_geom = UsdGeom.Mesh(mesh_prims[0])
    points = np.array(mesh_geom.GetPointsAttr().Get())
    assert len(points) == 8  # a box has 8 vertices


def test_write_collision_usd_sets_collision_and_invisible():
    from pxr import UsdPhysics
    out_path_str = "/tmp/_phase3_task2_test_collision.usd"  # UsdPhysics API check doesn't need tmp_path fixture isolation

    write_collision_usd(_simple_mesh(), Path(out_path_str))

    stage = Usd.Stage.Open(out_path_str)
    mesh_prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    assert mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
    imageable = UsdGeom.Imageable(mesh_prim)
    assert imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_occupancy_map.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'navmesh.occupancy_map'`.

- [ ] **Step 4: Write the implementation**

Create `navmesh/occupancy_map.py`:

```python
"""Wraps a collision mesh into a minimal USD stage (invisible, collision-
enabled), and invokes Isaac Sim's headless occupancy map generator on it
(docs/plans/2026-08-17-phase3-navmesh.md Task 2)."""
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, UsdPhysics

ISAAC_SIM_PYTHON = "/home/ubuntu/isaacsim/python.sh"
OCCUPANCY_MAP_SCRIPT = (
    "/home/ubuntu/isaacsim/extscache/omni.cip.mega.occupancy_map_ui-2.0.7+lx64.cp312"
    "/omni/cip/mega/occupancy_map_ui/scripts/generate_occupancy_map.py"
)


def write_collision_usd(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> None:
    output_path = Path(output_path)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/CollisionMesh")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    mesh_prim.GetPointsAttr().Set(vertices.tolist())
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    mesh_prim.GetFaceVertexIndicesAttr().Set(triangles.flatten().tolist())

    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    UsdGeom.Imageable(mesh_prim.GetPrim()).MakeInvisible()

    stage.GetRootLayer().Save()


def generate_occupancy_map(usd_path: Path, output_dir: Path, resolution: float = 0.05,
                            floor_height: float = 0.0,
                            flood_fill_xy: tuple[float, float] = (0.0, 0.0)) -> subprocess.CompletedProcess:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [ISAAC_SIM_PYTHON, OCCUPANCY_MAP_SCRIPT,
         "--usd-path", str(usd_path), "--output-dir", str(output_dir),
         "--resolution", str(resolution), "--robot-height", "1.5",
         "--floor-height", str(floor_height), "--no-auto-scan",
         "--flood-fill-x", str(flood_fill_xy[0]), "--flood-fill-y", str(flood_fill_xy[1])],
        check=True, capture_output=True, text=True,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_occupancy_map.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Manually verify `generate_occupancy_map` against a real USD once Phase 1 produces one**

This function is deliberately NOT called by any automated test (needs a GPU + full Isaac Sim runtime,
several seconds to minutes per call) — run it by hand once there's a real collision mesh:

```bash
unset PYTHONPATH && .venv/bin/python -c "
from pathlib import Path
from navmesh.build_collision_mesh import build_collision_mesh
from navmesh.occupancy_map import write_collision_usd, generate_occupancy_map
import open3d as o3d

cloud = o3d.io.read_point_cloud('<session>/align/mesh_aligned.ply')
mesh = build_collision_mesh(cloud)
write_collision_usd(mesh, Path('/tmp/collision.usd'))
result = generate_occupancy_map(Path('/tmp/collision.usd'), Path('/tmp/occmap_out'))
print(result.stdout, result.stderr)
"
```

First confirm the script's real CLI flags with `/home/ubuntu/isaacsim/python.sh <script> --help` —
this plan's flag names come from reading source, not from running `--help`; fix `generate_occupancy_map`'s
argument list if they don't match.

- [ ] **Step 7: Commit**

```bash
git add navmesh/occupancy_map.py tests/test_occupancy_map.py requirements.txt
git commit -m "feat: add collision-mesh-to-USD wrapper and occupancy map generator"
```
