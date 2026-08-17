# Phase 0: Isaac Sim Gaussian Splat Import Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the go/no-go question from `docs/design.md` §3 Phase 0 — can a `.ply` Gaussian Splat be converted to `.usdz` and imported into Isaac Sim 6.0.1 on this machine without hitting the "layered artifact" bug reported on the NVIDIA forums — before any effort goes into `train/`, `navmesh/`, or `isaacsim_import/`.

**Architecture:** Generate a small, correctly-formatted synthetic Gaussian Splat `.ply` fixture ourselves (no external dataset download, no GPU training needed — this isolates the question "does the conversion+import pipeline work" from "does training work"). Convert it with the official `nv-tlabs/3dgrut` NuRec exporter, then import the result into Isaac Sim 6.0.1 and visually confirm correct rendering. Record the outcome in `docs/design.md`.

**Tech Stack:** Python 3.12 (`camera-3d2sim/.venv`, `uv`-managed) for fixture generation; `nv-tlabs/3dgrut` in its own `uv`-managed venv (separate CUDA/PyTorch stack) for the PLY→USDZ conversion; Isaac Sim 6.0.1 (already installed on this machine) for import/render verification.

**Spec:** [docs/design.md](../design.md) — §3 "Phase 0", §9 "Rủi ro còn mở" (format .ply / layered artifact risk)

## Global Constraints

- System CUDA toolkit is 11.5 (`nvcc --version`), below 3DGRUT's minimum supported 11.8 — install 3DGRUT with a **local, venv-scoped CUDA 12 toolkit** (`FORCE_LOCAL_CUDA=1 CUDA_VERSION=12`), never touch the system-wide CUDA install (other tooling, e.g. ROS2, may depend on it).
- GPU: NVIDIA RTX 3060, 12GB VRAM, driver 595.84, compute capability 8.6 — meets 3DGRUT's minimum (sm_70+).
- gcc/g++ 11.4.0 present — compatible with CUDA 12 builds (3DGRUT recommends GCC ≤11).
- Install `nv-tlabs/3dgrut` **outside** the `camera-3d2sim` git repo, at `~/tools/3dgrut` — it is an external tool dependency (like Isaac Sim itself), not project source.
- Isaac Sim 6.0.1 is already installed and confirmed running on this machine (2026-08-17) — no installation step needed for it in this plan.
- Known bug (NVIDIA forum, thread on `forums.developer.nvidia.com`, id 360930): the "layered artifact" rendering bug in NuRec USDZ import is caused by **float16 precision loss at coordinate magnitudes ≥300m from the world origin**. Our synthetic test scene sits within ±0.5m of the origin, so it is not expected to trigger this — if it renders broken anyway, that is a stronger, more general finding worth escalating rather than dismissing.
- Prefer the `nurec` USDZ format for this spike (it's what the original spec and the forum bug both reference). Isaac Sim 6.0's `ParticleField` schema is the longer-term replacement (NuRec is being deprecated) — noted as a follow-up, not required for this spike's go/no-go.
- Run all `pytest` invocations in this project with `env -u PYTHONPATH` — a system-wide ROS2 `PYTHONPATH` shadows the project's `.venv` packages otherwise (see `docs/design.md` §8).
- Artifacts this plan generates (`.ply`, `.usdz` fixture files) are throwaway/regenerable — gitignore them, commit only the generator script, its test, and the written-up result.

---

## File Structure

- `spikes/phase0_isaacsim_import/generate_test_splat.py` — creates a small synthetic Gaussian Splat `.ply` fixture (colored sphere), in the exact pre-activation field layout 3DGRUT's PLY importer expects. No dataset download, no training.
- `spikes/phase0_isaacsim_import/test_generate_test_splat.py` — pytest validating the fixture's PLY schema and value ranges.
- `spikes/phase0_isaacsim_import/README.md` — how to run the spike end-to-end, and (filled in by Task 5) the recorded go/no-go result.
- `spikes/phase0_isaacsim_import/out/` — gitignored; holds the generated `.ply` and converted `.usdz`.
- `.gitignore` — add `spikes/*/out/`.
- `docs/design.md` — Task 5 updates §3 Phase 0 status and §9 risks with the outcome.

## Interfaces

- `generate_test_splat.generate_sphere_splat(output_path: Path, n_points: int = 8000, radius: float = 0.5) -> None` — writes the PLY fixture. Isotropic scale (all three `scale_i` equal) is a deliberate choice: it makes the quaternion rotation convention (wxyz vs xyzw) irrelevant to correctness, since a sphere is rotation-invariant — avoids guessing an unverified convention.
- Fixture PLY schema (verified against `threedgrut/export/importers/ply.py` in the cloned `nv-tlabs/3dgrut` repo): `x,y,z,nx,ny,nz,f_dc_0,f_dc_1,f_dc_2,opacity,scale_0,scale_1,scale_2,rot_0,rot_1,rot_2,rot_3`, all `float32`. No `f_rest_*` columns (degree-0 SH only — the importer handles this fine, logs "PLY file only contains DC components"). Values are **pre-activation**: `opacity` is a sigmoid logit, `scale_i` is `log(meters)`, colors are SH DC coefficients (`f_dc = (rgb - 0.5) / 0.28209479177387814`).

---

### Task 1: Install 3DGRUT with a local CUDA 12 toolkit

**Files:**
- Create (outside repo): `~/tools/3dgrut/` (git clone of `nv-tlabs/3dgrut`, with its own `.venv`)

**Interfaces:**
- Produces: a working `~/tools/3dgrut/.venv` with `python -m threedgrut.export.scripts.ply_to_usd` importable — Task 3 depends on this exact venv path and module.

- [ ] **Step 1: Create the tools directory and clone the repo**

```bash
mkdir -p ~/tools
cd ~/tools
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd ~/tools/3dgrut
```

- [ ] **Step 2: Install `uv` prerequisites for the build**

```bash
sudo apt-get install -y libgl1-mesa-dev wget
```

Expected: both already satisfied on this machine (`libgl1-mesa-dev` and `wget` are already installed) — this step should report "already the newest version" for both, not fail.

- [ ] **Step 3: Provision a local, venv-scoped CUDA 12 toolkit (does not touch system CUDA)**

```bash
cd ~/tools/3dgrut
FORCE_LOCAL_CUDA=1 CUDA_VERSION=12 ./scripts/create_venv_cuda.sh
```

Expected: downloads the CUDA 12 runfile (~4GB, cached at `/tmp/cuda_12_linux.run` for reuse) and installs it into `~/tools/3dgrut/.venv/cuda-12/`. Takes several minutes depending on network speed.

- [ ] **Step 4: Activate the venv and install 3DGRUT's Python dependencies**

```bash
source ~/tools/3dgrut/.venv/bin/activate
cd ~/tools/3dgrut
./install_env_uv.sh
```

Expected: installs PyTorch (CUDA 12 build), the project itself (`uv pip install -e .[gui]`), `tiny-cuda-nn` bindings (compiled from the pinned submodule — this is the step most likely to fail if the CUDA/GCC toolchain isn't detected correctly), Kaolin, and `slangc`. This step compiles CUDA extensions and can take 15-30+ minutes.

- [ ] **Step 5: Verify the export tooling imports and runs**

```bash
source ~/tools/3dgrut/.venv/bin/activate
python -m threedgrut.export.scripts.ply_to_usd --help
```

Expected: prints the argparse usage (`usage: ply_to_usd.py [-h] [--output_file OUTPUT_FILE] input_file`) with no `ImportError`/`ModuleNotFoundError`/CUDA-related traceback. If it fails with a CUDA/compiler error, stop and diagnose before Task 2 — Tasks 2-3 are pointless without this working.

- [ ] **Step 6: Record the install outcome**

No commit here — `~/tools/3dgrut` is outside the `camera-3d2sim` repo. Just confirm Step 5's output before moving on; nothing to stage.

---

### Task 2: Generate a synthetic Gaussian Splat PLY fixture

**Files:**
- Create: `spikes/phase0_isaacsim_import/generate_test_splat.py`
- Test: `spikes/phase0_isaacsim_import/test_generate_test_splat.py`
- Modify: `.gitignore` (add `spikes/*/out/`)

**Interfaces:**
- Produces: `generate_sphere_splat(output_path: Path, n_points: int = 8000, radius: float = 0.5) -> None`, used directly (as a script) to write the fixture Task 3 converts.
- Consumes: `numpy`, `plyfile` (neither yet in `camera-3d2sim/.venv` — installed in Step 1 below).

- [ ] **Step 1: Add `plyfile` to the project venv**

```bash
cd /home/ubuntu/camera-3d2sim
uv pip install plyfile
```

Expected: installs `plyfile` (small pure-Python PLY reader/writer) into `.venv`. `numpy` is already present.

- [ ] **Step 2: Write the failing test**

Create `spikes/phase0_isaacsim_import/test_generate_test_splat.py`:

```python
import numpy as np
from plyfile import PlyData

from generate_test_splat import generate_sphere_splat


def test_generate_sphere_splat_writes_expected_point_count(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    ply = PlyData.read(str(out_path))
    vertex = ply["vertex"]
    assert len(vertex) == 200


def test_generate_sphere_splat_has_expected_ply_fields(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    ply = PlyData.read(str(out_path))
    field_names = {p.name for p in ply["vertex"].properties}
    expected = {
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    assert expected.issubset(field_names)


def test_generate_sphere_splat_points_lie_on_sphere_surface(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    vertex = PlyData.read(str(out_path))["vertex"]
    positions = np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1
    )
    radii = np.linalg.norm(positions, axis=1)
    assert np.allclose(radii, 0.5, atol=1e-4)


def test_generate_sphere_splat_opacity_is_high_confidence_logit(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    vertex = PlyData.read(str(out_path))["vertex"]
    opacity = np.asarray(vertex["opacity"])
    # logit(0.95) ~= 2.944 -- a visibly-opaque splat, not a near-invisible one
    assert np.all(opacity > 2.9)
    assert np.all(opacity < 3.0)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /home/ubuntu/camera-3d2sim
env -u PYTHONPATH .venv/bin/python -m pytest spikes/phase0_isaacsim_import/test_generate_test_splat.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'generate_test_splat'` (the file doesn't exist yet).

- [ ] **Step 4: Write the minimal implementation**

Create `spikes/phase0_isaacsim_import/generate_test_splat.py`:

```python
"""Generate a small synthetic Gaussian Splat PLY fixture for the Phase 0
Isaac Sim import spike (docs/plans/2026-08-17-phase0-isaacsim-import-spike.md).

No dataset download, no GPU training -- this isolates "does PLY-to-USDZ
conversion and Isaac Sim import work" from "does training work". The PLY
schema matches what nv-tlabs/3dgrut's PLYImporter expects (verified by
reading threedgrut/export/importers/ply.py directly): pre-activation
values, degree-0 SH only (no f_rest_* columns).
"""
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814  # first spherical-harmonic band constant, degree-0 DC term


def generate_sphere_splat(output_path: Path, n_points: int = 8000, radius: float = 0.5) -> None:
    """Write a PLY of Gaussians arranged on a sphere surface, colored by height (Z).

    Isotropic scale (scale_0 == scale_1 == scale_2 for every point) is deliberate:
    a sphere is rotation-invariant, so the quaternion rotation convention (wxyz vs
    xyzw) cannot affect the rendered result -- we don't have to guess it.
    """
    rng = np.random.default_rng(seed=0)

    # Uniform points on a sphere surface (Marsaglia method).
    u = rng.uniform(-1.0, 1.0, size=n_points)
    theta = rng.uniform(0.0, 2 * math.pi, size=n_points)
    xy_radius = np.sqrt(1.0 - u**2)
    positions = radius * np.stack(
        [xy_radius * np.cos(theta), xy_radius * np.sin(theta), u], axis=1
    ).astype(np.float32)

    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # placeholder, unused by the importer (see ply.py's own comment)

    # Color by height: blue at the bottom (z=-radius) to red at the top (z=+radius).
    height_t = ((positions[:, 2] / radius) + 1.0) / 2.0  # 0..1
    rgb = np.stack([height_t, np.zeros(n_points), 1.0 - height_t], axis=1).astype(np.float32)
    f_dc = (rgb - 0.5) / SH_C0

    opacity_logit = np.full((n_points, 1), 3.0, dtype=np.float32)  # sigmoid(3.0) ~= 0.953

    gaussian_radius_m = 0.015
    scale_log = np.full((n_points, 3), math.log(gaussian_radius_m), dtype=np.float32)

    rotation = np.zeros((n_points, 4), dtype=np.float32)
    rotation[:, 0] = 1.0  # identity quaternion; irrelevant here since scale is isotropic

    attrs = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    rows = np.concatenate(
        [positions, normals, f_dc, opacity_logit, scale_log, rotation], axis=1
    )
    vertex_data = np.empty(n_points, dtype=attrs)
    vertex_data[:] = list(map(tuple, rows))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex_data, "vertex")]).write(str(output_path))


if __name__ == "__main__":
    fixture_path = Path(__file__).parent / "out" / "test_scene.ply"
    generate_sphere_splat(fixture_path, n_points=8000, radius=0.5)
    print(f"Wrote {fixture_path}")
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /home/ubuntu/camera-3d2sim
env -u PYTHONPATH .venv/bin/python -m pytest spikes/phase0_isaacsim_import/test_generate_test_splat.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Generate the actual fixture file used by Task 3**

```bash
cd /home/ubuntu/camera-3d2sim
env -u PYTHONPATH .venv/bin/python spikes/phase0_isaacsim_import/generate_test_splat.py
```

Expected: prints `Wrote spikes/phase0_isaacsim_import/out/test_scene.ply`.

- [ ] **Step 7: Gitignore the output directory and commit**

```bash
cd /home/ubuntu/camera-3d2sim
```

Add to `.gitignore`:
```
spikes/*/out/
```

```bash
git add spikes/phase0_isaacsim_import/generate_test_splat.py \
        spikes/phase0_isaacsim_import/test_generate_test_splat.py \
        .gitignore
git commit -m "feat: add synthetic Gaussian Splat PLY fixture generator for Phase 0 spike"
```

---

### Task 3: Convert the fixture PLY to USDZ

**Files:**
- Create: `spikes/phase0_isaacsim_import/test_convert_to_usdz.py`

**Interfaces:**
- Consumes: `spikes/phase0_isaacsim_import/out/test_scene.ply` (from Task 2), `~/tools/3dgrut/.venv` (from Task 1).
- Produces: `spikes/phase0_isaacsim_import/out/test_scene.usdz`, used by Task 4.

- [ ] **Step 1: Run the conversion**

```bash
source ~/tools/3dgrut/.venv/bin/activate
cd /home/ubuntu/camera-3d2sim
python -m threedgrut.export.scripts.ply_to_usd \
    spikes/phase0_isaacsim_import/out/test_scene.ply \
    --output_file spikes/phase0_isaacsim_import/out/test_scene.usdz
deactivate
```

Expected: log lines `Converting ... to ...`, `Loading default configuration`, `Loading PLY with init_from_ply`, `Exporting with NuRecExporter`, `Successfully exported to spikes/phase0_isaacsim_import/out/test_scene.usdz`. Exit code 0.

- [ ] **Step 2: Write an automated check that the USDZ was produced correctly**

Create `spikes/phase0_isaacsim_import/test_convert_to_usdz.py`:

```python
import zipfile
from pathlib import Path

USDZ_PATH = Path(__file__).parent / "out" / "test_scene.usdz"


def test_usdz_file_exists_and_is_non_trivial():
    assert USDZ_PATH.exists(), f"{USDZ_PATH} was not produced -- run Task 3 Step 1 first"
    assert USDZ_PATH.stat().st_size > 1024, "USDZ file is suspiciously small (near-empty export)"


def test_usdz_is_a_valid_zip_archive():
    # .usdz is a ZIP container (uncompressed, per the OpenUSD spec) -- a corrupt/truncated
    # export would fail zipfile's own CRC check here before Isaac Sim ever sees it.
    assert zipfile.is_zipfile(USDZ_PATH)
    with zipfile.ZipFile(USDZ_PATH) as zf:
        bad_file = zf.testzip()
        assert bad_file is None, f"corrupt member in USDZ archive: {bad_file}"
        names = zf.namelist()
        assert any(name.endswith((".usd", ".usda", ".usdc")) for name in names), (
            f"no USD layer found inside the USDZ archive, contents: {names}"
        )
```

- [ ] **Step 3: Run the check**

```bash
cd /home/ubuntu/camera-3d2sim
env -u PYTHONPATH .venv/bin/python -m pytest spikes/phase0_isaacsim_import/test_convert_to_usdz.py -v
```

Expected: 2 passed. If Task 3 Step 1 hasn't been run yet, the first test fails with the explicit "run Task 3 Step 1 first" message rather than a bare `FileNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git add spikes/phase0_isaacsim_import/test_convert_to_usdz.py
git commit -m "test: verify Phase 0 PLY-to-USDZ conversion output"
```

---

### Task 4: Import into Isaac Sim and visually verify (manual — requires the GUI)

This task cannot be automated by an agent without deeper Isaac Sim scripting knowledge than this spike warrants; it is a short human-in-the-loop check. **A human runs these steps directly**, not a subagent.

**Files:** none (verification only)

- [ ] **Step 1: Open Isaac Sim 6.0.1**

Launch Isaac Sim as you normally do on this machine.

- [ ] **Step 2: Import the USDZ**

`File > Import`, navigate to `/home/ubuntu/camera-3d2sim/spikes/phase0_isaacsim_import/out/test_scene.usdz`, import it. (Alternative: drag-and-drop the file from the content browser into the viewport.)

- [ ] **Step 3: Visually verify against this checklist**

**PASS** looks like: a smooth, solid-looking sphere (~1m diameter) shaded blue at the bottom fading to red at the top, viewable from any angle without gaps or seams.

**FAIL — layered artifact bug** looks like: the sphere appears sliced into flat, stacked planes/layers, or depth ordering flips (far points render in front of near points) when orbiting the camera around it.

**FAIL — other**: nothing renders (blank/invisible), Isaac Sim errors on import, or the geometry is at the wrong scale/position (should be a small ~1m sphere near the world origin).

- [ ] **Step 4: Note the result**

Write down which of the three outcomes occurred, and if possible a screenshot — needed for Task 5.

---

### Task 5: Record the go/no-go outcome

**Files:**
- Create: `spikes/phase0_isaacsim_import/README.md`
- Modify: `docs/design.md:33` (Phase 0 section — currently reads "Chưa chạy phase này")

**Interfaces:** none (documentation only)

- [ ] **Step 1: Write the spike README**

Create `spikes/phase0_isaacsim_import/README.md`:

```markdown
# Phase 0 spike: Isaac Sim Gaussian Splat import

Answers the go/no-go question in `docs/design.md` §3 Phase 0: does a `.ply`
Gaussian Splat survive conversion to `.usdz` (via `nv-tlabs/3dgrut`) and
import into Isaac Sim 6.0.1 without the "layered artifact" bug reported on
the NVIDIA forums?

## Running it

1. Install 3DGRUT once, outside this repo: see
   `docs/plans/2026-08-17-phase0-isaacsim-import-spike.md` Task 1.
2. `env -u PYTHONPATH .venv/bin/python spikes/phase0_isaacsim_import/generate_test_splat.py`
3. `source ~/tools/3dgrut/.venv/bin/activate && python -m threedgrut.export.scripts.ply_to_usd spikes/phase0_isaacsim_import/out/test_scene.ply --output_file spikes/phase0_isaacsim_import/out/test_scene.usdz && deactivate`
4. Import `spikes/phase0_isaacsim_import/out/test_scene.usdz` into Isaac Sim
   (`File > Import`) and check against the pass/fail checklist in Task 4 of
   the plan above.

## Result

<!-- Filled in after running Task 4: PASS / FAIL, date, Isaac Sim build,
     screenshot description, and if FAIL, which failure mode. -->
```

- [ ] **Step 2: Fill in the Result section**

Replace the placeholder comment in the README with the actual outcome from Task 4 Step 4 (date, pass/fail, notes).

- [ ] **Step 3: Update `docs/design.md`**

In `docs/design.md`, §3 "Phase 0", replace the sentence `**Chưa chạy phase này** — Isaac Sim đã sẵn sàng trên máy nên có thể thực hiện ngay.` with the actual outcome, e.g. (if PASS):

```markdown
**Kết quả (2026-08-17)**: PASS — xem `spikes/phase0_isaacsim_import/`. Gate mở, có thể triển
khai `train/`, `navmesh/`, `isaacsim_import/`.
```

or (if FAIL):

```markdown
**Kết quả (2026-08-17)**: FAIL — xem `spikes/phase0_isaacsim_import/README.md` cho chi tiết lỗi.
Cần phương án dự phòng trước khi triển khai `isaacsim_import/` (convert splat sang textured mesh,
hoặc point cloud renderer — xem §9 rủi ro).
```

Also update the §7 status table row for `isaacsim_import/` to reflect the outcome (e.g. "Phase 0: PASS, sẵn sàng bắt đầu" or "Phase 0: FAIL, cần phương án dự phòng").

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/camera-3d2sim
git add spikes/phase0_isaacsim_import/README.md docs/design.md
git commit -m "docs: record Phase 0 Isaac Sim import spike result"
```

---

## Self-Review

**Spec coverage**: `docs/design.md` §3 Phase 0 asks for exactly one thing — convert a sample `.ply` via `ply_to_usd`, import into Isaac Sim 6.0.1, confirm no layered-artifact rendering bug. Task 2 (fixture) + Task 3 (convert) + Task 4 (import/verify) + Task 5 (record) cover this end-to-end. §9's "Format .ply khác biệt giữa trainer" risk is explicitly addressed by generating the fixture in the exact schema 3DGRUT's own importer expects (verified against source, not guessed), sidestepping that risk for this spike (it resurfaces later when the real trainer for `train/` is chosen — out of scope here).

**Placeholder scan**: no TBD/TODO; every command, file path, and code block is concrete and was verified either by reading the cloned `nv-tlabs/3dgrut` source directly or by running commands on this machine. Task 4 is manual by necessity (GUI verification), not a placeholder — it has an explicit, concrete pass/fail checklist.

**Type consistency**: `generate_sphere_splat(output_path: Path, n_points: int = 8000, radius: float = 0.5) -> None` is defined once in Task 2 and used identically in its own test file, its `__main__` block, and Task 5's README — no signature drift.
