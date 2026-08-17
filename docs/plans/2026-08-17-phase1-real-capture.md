# Phase 1: Real Capture Session — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a real RTAB-Map SLAM tool installed and a real CLI capture entrypoint built, so a human can run an actual capture session with the D435 through `capture/` → `align_verify/` → `ingest/` and produce the project's first real (non-mock) capture session.

**Architecture:** RTAB-Map standalone GUI is built from source using only its core dependencies (PCL, OpenCV, Qt5, sqlite3, proj) — **no `librealsense2-dev`**, because RTAB-Map's own "Images" RGB-D source mode reads two folders of already-captured RGB/depth images plus a calibration YAML, which is exactly what `capture/record.py` already produces. This sidesteps `librealsense2-dev`'s broken/unreachable official apt repo entirely. A new `capture/cli.py` wires the already-tested library functions (`RealSenseFrameSource`, `create_capture_session`, `record_session`, manifest) into a runnable script, and writes the RTAB-Map calibration YAML from the session's saved intrinsics.

**Tech Stack:** RTAB-Map (C++ standalone, built from source, apt deps only). Python 3.12 (`camera-3d2sim/.venv`) for the new CLI + YAML generator.

**Spec:** [docs/design.md](../design.md) §11 Phase 1

## Global Constraints

- RTAB-Map build uses ONLY these apt packages (confirmed available in this Ubuntu 22.04 machine's configured repos): `libsqlite3-dev libpcl-dev libopencv-dev git cmake libproj-dev libqt5svg5-dev`. Do NOT attempt to install `librealsense2-dev` — it is not available in this machine's configured apt repos (`apt-cache policy librealsense2-dev` returns no candidate) and Intel's official repo is currently unreachable (confirmed via web search, known ongoing issue). RTAB-Map does not need it for this project's workflow.
- RTAB-Map official build: `git clone https://github.com/introlab/rtabmap.git rtabmap && cd rtabmap/build && cmake .. && make -j4 && sudo make install`. Build into `~/tools/rtabmap` (outside this repo, same convention as `~/tools/3dgrut` from Phase 0) — it is an external tool dependency, not project source. `sudo make install` requires interactive sudo — if the executing session has no TTY/passwordless sudo (confirmed in Phase 0's Task 1 report: this machine's sudo requires a password), install to a user-writable prefix instead: `cmake -DCMAKE_INSTALL_PREFIX=$HOME/tools/rtabmap/install ..`, then `make install` (no sudo), and add `$HOME/tools/rtabmap/install/bin` to PATH when running `rtabmap`.
- RTAB-Map's calibration YAML format (standard ROS `sensor_msgs/CameraInfo` YAML, confirmed via `introlab/rtabmap_ros` camera_info conversion code): `image_width`, `image_height`, `camera_name`, `camera_matrix{rows:3,cols:3,data:[fx,0,cx, 0,fy,cy, 0,0,1]}`, `distortion_model`, `distortion_coefficients{rows:1,cols:5,data:[...]}`, `rectification_matrix{rows:3,cols:3,data:<identity>}`, `projection_matrix{rows:3,cols:4,data:[fx,0,cx,0, 0,fy,cy,0, 0,0,1,0]}`.
- `capture/realsense_source.py`'s saved intrinsics dict currently omits distortion coefficients (`manifest.json`'s documented schema in `docs/design.md` §5 has `intrinsics{fx,fy,cx,cy,dist}` but the field is never populated) — Task 1 fixes this using `rs.intrinsics.coeffs` (a 5-float list, standard pyrealsense2 field).
- Run all `pytest` invocations with `unset PYTHONPATH` (ROS2 Humble's `PYTHONPATH` shadows this project's `.venv` — see `docs/design.md` §8).
- `capture/realsense_source.py` has no existing unit test (it requires physical hardware to import/construct — matches the project's existing convention, confirmed by its absence from `tests/`). Do not add hardware-dependent tests for it; the distortion-coefficient fix in Task 1 is a small, visually-verifiable code change, not a new test target.

---

## File Structure

- `capture/realsense_source.py` — modify: add `dist` (distortion coefficients) to the saved `intrinsics` dict.
- `capture/rtabmap_calib.py` — create: builds the RTAB-Map calibration YAML dict from a manifest's `intrinsics`.
- `tests/test_rtabmap_calib.py` — create: tests for the YAML generator.
- `capture/cli.py` — create: `python -m capture.cli --env-id <id> [--max-frames N] [--no-lock-exposure]` entrypoint wiring `RealSenseFrameSource` → `create_capture_session` → `record_session` → writes manifest + calibration YAML. Requires physical hardware to run (not unit-testable beyond argument parsing / wiring logic, which IS tested with a fake frame source).
- `tests/test_cli.py` — create: tests the wiring logic (session creation, manifest writing, calibration YAML writing) using a fake frame source, without touching `pyrealsense2`.

## Interfaces

- `capture.rtabmap_calib.build_calibration_dict(intrinsics: dict, camera_name: str = "d435_color") -> dict` — pure function, returns the nested dict structure to serialize as YAML. Consumed by `capture/cli.py`.
- `capture.rtabmap_calib.write_calibration_yaml(intrinsics: dict, output_path: Path, camera_name: str = "d435_color") -> None` — writes it to disk (RTAB-Map's calibration YAML needs a `%YAML:1.0` header line before the mapping, OpenCV's `cv::FileStorage` convention — use `yaml.dump` for the body and prepend the header line manually).
- `capture.cli.main(argv: list[str] | None = None) -> int` — CLI entrypoint; `argv=None` reads `sys.argv[1:]` (standard argparse convention), used directly by tests via `main(["--env-id", "test_room"])`. Depends on a module-level `FRAME_SOURCE_FACTORY` hook (see Task 2) so tests can inject a fake source instead of `RealSenseFrameSource`.

---

### Task 1: Build RTAB-Map from source + fix missing distortion coefficients

**Files:**
- Create (outside repo): `~/tools/rtabmap/` (git clone + build)
- Modify: `capture/realsense_source.py`

**Interfaces:**
- Produces: a `rtabmap` binary runnable from PATH (or a known install path recorded in the task report) — Task 3 (manual) depends on this.
- Produces: `intrinsics["dist"]` — a list of 5 floats — added to the dict `RealSenseFrameSource.__init__` builds (currently ends at `"exposure_mode": exposure_mode` around line 32-33 of `capture/realsense_source.py`).

- [ ] **Step 1: Install RTAB-Map's build dependencies**

```bash
sudo apt-get update
sudo apt-get install -y libsqlite3-dev libpcl-dev libopencv-dev git cmake libproj-dev libqt5svg5-dev
```

- [ ] **Step 2: Clone and build RTAB-Map (user-writable prefix, no sudo needed for install)**

```bash
mkdir -p ~/tools
git clone https://github.com/introlab/rtabmap.git ~/tools/rtabmap
cd ~/tools/rtabmap/build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/tools/rtabmap/install ..
make -j4
make install
```

Expected: `cmake` configure output should NOT show RealSense2 as found (that's fine/expected — we're not building with it) but SHOULD show PCL, OpenCV, Qt5, and SQLite3 as found. Build can take 20-40+ minutes (large C++ codebase). If `cmake` fails on a dependency, re-check Step 1 installed correctly — do not attempt to work around a missing PCL/OpenCV/Qt5 by disabling it, those are required for the GUI application.

- [ ] **Step 3: Verify the GUI binary runs**

```bash
$HOME/tools/rtabmap/install/bin/rtabmap --version
```

Expected: prints a version string, exits 0, no missing-shared-library errors (`error while loading shared libraries`). If there's a shared-library error, the fix is `export LD_LIBRARY_PATH=$HOME/tools/rtabmap/install/lib:$LD_LIBRARY_PATH` before running — record this in the report if needed, since Task 3 (manual, run by a human later) will need to know.

- [ ] **Step 4: Add distortion coefficients to the saved intrinsics**

Read `capture/realsense_source.py` first. In `RealSenseFrameSource.__init__`, the `self.intrinsics` dict is built from `color_stream.get_intrinsics()` (variable `intr`). Add `"dist": list(intr.coeffs)` to that dict (pyrealsense2's `intrinsics.coeffs` is a list of 5 floats — Brown-Conrady distortion coefficients for the D435 color stream, `[k1, k2, p1, p2, k3]` order matching OpenCV's `plumb_bob`/standard 5-coefficient model).

- [ ] **Step 5: Confirm existing tests still pass (this file has no direct test, but confirm nothing else broke)**

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -m pytest -v
```

Expected: 39 passed (unchanged — `realsense_source.py` isn't imported by any existing test).

- [ ] **Step 6: Commit**

```bash
git add capture/realsense_source.py
git commit -m "feat: save color-stream distortion coefficients in captured intrinsics"
```

(Nothing to commit for the RTAB-Map build itself — it lives outside the repo at `~/tools/rtabmap`, same convention as `~/tools/3dgrut`.)

---

### Task 2: RTAB-Map calibration YAML generator

**Files:**
- Create: `capture/rtabmap_calib.py`
- Test: `tests/test_rtabmap_calib.py`

**Interfaces:**
- Consumes: an `intrinsics` dict shaped like `capture/realsense_source.py`'s (after Task 1): `{"fx", "fy", "cx", "cy", "width", "height", "dist": [5 floats], ...}` (extra keys like `depth_scale`/`exposure_mode` are ignored).
- Produces: `build_calibration_dict(...) -> dict` and `write_calibration_yaml(...) -> None`, both used by Task 3's `capture/cli.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rtabmap_calib.py`:

```python
import yaml
from capture.rtabmap_calib import build_calibration_dict, write_calibration_yaml


def _sample_intrinsics():
    return {
        "fx": 911.3, "fy": 911.5, "cx": 640.0, "cy": 360.0,
        "width": 1280, "height": 720,
        "dist": [0.1, -0.2, 0.001, -0.002, 0.05],
        "depth_scale": 0.001, "exposure_mode": "locked",
    }


def test_build_calibration_dict_has_expected_top_level_fields():
    calib = build_calibration_dict(_sample_intrinsics(), camera_name="d435_color")

    assert calib["image_width"] == 1280
    assert calib["image_height"] == 720
    assert calib["camera_name"] == "d435_color"
    assert calib["distortion_model"] == "plumb_bob"


def test_build_calibration_dict_camera_matrix_matches_intrinsics():
    calib = build_calibration_dict(_sample_intrinsics())

    m = calib["camera_matrix"]
    assert m["rows"] == 3 and m["cols"] == 3
    assert m["data"] == [911.3, 0.0, 640.0, 0.0, 911.5, 360.0, 0.0, 0.0, 1.0]


def test_build_calibration_dict_distortion_coefficients_match_intrinsics():
    calib = build_calibration_dict(_sample_intrinsics())

    d = calib["distortion_coefficients"]
    assert d["rows"] == 1 and d["cols"] == 5
    assert d["data"] == [0.1, -0.2, 0.001, -0.002, 0.05]


def test_build_calibration_dict_rectification_matrix_is_identity():
    calib = build_calibration_dict(_sample_intrinsics())

    r = calib["rectification_matrix"]
    assert r["data"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_build_calibration_dict_projection_matrix_matches_camera_matrix_padded():
    calib = build_calibration_dict(_sample_intrinsics())

    p = calib["projection_matrix"]
    assert p["rows"] == 3 and p["cols"] == 4
    assert p["data"] == [911.3, 0.0, 640.0, 0.0, 0.0, 911.5, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_write_calibration_yaml_writes_opencv_header_and_valid_yaml(tmp_path):
    out_path = tmp_path / "d435_color.yaml"

    write_calibration_yaml(_sample_intrinsics(), out_path, camera_name="d435_color")

    text = out_path.read_text()
    assert text.startswith("%YAML:1.0\n")

    body = "\n".join(text.splitlines()[1:])
    parsed = yaml.safe_load(body)
    assert parsed["camera_name"] == "d435_color"
    assert parsed["camera_matrix"]["data"][0] == 911.3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_rtabmap_calib.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'capture.rtabmap_calib'`.

- [ ] **Step 3: Write the implementation**

Create `capture/rtabmap_calib.py`:

```python
"""Builds RTAB-Map's OpenCV-style camera_info calibration YAML from captured
intrinsics (docs/plans/2026-08-17-phase1-real-capture.md Task 2)."""
from pathlib import Path

import yaml


def build_calibration_dict(intrinsics: dict, camera_name: str = "d435_color") -> dict:
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    return {
        "image_width": intrinsics["width"],
        "image_height": intrinsics["height"],
        "camera_name": camera_name,
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1, "cols": 5,
            "data": list(intrinsics["dist"]),
        },
        "rectification_matrix": {
            "rows": 3, "cols": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }


def write_calibration_yaml(intrinsics: dict, output_path: Path, camera_name: str = "d435_color") -> None:
    calib = build_calibration_dict(intrinsics, camera_name=camera_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("%YAML:1.0\n" + yaml.dump(calib, default_flow_style=None, sort_keys=False))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_rtabmap_calib.py -v
```

Expected: 6 passed. If `yaml` (PyYAML) isn't installed in `.venv`, run `uv pip install pyyaml` first — but check `requirements.txt` first, it may already be listed (it is, for `_meta/gates.yaml` parsing).

- [ ] **Step 5: Commit**

```bash
git add capture/rtabmap_calib.py tests/test_rtabmap_calib.py
git commit -m "feat: generate RTAB-Map calibration YAML from captured intrinsics"
```

---

### Task 3: `capture/cli.py` entrypoint

**Files:**
- Create: `capture/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `capture.session.create_capture_session`, `capture.manifest.new_manifest`/`save_manifest`, `capture.record.record_session`, `capture.rtabmap_calib.write_calibration_yaml` (Task 2), `capture.realsense_source.RealSenseFrameSource` (Task 1).
- Produces: `main(argv: list[str] | None = None) -> int`, and a module-level `FRAME_SOURCE_FACTORY` callable (defaulting to `RealSenseFrameSource`) that tests monkeypatch to inject a fake source.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import numpy as np

import capture.cli as cli
from capture.record import Frame


class _FakeFrameSource:
    """Stands in for RealSenseFrameSource without touching pyrealsense2/hardware."""

    def __init__(self, width=640, height=480, fps=30, lock_exposure_us=8000):
        self.intrinsics = {
            "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "width": width, "height": height,
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "depth_scale": 0.001, "exposure_mode": "locked",
        }
        self._n_frames = 5

    def frames(self):
        for i in range(self._n_frames):
            color = np.full((4, 4, 3), i, dtype=np.uint8)
            depth = np.full((4, 4), i * 1000, dtype=np.uint16)
            yield Frame(index=i, timestamp=i * 0.1, color=color, depth=depth)


def test_main_writes_session_manifest_and_calibration_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "FRAME_SOURCE_FACTORY", _FakeFrameSource)

    exit_code = cli.main([
        "--scenes-root", str(tmp_path),
        "--env-id", "test_room",
        "--sop-version", "sop-v1",
        "--capture-profile-id", "p01",
        "--max-frames", "5",
    ])

    assert exit_code == 0
    sessions = list((tmp_path / "test_room" / "captures").iterdir())
    assert len(sessions) == 1
    session_dir = sessions[0]

    manifest_path = session_dir / "manifest.json"
    assert manifest_path.exists()
    import json
    manifest = json.loads(manifest_path.read_text())
    assert manifest["env_id"] == "test_room"
    assert manifest["capture_stats"]["n_frames"] == 5
    assert manifest["hardware"]["intrinsics"]["fx"] == 600.0

    calib_path = session_dir / "raw" / "calib" / "d435_color.yaml"
    assert calib_path.exists()
    assert calib_path.read_text().startswith("%YAML:1.0\n")

    assert (session_dir / "raw" / "rgb" / "000000.png").exists()
    assert (session_dir / "raw" / "depth" / "000004.png").exists()


def test_main_respects_max_frames_argument(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "FRAME_SOURCE_FACTORY", _FakeFrameSource)

    cli.main([
        "--scenes-root", str(tmp_path),
        "--env-id", "test_room",
        "--sop-version", "sop-v1",
        "--capture-profile-id", "p01",
        "--max-frames", "2",
    ])

    session_dir = next((tmp_path / "test_room" / "captures").iterdir())
    manifest = __import__("json").loads((session_dir / "manifest.json").read_text())
    assert manifest["capture_stats"]["n_frames"] == 2
    assert not (session_dir / "raw" / "rgb" / "000002.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'capture.cli'`.

- [ ] **Step 3: Write the implementation**

Create `capture/cli.py`:

```python
"""CLI entrypoint wiring capture/ library functions into a runnable capture
session (docs/plans/2026-08-17-phase1-real-capture.md Task 3).

Usage: python -m capture.cli --env-id <env_id> [--scenes-root PATH]
       [--max-frames N] [--sop-version SOP] [--capture-profile-id ID]
"""
import argparse
import sys
from pathlib import Path

from capture.manifest import new_manifest, save_manifest
from capture.realsense_source import RealSenseFrameSource
from capture.record import record_session
from capture.rtabmap_calib import write_calibration_yaml
from capture.session import create_capture_session

FRAME_SOURCE_FACTORY = RealSenseFrameSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a capture session from the D435")
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--sop-version", type=str, default="sop-v1")
    parser.add_argument("--capture-profile-id", type=str, default="p01")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    session = create_capture_session(args.scenes_root, args.env_id)
    manifest = new_manifest(
        capture_id=session.root.name,
        env_id=args.env_id,
        capture_version=1,
        sop_version=args.sop_version,
        capture_profile_id=args.capture_profile_id,
    )

    source = FRAME_SOURCE_FACTORY()
    manifest = record_session(source.frames(), source.intrinsics, session, manifest, max_frames=args.max_frames)
    save_manifest(manifest, session.manifest_path)

    write_calibration_yaml(source.intrinsics, session.raw_calib / "d435_color.yaml")

    print(f"Wrote capture session to {session.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest -v
```

Expected: 47 passed (39 original + 6 from Task 2 + 2 from Task 3).

- [ ] **Step 6: Commit**

```bash
git add capture/cli.py tests/test_cli.py
git commit -m "feat: add capture CLI entrypoint wiring session/record/calibration"
```

---

### Task 4: Physical capture + RTAB-Map SLAM processing (manual — requires the physical camera and a room)

This task cannot be automated by an agent — it requires a human to physically move around a room with the D435, and drive the RTAB-Map GUI interactively. **A human runs these steps directly.**

**Files:** none (produces a capture session directory under wherever `--scenes-root` points, e.g. `~/scenes/`)

- [ ] **Step 1: Physical prep per `_meta/sop_capture.md`**

Print/place the AprilTag anchor (tag id 0) fixed in the room, place the 1.000m scale bar + checkerboard visible in the first frame. Plug in D435 to a **USB3** port (verify with `lsusb -t`, look for `5000M` — see `docs/design.md` §8), let it thermally stabilize ≥5 minutes.

- [ ] **Step 2: Record**

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -m capture.cli --scenes-root ~/scenes --env-id <choose a name, e.g. lab_room_a>
```

Move slowly (target p95 angular velocity < 30°/s per SOP), cover each 1×1m floor cell from ≥2 angles >30° apart, close the loop if the trajectory allows.

- [ ] **Step 3: Process in RTAB-Map GUI**

Open `$HOME/tools/rtabmap/install/bin/rtabmap` (from Task 1). `Preferences → Source → RGB-D → Images`: point the RGB folder at `<session>/raw/rgb/`, the depth folder at `<session>/raw/depth/`, and load the calibration file at `<session>/raw/calib/d435_color.yaml` (produced by Task 3). Start mapping, let it process to the end, watch for loop closures in the log. `File → Export poses → RGBD-SLAM format (*.txt)` → save to `<session>/slam/rtabmap_poses.txt`. `Tools → Post-processing` → run default cleanup. `Edit → View Point Clouds → Export` → save point cloud to `<session>/slam/rtabmap_cloud.ply` (and mesh to `<session>/slam/rtabmap_mesh.ply` if available).

- [ ] **Step 4: Record the outcome**

Note whether RTAB-Map's "Images" source mode worked as expected (this is the first real test of that approach) and whether the calibration YAML from Task 2 loaded correctly — needed for Task 5's report and for fixing Task 2 if the format turns out to be wrong in practice.

---

### Task 5: `align_verify/run.py` orchestrator (glue code that doesn't exist yet)

**Files:**
- Create: `align_verify/run.py`
- Test: `tests/test_align_verify_run.py`

**Context:** `align_verify/gravity_align.py`, `anchor.py`, `gate_a.py` are unit-tested against
in-memory arrays/point clouds but nothing currently reads a session's real `slam/cloud.ply` +
`slam/trajectory.tum`, applies the gravity transform, writes `align/mesh_aligned.ply` +
`align/T_world_from_slam.json`, and updates the manifest — this glue is required before Task 6 can
run Gate A on a real session.

**Interfaces:**
- Produces: `run_alignment(session, manifest: dict) -> dict` — mutates and returns `manifest` with
  `manifest["align"]` populated (`T_world_from_slam`, `floor_inlier_ratio`, `gravity_residual_deg`,
  `T_anchor_from_world` if an anchor tag was found), and writes `session.align_dir / "mesh_aligned.ply"`
  (the gravity-aligned point cloud — named to match the schema in `docs/design.md` §4; it is a point
  cloud, not a triangle mesh, since no meshing step exists in this pipeline) and
  `session.align_dir / "T_world_from_slam.json"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_align_verify_run.py`:

```python
import json
import numpy as np
import open3d as o3d

from capture.session import create_capture_session
from align_verify.run import run_alignment


def _tilted_floor_cloud(n=2000, tilt_deg=5.0):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    pts = np.column_stack([xy, np.zeros(n)])
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("x", tilt_deg, degrees=True).as_matrix()
    pts = pts @ R.T
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    return cloud


def test_run_alignment_writes_aligned_cloud_and_transform(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    cloud = _tilted_floor_cloud()
    o3d.io.write_point_cloud(str(session.slam_dir / "cloud.ply"), cloud)

    poses = [
        {"timestamp": 0.0, "t": np.array([0.0, 0.0, 1.0]), "q": np.array([0.0, 0.0, 0.0, 1.0])},
        {"timestamp": 0.1, "t": np.array([0.1, 0.0, 1.0]), "q": np.array([0.0, 0.0, 0.0, 1.0])},
    ]
    manifest = {"align": {}}

    result = run_alignment(session, manifest, poses=poses)

    assert result is manifest
    aligned_path = session.align_dir / "mesh_aligned.ply"
    assert aligned_path.exists()
    aligned = o3d.io.read_point_cloud(str(aligned_path))
    aligned_z = np.asarray(aligned.points)[:, 2]
    assert np.abs(aligned_z).max() < 0.05  # flattened to the z=0 plane, within RANSAC tolerance

    transform_path = session.align_dir / "T_world_from_slam.json"
    assert transform_path.exists()
    saved = json.loads(transform_path.read_text())
    assert len(saved["T_world_from_slam"]) == 4 and len(saved["T_world_from_slam"][0]) == 4

    assert manifest["align"]["gravity_residual_deg"] < 1.0
    assert manifest["align"]["floor_inlier_ratio"] > 0.9


def test_run_alignment_without_anchor_detections_leaves_anchor_transform_absent(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    cloud = _tilted_floor_cloud()
    o3d.io.write_point_cloud(str(session.slam_dir / "cloud.ply"), cloud)
    poses = [{"timestamp": 0.0, "t": np.array([0.0, 0.0, 1.0]), "q": np.array([0.0, 0.0, 0.0, 1.0])}]
    manifest = {"align": {}}

    run_alignment(session, manifest, poses=poses, anchor_detections=None)

    assert "T_anchor_from_world" not in manifest["align"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_align_verify_run.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'align_verify.run'`.

- [ ] **Step 3: Write the implementation**

Create `align_verify/run.py`:

```python
"""Orchestrates gravity alignment (+ optional anchor registration) on a real
capture session: reads slam/cloud.ply + poses, writes align/mesh_aligned.ply +
align/T_world_from_slam.json, updates manifest["align"]
(docs/plans/2026-08-17-phase1-real-capture.md Task 5)."""
import json

import numpy as np
import open3d as o3d

from align_verify.anchor import compute_anchor_from_world
from align_verify.gravity_align import apply_transform, apply_transform_to_poses, fit_floor_plane, gravity_alignment_transform


def run_alignment(session, manifest: dict, poses: list[dict], anchor_detections: list[dict] | None = None) -> dict:
    cloud = o3d.io.read_point_cloud(str(session.slam_dir / "cloud.ply"))
    plane_model, floor_inlier_ratio = fit_floor_plane(cloud)
    T, residual_deg = gravity_alignment_transform(plane_model)

    aligned_points = apply_transform(T, np.asarray(cloud.points))
    aligned_cloud = o3d.geometry.PointCloud()
    aligned_cloud.points = o3d.utility.Vector3dVector(aligned_points)
    session.align_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(session.align_dir / "mesh_aligned.ply"), aligned_cloud)

    aligned_poses = apply_transform_to_poses(T, poses)

    manifest["align"]["T_world_from_slam"] = T.tolist()
    manifest["align"]["floor_inlier_ratio"] = floor_inlier_ratio
    manifest["align"]["gravity_residual_deg"] = residual_deg
    (session.align_dir / "T_world_from_slam.json").write_text(
        json.dumps({"T_world_from_slam": T.tolist()}, indent=2)
    )

    if anchor_detections:
        T_world_from_camera_by_frame = {
            i: _pose_to_matrix(p) for i, p in enumerate(aligned_poses)
        }
        T_anchor_from_world = compute_anchor_from_world(anchor_detections, T_world_from_camera_by_frame)
        if T_anchor_from_world is not None:
            manifest["align"]["T_anchor_from_world"] = T_anchor_from_world.tolist()

    return manifest


def _pose_to_matrix(pose: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(pose["q"]).as_matrix()
    M[:3, 3] = pose["t"]
    return M
```

- [ ] **Step 4: Run test to verify it passes**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_align_verify_run.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest -v
```

Expected: 49 passed.

- [ ] **Step 6: Commit**

```bash
git add align_verify/run.py tests/test_align_verify_run.py
git commit -m "feat: add align_verify orchestrator (gravity align + anchor on a real session)"
```

---

### Task 6: Run align_verify + ingest on the real session, record the outcome

**Files:**
- Modify: `docs/design.md` (§7 status table, §11 Phase 1 section)
- Modify: `_meta/defects_log.md` (if Gate A rejects)

**Interfaces:**
- Consumes: `<session>/slam/rtabmap_poses.txt` and `<session>/slam/rtabmap_cloud.ply` from Task 4 (manual).

- [ ] **Step 1: Import SLAM output**

`capture/slam_import.py`'s real signature is `import_slam_output(rtabmap_poses_path: Path, rtabmap_cloud_path: Path, session) -> dict` — it copies both files into `session.slam_dir` and returns parsed poses. `session` needs a `SessionPaths` object, easiest obtained by re-deriving it from the known capture directory rather than re-running `create_capture_session` (which would allocate a NEW version number):

```bash
unset PYTHONPATH && .venv/bin/python -c "
from pathlib import Path
from capture.session import SessionPaths
from capture.slam_import import import_slam_output

root = Path('~/scenes/<env_id>/captures/<capture_id>').expanduser()  # from Task 4
session = SessionPaths(
    root=root, raw_rgb=root/'raw/rgb', raw_depth=root/'raw/depth', raw_calib=root/'raw/calib',
    slam_dir=root/'slam', align_dir=root/'align', eval_dir=root/'eval',
    manifest_path=root/'manifest.json', capture_log_path=root/'capture_log.md',
)
result = import_slam_output(root/'slam/rtabmap_poses.txt', root/'slam/rtabmap_cloud.ply', session)
print(result['n_poses'], 'poses imported')
"
```

- [ ] **Step 2: Run gravity align + anchor registration (Task 5's orchestrator)**

```bash
unset PYTHONPATH && .venv/bin/python -c "
from pathlib import Path
from capture.session import SessionPaths
from capture.manifest import load_manifest, save_manifest
from align_verify.run import run_alignment

root = Path('~/scenes/<env_id>/captures/<capture_id>').expanduser()
session = SessionPaths(root=root, raw_rgb=root/'raw/rgb', raw_depth=root/'raw/depth',
    raw_calib=root/'raw/calib', slam_dir=root/'slam', align_dir=root/'align', eval_dir=root/'eval',
    manifest_path=root/'manifest.json', capture_log_path=root/'capture_log.md')
manifest = load_manifest(session.manifest_path)
# poses: re-parse from the copied trajectory.tum (Task 5's run_alignment takes poses directly,
# not a file path, to keep it decoupled from slam_import's copy step)
from capture.slam_import import parse_tum_poses
poses = parse_tum_poses(session.slam_dir / 'trajectory.tum')
run_alignment(session, manifest, poses=poses)  # anchor_detections=None until AprilTag detection is wired up
save_manifest(manifest, session.manifest_path)
print('gravity_residual_deg:', manifest['align']['gravity_residual_deg'])
print('floor_inlier_ratio:', manifest['align']['floor_inlier_ratio'])
"
```

Note: `run_alignment` is called with `anchor_detections=None` here — Task 5's orchestrator supports
anchor registration but nothing in this plan yet wires `align_verify/anchor.py`'s
`detect_tags_in_frame` across all captured RGB frames to build the `anchor_detections` list. For
this first real session, anchor registration can be skipped (`manifest["align"]` will simply lack
`T_anchor_from_world` — Gate A does not require it, only gravity+scale+trajectory+coverage do); wiring
AprilTag detection across the frame sequence is worth its own small follow-up task once this session
proves the rest of the pipeline works.

- [ ] **Step 2b: Manual measurement entry + Gate A**

Measure 3 known distances by tape per the SOP, add them to `manifest["ground_truth"]["measurements"]`
as `{"name": ..., "tape_m": ..., "cloud_m": <measured on mesh_aligned.ply>}`. Then:

```bash
unset PYTHONPATH && .venv/bin/python -c "
from pathlib import Path
import yaml
from capture.manifest import load_manifest, save_manifest
from capture.slam_import import parse_tum_poses
from align_verify.gate_a import load_thresholds, run_gate_a

root = Path('~/scenes/<env_id>/captures/<capture_id>').expanduser()
manifest = load_manifest(root / 'manifest.json')
poses = parse_tum_poses(root / 'slam/trajectory.tum')
thresholds = load_thresholds(Path('_meta/gates.yaml'))
# blur_flags and floor_points_xy: this plan has no blur-detection or floor-point-extraction code yet
# (both are needed by check_trajectory/check_coverage) -- see the concern noted in Task 6 Step 3.
"
```

**Open gap found while writing this task**: `run_gate_a` needs `blur_flags` (per-frame blur
detection — not implemented anywhere in this codebase yet) and `floor_points_xy` (2D floor points for
coverage checking — `run_alignment`'s aligned cloud has this, but nothing extracts just the
near-z=0 points into the `(N,2)` array `check_coverage` expects). Do not fabricate placeholder values
for these — implement them for real (small, focused additions to `align_verify/`) or explicitly mark
Gate A's trajectory/coverage checks as skipped-with-reason for this first run, and record that gap in
Step 3's report rather than silently passing fake data through the gate.

- [ ] **Step 3: Record the outcome**

If **APPROVED**: run `ingest/package.py` to finalize the session (read-only), then `_meta/scripts/reindex.py` to update the catalog. Update `docs/design.md` §7's `capture/`/`align_verify/`/`ingest/` rows to say a real session passed, and §11 Phase 1 with the result and date.

If **REJECTED**: add the defect code(s) to `_meta/defects_log.md` per the taxonomy in `docs/design.md` §9, and check whether any code appears ≥2 times (triggering the SOP-bump rule already documented there). Update §11 Phase 1 with what failed and why — this is real, valuable flywheel data either way.

- [ ] **Step 4: Commit**

```bash
git add docs/design.md _meta/defects_log.md  # whichever changed
git commit -m "docs: record first real capture session outcome (Phase 1)"
```
