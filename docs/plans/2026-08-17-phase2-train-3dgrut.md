# Phase 2: `train/` (3DGRUT) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a capture session (gravity-aligned poses + RGB frames + point cloud) into a 3DGRUT-trainable dataset, train a metric-scale Gaussian Splat, and score it against Gate B — reusing the exact `~/tools/3dgrut` install and `ply_to_usd`/`export_usd` export path already validated in Phase 0.

**Architecture:** 3DGRUT's `colmap` dataset loader reads a hand-writable COLMAP-format sparse reconstruction (`sparse/0/cameras.txt` + `images.txt`, text format, no COLMAP binary ever invoked) plus a plain PLY for Gaussian initialization (`initialization: fused_point_cloud`) — confirmed by reading `~/tools/3dgrut/threedgrut/datasets/dataset_colmap.py` directly. Camera poses come from this project's own gravity-aligned SLAM trajectory, not from re-running structure-from-motion. `dataset.normalize_world_space` defaults to `false` (confirmed in `~/tools/3dgrut/configs/dataset/colmap.yaml:5`) — metric scale is preserved without needing to disable anything, unlike the nerfstudio assumption in the original spec draft.

**Tech Stack:** Python (`camera-3d2sim/.venv`) for the dataset converter + Gate B scorer. `~/tools/3dgrut/.venv` (separate venv, already built in Phase 0) for training itself.

**Spec:** [docs/design.md](../design.md) §3 step [4], §11 Phase 2

## Global Constraints

- **Dataset format** (verified by reading `~/tools/3dgrut/threedgrut/datasets/dataset_colmap.py` and `colmap_gsplat.py` directly, not guessed):
  ```
  <dataset_root>/
    images/<frame>.png                  # copies of session's raw/rgb/*.png
    sparse/0/cameras.txt                 # one line: "1 OPENCV <w> <h> fx fy cx cy k1 k2 p1 p2"
    sparse/0/images.txt                  # two lines per image (see Task 1)
    fused_cloud.ply                      # x,y,z[,red,green,blue] — the gravity-aligned point cloud
  ```
- `images.txt` pose convention: `QW QX QY QZ TX TY TZ` is **world-to-camera**; 3DGRUT inverts it internally to camera-to-world. This project's poses (from `capture/slam_import.py`'s TUM parser, then gravity-aligned by `align_verify/run.py`) are camera-to-world with quaternion in **xyzw** order (scipy convention) — Task 1 must invert the pose and reorder the quaternion to **wxyz** when writing `images.txt`.
- COLMAP's `OPENCV` camera model takes 8 params (`fx fy cx cy k1 k2 p1 p2`) — no `k3`. This project's saved `intrinsics["dist"]` (Phase 1 Task 1) is `[k1, k2, p1, p2, k3]` (5-element plumb_bob/Brown-Conrady) — Task 1 uses the first 4 and drops `k3`. This is a reasonable simplification for the D435's RGB stream (already near-rectilinear; higher-order `k3` is typically negligible) — flag it, don't silently ignore it.
- `dataset.normalize_world_space` must stay `false` (the 3DGRUT default — do not set it to `true` anywhere in this project's configs) to preserve real-world meter scale established by gravity alignment + anchor registration.
- Config to train with: `apps/colmap_3dgut_mcmc.yaml` (MCMC densification, matches the command shown in NVIDIA's own official blog post for this exact GSplat→Isaac Sim workflow), with `dataset=colmap`, `initialization=fused_point_cloud`, `initialization.fused_point_cloud_path=<dataset_root>/fused_cloud.ply`, `dataset.normalize_world_space=false`, `export_usd.enabled=true`, `export_usd.format=nurec` (matching Phase 0's validated import path).
- Run all `pytest` invocations with `unset PYTHONPATH`.
- Training itself runs via `~/tools/3dgrut/.venv/bin/python`, invoked with `PATH` prefixed by that venv's `bin/` (not `source .../activate` — confirmed unreliable in this sandboxed environment, see Phase 0's Task 3 report).
- **Early development does not need a real capture session**: `~/tools/3dgrut`'s README documents `data/mipnerf360/garden` as a standard test dataset. Task 1-2 can be developed/tested by hand-converting a small slice of a public dataset if no real Phase 1 session exists yet when this plan executes — but Task 1's own unit tests use small synthetic fixtures (no external dataset needed for the tests themselves).

---

## File Structure

- `train/convert_session.py` — creates: converts a capture session directory into the 3DGRUT dataset layout above.
- `tests/test_convert_session.py` — creates: unit tests using a small synthetic session fixture.
- `train/gate_b.py` — creates: Gate B scoring (PSNR/SSIM on holdout frames, distance measurement error, floater check).
- `tests/test_gate_b.py` — creates: unit tests using synthetic image arrays.

## Interfaces

- `train.convert_session.write_colmap_dataset(session, manifest: dict, poses: list[dict], output_dir: Path) -> None` — `poses` are the **gravity-aligned** camera-to-world poses (xyzw quaternion, same shape as `align_verify.gravity_align.apply_transform_to_poses`'s output); `manifest["hardware"]["intrinsics"]` supplies `fx,fy,cx,cy,width,height,dist`. Reads RGB frames from `session.raw_rgb`, the aligned cloud from `session.align_dir / "mesh_aligned.ply"`.
- `train.gate_b.check_psnr_ssim(rendered: list[np.ndarray], reference: list[np.ndarray], psnr_min_db: float, psnr_warn_db: float, ssim_min: float) -> dict`
- `train.gate_b.check_measurements(measurements: list[dict], max_err_pct: float) -> dict` — same shape/contract as `align_verify.gate_a.check_scale`, reused conceptually (do not import Gate A's function directly — Gate B measures splat-render-vs-tape, not cloud-vs-tape; keep them separate per the plan's own file-structure principle of one responsibility per file).

---

### Task 1: `train/convert_session.py` — capture session → 3DGRUT COLMAP dataset

**Files:**
- Create: `train/convert_session.py`
- Test: `tests/test_convert_session.py`

**Interfaces:**
- Produces: dataset directory ready for `~/tools/3dgrut`'s `train.py --config-name apps/colmap_3dgut_mcmc.yaml path=<output_dir> ...` — Task 3 (manual training run) depends on this exact layout.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_convert_session.py`:

```python
import numpy as np
import imageio.v3 as iio
import open3d as o3d

from capture.session import create_capture_session
from train.convert_session import write_colmap_dataset


def _sample_manifest():
    return {"hardware": {"intrinsics": {
        "fx": 911.3, "fy": 911.5, "cx": 640.0, "cy": 360.0,
        "width": 1280, "height": 720,
        "dist": [0.1, -0.2, 0.001, -0.002, 0.05],
    }}}


def _sample_poses():
    # camera-to-world, xyzw quaternion (this project's convention)
    return [
        {"timestamp": 0.0, "t": np.array([0.0, 0.0, 1.5]), "q": np.array([0.0, 0.0, 0.0, 1.0])},
        {"timestamp": 0.1, "t": np.array([1.0, 0.0, 1.5]), "q": np.array([0.0, 0.0, 0.0, 1.0])},
    ]


def _make_session_with_frames(tmp_path, n=2):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    for i in range(n):
        img = np.full((4, 4, 3), i * 10, dtype=np.uint8)
        iio.imwrite(session.raw_rgb / f"{i:06d}.png", img)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    session.align_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(session.align_dir / "mesh_aligned.ply"), cloud)
    return session


def test_write_colmap_dataset_copies_images(tmp_path):
    session = _make_session_with_frames(tmp_path)
    out_dir = tmp_path / "dataset"

    write_colmap_dataset(session, _sample_manifest(), _sample_poses(), out_dir)

    assert (out_dir / "images" / "000000.png").exists()
    assert (out_dir / "images" / "000001.png").exists()


def test_write_colmap_dataset_cameras_txt_has_opencv_model_dropping_k3(tmp_path):
    session = _make_session_with_frames(tmp_path)
    out_dir = tmp_path / "dataset"

    write_colmap_dataset(session, _sample_manifest(), _sample_poses(), out_dir)

    lines = (out_dir / "sparse" / "0" / "cameras.txt").read_text().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[1] == "OPENCV"
    assert parts[2] == "1280" and parts[3] == "720"
    params = [float(x) for x in parts[4:]]
    assert len(params) == 8  # fx fy cx cy k1 k2 p1 p2 -- no k3
    assert params == [911.3, 911.5, 640.0, 360.0, 0.1, -0.2, 0.001, -0.002]


def test_write_colmap_dataset_images_txt_inverts_pose_and_reorders_quaternion(tmp_path):
    session = _make_session_with_frames(tmp_path)
    out_dir = tmp_path / "dataset"

    write_colmap_dataset(session, _sample_manifest(), _sample_poses(), out_dir)

    lines = (out_dir / "sparse" / "0" / "images.txt").read_text().splitlines()
    assert len(lines) == 4  # 2 images x 2 lines each

    first = lines[0].split()
    # identity rotation (q=[0,0,0,1] xyzw) -> wxyz [1,0,0,0]; world-to-camera of a
    # camera-to-world translation [0,0,1.5] with identity rotation is just [0,0,-1.5]
    assert first[0] == "1"
    qw, qx, qy, qz = (float(x) for x in first[1:5])
    assert abs(qw - 1.0) < 1e-6 and abs(qx) < 1e-6 and abs(qy) < 1e-6 and abs(qz) < 1e-6
    tx, ty, tz = (float(x) for x in first[5:8])
    assert abs(tx) < 1e-6 and abs(ty) < 1e-6 and abs(tz - (-1.5)) < 1e-6
    assert first[9] == "000000.png"
    assert lines[1] == ""  # required-but-empty POINTS2D line


def test_write_colmap_dataset_copies_aligned_cloud_as_fused_cloud(tmp_path):
    session = _make_session_with_frames(tmp_path)
    out_dir = tmp_path / "dataset"

    write_colmap_dataset(session, _sample_manifest(), _sample_poses(), out_dir)

    assert (out_dir / "fused_cloud.ply").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_convert_session.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'train'` (or `train.convert_session`).

- [ ] **Step 3: Write the implementation**

Create `train/__init__.py` (empty) and `train/convert_session.py`:

```python
"""Converts a capture session into 3DGRUT's COLMAP-format dataset layout
(docs/plans/2026-08-17-phase2-train-3dgrut.md Task 1). Poses in, poses are
world-to-camera QW QX QY QZ TX TY TZ per COLMAP's images.txt convention --
this project's poses are camera-to-world with xyzw quaternions, so both the
inversion and the quaternion reorder happen here."""
import shutil
from pathlib import Path

from scipy.spatial.transform import Rotation


def write_colmap_dataset(session, manifest: dict, poses: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(session.raw_rgb.glob("*.png"))
    for src in frame_paths:
        shutil.copy(src, images_dir / src.name)

    intr = manifest["hardware"]["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    k1, k2, p1, p2 = intr["dist"][:4]  # drop k3 -- OPENCV model has no 5th coefficient
    cameras_line = f"1 OPENCV {intr['width']} {intr['height']} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}"
    (sparse_dir / "cameras.txt").write_text(cameras_line + "\n")

    lines = []
    for image_id, (pose, frame_path) in enumerate(zip(poses, frame_paths), start=1):
        R_c2w = Rotation.from_quat(pose["q"]).as_matrix()  # camera-to-world, xyzw in
        t_c2w = pose["t"]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        qx, qy, qz, qw = Rotation.from_matrix(R_w2c).as_quat()  # scipy returns xyzw
        lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {frame_path.name}"
        )
        lines.append("")  # required-but-unused POINTS2D line

    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")

    shutil.copy(session.align_dir / "mesh_aligned.ply", output_dir / "fused_cloud.ply")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_convert_session.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full suite**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest -v
```

Expected: 53 passed (49 from Phase 1 + 4 here — run this task's numbers assuming Phase 1 already landed; if not, just confirm no existing test broke).

- [ ] **Step 6: Commit**

```bash
git add train/__init__.py train/convert_session.py tests/test_convert_session.py
git commit -m "feat: add capture-session to 3DGRUT COLMAP-dataset converter"
```

---

### Task 2: `train/gate_b.py` — PSNR/SSIM + measurement scoring

**Files:**
- Create: `train/gate_b.py`
- Test: `tests/test_gate_b.py`

**Interfaces:**
- Consumes: `_meta/gates.yaml`'s `gate_b` section (`psnr_min_db`, `psnr_warn_db`, `ssim_min`, `measurement_max_err_pct` — already defined).
- Produces: `check_psnr_ssim(...)`, `check_measurements(...)`, `run_gate_b(...)` — used by Task 3's real training run (manual/GPU-bound, not unit-tested end-to-end).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_b.py`:

```python
import numpy as np
from train.gate_b import check_measurements, check_psnr_ssim, run_gate_b


def test_check_psnr_ssim_identical_images_score_perfectly():
    img = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    result = check_psnr_ssim([img], [img], psnr_min_db=25.0, psnr_warn_db=22.0, ssim_min=0.80)

    assert result["pass"] is True
    assert result["psnr_db"] > 40.0
    assert result["ssim"] > 0.99


def test_check_psnr_ssim_noisy_image_fails_threshold():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    noisy = np.clip(ref.astype(int) + rng.integers(-80, 80, size=ref.shape), 0, 255).astype(np.uint8)

    result = check_psnr_ssim([noisy], [ref], psnr_min_db=25.0, psnr_warn_db=22.0, ssim_min=0.80)

    assert result["pass"] is False


def test_check_measurements_within_tolerance_passes():
    measurements = [
        {"name": "wall_a", "tape_m": 1.000, "cloud_m": 1.005},
        {"name": "wall_b", "tape_m": 2.000, "cloud_m": 1.990},
        {"name": "wall_c", "tape_m": 0.500, "cloud_m": 0.503},
    ]
    result = check_measurements(measurements, max_err_pct=2.0)
    assert result["pass"] is True


def test_check_measurements_over_tolerance_fails():
    measurements = [{"name": "wall_a", "tape_m": 1.000, "cloud_m": 1.05}]
    result = check_measurements(measurements, max_err_pct=2.0)
    assert result["pass"] is False


def test_run_gate_b_combines_checks_into_verdict():
    img = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    thresholds = {"psnr_min_db": 25.0, "psnr_warn_db": 22.0, "ssim_min": 0.80, "measurement_max_err_pct": 2.0}
    measurements = [{"name": "a", "tape_m": 1.0, "cloud_m": 1.005}]

    scorecard = run_gate_b([img], [img], measurements, thresholds)

    assert scorecard["verdict"] == "APPROVED"
    assert "psnr_ssim" in scorecard["checks"]
    assert "measurements" in scorecard["checks"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_gate_b.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'train.gate_b'`.

- [ ] **Step 3: Write the implementation**

Add `scikit-image` to `requirements.txt` if not already present (needed for SSIM — check first, `scikit-learn` is already listed but that's a different package). Create `train/gate_b.py`:

```python
"""Gate B: PSNR/SSIM on holdout frames + splat-vs-tape distance measurements
(docs/plans/2026-08-17-phase2-train-3dgrut.md Task 2)."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def check_psnr_ssim(rendered: list[np.ndarray], reference: list[np.ndarray],
                     psnr_min_db: float, psnr_warn_db: float, ssim_min: float) -> dict:
    psnrs = [peak_signal_noise_ratio(r, g) for r, g in zip(reference, rendered)]
    ssims = [structural_similarity(r, g, channel_axis=-1) for r, g in zip(reference, rendered)]
    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    ok = mean_psnr >= psnr_min_db and mean_ssim >= ssim_min
    return {"pass": ok, "psnr_db": mean_psnr, "ssim": mean_ssim,
            "warn": psnr_warn_db <= mean_psnr < psnr_min_db}


def check_measurements(measurements: list[dict], max_err_pct: float) -> dict:
    errs = []
    for m in measurements:
        err_pct = abs(m["cloud_m"] - m["tape_m"]) / m["tape_m"] * 100.0
        m["err_pct"] = err_pct
        errs.append(err_pct)
    worst = max(errs) if errs else float("inf")
    return {"pass": worst <= max_err_pct, "worst_err_pct": worst, "measurements": measurements}


def run_gate_b(rendered: list[np.ndarray], reference: list[np.ndarray],
               measurements: list[dict], thresholds: dict) -> dict:
    psnr_ssim = check_psnr_ssim(rendered, reference, thresholds["psnr_min_db"],
                                 thresholds["psnr_warn_db"], thresholds["ssim_min"])
    meas = check_measurements(measurements, thresholds["measurement_max_err_pct"])
    checks = {"psnr_ssim": psnr_ssim, "measurements": meas}
    verdict = "APPROVED" if all(c["pass"] for c in checks.values()) else "REJECTED"
    return {"verdict": verdict, "checks": checks}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
unset PYTHONPATH && .venv/bin/python -m pytest tests/test_gate_b.py -v
```

Expected: 5 passed. (If `scikit-image` wasn't already installed: `uv pip install scikit-image` first.)

- [ ] **Step 5: Commit**

```bash
git add train/gate_b.py tests/test_gate_b.py requirements.txt
git commit -m "feat: add Gate B scoring (PSNR/SSIM, measurement error)"
```

---

### Task 3: First real (or public-dataset) training run (manual — GPU-bound, not unit-testable)

This task runs actual 3DGRUT training, which takes real GPU time and produces genuinely varying
results — it is validated by inspection and Gate B's metrics, not a fixed pytest assertion. **A human
runs and watches this.**

**Files:** none in this repo (writes to `~/scenes/<env_id>/builds/<build_id>/` per `docs/design.md` §4)

- [ ] **Step 1: Convert a session**

Using a real Phase 1 session if one exists, or a small slice of `data/mipnerf360/garden` (see
`~/tools/3dgrut`'s README for how to download it) reformatted into this project's session directory
shape for an early dry run:

```bash
cd /home/ubuntu/camera-3d2sim
unset PYTHONPATH && .venv/bin/python -c "
from pathlib import Path
from capture.manifest import load_manifest
from capture.slam_import import parse_tum_poses
from train.convert_session import write_colmap_dataset

root = Path('~/scenes/<env_id>/captures/<capture_id>').expanduser()
manifest = load_manifest(root / 'manifest.json')
poses = parse_tum_poses(root / 'slam/trajectory.tum')  # NOTE: raw, not gravity-aligned -- see below
write_colmap_dataset(_session(root), manifest, poses, Path('~/scenes/<env_id>/builds/v01_gsA/config/dataset').expanduser())
"
```

**Open gap found while writing this task**: like Phase 1 Task 6, the gravity-aligned poses aren't
persisted anywhere by `align_verify/run.py` (only the aligned point cloud is). Either extend
`align_verify/run.py` to also write `align/trajectory_aligned.tum`, or re-apply
`align_verify.gravity_align.apply_transform_to_poses` here using the saved
`align/T_world_from_slam.json` before calling `write_colmap_dataset` — do not feed raw un-aligned
poses into the dataset, it would silently defeat the entire point of gravity alignment (this is
exactly the "lỗi im lặng" class of bug `docs/design.md` §3 warns about twice already).

- [ ] **Step 2: Sanity-check render BEFORE full training**

Per `docs/design.md` §3's explicit warning, render one frame at a known pose using the initialized
(pre-training) model and visually compare to the real source image — confirms axes/scale are right
before spending GPU time on a bad config. Use `~/tools/3dgrut`'s `render.py` or the GUI
(`with_gui=True`) pointed at iteration 0 of a training run.

- [ ] **Step 3: Train**

```bash
PATH="/home/ubuntu/tools/3dgrut/.venv/bin:$PATH" /home/ubuntu/tools/3dgrut/.venv/bin/python \
    -m train \
    --config-name apps/colmap_3dgut_mcmc.yaml \
    path=<dataset_dir> out_dir=<builds_dir> experiment_name=<build_id> \
    dataset.normalize_world_space=false \
    initialization=fused_point_cloud \
    initialization.fused_point_cloud_path=<dataset_dir>/fused_cloud.ply \
    export_usd.enabled=true export_usd.format=nurec
```

(Run from inside `~/tools/3dgrut` — `train.py` is at the repo root, invoke as
`python train.py --config-name ...` not `-m train`; correct the exact invocation against
`~/tools/3dgrut/README.md`'s current examples at execution time, this plan's command sketch may drift
from the tool's actual CLI over time.)

- [ ] **Step 4: Score Gate B**

Using `train/gate_b.run_gate_b(...)` (Task 2) with holdout-frame renders vs originals, and the 3
manual tape measurements re-measured on the trained splat render. Write `build.json` per the schema
in `docs/design.md` §5.

- [ ] **Step 5: Record the outcome**

Update `docs/design.md` §7 and §11 Phase 2 with the real result (APPROVED/USABLE_WITH_CAVEATS/REJECTED,
PSNR/SSIM numbers, whether the fused-point-cloud-init + no-COLMAP-SFM approach worked as expected).
