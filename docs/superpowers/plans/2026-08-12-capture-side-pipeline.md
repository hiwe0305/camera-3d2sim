# Capture-Side Pipeline (Laptop Slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the part of the RealSense→GSplat→Isaac Sim pipeline that runs entirely on the
laptop with the D435 attached: record a capture session, get camera poses via RTAB-Map, gravity-
align and anchor-register the session, run Gate A quality scoring, and package the session
read-only for later transfer to the Isaac Sim 6.0.1 server.

**Architecture:** A chain of small, independently-testable Python modules connected by plain
dicts/dataclasses and files on disk (per the spec's "filesystem is the database" principle). Every
module that touches hardware or a third-party GUI (RealSense camera, RTAB-Map, AprilTag detector)
is kept as a thin adapter behind a fake-able interface, so the actual decision logic (scoring,
math, file layout) is unit-tested without hardware in the loop.

**Tech Stack:** Python 3.10+, pyrealsense2, open3d, numpy, scipy, pyyaml, pupil-apriltags, pytest.

## Global Constraints

- No ROS2, no Docker (per spec §1 YAGNI principle).
- No DVC/MLflow — filesystem + JSON manifests only (spec §3.1).
- SLAM processing (RTAB-Map) is a **manual GUI step** guided by a checklist in
  `_meta/sop_capture.md`, not scripted — RTAB-Map's standalone CLI flags for the RealSense2
  driver are not reliably documented, so this plan does not fabricate them (user-approved
  adjustment, 2026-08-12).
- Capture sessions are named `YYYY-MM-DD_vNN` and become **read-only** after packaging (spec §3.2)
  — no in-place edits after `ingest/package.py` runs.
- Gate A thresholds live in `_meta/gates.yaml`, never hardcoded in Python (spec §3.4).
- Isaac Sim version for the whole pipeline is pinned to **6.0.1**; this slice does not touch Isaac
  Sim but `build.json`'s `versions.lock.isaacsim` (future slice) must record this value.
- Every module that wraps a hardware/GUI dependency (`capture/realsense_source.py`, the
  `pupil_apriltags.Detector` call inside `align_verify/anchor.py`) is exempted from unit tests and
  instead gets a documented manual test procedure — do not attempt to mock the SDK/GUI itself.

---

## File Structure

```
camera-3d2sim/
  requirements.txt
  pytest.ini
  capture/
    __init__.py
    session.py           # session directory scaffolding
    manifest.py           # manifest.json schema + load/save
    record.py             # frame-recording logic (hardware-agnostic)
    realsense_source.py   # pyrealsense2 adapter (manual-tested)
    slam_import.py         # parses RTAB-Map GUI export into slam/
  align_verify/
    __init__.py
    gravity_align.py      # RANSAC floor fit + T_world_from_slam
    anchor.py              # AprilTag anchor registration -> T_anchor_from_world
    gate_a.py               # Gate A scoring
  ingest/
    __init__.py
    package.py             # finalize manifest, sha256, chmod read-only
  _meta/
    gates.yaml
    sop_capture.md
    scripts/
      reindex.py            # catalog.md/json generator
  tests/
    test_session.py
    test_manifest.py
    test_record.py
    test_slam_import.py
    test_gravity_align.py
    test_anchor.py
    test_gate_a.py
    test_package.py
    test_reindex.py
```

---

### Task 1: Repo scaffolding + capture session directories

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `capture/__init__.py`
- Create: `capture/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `capture.session.SessionPaths` dataclass with fields `root, raw_rgb, raw_depth,
  raw_calib, slam_dir, align_dir, eval_dir, manifest_path, capture_log_path` (all `pathlib.Path`).
- Produces: `capture.session.create_capture_session(scenes_root: Path, env_id: str, day: str |
  None = None) -> SessionPaths` — `day` defaults to today (`YYYY-MM-DD`), auto-picks the next
  `vNN` for that env+day, creates all directories, returns `SessionPaths`.
- Produces: `capture.session.next_capture_version(env_dir: Path, day: str) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session.py
from pathlib import Path
from capture.session import create_capture_session, next_capture_version

def test_next_capture_version_starts_at_1(tmp_path):
    assert next_capture_version(tmp_path / "lab_room_a", "2026-08-12") == 1

def test_next_capture_version_increments(tmp_path):
    env_dir = tmp_path / "lab_room_a"
    (env_dir / "captures" / "2026-08-12_v01").mkdir(parents=True)
    (env_dir / "captures" / "2026-08-12_v02").mkdir(parents=True)
    assert next_capture_version(env_dir, "2026-08-12") == 3

def test_next_capture_version_is_per_day(tmp_path):
    env_dir = tmp_path / "lab_room_a"
    (env_dir / "captures" / "2026-08-11_v01").mkdir(parents=True)
    assert next_capture_version(env_dir, "2026-08-12") == 1

def test_create_capture_session_layout(tmp_path):
    paths = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    assert paths.root == tmp_path / "lab_room_a" / "captures" / "2026-08-12_v01"
    for d in [paths.raw_rgb, paths.raw_depth, paths.raw_calib, paths.slam_dir,
              paths.align_dir, paths.eval_dir]:
        assert d.is_dir()
    assert paths.manifest_path == paths.root / "manifest.json"
    assert paths.capture_log_path == paths.root / "capture_log.md"

def test_create_capture_session_increments_version(tmp_path):
    create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    paths2 = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    assert paths2.root.name == "2026-08-12_v02"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pip install pytest && pytest tests/test_session.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'capture'`)

- [ ] **Step 3: Write `requirements.txt` and `pytest.ini`**

```
# requirements.txt
pyrealsense2
open3d
numpy
scipy
pyyaml
pupil-apriltags
pytest
```

```ini
# pytest.ini
[pytest]
testpaths = tests
```

- [ ] **Step 4: Write `capture/__init__.py` (empty) and `capture/session.py`**

```python
# capture/session.py
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path


@dataclass
class SessionPaths:
    root: Path
    raw_rgb: Path
    raw_depth: Path
    raw_calib: Path
    slam_dir: Path
    align_dir: Path
    eval_dir: Path
    manifest_path: Path
    capture_log_path: Path


def next_capture_version(env_dir: Path, day: str) -> int:
    captures_dir = env_dir / "captures"
    if not captures_dir.exists():
        return 1
    prefix = f"{day}_v"
    versions = []
    for p in captures_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix):]
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=0) + 1


def create_capture_session(scenes_root: Path, env_id: str, day: str | None = None) -> SessionPaths:
    day = day or date_cls.today().isoformat()
    env_dir = scenes_root / env_id
    version = next_capture_version(env_dir, day)
    root = env_dir / "captures" / f"{day}_v{version:02d}"
    raw_rgb = root / "raw" / "rgb"
    raw_depth = root / "raw" / "depth"
    raw_calib = root / "raw" / "calib"
    slam_dir = root / "slam"
    align_dir = root / "align"
    eval_dir = root / "eval"
    for d in [raw_rgb, raw_depth, raw_calib, slam_dir, align_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return SessionPaths(
        root=root,
        raw_rgb=raw_rgb,
        raw_depth=raw_depth,
        raw_calib=raw_calib,
        slam_dir=slam_dir,
        align_dir=align_dir,
        eval_dir=eval_dir,
        manifest_path=root / "manifest.json",
        capture_log_path=root / "capture_log.md",
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_session.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini capture/__init__.py capture/session.py tests/test_session.py
git commit -m "feat: add capture session directory scaffolding"
```

---

### Task 2: Manifest schema module

**Files:**
- Create: `capture/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `capture.manifest.SCHEMA_VERSION: int = 1`.
- Produces: `capture.manifest.new_manifest(capture_id: str, env_id: str, capture_version: int,
  sop_version: str, capture_profile_id: str) -> dict` matching spec §3.3 key groups:
  `identity` fields at top level, plus `hardware`, `conditions`, `capture_stats`, `align`,
  `ground_truth` (`{"measurements": [], "holdout_frames": []}`), `artifacts` (`[]`), `gate`
  (`{"gateA": None, "defects": []}`).
- Produces: `capture.manifest.save_manifest(manifest: dict, path: Path) -> None`.
- Produces: `capture.manifest.load_manifest(path: Path) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'capture.manifest'`)

- [ ] **Step 3: Write `capture/manifest.py`**

```python
# capture/manifest.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest schema module"
```

---

### Task 3: Frame recording core logic (hardware-agnostic)

**Files:**
- Create: `capture/record.py`
- Test: `tests/test_record.py`

**Interfaces:**
- Consumes: `capture.session.SessionPaths` (Task 1).
- Produces: `capture.record.Frame` dataclass: `index: int, timestamp: float, color: np.ndarray
  (HxWx3 uint8), depth: np.ndarray (HxW uint16, millimeters)`.
- Produces: `capture.record.record_session(frames: Iterable[Frame], intrinsics: dict, session:
  SessionPaths, manifest: dict, max_frames: int | None = None) -> dict` — writes each frame's
  color to `session.raw_rgb/{index:06d}.png`, depth to `session.raw_depth/{index:06d}.png` (16-bit
  PNG), sets `manifest["hardware"]["intrinsics"] = intrinsics`, sets
  `manifest["capture_stats"]["n_frames"]` and `manifest["capture_stats"]["duration_s"]`. Returns
  the mutated `manifest`. Later tasks (`slam_import`, `gate_a`) consume
  `manifest["capture_stats"]["n_frames"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_record.py
import numpy as np
from pathlib import Path
from capture.session import create_capture_session
from capture.record import Frame, record_session
from capture.manifest import new_manifest

def _fake_frames(n):
    for i in range(n):
        color = np.full((4, 4, 3), i, dtype=np.uint8)
        depth = np.full((4, 4), i * 100, dtype=np.uint16)
        yield Frame(index=i, timestamp=i * 0.1, color=color, depth=depth)

def test_record_session_writes_files_and_updates_manifest(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    manifest = new_manifest(session.root.name, "lab_room_a", 1, "sop-v1", "p01")
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}

    result = record_session(_fake_frames(3), intrinsics, session, manifest)

    assert result is manifest
    assert manifest["hardware"]["intrinsics"] == intrinsics
    assert manifest["capture_stats"]["n_frames"] == 3
    assert manifest["capture_stats"]["duration_s"] == 0.2
    for i in range(3):
        assert (session.raw_rgb / f"{i:06d}.png").exists()
        assert (session.raw_depth / f"{i:06d}.png").exists()

def test_record_session_respects_max_frames(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    manifest = new_manifest(session.root.name, "lab_room_a", 1, "sop-v1", "p01")
    intrinsics = {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 4, "height": 4}

    record_session(_fake_frames(10), intrinsics, session, manifest, max_frames=3)

    assert manifest["capture_stats"]["n_frames"] == 3
    assert not (session.raw_rgb / "000003.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_record.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'capture.record'`)

- [ ] **Step 3: Write `capture/record.py`**

```python
# capture/record.py
from dataclasses import dataclass
from typing import Iterable
import numpy as np

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    iio = None


@dataclass
class Frame:
    index: int
    timestamp: float
    color: np.ndarray
    depth: np.ndarray


def record_session(frames: Iterable[Frame], intrinsics: dict, session, manifest: dict,
                    max_frames: int | None = None) -> dict:
    count = 0
    first_ts = None
    last_ts = None
    for frame in frames:
        if max_frames is not None and count >= max_frames:
            break
        if first_ts is None:
            first_ts = frame.timestamp
        last_ts = frame.timestamp
        iio.imwrite(session.raw_rgb / f"{frame.index:06d}.png", frame.color)
        iio.imwrite(session.raw_depth / f"{frame.index:06d}.png", frame.depth)
        count += 1

    manifest["hardware"]["intrinsics"] = intrinsics
    manifest["capture_stats"]["n_frames"] = count
    manifest["capture_stats"]["duration_s"] = round((last_ts - first_ts), 6) if count > 1 else 0.0
    return manifest
```

- [ ] **Step 4: Add `imageio` to `requirements.txt`**

```
imageio
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pip install imageio && pytest tests/test_record.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add capture/record.py requirements.txt tests/test_record.py
git commit -m "feat: add hardware-agnostic frame recording logic"
```

---

### Task 4: RealSense hardware adapter (manual-tested)

**Files:**
- Create: `capture/realsense_source.py`

**Interfaces:**
- Consumes: `capture.record.Frame` (Task 3).
- Produces: `capture.realsense_source.RealSenseFrameSource` class with `.intrinsics -> dict`
  (same shape as Task 3's `intrinsics` argument) and `.frames() -> Iterator[Frame]`, so it can be
  passed directly as the first argument to `record_session`.
- Produces: `capture.realsense_source.RealSenseFrameSource.__init__(self, width=1280, height=720,
  fps=30, lock_exposure_us: int | None = 8000)` — when `lock_exposure_us` is not `None`, disables
  `rs.option.enable_auto_exposure` and `rs.option.enable_auto_white_balance` and sets
  `rs.option.exposure` to the given value, then records `"exposure_mode": "locked"` (else
  `"auto"`) into `.intrinsics["exposure_mode"]`. This exists because inconsistent
  auto-exposure/white-balance between frames visibly degrades Gaussian Splatting training quality.

No unit test — this module is a thin pyrealsense2 adapter and cannot run without the physical
D435 attached. Verification is manual.

- [ ] **Step 1: Write `capture/realsense_source.py`**

```python
# capture/realsense_source.py
import time
import numpy as np
import pyrealsense2 as rs

from capture.record import Frame


class RealSenseFrameSource:
    def __init__(self, width=1280, height=720, fps=30, lock_exposure_us: int | None = 8000):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)

        color_sensor = profile.get_device().first_color_sensor()
        exposure_mode = "auto"
        if lock_exposure_us is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.exposure, lock_exposure_us)
            exposure_mode = "locked"

        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = depth_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.intrinsics = {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
            "depth_scale": depth_sensor.get_depth_scale(),
            "exposure_mode": exposure_mode,
        }
        self._align = rs.align(rs.stream.color)

    def frames(self):
        index = 0
        start = time.monotonic()
        try:
            while True:
                frameset = self.pipeline.wait_for_frames()
                frameset = self._align.process(frameset)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())[:, :, ::-1]  # BGR -> RGB
                depth = np.asanyarray(depth_frame.get_data())
                yield Frame(index=index, timestamp=time.monotonic() - start, color=color, depth=depth)
                index += 1
        finally:
            self.pipeline.stop()
```

- [ ] **Step 2: Manual verification procedure (write into `_meta/sop_capture.md` in Task 5)**

With the D435 plugged in, run:

```bash
python -c "
from capture.realsense_source import RealSenseFrameSource
from capture.session import create_capture_session
from capture.manifest import new_manifest, save_manifest
from capture.record import record_session
from pathlib import Path

session = create_capture_session(Path.home() / 'scenes', 'smoke_test')
manifest = new_manifest(session.root.name, 'smoke_test', 1, 'sop-v1', 'manual')
source = RealSenseFrameSource()
record_session(source.frames(), source.intrinsics, session, manifest, max_frames=30)
save_manifest(manifest, session.manifest_path)
print('OK', manifest['capture_stats'])
"
```

Expected: prints `OK {'n_frames': 30, 'duration_s': ...}`, and `raw/rgb/000000.png` .. `000029.png`
plus matching `raw/depth/*.png` exist and open as valid images.

- [ ] **Step 3: Commit**

```bash
git add capture/realsense_source.py
git commit -m "feat: add RealSense D435 hardware adapter"
```

---

### Task 5: Capture SOP checklist (RTAB-Map GUI step)

**Files:**
- Create: `_meta/sop_capture.md`

**Interfaces:**
- Produces: `_meta/sop_capture.md`, versioned document. Its top line states
  `sop_version: sop-v1`, the value that goes into `manifest["sop_version"]` (Task 2). This is a
  content-only task — no code, no tests — but is a required deliverable per spec §3.5 ("SOP sống
  trong git").

- [ ] **Step 1: Write `_meta/sop_capture.md`**

```markdown
# Capture SOP

sop_version: sop-v1

## Trước khi quay

- [ ] Cắm D435 vào laptop, đợi ổn định nhiệt ≥5 phút trước khi quay (giảm intrinsics drift).
- [ ] Đặt scale bar 1.000 m + checkerboard trong khung hình đầu tiên, nằm phẳng trên sàn.
- [ ] Xác nhận AprilTag anchor (tag id 0) đã dán cố định và nhìn thấy được từ vị trí bắt đầu.
- [ ] Ghi `capture_profile_id` sẽ dùng (xem `_meta/capture_profiles/`).

## Trong khi quay (chạy `capture/realsense_source.py` qua `capture/record.py`)

- [ ] Đi chậm, tránh xoay nhanh (mục tiêu p95 vận tốc góc < 30°/s — Gate A sẽ tự kiểm tra).
- [ ] Đảm bảo mỗi ô sàn 1×1 m được nhìn từ ≥2 hướng lệch nhau >30°.
- [ ] Nếu quỹ đạo khép kín được (quay lại điểm xuất phát), cố tình đi qua lại điểm đó một lần
      để tạo loop closure cho RTAB-Map.

## Xử lý SLAM (RTAB-Map GUI, thủ công)

- [ ] Mở RTAB-Map GUI → Preferences → Source → chọn driver **RealSense2**, trỏ tới `.bag` đã ghi
      (hoặc camera trực tiếp nếu quay lại tại chỗ).
- [ ] Start mapping, để chạy tới hết đoạn ghi. Theo dõi loop closure trong log.
- [ ] `File → Export poses → RGBD-SLAM format (*.txt)` → lưu vào
      `<session>/slam/rtabmap_poses.txt`.
- [ ] `Tools → Post-processing` → chạy các bước làm sạch mặc định.
- [ ] `Edit → View Point Clouds` → Export → lưu point cloud thành
      `<session>/slam/rtabmap_cloud.ply`, và mesh (nếu có) thành
      `<session>/slam/rtabmap_mesh.ply`.

## Sau khi quay

- [ ] Chạy `capture/slam_import.py` để nạp export của RTAB-Map vào `slam/`.
- [ ] Chạy `align_verify/gravity_align.py` + `align_verify/anchor.py`.
- [ ] Đo tay 3 khoảng cách bằng scale bar/tape, ghi vào `manifest["ground_truth"]["measurements"]`.
- [ ] Chạy `align_verify/gate_a.py`. Nếu `REJECTED`, ghi defect code vào manifest và xem mục
      "Defect taxonomy" bên dưới trước khi quay lại.
- [ ] Nếu `APPROVED`, chạy `ingest/package.py` để đóng gói session (read-only).

## Defect taxonomy

| Code | Ý nghĩa |
|---|---|
| DARK | Thiếu sáng, ảnh RGB nhiễu |
| BLUR-FAST | Đi/xoay quá nhanh, motion blur |
| COV-GAP | Có ô sàn chưa được nhìn từ đủ hướng |
| GLASS | Có mặt kính/gương gây depth rác |
| DYN-OBJ | Có vật thể di chuyển trong khung hình khi quay |
| DRIFT-NOLOOP | Quỹ đạo dài không có loop closure, nghi ngờ drift |
| FLOOR-NOTEX | Sàn ít texture, RANSAC floor fit inlier ratio thấp |
| SCALE-DRIFT | Sai số đo khoảng cách vượt ngưỡng |

Quy tắc: một defect code xuất hiện ≥2 lần trong `defects_log.md` thì bắt buộc thêm một dòng
checklist mới ở trên, bump `sop_version`, và ghi capture_id gây ra nó vào `defects_log.md`.
```

- [ ] **Step 2: Create empty `_meta/defects_log.md` and `_meta/capture_profiles/p01.yaml`**

```markdown
<!-- _meta/defects_log.md -->
# Defect Log

| date | capture_id | code | note |
|---|---|---|---|
```

```yaml
# _meta/capture_profiles/p01.yaml
profile_id: p01
width: 1280
height: 720
fps: 30
lock_exposure_us: 8000
max_linear_speed_m_s: 0.3
max_wall_distance_m: 3.0
```

- [ ] **Step 3: Commit**

```bash
git add _meta/sop_capture.md _meta/defects_log.md _meta/capture_profiles/p01.yaml
git commit -m "docs: add capture SOP, defect taxonomy, and capture profile p01"
```

---

### Task 6: RTAB-Map export parser

**Files:**
- Create: `capture/slam_import.py`
- Test: `tests/test_slam_import.py`

**Interfaces:**
- Consumes: `capture.session.SessionPaths` (Task 1).
- Produces: `capture.slam_import.parse_tum_poses(path: Path) -> list[dict]` — each dict is
  `{"timestamp": float, "t": np.ndarray(3,), "q": np.ndarray(4,)}` with `q` in `[qx, qy, qz, qw]`
  order (scipy `Rotation.from_quat` convention). Raises `ValueError` on malformed lines.
- Produces: `capture.slam_import.import_slam_output(rtabmap_poses_path: Path,
  rtabmap_cloud_path: Path, session) -> dict` — copies the cloud to
  `session.slam_dir / "cloud.ply"`, the poses to `session.slam_dir / "trajectory.tum"`, and
  returns `{"poses": list[dict], "n_poses": int}`. The `poses` list is what `align_verify` (Tasks
  7-9) consumes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slam_import.py
import numpy as np
import pytest
from pathlib import Path
from capture.session import create_capture_session
from capture.slam_import import parse_tum_poses, import_slam_output

TUM_SAMPLE = """# comment line
0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.1 1.0 0.0 0.0 0.0 0.0 0.0 1.0
"""

def test_parse_tum_poses(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text(TUM_SAMPLE)
    poses = parse_tum_poses(p)
    assert len(poses) == 2
    assert poses[0]["timestamp"] == 0.0
    np.testing.assert_array_equal(poses[1]["t"], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(poses[1]["q"], [0.0, 0.0, 0.0, 1.0])

def test_parse_tum_poses_rejects_bad_row_length(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text("0.0 1.0 2.0\n")
    with pytest.raises(ValueError):
        parse_tum_poses(p)

def test_import_slam_output_copies_and_returns_poses(tmp_path):
    session = create_capture_session(tmp_path, "lab_room_a", day="2026-08-12")
    poses_src = tmp_path / "rtabmap_poses.txt"
    poses_src.write_text(TUM_SAMPLE)
    cloud_src = tmp_path / "rtabmap_cloud.ply"
    cloud_src.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")

    result = import_slam_output(poses_src, cloud_src, session)

    assert result["n_poses"] == 2
    assert len(result["poses"]) == 2
    assert (session.slam_dir / "trajectory.tum").exists()
    assert (session.slam_dir / "cloud.ply").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_slam_import.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'capture.slam_import'`)

- [ ] **Step 3: Write `capture/slam_import.py`**

```python
# capture/slam_import.py
import shutil
import numpy as np
from pathlib import Path


def parse_tum_poses(path: Path) -> list[dict]:
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"expected 8 columns (TUM format), got {len(parts)}: {line!r}")
        ts, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in parts)
        poses.append({
            "timestamp": ts,
            "t": np.array([tx, ty, tz]),
            "q": np.array([qx, qy, qz, qw]),
        })
    return poses


def import_slam_output(rtabmap_poses_path: Path, rtabmap_cloud_path: Path, session) -> dict:
    poses = parse_tum_poses(rtabmap_poses_path)
    shutil.copy(rtabmap_poses_path, session.slam_dir / "trajectory.tum")
    shutil.copy(rtabmap_cloud_path, session.slam_dir / "cloud.ply")
    return {"poses": poses, "n_poses": len(poses)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_slam_import.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add capture/slam_import.py tests/test_slam_import.py
git commit -m "feat: parse and import RTAB-Map GUI export"
```

---

### Task 7: Gravity alignment

**Files:**
- Create: `align_verify/__init__.py`
- Create: `align_verify/gravity_align.py`
- Test: `tests/test_gravity_align.py`

**Interfaces:**
- Consumes: `open3d.geometry.PointCloud` (from `capture.slam_import`'s `cloud.ply`, loaded via
  `open3d.io.read_point_cloud`).
- Produces: `align_verify.gravity_align.fit_floor_plane(cloud, distance_threshold=0.02, ransac_n=3,
  num_iterations=2000) -> tuple[np.ndarray, float]` — returns `(plane_model[4], inlier_ratio)`.
- Produces: `align_verify.gravity_align.gravity_alignment_transform(plane_model: np.ndarray) ->
  tuple[np.ndarray, float]` — returns `(T_world_from_slam[4,4], gravity_residual_deg)`. The
  transform maps a point in SLAM's raw frame to the gravity-aligned world frame: floor at Z=0,
  floor normal along +Z.
- Produces: `align_verify.gravity_align.apply_transform(T: np.ndarray, points: np.ndarray) ->
  np.ndarray` — `points` is `(N, 3)`, returns transformed `(N, 3)`.
- Produces: `align_verify.gravity_align.apply_transform_to_poses(T: np.ndarray, poses: list[dict])
  -> list[dict]` — applies `T` to each pose's translation and left-multiplies rotation, returns a
  new list (same `dict` shape as `capture.slam_import.parse_tum_poses`'s output). Consumed by
  Task 9 (`gate_a`) and Task 8 (`anchor`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gravity_align.py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from align_verify.gravity_align import (
    fit_floor_plane, gravity_alignment_transform, apply_transform, apply_transform_to_poses,
)

def _tilted_floor_cloud(tilt_deg=5.0, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-2, 2, size=(n, 2))
    z = np.zeros(n)
    pts = np.column_stack([xy, z])
    R = Rotation.from_euler("x", tilt_deg, degrees=True).as_matrix()
    pts = pts @ R.T
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    return cloud

def test_fit_floor_plane_finds_high_inlier_ratio():
    cloud = _tilted_floor_cloud()
    plane_model, inlier_ratio = fit_floor_plane(cloud)
    assert inlier_ratio > 0.95

def test_gravity_alignment_transform_recovers_tilt_angle():
    cloud = _tilted_floor_cloud(tilt_deg=5.0)
    plane_model, _ = fit_floor_plane(cloud)
    T, residual_deg = gravity_alignment_transform(plane_model)
    assert abs(residual_deg - 5.0) < 0.5

def test_apply_transform_flattens_floor_to_z_zero():
    cloud = _tilted_floor_cloud(tilt_deg=5.0)
    plane_model, _ = fit_floor_plane(cloud)
    T, _ = gravity_alignment_transform(plane_model)
    pts = np.asarray(cloud.points)
    transformed = apply_transform(T, pts)
    assert np.abs(transformed[:, 2]).max() < 0.01

def test_apply_transform_to_poses_translates_and_rotates():
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    poses = [{"timestamp": 0.0, "t": np.array([0.0, 0.0, 0.0]), "q": np.array([0, 0, 0, 1.0])}]
    out = apply_transform_to_poses(T, poses)
    np.testing.assert_allclose(out[0]["t"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out[0]["q"], [0.0, 0.0, 0.0, 1.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gravity_align.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'align_verify'`)

- [ ] **Step 3: Write `align_verify/__init__.py` (empty) and `align_verify/gravity_align.py`**

```python
# align_verify/gravity_align.py
import numpy as np
from scipy.spatial.transform import Rotation


def fit_floor_plane(cloud, distance_threshold=0.02, ransac_n=3, num_iterations=2000):
    plane_model, inliers = cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_ratio = len(inliers) / max(len(cloud.points), 1)
    return np.array(plane_model), inlier_ratio


def gravity_alignment_transform(plane_model: np.ndarray) -> tuple[np.ndarray, float]:
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
        d = -d

    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
    residual_deg = float(np.degrees(np.arccos(dot)))

    axis = np.cross(normal, z_axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(dot)
        R = Rotation.from_rotvec(axis * angle).as_matrix()

    p0 = -d * normal  # foot of perpendicular from origin to the plane
    t = -R @ p0

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, residual_deg


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def apply_transform_to_poses(T: np.ndarray, poses: list[dict]) -> list[dict]:
    R_t = T[:3, :3]
    t_t = T[:3, 3]
    out = []
    for p in poses:
        new_t = R_t @ p["t"] + t_t
        new_R = R_t @ Rotation.from_quat(p["q"]).as_matrix()
        out.append({
            "timestamp": p["timestamp"],
            "t": new_t,
            "q": Rotation.from_matrix(new_R).as_quat(),
        })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gravity_align.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add align_verify/__init__.py align_verify/gravity_align.py tests/test_gravity_align.py
git commit -m "feat: add gravity alignment (RANSAC floor fit)"
```

---

### Task 8: AprilTag anchor registration

**Files:**
- Create: `align_verify/anchor.py`
- Test: `tests/test_anchor.py`

**Interfaces:**
- Consumes: gravity-aligned poses (Task 7's `apply_transform_to_poses` output) as
  `T_world_from_camera_by_frame: dict[int, np.ndarray[4,4]]` (built by the caller from `poses`,
  keyed by frame index — construction shown in Task 9's wiring, not this task's concern).
- Produces: `align_verify.anchor.ANCHOR_TAG_ID: int = 0`.
- Produces: `align_verify.anchor.detections_to_world(detections: list[dict],
  T_world_from_camera_by_frame: dict[int, np.ndarray]) -> list[np.ndarray]` — `detections` items
  are `{"frame_index": int, "tag_id": int, "R_cam_tag": np.ndarray[3,3], "t_cam_tag":
  np.ndarray[3]}`. Filters to `tag_id == ANCHOR_TAG_ID` and frames present in the pose dict,
  returns `T_world_from_tag` candidates.
- Produces: `align_verify.anchor.average_transform(T_list: list[np.ndarray]) -> np.ndarray`.
- Produces: `align_verify.anchor.compute_anchor_from_world(detections, T_world_from_camera_by_frame)
  -> np.ndarray | None` — returns `T_anchor_from_world` (4x4), or `None` if the tag was never
  detected. This is what Task 9 writes into `manifest["align"]["T_anchor_from_world"]`.
- Produces (manual-tested, no unit test): `align_verify.anchor.detect_tags_in_frame(gray_image:
  np.ndarray, intrinsics: dict, tag_size_m: float) -> list[dict]` — thin `pupil_apriltags.Detector`
  adapter returning items in the same shape `detections_to_world` expects.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anchor.py
import numpy as np
from scipy.spatial.transform import Rotation
from align_verify.anchor import (
    ANCHOR_TAG_ID, detections_to_world, average_transform, compute_anchor_from_world,
)

def test_detections_to_world_filters_tag_id_and_known_frames():
    T_wc = {0: np.eye(4), 1: np.eye(4)}
    detections = [
        {"frame_index": 0, "tag_id": ANCHOR_TAG_ID, "R_cam_tag": np.eye(3), "t_cam_tag": np.array([1.0, 0, 0])},
        {"frame_index": 1, "tag_id": 99, "R_cam_tag": np.eye(3), "t_cam_tag": np.array([0, 0, 0])},
        {"frame_index": 5, "tag_id": ANCHOR_TAG_ID, "R_cam_tag": np.eye(3), "t_cam_tag": np.array([0, 0, 0])},
    ]
    result = detections_to_world(detections, T_wc)
    assert len(result) == 1
    np.testing.assert_allclose(result[0][:3, 3], [1.0, 0, 0])

def test_average_transform_averages_translation_and_rotation():
    T1 = np.eye(4); T1[:3, 3] = [1.0, 0.0, 0.0]
    T2 = np.eye(4); T2[:3, 3] = [1.2, 0.0, 0.0]
    avg = average_transform([T1, T2])
    np.testing.assert_allclose(avg[:3, 3], [1.1, 0.0, 0.0])
    np.testing.assert_allclose(avg[:3, :3], np.eye(3), atol=1e-9)

def test_average_transform_raises_on_empty_list():
    import pytest
    with pytest.raises(ValueError):
        average_transform([])

def test_compute_anchor_from_world_is_inverse_of_tag_pose():
    T_wc = {0: np.eye(4)}
    detections = [{"frame_index": 0, "tag_id": ANCHOR_TAG_ID, "R_cam_tag": np.eye(3),
                   "t_cam_tag": np.array([2.0, 0.0, 0.0])}]
    T_anchor_from_world = compute_anchor_from_world(detections, T_wc)
    # tag is at world (2,0,0); anchor frame origin is the tag, so a world point at (2,0,0)
    # must map to the anchor origin (0,0,0)
    p_world = np.array([2.0, 0.0, 0.0])
    p_anchor = T_anchor_from_world[:3, :3] @ p_world + T_anchor_from_world[:3, 3]
    np.testing.assert_allclose(p_anchor, [0, 0, 0], atol=1e-9)

def test_compute_anchor_from_world_returns_none_when_tag_never_seen():
    result = compute_anchor_from_world([], {0: np.eye(4)})
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_anchor.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'align_verify.anchor'`)

- [ ] **Step 3: Write `align_verify/anchor.py`**

```python
# align_verify/anchor.py
import numpy as np
from scipy.spatial.transform import Rotation

ANCHOR_TAG_ID = 0


def detections_to_world(detections: list[dict], T_world_from_camera_by_frame: dict) -> list[np.ndarray]:
    results = []
    for det in detections:
        if det["tag_id"] != ANCHOR_TAG_ID:
            continue
        fi = det["frame_index"]
        if fi not in T_world_from_camera_by_frame:
            continue
        T_wc = T_world_from_camera_by_frame[fi]
        T_ct = np.eye(4)
        T_ct[:3, :3] = det["R_cam_tag"]
        T_ct[:3, 3] = det["t_cam_tag"]
        results.append(T_wc @ T_ct)
    return results


def average_transform(T_list: list[np.ndarray]) -> np.ndarray:
    if not T_list:
        raise ValueError("no transforms to average")
    ts = np.array([T[:3, 3] for T in T_list])
    quats = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_list])
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(ref, quats[i]) < 0:
            quats[i] = -quats[i]
    mean_q = quats.mean(axis=0)
    mean_q = mean_q / np.linalg.norm(mean_q)

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(mean_q).as_matrix()
    T[:3, 3] = ts.mean(axis=0)
    return T


def compute_anchor_from_world(detections: list[dict], T_world_from_camera_by_frame: dict) -> np.ndarray | None:
    samples = detections_to_world(detections, T_world_from_camera_by_frame)
    if not samples:
        return None
    T_world_from_tag = average_transform(samples)
    return np.linalg.inv(T_world_from_tag)


def detect_tags_in_frame(gray_image: np.ndarray, intrinsics: dict, tag_size_m: float) -> list[dict]:
    from pupil_apriltags import Detector

    detector = Detector(families="tag36h11")
    camera_params = (intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"])
    results = detector.detect(gray_image, estimate_tag_pose=True, camera_params=camera_params,
                               tag_size=tag_size_m)
    return [
        {"frame_index": None, "tag_id": r.tag_id, "R_cam_tag": r.pose_R, "t_cam_tag": r.pose_t.flatten()}
        for r in results
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_anchor.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Manual verification of `detect_tags_in_frame`**

With a printed AprilTag (`tag36h11`, id 0) of known size in front of the D435: capture one frame
with `RealSenseFrameSource`, convert to grayscale, call `detect_tags_in_frame(gray, source.intrinsics,
tag_size_m=<measured tag edge length>)`, confirm one detection with `tag_id == 0` and a plausible
`t_cam_tag` (distance roughly matching the tape-measured distance to the tag).

- [ ] **Step 6: Commit**

```bash
git add align_verify/anchor.py tests/test_anchor.py
git commit -m "feat: add AprilTag anchor registration"
```

---

### Task 9: Gate A scoring

**Files:**
- Create: `align_verify/gate_a.py`
- Create: `_meta/gates.yaml`
- Test: `tests/test_gate_a.py`

**Interfaces:**
- Consumes: `manifest["ground_truth"]["measurements"]` (Task 2), `manifest["align"]
  ["gravity_residual_deg"]` and `["floor_inlier_ratio"]` (Task 7's output, written by caller),
  gravity-aligned `poses` (Task 7), per-frame blur flags (`list[bool]`, computed by caller —
  out of scope for this task), and floor point XY coordinates (`np.ndarray[N,2]`, the
  gravity-aligned cloud's points with `|z| < 0.05` filtered by the caller).
- Produces: `align_verify.gate_a.load_thresholds(gates_yaml_path: Path) -> dict`.
- Produces: `align_verify.gate_a.check_scale`, `check_gravity`, `check_trajectory`,
  `check_coverage` — each returns a dict with at least `{"pass": bool, ...details}`.
- Produces: `align_verify.gate_a.run_gate_a(manifest: dict, poses: list[dict], blur_flags:
  list[bool], floor_points_xy: np.ndarray, thresholds: dict) -> dict` — writes the returned
  scorecard into `manifest["gate"]["gateA"]` and returns it. `scorecard["verdict"]` is
  `"APPROVED"` or `"REJECTED"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate_a.py
import numpy as np
from pathlib import Path
from align_verify.gate_a import (
    load_thresholds, check_scale, check_gravity, check_trajectory, check_coverage, run_gate_a,
)

THRESHOLDS = {
    "scale_max_err_pct": 2.0, "scale_min_measurements": 3,
    "gravity_max_residual_deg": 1.0, "gravity_min_floor_inlier_ratio": 0.60,
    "trajectory_max_gap_s": 0.5, "trajectory_max_p95_angular_vel_deg_s": 30.0,
    "trajectory_max_blur_ratio": 0.10,
    "coverage_cell_size_m": 1.0, "coverage_min_angle_spread_deg": 30.0,
}

def test_load_thresholds_reads_gate_a_section(tmp_path):
    p = tmp_path / "gates.yaml"
    p.write_text("gate_a:\n  scale_max_err_pct: 2.0\n")
    t = load_thresholds(p)
    assert t["scale_max_err_pct"] == 2.0

def test_check_scale_passes_within_tolerance():
    measurements = [
        {"name": "wall", "tape_m": 2.0, "cloud_m": 2.01},
        {"name": "table", "tape_m": 1.0, "cloud_m": 1.02},
        {"name": "door", "tape_m": 0.9, "cloud_m": 0.905},
    ]
    result = check_scale(measurements, THRESHOLDS["scale_max_err_pct"], THRESHOLDS["scale_min_measurements"])
    assert result["pass"] is True

def test_check_scale_fails_too_few_measurements():
    result = check_scale([{"name": "wall", "tape_m": 2.0, "cloud_m": 2.01}],
                          THRESHOLDS["scale_max_err_pct"], THRESHOLDS["scale_min_measurements"])
    assert result["pass"] is False

def test_check_scale_fails_over_tolerance():
    measurements = [
        {"name": "wall", "tape_m": 2.0, "cloud_m": 2.10},
        {"name": "table", "tape_m": 1.0, "cloud_m": 1.00},
        {"name": "door", "tape_m": 0.9, "cloud_m": 0.90},
    ]
    result = check_scale(measurements, THRESHOLDS["scale_max_err_pct"], THRESHOLDS["scale_min_measurements"])
    assert result["pass"] is False

def test_check_gravity_pass_and_fail():
    ok = check_gravity(0.5, 0.7, 1.0, 0.60)
    assert ok["pass"] is True
    bad = check_gravity(2.0, 0.7, 1.0, 0.60)
    assert bad["pass"] is False

def _poses(n, dt=0.1, angular_step_deg=0.0):
    from scipy.spatial.transform import Rotation
    poses = []
    for i in range(n):
        q = Rotation.from_euler("z", angular_step_deg * i, degrees=True).as_quat()
        poses.append({"timestamp": i * dt, "t": np.array([0.0, 0.0, 0.0]), "q": q})
    return poses

def test_check_trajectory_passes_smooth_motion():
    poses = _poses(20, dt=0.1, angular_step_deg=1.0)
    result = check_trajectory(poses, 0.5, 30.0, 0.10, [False] * 20)
    assert result["pass"] is True

def test_check_trajectory_fails_on_gap():
    poses = _poses(5, dt=0.1)
    poses[3]["timestamp"] = 10.0  # huge gap
    result = check_trajectory(poses, 0.5, 30.0, 0.10, [False] * 5)
    assert result["pass"] is False

def test_check_trajectory_fails_on_fast_rotation():
    poses = _poses(10, dt=0.1, angular_step_deg=10.0)  # 100 deg/s
    result = check_trajectory(poses, 0.5, 30.0, 0.10, [False] * 10)
    assert result["pass"] is False

def test_check_coverage_passes_with_multi_angle_views():
    floor_points_xy = np.array([[0.5, 0.5]])
    from scipy.spatial.transform import Rotation
    poses = [
        {"timestamp": 0.0, "t": np.array([0.5, -0.5, 1.0]), "q": Rotation.from_euler("z", 90, degrees=True).as_quat()},
        {"timestamp": 1.0, "t": np.array([-0.5, 0.5, 1.0]), "q": Rotation.from_euler("z", -90, degrees=True).as_quat()},
    ]
    result = check_coverage(poses, floor_points_xy, 1.0, 30.0)
    assert result["pass"] is True

def test_run_gate_a_writes_manifest_and_verdict():
    manifest = {
        "align": {"gravity_residual_deg": 0.5, "floor_inlier_ratio": 0.7},
        "ground_truth": {"measurements": [
            {"name": "a", "tape_m": 2.0, "cloud_m": 2.01},
            {"name": "b", "tape_m": 1.0, "cloud_m": 1.0},
            {"name": "c", "tape_m": 0.9, "cloud_m": 0.905},
        ]},
        "gate": {"gateA": None, "defects": []},
    }
    poses = _poses(20, dt=0.1, angular_step_deg=1.0)
    floor_points_xy = np.zeros((1, 2))
    scorecard = run_gate_a(manifest, poses, [False] * 20, floor_points_xy, THRESHOLDS)
    assert manifest["gate"]["gateA"] is scorecard
    assert scorecard["verdict"] in {"APPROVED", "REJECTED"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gate_a.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'align_verify.gate_a'`)

- [ ] **Step 3: Write `_meta/gates.yaml`**

```yaml
gate_a:
  scale_max_err_pct: 2.0
  scale_min_measurements: 3
  gravity_max_residual_deg: 1.0
  gravity_min_floor_inlier_ratio: 0.60
  trajectory_max_gap_s: 0.5
  trajectory_max_p95_angular_vel_deg_s: 30.0
  trajectory_max_blur_ratio: 0.10
  coverage_cell_size_m: 1.0
  coverage_min_angle_spread_deg: 30.0
gate_b:
  psnr_min_db: 25.0
  psnr_warn_db: 22.0
  ssim_min: 0.80
  measurement_max_err_pct: 2.0
gate_c:
  drop_test_points: 20
  drop_test_max_height_err_m: 0.03
  collision_max_hole_m: 0.05
  min_fps: 30.0
```

- [ ] **Step 4: Write `align_verify/gate_a.py`**

```python
# align_verify/gate_a.py
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_thresholds(gates_yaml_path: Path) -> dict:
    return yaml.safe_load(gates_yaml_path.read_text())["gate_a"]


def check_scale(measurements: list[dict], max_err_pct: float, min_measurements: int) -> dict:
    if len(measurements) < min_measurements:
        return {"pass": False, "reason": f"need >= {min_measurements} measurements, got {len(measurements)}"}
    errs = []
    for m in measurements:
        err_pct = abs(m["cloud_m"] - m["tape_m"]) / m["tape_m"] * 100.0
        m["err_pct"] = err_pct
        errs.append(err_pct)
    worst = max(errs)
    return {"pass": worst <= max_err_pct, "worst_err_pct": worst, "measurements": measurements}


def check_gravity(residual_deg: float, floor_inlier_ratio: float, max_residual_deg: float,
                   min_inlier_ratio: float) -> dict:
    ok = residual_deg < max_residual_deg and floor_inlier_ratio >= min_inlier_ratio
    return {"pass": ok, "residual_deg": residual_deg, "floor_inlier_ratio": floor_inlier_ratio}


def _angular_velocities_deg_s(poses):
    vels = []
    for a, b in zip(poses, poses[1:]):
        dt = b["timestamp"] - a["timestamp"]
        if dt <= 0:
            continue
        Ra = Rotation.from_quat(a["q"])
        Rb = Rotation.from_quat(b["q"])
        rel = Ra.inv() * Rb
        angle_deg = np.degrees(rel.magnitude())
        vels.append(angle_deg / dt)
    return np.array(vels)


def check_trajectory(poses: list[dict], max_gap_s: float, max_p95_angular_vel_deg_s: float,
                      max_blur_ratio: float, blur_flags: list[bool]) -> dict:
    ts = np.array([p["timestamp"] for p in poses])
    if np.isnan(ts).any():
        return {"pass": False, "reason": "NaN timestamp"}
    for p in poses:
        if np.isnan(p["t"]).any() or np.isnan(p["q"]).any():
            return {"pass": False, "reason": "NaN in pose"}

    gaps = np.diff(ts)
    max_gap = float(gaps.max()) if len(gaps) else 0.0
    angular_vels = _angular_velocities_deg_s(poses)
    p95_angular_vel = float(np.percentile(angular_vels, 95)) if len(angular_vels) else 0.0
    blur_ratio = sum(blur_flags) / len(blur_flags) if blur_flags else 0.0

    ok = (max_gap <= max_gap_s and p95_angular_vel <= max_p95_angular_vel_deg_s
          and blur_ratio <= max_blur_ratio)
    return {"pass": ok, "max_gap_s": max_gap, "p95_angular_vel_deg_s": p95_angular_vel,
            "blur_ratio": blur_ratio}


def _floor_headings_deg(poses):
    headings = []
    for p in poses:
        fwd = Rotation.from_quat(p["q"]).apply([0, 0, 1])
        headings.append(np.degrees(np.arctan2(fwd[1], fwd[0])) % 360)
    return np.array(headings)


def _max_pairwise_angular_spread_deg(headings_deg):
    diffs = []
    for i in range(len(headings_deg)):
        for j in range(i + 1, len(headings_deg)):
            d = abs(headings_deg[i] - headings_deg[j]) % 360
            diffs.append(min(d, 360 - d))
    return max(diffs) if diffs else 0.0


def check_coverage(poses: list[dict], floor_points_xy: np.ndarray, cell_size_m: float,
                    min_angle_spread_deg: float) -> dict:
    if floor_points_xy.size == 0 or not poses:
        return {"pass": False, "reason": "no floor points or poses"}

    mins = floor_points_xy.min(axis=0)
    cells = set(map(tuple, np.floor((floor_points_xy - mins) / cell_size_m).astype(int)))
    positions = np.array([p["t"][:2] for p in poses])
    headings = _floor_headings_deg(poses)
    radius = 1.5 * cell_size_m

    uncovered = []
    for cell in cells:
        center = mins + (np.array(cell) + 0.5) * cell_size_m
        dists = np.linalg.norm(positions - center, axis=1)
        nearby_idx = np.where(dists <= radius)[0]
        if len(nearby_idx) < 2:
            uncovered.append(cell)
            continue
        spread = _max_pairwise_angular_spread_deg(headings[nearby_idx])
        if spread < min_angle_spread_deg:
            uncovered.append(cell)

    coverage_ratio = 1 - len(uncovered) / len(cells)
    return {"pass": len(uncovered) == 0, "coverage_ratio": coverage_ratio,
            "uncovered_cells": len(uncovered), "total_cells": len(cells)}


def run_gate_a(manifest: dict, poses: list[dict], blur_flags: list[bool],
                floor_points_xy: np.ndarray, thresholds: dict) -> dict:
    scale = check_scale(manifest["ground_truth"]["measurements"],
                         thresholds["scale_max_err_pct"], thresholds["scale_min_measurements"])
    gravity = check_gravity(manifest["align"]["gravity_residual_deg"],
                             manifest["align"]["floor_inlier_ratio"],
                             thresholds["gravity_max_residual_deg"],
                             thresholds["gravity_min_floor_inlier_ratio"])
    trajectory = check_trajectory(poses, thresholds["trajectory_max_gap_s"],
                                   thresholds["trajectory_max_p95_angular_vel_deg_s"],
                                   thresholds["trajectory_max_blur_ratio"], blur_flags)
    coverage = check_coverage(poses, floor_points_xy, thresholds["coverage_cell_size_m"],
                               thresholds["coverage_min_angle_spread_deg"])

    checks = {"scale": scale, "gravity": gravity, "trajectory": trajectory, "coverage": coverage}
    verdict = "APPROVED" if all(c["pass"] for c in checks.values()) else "REJECTED"
    scorecard = {"verdict": verdict, "checks": checks}
    manifest["gate"]["gateA"] = scorecard
    return scorecard
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pip install pyyaml && pytest tests/test_gate_a.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add align_verify/gate_a.py _meta/gates.yaml tests/test_gate_a.py
git commit -m "feat: add Gate A quality scoring"
```

---

### Task 10: Ingest packaging

**Files:**
- Create: `ingest/__init__.py`
- Create: `ingest/package.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `capture.session.SessionPaths.root` (Task 1), a `manifest` dict with
  `manifest["gate"]["gateA"]["verdict"] == "APPROVED"` (Task 9) — `finalize_session` does not
  itself enforce the verdict (the SOP in Task 5 already gates this manually); it only packages
  what it's given.
- Produces: `ingest.package.sha256_file(path: Path) -> str`.
- Produces: `ingest.package.collect_artifacts(session_root: Path) -> list[dict]` — one
  `{"path": str, "sha256": str, "bytes": int}` per file under `session_root` except
  `manifest.json` itself, paths relative to `session_root`.
- Produces: `ingest.package.finalize_session(session_root: Path, manifest: dict) -> dict` — sets
  `manifest["artifacts"]`, writes `manifest.json`, chmods every file under `session_root`
  read-only. Returns the final manifest.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_package.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ingest'`)

- [ ] **Step 3: Write `ingest/package.py`**

```python
# ingest/package.py
import hashlib
import json
import stat
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(session_root: Path) -> list[dict]:
    artifacts = []
    for p in sorted(session_root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            artifacts.append({
                "path": str(p.relative_to(session_root)),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            })
    return artifacts


def finalize_session(session_root: Path, manifest: dict) -> dict:
    manifest["artifacts"] = collect_artifacts(session_root)
    manifest_path = session_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for p in session_root.rglob("*"):
        if p.is_file():
            mode = p.stat().st_mode
            p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_package.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/__init__.py ingest/package.py tests/test_package.py
git commit -m "feat: add ingest packaging (read-only session finalization)"
```

---

### Task 11: Catalog reindex

**Files:**
- Create: `_meta/scripts/reindex.py`
- Test: `tests/test_reindex.py`

**Interfaces:**
- Consumes: `manifest.json` files under `<scenes_root>/*/captures/*/manifest.json` (Task 9/10's
  output shape).
- Produces: `_meta.scripts.reindex.scan_manifests(scenes_root: Path) -> list[dict]`.
- Produces: `_meta.scripts.reindex.build_catalog(scenes_root: Path) -> dict` — `{"captures":
  [{"env_id", "capture_id", "started_at", "verdict", "scale_err_pct"}, ...]}`.
- Produces: `_meta.scripts.reindex.render_markdown(catalog: dict) -> str`.
- Produces: `_meta.scripts.reindex.reindex(scenes_root: Path) -> None` — writes
  `<scenes_root>/_meta/catalog.json` and `<scenes_root>/_meta/catalog.md`.

- [ ] **Step 1: Write `_meta/scripts/__init__.py` (empty) and the failing test**

```python
# tests/test_reindex.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reindex.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'reindex'`)

- [ ] **Step 3: Write `_meta/scripts/reindex.py`**

```python
# _meta/scripts/reindex.py
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
    (scenes_root / "_meta" / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (scenes_root / "_meta" / "catalog.md").write_text(render_markdown(catalog))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reindex.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add _meta/scripts/reindex.py tests/test_reindex.py
git commit -m "feat: add catalog reindex script"
```

---

## End-to-end manual smoke test (after all tasks)

Not automated — requires the physical D435 + a printed AprilTag + a scale bar. Follow
`_meta/sop_capture.md` start to finish once, targeting `~/scenes` as `scenes_root` and any test
room as `env_id`. Confirm: `manifest.json` exists and is valid JSON, `gate.gateA.verdict` is set,
session files are read-only, and `_meta/scripts/reindex.py` picks it up in `catalog.md`.
