# Thiết kế: camera-3d2sim

Pipeline: RealSense D435 → 3D Gaussian Splatting → Isaac Sim

Status: Living document (thay thế `docs/superpowers/plans/` và `docs/superpowers/specs/`)
Cập nhật: 2026-08-17

## 1. Mục tiêu

Dựng lại không gian thật (phòng/khu vực) bằng camera Intel RealSense D435 (RGB-D, không IMU) thành
3D Gaussian Splatting đúng tỉ lệ mét thật, rồi nạp vào NVIDIA Isaac Sim để làm environment cho robot
navigation.

**Mục tiêu tối thượng**: một thư viện environment chất lượng cao trong Isaac Sim, để toàn bộ công
việc phát triển robot (navigation, RL training, testing) diễn ra thuần trong sim, không cần quay lại
camera thật cho phần robot logic. Đây là một **quy trình lặp lại được** (capture nhiều scene, capture
lại cùng một phòng nhiều lần để cải thiện chất lượng), không phải một lần làm rồi thôi.

**Nguyên tắc dẫn dắt thiết kế**: đơn giản trước, phức tạp/độ phân giải cao sau. YAGNI — không dùng
ROS2/Docker cho phần capture, không dùng DVC/MLflow. Mọi công cụ chọn ở mức tối thiểu đủ dùng.

## 2. Bố trí máy móc

Thiết kế ban đầu giả định hai máy tách biệt (laptop capture-only, server riêng train+Isaac Sim). Trên
máy hiện tại (`ubuntu-desktop`), thực tế đã gộp được cả hai vai trò:

- **GPU**: NVIDIA RTX 3060, 12GB VRAM, driver 595.84 — đủ chạy Isaac Sim và train GSplat quy mô phòng
  đơn lẻ (chưa kiểm chứng throughput train GSplat thực tế).
- **Camera**: D435 kết nối qua USB3 (SuperSpeed 5000M, xác nhận qua `lsusb -t`) — bắt buộc USB3 để
  đạt 1280x720@30fps cho cả color lẫn depth; USB2 chỉ hỗ trợ depth 1280x720 tới 6fps.
- **Isaac Sim 6.0.1**: đã chạy được trên máy này (xác nhận 2026-08-17).

Vì vậy Phase 0 (mục 3) và các bước phía "server" trong pipeline chính không còn bị chặn bởi việc thiếu
quyền truy cập máy khác — có thể triển khai tuần tự trên cùng một máy. Kiến trúc hai máy (rsync giữa
laptop capture và server train) vẫn giữ làm phương án khi capture ở hiện trường khác với máy có GPU.

## 3. Kiến trúc pipeline

### Phase 0 — Spike xác nhận Isaac Sim import (gate go/no-go, làm trước tiên)

Rủi ro lớn nhất tưởng là "Isaac Sim có hỗ trợ Gaussian Splatting không" — thực tế đã hỗ trợ native từ
bản 6.0 (qua Omniverse NuRec). Việc cần làm: chuẩn bị một `.ply` Gaussian Splat mẫu (tải công khai
hoặc tự sinh) → chạy
`ply_to_usd` (công cụ 3DGRUT) → mở `.usdz` kết quả trong Isaac Sim 6.0.1 → xác nhận render đúng,
không gặp lỗi "layered artifact" đã báo cáo trên forum NVIDIA.

**Kết quả (2026-08-17)**: PASS trên fixture Gaussian Splat tự sinh (hình cầu tổng hợp, không tải mẫu
công khai) — xem `spikes/phase0_isaacsim_import/`. Xác nhận pipeline convert + import chạy đúng,
không gặp lỗi "layered artifact" ở toạ độ gần gốc (bán kính 0.5m). Lưu ý: bug layered-artifact đã biết
trên forum NVIDIA chỉ được báo cáo ở toạ độ ≥300m từ gốc (do float16 precision) — Phase 0 CHƯA kiểm
chứng ở quy mô đó; capture thật trong phòng (vài mét) nhiều khả năng vẫn an toàn nhưng chưa test trực
tiếp. Gate mở, có thể triển khai `train/`, `navmesh/`, `isaacsim_import/`.

### Pipeline chính (6 bước, sau khi Phase 0 pass)

```
D435 (máy có camera)
   │
   ▼
[1] CAPTURE ── RTAB-Map standalone (không ROS2/Docker), RGB+depth đồng bộ
   │              → trajectory mét thật (chưa gravity-align) + point cloud/mesh
   ▼
[2] GRAVITY ALIGN + VERIFY ── Gate A
   │   RANSAC fit mặt sàn → xoay về +Z → "T_world_from_slam" duy nhất áp cho toàn bộ
   │   trajectory + mesh. Đăng ký về anchor vật lý cố định (AprilTag) → T_anchor_from_world
   ▼
[3] TRANSFER (rsync/scp nếu khác máy, hoặc bỏ qua nếu cùng máy)
   │
   ├──▼ [4] TRAIN GSPLAT ── Gate B
   │      convert → transforms.json, đổi trục OpenCV→OpenGL, TẮT auto_scale_poses/
   │      center_method/orientation_method của nerfstudio, dùng TSDF cloud làm init,
   │      freeze camera pose optimizer. Sanity check: render 1 frame ở pose đã biết,
   │      so khớp ảnh gốc TRƯỚC khi train full.
   │
   └──▼ [5] NAV GEOMETRY (song song, dùng chung mesh đã gravity-align)
          ground plane segmentation + lấp lỗ (Poisson+trim) + lọc nhiễu → collision mesh
          → Isaac Sim Occupancy Map Generator (built-in) cho occupancy map 2D
   │
   ▼
[6] ISAAC SIM IMPORT ── Gate C
    ply→usdz (pipeline đã validate ở Phase 0), dựng USD stage (collision mesh ẩn + GSplat
    hiển thị, cùng gốc T_world_from_slam / T_anchor_from_world). Verify: physics drop-test.
```

**Vì sao RTAB-Map thay vì Open3D odometry**: Open3D's built-in RGB-D odometry tune cho chuyển động
chậm/mượt, dễ mất tracking khi cầm tay đi vòng quanh phòng. RTAB-Map có bản standalone, loop closure
bag-of-words trưởng thành hơn.

**Vì sao gravity alignment là bước bắt buộc riêng**: D435 không IMU, pose từ SLAM chỉ đúng tương đối,
không đảm bảo mặt sàn song song mặt phẳng XY. Bỏ qua → scene nghiêng vài độ trong Isaac Sim (Z-up) →
robot ảo "trượt dốc", occupancy map méo — lỗi im lặng, chỉ lộ ra khi robot chạy sai.

**Vì sao tắt auto_scale_poses/center_method/orientation_method của nerfstudio**: mặc định bật sẽ
normalize scene về hộp đơn vị — xoá sạch nỗ lực lấy pose mét thật ở bước 1-2. Train vẫn chạy bình
thường, chỉ kết quả sai tỉ lệ — lỗi im lặng thứ hai.

## 4. Data Flywheel — tổ chức dữ liệu

Nguyên tắc: filesystem là database, mỗi thư mục tự mô tả bằng manifest JSON, catalog là sản phẩm
derived — xoá và sinh lại bất cứ lúc nào. Không DVC, không MLflow.

Ba khái niệm, quan hệ 1-N-N: **environment** (không gian vật lý cố định, có AprilTag anchor) →
**capture session** (một lần đi quay, read-only sau ingest) → **build** (một lần train/đóng gói ra
USD từ một capture, có thể nhiều build/hyperparameter khác nhau).

```
~/scenes/
  _meta/                                <- git repo (chỉ text: script, config, log)
      scripts/  gates.yaml  sop_capture.md  capture_profiles/  defects_log.md
      catalog.json  catalog.md
  <env_id>/                             <- ổn định vĩnh viễn
      scene.json                        <- anchor def, kích thước phòng, con trỏ BEST
      captures/
          YYYY-MM-DD_vNN/
              raw/      rgb/ depth/ *.bag  calib/
              slam/     rtabmap.db  trajectory.tum  cloud.ply  mesh.ply
              align/    T_world_from_slam.json  mesh_aligned.ply
              eval/     holdout_frames.txt  measurements.csv  gateA.json
              manifest.json  capture_log.md
      builds/
          <cap_ver>_gs<Letter>/
              config/  transforms.json  train_cmd.sh  versions.lock
              gsplat/  splat.ply    nav/  collision.obj  occupancy.png+yaml
              usd/     env.usd      report/ scorecard.json  renders/
              build.json
      BEST -> builds/<best_build>       <- symlink, cập nhật bởi script
```

Quy ước tên: capture = `YYYY-MM-DD_vNN`, build = `<capture_ver>_gs<Letter>`. Sau ingest, capture thành
read-only (`chmod -R a-w`) — mọi thay đổi tạo version mới.

**Anchor & scale bar vật lý** (bắt buộc mỗi lần capture): AprilTag dán cố định tại một vị trí trong
phòng (để nhiều lần capture cùng phòng register về chung gốc toạ độ), scale bar 1.000 m + checkerboard
trong khung hình đầu (đo scale error tự động, lặp lại được).

**Retention**: giữ `raw/` của bản APPROVED mới nhất + 1 bản trước đó; REJECTED xoá `raw/` sau 30 ngày
nhưng giữ `manifest.json` vĩnh viễn.

## 5. Manifest schema (key chính)

`manifest.json` (capture session):
- **identity**: `schema_version, capture_id, env_id, capture_version, started_at, ended_at, sop_version, capture_profile_id`
- **hardware/calib**: `camera_model, serial, firmware, depth_preset, resolution, fps, exposure_mode, emitter_on, depth_scale, calib_id, calib_date, intrinsics{fx,fy,cx,cy,dist}`
- **conditions**: `lighting_type, dynamic_objects_present, floor_material, glass_or_mirror_present, texture_poor_regions[]`
- **capture stats**: `n_frames, duration_s, trajectory_length_m, mean_linear_vel, p95_angular_vel, blur_frame_ratio, loop_closures, room_bbox`
- **align**: `T_world_from_slam(4x4), floor_inlier_ratio, floor_rms_mm, gravity_residual_deg, T_anchor_from_world`
- **ground truth**: `measurements[{name, tape_m, cloud_m, err_pct}]`, `holdout_frames[]` (5-10% tách riêng ngay từ lúc capture)
- **artifacts**: `[{path, sha256, bytes}]`
- **gate**: `gateA{scores, verdict}`, `defects[{code, note}]`

`build.json` (trained environment): `build_id, env_id, source_capture_id, source_manifest_sha`,
`versions.lock{nerfstudio, gsplat, 3dgrut, isaacsim, cuda, driver, repo_git_sha}`, `train_config_sha`
(+ xác nhận auto_scale_poses/center_method/orientation_method đã tắt), `coords{metersPerUnit, up_axis,
T_applied, anchor}`, `metrics{psnr_holdout, ssim, n_gaussians, train_time_h}`,
`nav{watertight, max_hole_area_m2, occ_resolution, free_area_m2}`, `sim{drop_test, load_time_s, fps_1080p}`,
`scorecard`, `status ∈ {candidate, approved, usable_caveats, rejected, deprecated}`, `superseded_by`.

## 6. Quality gate — 3 tầng, fail-fast

Threshold trong `_meta/gates.yaml` (versioned trong git).

**Gate A — chạy TRƯỚC KHI rời hiện trường** (điểm ăn tiền nhất):
- scale error < 2% trên ≥3 phép đo dùng scale bar — hard fail
- gravity residual < 1.0°, floor inlier ratio ≥ 60%
- trajectory: 0 NaN, không gap > 0.5s, p95 vận tốc góc < 30°/s, blur ratio < 10%
- coverage: mỗi ô sàn 1×1m nhìn từ ≥2 hướng lệch >30°; ≥1 loop closure nếu quỹ đạo khép kín

**Gate B — sau train**:
- PSNR holdout ≥ 25 dB (warn 22-25), SSIM ≥ 0.80 — tham khảo: RoboGSim (arXiv:2411.11839) báo cáo
  31-34 dB cho static scene bằng 3DGS thuần, nên ngưỡng 25dB là bảo thủ, còn dư địa nếu cần siết lại
- đo lại 3 khoảng cách trên splat render vs thước < 2%
- không có floater trong vùng free space quanh đường đi robot

**Gate C — Isaac Sim**:
- physics drop-test 20 điểm rải đều sàn, 100% dừng trong ±3cm cao độ sàn kỳ vọng, 0 ca rơi xuyên
- collision mesh không lỗ > 5cm trong vùng navigable
- chạy thử 10 waypoint không kẹt; ≥30 fps

**Verdict**: `APPROVED` → dùng cho mọi việc robotics. `USABLE_WITH_CAVEATS` (geometry pass, visual chỉ
warn) → dùng cho navigation/planning, KHÔNG dùng cho RL/perception dựa trên ảnh. `REJECTED` → capture
lại. Geometry là hard gate, visual là soft gate — sàn lệch nguy hiểm hơn splat mờ.

### Feedback loop

Defect taxonomy cố định, ghi vào manifest:

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

Một defect code xuất hiện ≥2 lần trong `_meta/defects_log.md` → bắt buộc thêm checklist mới vào
`_meta/sop_capture.md`, bump `sop_version`, ghi capture_id gây ra nó. Git log của SOP là log của
flywheel. `first_pass_yield` = % session APPROVED ngay lần đầu — chỉ số sức khỏe flywheel, hiển thị
trong catalog.

### Registry / catalog

`_meta/scripts/reindex.py` quét toàn bộ `**/manifest.json` + `**/build.json` → sinh `catalog.md`
(bảng env_id | build BEST | ngày | scale err | PSNR | drop-test | status | đường dẫn USD),
`catalog.json`, cập nhật symlink `BEST`. SQLite chỉ thêm khi vượt ~30 build.

## 7. Thành phần & trạng thái triển khai

| Thành phần | Chạy ở đâu | Việc chính | Trạng thái |
|---|---|---|---|
| [capture/](../capture/) | máy có camera | wrap pyrealsense2 (`realsense_source.py`), ghi raw/ theo session (`session.py`, `record.py`), import export RTAB-Map (`slam_import.py`), schema manifest (`manifest.py`) | Code + 39 unit test PASS (mock data). Camera D435 xác nhận kết nối thật qua USB3, chưa chạy capture session thật end-to-end |
| [align_verify/](../align_verify/) | máy có camera | gravity alignment (`gravity_align.py`), AprilTag anchor (`anchor.py`), Gate A (`gate_a.py`) | Code + test PASS. Chưa chạy trên dữ liệu thật |
| [ingest/](../ingest/) | máy có camera | đóng gói session thành read-only (`package.py`) | Code + test PASS |
| [_meta/scripts/reindex.py](../_meta/scripts/reindex.py) | mọi máy | sinh catalog.md/json từ manifest/build | Code + test PASS |
| `train/` (nerfstudio/3DGS) | máy có GPU | convert transforms.json, train, Gate B | Chưa triển khai |
| `navmesh/` | máy có GPU | ground segmentation, collision mesh, occupancy map | Chưa triển khai |
| `isaacsim_import/` | máy có GPU | ply→usdz, dựng USD stage, Gate C drop-test | Phase 0: PASS (2026-08-17), sẵn sàng triển khai Gate C |

RTAB-Map GUI (xử lý SLAM thủ công theo `_meta/sop_capture.md`) chưa được xác nhận cài/chạy trên máy
này.

## 8. Môi trường phát triển

Quản lý dependency bằng `uv` (venv tại `.venv/`, gitignored):

```bash
uv venv
uv pip install -r requirements.txt
```

Lưu ý: nếu máy có cài ROS2 (vd. Humble), biến `PYTHONPATH` hệ thống có thể đè vào venv gây lỗi import
(`ModuleNotFoundError` cho package chỉ có trong venv). Chạy với `env -u PYTHONPATH` khi cần:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest
```

Camera D435 cần cổng **USB3** (SuperSpeed) để đạt 1280x720@30fps cho cả color và depth — USB2 giới
hạn depth 1280x720 xuống còn 6fps. Kiểm tra tốc độ kết nối: `lsusb -t` (tìm dòng `5000M`).

## 9. Rủi ro còn mở / cần theo dõi

- **Format .ply khác biệt giữa trainer**: `.ply` của splatfacto có layout SH/field khác bản INRIA gốc
  mà `ply_to_usd` (3DGRUT) mong đợi. Phase 0 xác nhận pipeline convert+import chạy đúng với fixture tự
  sinh (đúng schema 3DGRUT) — vẫn cần xác nhận riêng `.ply` do trainer thật (splatfacto/INRIA) sinh ra
  có tương thích thẳng không, khi `train/` được triển khai.
- **Intrinsics drift theo nhiệt độ**: D435 cần calib định kỳ; `calib_id` trong manifest giúp truy vết
  nhưng chưa có quy trình calib tự động.
- **Weak texture / phản chiếu**: RoboGSim (arXiv:2411.11839) dùng GIM feature matcher trước COLMAP để
  xử lý vùng ít texture/ánh sáng yếu/bề mặt phản chiếu — kỹ thuật cụ thể đáng cân nhắc cho bước
  `train/` để giảm defect `FLOOR-NOTEX`/`GLASS`.
- **Số lượng scene dự kiến chưa rõ**: catalog hiện nhắm quy mô chục scene (grep/markdown đủ dùng);
  vượt xa con số đó thì bổ sung SQLite (đã tính trước, không phải thiết kế lại).

## 10. Ngoài phạm vi

- Tự động hoá calib camera định kỳ.
- Chia sẻ dataset ra ngoài máy cá nhân (không cần DVC/git-annex ở quy mô hiện tại).
- Multi-room stitching (ghép nhiều phòng capture riêng lẻ thành một map liền mạch).
- Isaac ROS / nvblox.
- Reconstruct robot arm bằng Gaussian Splatting kiểu RoboGSim (kinematic-driven Gaussian theo khớp) —
  robot dùng asset sẵn có của Isaac Sim, không cần dựng bằng GS.
