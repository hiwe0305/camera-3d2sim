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
   ├──▼ [4] TRAIN GSPLAT (3DGRUT) ── Gate B
   │      convert capture session → COLMAP-style dataset cho 3DGRUT, đổi trục OpenCV→OpenGL,
   │      TẮT mọi normalize/auto-scale pose mà 3DGRUT áp dụng mặc định (tương đương
   │      auto_scale_poses/center_method/orientation_method của nerfstudio — cần khảo sát
   │      cấu hình 3DGRUT cụ thể, xem §11 Phase 2), freeze camera pose optimizer. Sanity
   │      check: render 1 frame ở pose đã biết, so khớp ảnh gốc TRƯỚC khi train full.
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

**Vì sao tắt auto-normalize pose khi train**: hầu hết framework GSplat (nerfstudio, và có thể cả
3DGRUT) mặc định bật normalize scene về hộp đơn vị khi load dataset — xoá sạch nỗ lực lấy pose mét
thật ở bước 1-2. Train vẫn chạy bình thường, chỉ kết quả sai tỉ lệ — lỗi im lặng thứ hai. **Quyết
định (2026-08-17)**: dùng 3DGRUT (không phải nerfstudio splatfacto như bản nháp đầu) để train — cùng
toolchain với `ply_to_usd` đã validate ở Phase 0, tránh rủi ro lệch layout `.ply` giữa 2 trainer khác
nhau (xem §9). Cấu hình normalize cụ thể của 3DGRUT chưa khảo sát — việc cần làm đầu tiên của `train/`.

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
| `train/` (3DGRUT) | máy có GPU | convert dataset, train, export USD/PLY, Gate B | Chưa triển khai. `~/tools/3dgrut` đã cài + verify ở Phase 0 |
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

**Isaac Sim standalone cần `LD_LIBRARY_PATH` sạch** (xác nhận 2026-08-17): biến `LD_LIBRARY_PATH` của
shell trên máy này trỏ vào `/usr/local/cuda/lib64` (CUDA 12.1, từ setup ROS2/dev khác) — nếu để
nguyên khi chạy `/home/ubuntu/isaacsim/python.sh`, linker sẽ nạp nhầm `libnvJitLink.so.12` bản 12.1
(thiếu symbol `__nvJitLinkCreate_12_8`) thay vì bản 12.8 mà Isaac Sim đóng gói sẵn, khiến
`SimulationApp()` crash ngay khi khởi động — không phải lỗi cài đặt Isaac Sim, không cần cài lại
torch. Luôn `unset LD_LIBRARY_PATH` trước khi gọi `/home/ubuntu/isaacsim/python.sh` (script Python nào
tự gọi Isaac Sim qua `subprocess` cũng nên tự strip biến này khỏi `env` truyền vào, xem
`navmesh/occupancy_map.py`).

## 9. Rủi ro còn mở / cần theo dõi

- **Format .ply khác biệt giữa trainer — ĐÃ GIẢM RỦI RO (2026-08-17)**: rủi ro gốc là dùng nerfstudio
  splatfacto để train rồi convert `.ply` sang USDZ bằng 3DGRUT — 2 toolchain khác nhau, layout SH/field
  có thể lệch. Đã quyết định dùng 3DGRUT cho cả train lẫn export (§3, §11 Phase 2) — cùng một toolchain
  nên không còn rủi ro lệch layout `.ply` giữa 2 bên. Rủi ro còn lại: chưa khảo sát cấu hình
  normalize-pose mặc định của 3DGRUT khi train (xem §11 Phase 2, việc cần làm đầu tiên).
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

## 11. Lộ trình triển khai tiếp theo (sau Phase 0)

Trạng thái tại 2026-08-17: `capture/`, `align_verify/`, `ingest/`, `reindex.py` đã code + test (mock
data) xong nhưng **chưa chạy trên dữ liệu thật**. Phase 0 (Isaac Sim import) PASS. `train/`,
`navmesh/`, `isaacsim_import/` chưa code dòng nào.

### Thứ tự khuyến nghị

```
Phase 1 (Real Capture)  ─┐
                          ├─→ Phase 3 (navmesh/) ─┐
Phase 2 (train/ skeleton)┘                        ├─→ Phase 4 (isaacsim_import/)
        (bắt đầu song song, không cần chờ Phase 1)─┘
```

- **Phase 1** và **Phase 2** có thể làm **song song** — Phase 2 (viết code `train/`) không cần chờ có
  capture thật, dùng dataset công khai của 3DGRUT (MipNeRF360 `garden`/`bonsai`) để phát triển +test
  script convert/train trước, validate lại bằng dữ liệu thật của Phase 1 sau khi cả hai xong.
- **Phase 3** (`navmesh/`) chỉ cần mesh đã gravity-align từ Phase 1 (không cần GSplat đã train) — có
  thể bắt đầu ngay sau Phase 1, chạy song song với Phase 2.
- **Phase 4** (`isaacsim_import/`) cần input từ cả Phase 2 (GSplat đã train) và Phase 3 (collision
  mesh + occupancy map) — làm sau cùng, nhưng phần USD-stage-assembly đã được de-risk phần lớn bởi
  Phase 0 (convert + import đã xác nhận chạy được).

### Phase 1 — Real Capture Session (validate code đã build trên phần cứng thật)

**Mục tiêu**: chạy `capture/` → `align_verify/` → `ingest/` trên một phòng thật, xác nhận toàn bộ code
đã viết (39 unit test hiện tại đều dùng mock) hoạt động đúng ngoài đời, tạo capture session đầu tiên
trong flywheel.

**Việc cần làm**:
1. Cài + xác nhận RTAB-Map GUI chạy được trên máy này (chưa xác nhận — xem §7).
2. Chuẩn bị vật lý: in/dán AprilTag anchor cố định trong phòng, scale bar 1.000m + checkerboard.
3. Viết `capture/record.py` CLI entrypoint thật (hiện `record_session()` là hàm thư viện, chưa có
   script chạy được từ dòng lệnh nối với `RealSenseFrameSource` + `create_capture_session`).
4. Quay thử theo `_meta/sop_capture.md`, chạy RTAB-Map GUI thủ công theo SOP, export trajectory+cloud.
5. Chạy `capture/slam_import.py` → `align_verify/gravity_align.py` + `anchor.py` → đo tay 3 khoảng
   cách → `align_verify/gate_a.py`.
6. Nếu APPROVED: `ingest/package.py` đóng gói session, `_meta/scripts/reindex.py` cập nhật catalog.
7. Nếu REJECTED: ghi defect code vào `_meta/defects_log.md` theo taxonomy đã định nghĩa (đây sẽ là
   dòng đầu tiên thật trong defects_log — hiện file này rỗng/mẫu).

**Rủi ro/quyết định mở**: RTAB-Map GUI chưa xác nhận cài được trên máy này — nên là việc đầu tiên kiểm
tra, có thể tự nó là một spike nhỏ giống Phase 0 nếu cài đặt gặp trục trặc.

**Gate**: Gate A (đã code, threshold trong `_meta/gates.yaml`).

### Phase 2 — `train/` (3DGRUT)

**Mục tiêu**: convert một capture session thành GSplat đã train, đúng tỉ lệ mét thật, qua Gate B.

**Việc cần làm** (theo đúng tinh thần TDD/simple-first của project):
1. **Khảo sát cấu hình normalize-pose của 3DGRUT** (việc đầu tiên, chặn mọi thứ sau) — đọc
   `configs/base_gs.yaml` và `configs/apps/colmap_3dgut.yaml` trong `~/tools/3dgrut`, tìm flag tương
   đương `auto_scale_poses`/`center_method`/`orientation_method` của nerfstudio, xác nhận tắt được.
2. `train/convert_session.py`: convert capture session (`raw/rgb`, `slam/trajectory.tum` đã
   gravity-align) → dataset COLMAP-style mà 3DGRUT đọc được (`configs/dataset/`), đổi trục
   OpenCV→OpenGL nếu cần, dùng `slam/cloud.ply` (đã align) làm init thay vì để 3DGRUT tự chạy COLMAP
   SFM từ đầu (SLAM đã có pose, không cần SFM lại — cần xác nhận 3DGRUT hỗ trợ "known poses" input,
   không phải luôn tự chạy COLMAP).
3. Sanity check bắt buộc TRƯỚC khi train full: render 1 frame ở pose đã biết bằng model init, so khớp
   ảnh gốc — bắt lỗi trục/tỉ lệ sớm, đúng nguyên tắc "lỗi im lặng" đã ghi ở §3.
4. Train (`python -m threedgrut...train.py --config-name apps/colmap_3dgut_mcmc.yaml ...`), với
   `export_usd.enabled=true` để xuất luôn USD/PLY sau train — tái dùng chính pipeline export đã
   validate ở Phase 0.
5. `train/gate_b.py`: đo PSNR/SSIM trên `holdout_frames` (đã tách sẵn lúc capture), đo lại 3 khoảng
   cách trên splat render vs thước đo tay, kiểm tra floater vùng free-space quanh đường đi robot.
6. Ghi `build.json` theo schema đã định nghĩa ở §5 (`versions.lock`, `train_config_sha`,
   `coords{...}`, `metrics{...}`).

**Phát triển sớm không cần chờ Phase 1**: bước 1-4 có thể test bằng dataset công khai
(`data/mipnerf360/garden`, theo README của 3DGRUT) — không cần capture thật, chỉ cần khi validate
Gate B cuối cùng mới cần dữ liệu từ Phase 1 (holdout frames, phép đo tay thật).

**Rủi ro mở**: bước 2 (dùng SLAM pose có sẵn thay vì SFM lại) là điểm chưa chắc chắn nhất — 3DGRUT có
thể mặc định luôn chạy COLMAP; cần khảo sát kỹ trước khi viết code, có thể phải chấp nhận chạy COLMAP
song song SLAM (tốn thêm thời gian nhưng an toàn hơn) nếu không có đường tắt.

**Gate**: Gate B (threshold có sẵn trong `_meta/gates.yaml`, PSNR ≥25dB warn 22-25 — Phase 0 tham khảo
paper RoboGSim thấy baseline thật ~31-34dB nên ngưỡng hiện tại khá bảo thủ).

### Phase 3 — `navmesh/`

**Mục tiêu**: từ mesh đã gravity-align (sản phẩm của Phase 1, không cần chờ Phase 2), dựng collision
mesh sạch + occupancy map 2D cho Isaac Sim.

**Việc cần làm**:
1. Ground plane segmentation (RANSAC, tái dùng logic đã có trong `align_verify/gravity_align.py` —
   cân nhắc refactor phần fit-floor-plane thành hàm dùng chung thay vì viết lại).
2. Lấp lỗ (Poisson reconstruction + trim) trên mesh RTAB-Map, lọc nhiễu.
3. Gọi Isaac Sim Occupancy Map Generator (built-in, cần xác nhận API/CLI cụ thể — chưa khảo sát) để
   sinh occupancy map 2D từ collision mesh.
4. Kiểm tra: max hole area, watertight — theo `nav{watertight, max_hole_area_m2, occ_resolution,
   free_area_m2}` đã định nghĩa trong `build.json` schema (§5).

**Rủi ro mở**: cách gọi Isaac Sim Occupancy Map Generator từ script (không qua GUI) chưa được khảo
sát — cần xác nhận có Python API hay chỉ có thao tác GUI trước khi viết plan chi tiết.

**Gate**: một phần của Gate C (watertight + hole area được kiểm tra ở đây, phần drop-test vật lý thuộc
Phase 4).

### Phase 4 — `isaacsim_import/`

**Mục tiêu**: dựng USD stage hoàn chỉnh (GSplat hiển thị + collision mesh ẩn, cùng gốc toạ độ), chạy
Gate C.

**Việc cần làm**:
1. Tái dùng chính xác lệnh `ply_to_usd`/`export_usd` đã validate ở Phase 0 (bao gồm workaround
   `export_cameras=False` đã ghi trong `spikes/phase0_isaacsim_import/README.md`) để convert output
   thật của Phase 2.
2. Dựng USD stage: reference GSplat USD (từ Phase 2) + collision mesh USD (từ Phase 3, set invisible)
   vào cùng một gốc `T_world_from_slam`/`T_anchor_from_world` — dùng `add_mesh_to_usdz.py` của 3DGRUT
   (đã thấy trong export README lúc nghiên cứu Phase 0) hoặc dựng USD stage thủ công bằng `pxr` API.
3. `isaacsim_import/gate_c.py`: physics drop-test script (20 điểm rải sàn, kiểm tra dừng đúng ±3cm) —
   cần chạy trong Isaac Sim (headless hoặc qua Python API), chưa khảo sát cách tự động hoá tốt nhất.
4. Cập nhật `environments.usda` tổng + symlink `BEST` theo `env_id` (§3.6 cũ / §4 catalog).

**Rủi ro mở**: tự động hoá physics drop-test trong Isaac Sim (headless) là phần khó nhất — có thể cần
một spike riêng giống Phase 0 nếu Isaac Sim's scripting API cho việc này chưa rõ.

**Gate**: Gate C đầy đủ (kết hợp phần navmesh ở Phase 3 + drop-test ở đây).

### Gợi ý bước tiếp theo

Trong 2 nhánh song song (Phase 1 / Phase 2), **Phase 1 rẻ hơn và giải toả nhiều ẩn số hơn** — nó xác
nhận toàn bộ code capture-side đã viết (nhưng chưa test thật) hoạt động đúng, và là input bắt buộc cho
cả Phase 3 lẫn phần validate cuối của Phase 2. Khuyến nghị viết implementation plan chi tiết
(TDD, bite-sized) cho Phase 1 trước, dùng `superpowers:writing-plans` giống Phase 0.
