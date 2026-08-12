# Pipeline: RealSense D435 → 3D Gaussian Splatting → Isaac Sim

Status: Draft — chờ user review
Ngày: 2026-08-12

## 1. Mục tiêu & bối cảnh

Dựng lại không gian thật (phòng/khu vực) bằng camera Intel RealSense D435 (RGB-D, không IMU) thành
3D Gaussian Splatting đúng tỉ lệ mét thật, chân thực như Scaniverse, rồi nạp vào NVIDIA Isaac Sim
để làm environment cho robot navigation.

**Mục tiêu tối thượng**: dựng được một thư viện environment chất lượng cao trong Isaac Sim, để sau
đó toàn bộ công việc phát triển robot (navigation, RL training, testing) diễn ra **thuần trong sim**,
không cần quay lại camera thật cho phần robot logic. Điều này có nghĩa: pipeline này được thiết kế
như một **quy trình lặp lại được** (capture nhiều scene, có thể capture lại cùng 1 phòng nhiều lần
để cải thiện chất lượng), không phải một lần làm rồi thôi.

**Bố trí máy móc**:
- Laptop: cắm D435, chỉ dùng để capture + verify tại chỗ. Không có GPU NVIDIA mạnh.
- Server/workstation riêng: vừa train Gaussian Splatting vừa chạy Isaac Sim (cùng một máy).

**Nguyên tắc dẫn dắt thiết kế**: đơn giản trước, phức tạp/độ phân giải cao sau — nhưng MVP đầu
tiên đã phải đi hết tới navigation cơ bản (robot ảo né vật cản được trong Isaac Sim), không dừng
ở visual đẹp. YAGNI: không dùng ROS2/Docker, không dùng DVC/MLflow — mọi công cụ chọn ở mức tối
thiểu đủ dùng cho một người, một laptop, một server.

## 2. Kiến trúc pipeline

### Phase 0 — Spike xác nhận Isaac Sim import (làm TRƯỚC TIÊN, tách biệt khỏi phần còn lại)

Rủi ro lớn nhất ban đầu tưởng là "Isaac Sim có hỗ trợ Gaussian Splatting không" — hoá ra Isaac Sim
đã hỗ trợ native từ bản 6.0 (qua Omniverse NuRec), nhưng cần xác nhận thực tế trên máy của mình
trước khi đầu tư công sức vào SLAM.

Việc cần làm: tải một file `.ply` Gaussian Splat mẫu công khai bất kỳ → chạy `ply_to_usd` (công cụ
3DGRUT) → mở `.usdz` kết quả trong Isaac Sim (yêu cầu bản ≥6.0) → xác nhận render đúng, không gặp
lỗi "layered artifact" đã được báo cáo trên forum NVIDIA.

**Đây là cổng go/no-go**: nếu thất bại, cần có phương án dự phòng (convert splat sang textured mesh,
hoặc dùng point cloud renderer) trước khi tiếp tục các bước sau. Không đi tiếp nếu Phase 0 chưa pass.

### Pipeline chính (6 bước, sau khi Phase 0 pass)

```
D435 (laptop)
   │
   ▼
[1] CAPTURE ── RTAB-Map standalone (không cần ROS2/Docker), RGB+depth đồng bộ
   │              → trajectory mét thật (chưa gravity-align) + point cloud/mesh
   ▼
[2] GRAVITY ALIGN + VERIFY (laptop) ── Gate A
   │   RANSAC fit mặt sàn → xoay về +Z → MỘT transform gốc "T_world_from_slam"
   │   duy nhất áp cho toàn bộ trajectory + mesh (không để mỗi nhánh sau tự làm riêng)
   │   Đăng ký về anchor vật lý cố định (AprilTag) → T_anchor_from_world
   │  rsync/scp (kèm manifest + transform)
   ▼
[3] TRANSFER → server
   │
   ├──▼ [4] TRAIN GSPLAT (server) ── Gate B
   │      convert → transforms.json, đổi trục OpenCV→OpenGL, TẮT auto_scale_poses/
   │      center_method/orientation_method của nerfstudio, dùng TSDF cloud làm init,
   │      freeze camera pose optimizer. Sanity check: render 1 frame ở pose đã biết,
   │      so khớp ảnh gốc TRƯỚC khi train full.
   │
   └──▼ [5] NAV GEOMETRY (server, song song, dùng chung mesh đã gravity-align)
          ground plane segmentation + lấp lỗ (Poisson+trim) + lọc nhiễu → collision mesh
          → Isaac Sim Occupancy Map Generator (built-in) cho occupancy map 2D
   │
   ▼
[6] ISAAC SIM IMPORT (server) ── Gate C
    dùng lại pipeline đã validate ở Phase 0 (ply→usdz), dựng USD stage
    (collision mesh ẩn + GSplat hiển thị, cùng gốc T_world_from_slam / T_anchor_from_world)
    Verify: physics drop-test — thả rigid body, kiểm tra nó nằm đúng mặt sàn
    nhìn thấy trong splat, không lơ lửng/chìm.
```

**Vì sao chọn RTAB-Map thay vì Open3D odometry**: Open3D's built-in RGB-D
odometry/reconstruction được tune cho chuyển động chậm/mượt, dễ mất tracking khi cầm tay đi vòng
quanh phòng. RTAB-Map có bản standalone (không cần ROS2/Docker), loop closure bag-of-words trưởng
thành hơn, vẫn giữ tiêu chí "đơn giản, không ROS2".

**Vì sao gravity alignment là một bước bắt buộc riêng**: D435 không có IMU, nên pose từ SLAM chỉ
đúng vị trí tương đối, KHÔNG đảm bảo mặt sàn song song mặt phẳng XY. Nếu bỏ qua, scene sẽ nghiêng
vài độ trong Isaac Sim (Z-up, trọng lực theo -Z) → robot ảo "trượt dốc", occupancy map méo, path
planning sai — một lỗi im lặng, chỉ lộ ra khi robot chạy sai chứ nhìn ảnh không thấy.

**Vì sao phải tắt auto_scale_poses/center_method/orientation_method của nerfstudio**: các flag này
mặc định bật, sẽ normalize scene về hộp đơn vị — xoá sạch nỗ lực lấy pose mét thật ở bước 1-2. Đây
là lỗi im lặng thứ hai: train vẫn chạy bình thường, chỉ là kết quả sai tỉ lệ.

## 3. Data Flywheel — tổ chức dữ liệu cho quy trình lặp lại

Vì mục tiêu là một thư viện environment dùng lâu dài (không phải one-shot), cần một lớp tổ chức dữ
liệu để: theo dõi version, đánh giá chất lượng nhất quán, và cải thiện dần quy trình capture.

**Nguyên tắc thiết kế**: filesystem là database, mỗi thư mục tự mô tả bằng một JSON manifest, catalog
là sản phẩm phái sinh (derived) — có thể xoá và sinh lại bất cứ lúc nào. Không DVC, không MLflow.

### 3.1 Ba khái niệm, quan hệ 1-N-N

- **environment**: một không gian vật lý cố định (vd. `lab_room_a`), có anchor vật lý cố định
  (AprilTag dán tại một góc phòng) để các lần capture lại sau vẫn register về đúng một frame.
- **capture session**: một lần đi quay (vd. `2026-08-12_v01`) — bất biến (read-only) sau khi ingest.
- **build**: một lần train/đóng gói ra USD từ một capture session (vd. `v01_gsA`) — có thể có nhiều
  build từ cùng một capture khi đổi hyperparameter.

### 3.2 Cấu trúc thư mục

```
~/scenes/
  _meta/                                <- git repo (chỉ text: script, config, log)
      scripts/  gates.yaml  sop_capture.md  capture_profiles/  defects_log.md
      catalog.json  catalog.md
  lab_room_a/                           <- env_id, ổn định vĩnh viễn
      scene.json                        <- anchor def, kích thước phòng, con trỏ BEST
      captures/
          2026-08-12_v01/
              raw/      rgb/ depth/ *.bag  calib/
              slam/     rtabmap.db  trajectory.tum  cloud.ply  mesh.ply
              align/    T_world_from_slam.json  mesh_aligned.ply
              eval/     holdout_frames.txt  measurements.csv  gateA.json
              manifest.json  capture_log.md
      builds/
          v01_gsA/
              config/  transforms.json  train_cmd.sh  versions.lock
              gsplat/  splat.ply    nav/  collision.obj  occupancy.png+yaml
              usd/     env.usd      report/ scorecard.json  renders/
              build.json
      BEST -> builds/v01_gsA            <- symlink, cập nhật bởi script
```

Quy ước tên: capture = `YYYY-MM-DD_vNN`, build = `<capture_ver>_gs<Letter>`. Sau khi ingest xong,
đặt thư mục capture thành read-only (`chmod -R a-w`) — mọi thay đổi phải tạo version mới, không sửa
tại chỗ.

**Anchor & scale bar vật lý** (bắt buộc mang theo mỗi lần capture): một AprilTag dán cố định tại một
vị trí trong phòng (để nhiều lần capture cùng phòng register về chung một gốc toạ độ), và một scale
bar 1.000 m + checkerboard đặt trong khung hình đầu (để đo scale error tự động và lặp lại được, thay
vì đo tay bằng thước).

**Versioning/backup**: không dùng DVC ở quy mô này. Git cho `_meta/` (script, SOP, gates, manifest
JSON — toàn bộ đều là text nhỏ). sha256 của từng artifact lớn (point cloud, splat, usd) ghi trong
manifest để phát hiện hỏng/đổi. rsync định kỳ sang ổ cứng ngoài làm backup.

**Retention**: giữ `raw/` của bản APPROVED mới nhất + 1 bản trước đó; capture bị REJECTED thì xoá
`raw/` sau 30 ngày nhưng giữ `manifest.json` vĩnh viễn (chỉ vài KB, là dữ liệu học của flywheel).

### 3.3 Manifest schema (key chính)

`manifest.json` (capture session):
- **identity**: `schema_version, capture_id, env_id, capture_version, started_at, ended_at, sop_version, capture_profile_id`
- **hardware/calib**: `camera_model, serial, firmware, depth_preset, resolution, fps, exposure_mode, emitter_on, depth_scale, calib_id, calib_date, intrinsics{fx,fy,cx,cy,dist}`
- **conditions**: `lighting_type, dynamic_objects_present, floor_material, glass_or_mirror_present, texture_poor_regions[]`
- **capture stats**: `n_frames, duration_s, trajectory_length_m, mean_linear_vel, p95_angular_vel, blur_frame_ratio, loop_closures, room_bbox`
- **align**: `T_world_from_slam(4x4), floor_inlier_ratio, floor_rms_mm, gravity_residual_deg, T_anchor_from_world`
- **ground truth**: `measurements[{name, tape_m, cloud_m, err_pct}]`, `holdout_frames[]` (5-10% frame tách riêng ngay từ lúc capture)
- **artifacts**: `[{path, sha256, bytes}]`
- **gate**: `gateA{scores, verdict}`, `defects[{code, note}]`

`build.json` (trained environment): `build_id, env_id, source_capture_id, source_manifest_sha`,
`versions.lock{nerfstudio, gsplat, 3dgrut, isaacsim, cuda, driver, repo_git_sha}`,
`train_config_sha` (+ xác nhận các flag auto_scale_poses/center_method/orientation_method đã tắt),
`coords{metersPerUnit, up_axis, T_applied, anchor}`,
`metrics{psnr_holdout, ssim, n_gaussians, train_time_h}`,
`nav{watertight, max_hole_area_m2, occ_resolution, free_area_m2}`,
`sim{drop_test, load_time_s, fps_1080p}`,
`scorecard`, `status ∈ {candidate, approved, usable_caveats, rejected, deprecated}`, `superseded_by`.

### 3.4 Quality gate — 3 tầng, fail-fast

Threshold nằm trong `_meta/gates.yaml` (versioned trong git — đổi ngưỡng là một commit, lịch sử điểm
số vẫn diễn giải được).

**Gate A — laptop, chạy TRƯỚC KHI rời hiện trường** (điểm ăn tiền nhất — tránh về tới nhà mới biết hỏng):
- scale error < 2% trên ≥3 phép đo dùng scale bar — hard fail
- gravity residual < 1.0°, floor inlier ratio ≥ 60%
- trajectory: 0 NaN, không gap > 0.5s, p95 vận tốc góc < 30°/s, blur ratio < 10%
- coverage: mỗi ô sàn 1×1m được nhìn từ ≥2 hướng lệch >30°; ≥1 loop closure nếu quỹ đạo khép kín

**Gate B — server, sau train**:
- PSNR holdout ≥ 25 dB (warn 22-25), SSIM ≥ 0.80
- đo lại 3 khoảng cách trên splat render vs thước < 2%
- không có floater trong vùng free space quanh đường đi robot

**Gate C — Isaac Sim**:
- physics drop-test 20 điểm rải đều sàn, 100% dừng trong ±3cm cao độ sàn kỳ vọng, 0 ca rơi xuyên
- collision mesh không lỗ > 5cm trong vùng navigable
- chạy thử 10 waypoint không kẹt; ≥30 fps

**Verdict**: `APPROVED` (không fail nào) → dùng được cho mọi việc robotics. `USABLE_WITH_CAVEATS`
(geometry pass, visual chỉ warn) → dùng cho navigation/planning, KHÔNG dùng cho RL/perception dựa
trên ảnh. `REJECTED` → capture lại.

**Nguyên tắc**: geometry là hard gate, visual là soft gate — vì mục tiêu cuối là navigation, sàn lệch
5cm nguy hiểm hơn splat mờ.

### 3.5 Feedback loop

- **Defect taxonomy cố định**, ghi vào manifest: `DARK, BLUR-FAST, COV-GAP, GLASS, DYN-OBJ, DRIFT-NOLOOP, FLOOR-NOTEX, SCALE-DRIFT`.
- `_meta/sop_capture.md` là checklist sống, versioned trong git. Quy tắc cứng: một defect code xuất
  hiện ≥2 lần thì bắt buộc sinh ra một dòng checklist mới hoặc một thay đổi tham số, kèm link tới
  capture_id đã gây ra nó. Git log của file này là log của flywheel.
- `_meta/capture_profiles/pNN.yaml`: config quay (exposure, emitter, resolution, tốc độ đi tối đa,
  khoảng cách tới tường, pattern quỹ đạo). SOP đổi → bump profile → `capture_profile_id` ghi vào
  manifest → so sánh được profile nào cho pass-rate cao nhất.
- **Chỉ số sức khỏe flywheel**: `first_pass_yield` = % session APPROVED ngay lần đầu, hiển thị trong
  catalog — theo dõi xem flywheel có đang cải thiện theo thời gian không.

### 3.6 Registry / catalog

Script `reindex.py` quét toàn bộ `**/manifest.json` + `**/build.json` → sinh:
- `catalog.md`: một bảng — `env_id | build BEST | ngày | scale err | PSNR | drop-test | status | đường dẫn USD`. Đây là thứ mở ra đọc hằng ngày.
- `catalog.json`: cùng dữ liệu, cho script khác dùng.
- cập nhật symlink `BEST` của từng environment theo verdict + ngày mới nhất APPROVED.

SQLite chỉ thêm khi vượt ~30 build và grep không còn đủ — vẫn cùng script sinh ra, không phải phụ
thuộc mới.

**Tích hợp phía Isaac Sim**: mỗi environment APPROVED xuất `usd/env.usd`; một stage tổng
`environments.usda` **reference theo `env_id`** chứ không hardcode đường dẫn build cụ thể. Khi capture
lại một phòng và promote build mới, chỉ cần đổi một reference — mọi scene robot đang dùng phòng đó
tự động cập nhật. Đây là chỗ data flywheel trả lãi cho giai đoạn sim-only về sau.

## 4. Thành phần (components)

| Thành phần | Chạy ở đâu | Việc chính |
|---|---|---|
| `capture/` | laptop | wrap RTAB-Map + pyrealsense2, ghi raw/ + slam/ theo cấu trúc session |
| `align_verify/` | laptop | gravity alignment (RANSAC floor fit), tính T_world_from_slam/T_anchor_from_world, chạy Gate A, ghi manifest.json |
| `ingest/` | laptop→server | rsync session sang server, đặt read-only, đăng ký vào `scene.json` của environment |
| `train/` | server | convert sang transforms.json (đổi trục, tắt auto-normalize, TSDF init), gọi trainer 3DGS, chạy Gate B |
| `navmesh/` | server | ground segmentation, lấp lỗ, collision mesh; gọi Isaac Sim Occupancy Map Generator |
| `isaacsim_import/` | server | ply→usdz (pipeline đã validate ở Phase 0), dựng USD stage, chạy Gate C (drop-test) |
| `_meta/scripts/qgate.py` | cả hai | đọc manifest, áp threshold từ gates.yaml, ghi scorecard.json |
| `_meta/scripts/reindex.py` | server | sinh catalog.md/json từ toàn bộ manifest/build |

## 5. Rủi ro còn mở / cần theo dõi

- **Format .ply khác biệt giữa trainer**: `.ply` của splatfacto có layout SH/field khác bản INRIA
  gốc mà `ply_to_usd` (3DGRUT) mong đợi — cần xác nhận trong Phase 0 dùng trainer nào cho ra format
  tương thích thẳng, tránh phải viết converter riêng.
- **Intrinsics drift theo nhiệt độ**: D435 cần calib định kỳ; `calib_id` trong manifest giúp truy vết
  nhưng chưa có quy trình calib tự động — để làm thủ công ở giai đoạn đầu, tự động hoá sau nếu cần.
- **Số lượng scene dự kiến chưa rõ**: thiết kế catalog hiện nhắm quy mô chục scene (grep/markdown đủ
  dùng); nếu vượt xa con số đó, cần bổ sung SQLite (đã tính trước trong §3.6, không phải thiết kế lại).

## 6. Ngoài phạm vi (out of scope cho thiết kế này)

- Tự động hoá calib camera định kỳ.
- Chia sẻ dataset ra ngoài máy cá nhân (không cần DVC/git-annex ở quy mô hiện tại).
- Multi-room stitching (ghép nhiều phòng capture riêng lẻ thành một map liền mạch) — mỗi environment
  hiện được coi là một không gian capture trong một lần đi liên tục.
- Isaac ROS / nvblox (hướng C đã cân nhắc và gác lại cho giai đoạn sau MVP).
