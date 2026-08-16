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
