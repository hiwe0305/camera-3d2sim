# Phase 0 spike: Isaac Sim Gaussian Splat import

Answers the go/no-go question in `docs/design.md` §3 Phase 0: does a `.ply`
Gaussian Splat survive conversion to `.usdz` (via `nv-tlabs/3dgrut`) and
import into Isaac Sim 6.0.1 without the "layered artifact" bug reported on
the NVIDIA forums?

## Running it

1. Install 3DGRUT once, outside this repo: see
   `docs/plans/2026-08-17-phase0-isaacsim-import-spike.md` Task 1.
2. `env -u PYTHONPATH .venv/bin/python spikes/phase0_isaacsim_import/generate_test_splat.py`
3. Convert to USDZ using `~/tools/3dgrut`. The straightforward form
   (`python -m threedgrut.export.scripts.ply_to_usd ...`) fails on the currently-installed 3dgrut
   (`NuRecExporter()` defaults `export_cameras=True`, which raises `ValueError` with no dataset).
   Prefix `PATH` with the venv's `bin` dir (do not rely on `source .../activate` in this
   environment), and pass `export_cameras=False` to `NuRecExporter` — e.g. by editing a local copy
   of `ply_to_usd.py` outside this repo, or waiting for upstream to add a CLI flag for it. See
   `.superpowers/sdd/2026-08-17-phase0-isaacsim-import-spike/task-3-report.md` for the exact
   commands used.
4. Import `spikes/phase0_isaacsim_import/out/test_scene.usdz` into Isaac Sim
   (`File > Import`) and check against the pass/fail checklist in Task 4 of
   the plan above.

## Result

**Kết quả (2026-08-17)**: PASS — xem `spikes/phase0_isaacsim_import/` cho chi tiết.
Nhập vào Isaac Sim 6.0.1 thành công: cầu hình cầu mịn, gradient màu xanh-đỏ, không có layered
artifact, tỉ lệ ~1m đúng tại gốc toạ độ. Gate mở, sẵn sàng triển khai `train/`, `navmesh/`,
`isaacsim_import/`.
