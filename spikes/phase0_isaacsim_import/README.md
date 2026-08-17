# Phase 0 spike: Isaac Sim Gaussian Splat import

Answers the go/no-go question in `docs/design.md` §3 Phase 0: does a `.ply`
Gaussian Splat survive conversion to `.usdz` (via `nv-tlabs/3dgrut`) and
import into Isaac Sim 6.0.1 without the "layered artifact" bug reported on
the NVIDIA forums?

## Running it

Note: this project's `pytest.ini` sets `testpaths = tests`, so the tests in this directory are NOT
part of the default `pytest` run — they must be invoked explicitly by path (as shown in the plan's
Task 2/3 steps). This is intentional: `test_convert_to_usdz.py` depends on the gitignored
`out/test_scene.usdz`, and including it in the default run would make a fresh clone's `pytest` fail
red.

1. Install 3DGRUT once, outside this repo: see
   `docs/plans/2026-08-17-phase0-isaacsim-import-spike.md` Task 1.
2. `env -u PYTHONPATH .venv/bin/python spikes/phase0_isaacsim_import/generate_test_splat.py`
3. Convert to USDZ using `~/tools/3dgrut`. The straightforward form
   (`python -m threedgrut.export.scripts.ply_to_usd ...`) fails on the currently-installed 3dgrut:
   `NuRecExporter()` defaults `export_cameras=True`, which raises `ValueError` with no dataset.
   Fix: edit `~/tools/3dgrut/threedgrut/export/scripts/ply_to_usd.py`, change
   `exporter = NuRecExporter()` to `exporter = NuRecExporter(export_cameras=False)`, then run:
   ```bash
   PATH="/home/ubuntu/tools/3dgrut/.venv/bin:$PATH" /home/ubuntu/tools/3dgrut/.venv/bin/python \
       -m threedgrut.export.scripts.ply_to_usd \
       spikes/phase0_isaacsim_import/out/test_scene.ply \
       --output_file spikes/phase0_isaacsim_import/out/test_scene.usdz
   ```
   (`PATH` is prefixed with the venv's `bin` dir instead of `source .../activate`, which doesn't work
   reliably in this environment.)
4. Import `spikes/phase0_isaacsim_import/out/test_scene.usdz` into Isaac Sim
   (`File > Import`) and check against the pass/fail checklist in Task 4 of
   `docs/plans/2026-08-17-phase0-isaacsim-import-spike.md`.

## Result

**Kết quả (2026-08-17)**: PASS — xem `spikes/phase0_isaacsim_import/` cho chi tiết.
Nhập vào Isaac Sim 6.0.1 thành công: hình cầu mịn, gradient màu xanh-đỏ, không có layered
artifact, tỉ lệ ~1m đúng tại gốc toạ độ. Gate mở, sẵn sàng triển khai `train/`, `navmesh/`,
`isaacsim_import/`.
