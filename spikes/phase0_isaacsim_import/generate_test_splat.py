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

    # Uniform points on a sphere surface (cylindrical/Archimedes projection: uniform z, then azimuth).
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

    opacity_logit = np.full((n_points, 1), 2.9444, dtype=np.float32)  # logit(0.95); sigmoid(2.9444) ~= 0.95

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
