import numpy as np
from plyfile import PlyData

from generate_test_splat import generate_sphere_splat


def test_generate_sphere_splat_writes_expected_point_count(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    ply = PlyData.read(str(out_path))
    vertex = ply["vertex"]
    assert len(vertex) == 200


def test_generate_sphere_splat_has_expected_ply_fields(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    ply = PlyData.read(str(out_path))
    field_names = {p.name for p in ply["vertex"].properties}
    expected = {
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    assert expected.issubset(field_names)


def test_generate_sphere_splat_points_lie_on_sphere_surface(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.3)

    vertex = PlyData.read(str(out_path))["vertex"]
    positions = np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1
    )
    radii = np.linalg.norm(positions, axis=1)
    assert np.allclose(radii, 0.3, atol=1e-4)


def test_generate_sphere_splat_opacity_is_high_confidence_logit(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    vertex = PlyData.read(str(out_path))["vertex"]
    opacity = np.asarray(vertex["opacity"])
    # logit(0.95) ~= 2.944 -- a visibly-opaque splat, not a near-invisible one
    assert np.all(opacity > 2.9)
    assert np.all(opacity < 3.0)


def test_generate_sphere_splat_scale_is_log_of_15mm(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    vertex = PlyData.read(str(out_path))["vertex"]
    scale_0 = np.asarray(vertex["scale_0"])
    assert np.allclose(np.exp(scale_0), 0.015, atol=1e-4)


def test_generate_sphere_splat_color_gradient_matches_height(tmp_path):
    out_path = tmp_path / "test_scene.ply"

    generate_sphere_splat(out_path, n_points=200, radius=0.5)

    vertex = PlyData.read(str(out_path))["vertex"]
    z = np.asarray(vertex["z"])
    f_dc_0 = np.asarray(vertex["f_dc_0"])  # red channel DC coefficient
    f_dc_2 = np.asarray(vertex["f_dc_2"])  # blue channel DC coefficient
    SH_C0 = 0.28209479177387814
    red = SH_C0 * f_dc_0 + 0.5
    blue = SH_C0 * f_dc_2 + 0.5

    lowest = np.argmin(z)   # bottom of sphere: should be near-pure blue
    highest = np.argmax(z)  # top of sphere: should be near-pure red
    assert red[lowest] < 0.05 and blue[lowest] > 0.95
    assert red[highest] > 0.95 and blue[highest] < 0.05
