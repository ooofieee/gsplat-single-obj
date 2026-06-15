import numpy as np
from PIL import Image

from scripts.standardize_co3d_colmap import (
    camera_center_from_co3d,
    co3d_to_colmap_w2c,
    image_is_black,
    mask_is_empty,
    ndc_to_colmap_pinhole,
    save_colmap_feature_mask,
    qvec_to_rotmat,
)


def test_ndc_isotropic_intrinsics_match_pytorch3d_screen_example():
    fx, fy, cx, cy = ndc_to_colmap_pinhole(
        focal_length=(1.2, 1.2),
        principal_point=(0.2, 0.5),
        width=256,
        height=128,
        intrinsics_format="ndc_isotropic",
    )

    assert fx == 76.8
    assert fy == 76.8
    assert cx == 115.2
    assert cy == 32.0


def test_ndc_norm_image_bounds_uses_axis_specific_scale():
    fx, fy, cx, cy = ndc_to_colmap_pinhole(
        focal_length=(2.0, 4.0),
        principal_point=(0.25, -0.5),
        width=200,
        height=100,
        intrinsics_format="ndc_norm_image_bounds",
    )

    assert fx == 200.0
    assert fy == 200.0
    assert cx == 75.0
    assert cy == 75.0


def test_co3d_pose_converts_to_colmap_opencv_w2c():
    qvec, tvec = co3d_to_colmap_w2c(
        R_co3d=np.eye(3),
        T_co3d=np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(qvec_to_rotmat(qvec), np.diag([-1.0, -1.0, 1.0]))
    np.testing.assert_allclose(tvec, np.array([-1.0, -2.0, 3.0]))

    colmap_center = -qvec_to_rotmat(qvec).T @ tvec
    np.testing.assert_allclose(
        colmap_center,
        camera_center_from_co3d(np.eye(3), np.array([1.0, 2.0, 3.0])),
    )


def test_co3d_pose_conversion_matches_known_couch_frame():
    qvec, tvec = co3d_to_colmap_w2c(
        R_co3d=[
            [-0.8552641868591309, -0.32096993923187256, -0.406818687915802],
            [0.3302117586135864, -0.9426088929176331, 0.04948350414633751],
            [-0.39935365319252014, -0.0920148491859436, 0.9121677279472351],
        ],
        T_co3d=[0.17308206856250763, 0.9126381278038025, 4.1726179122924805],
    )

    np.testing.assert_allclose(
        qvec,
        [0.96307331, -0.01104053, 0.20927076, 0.16903742],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tvec,
        [-0.17308207, -0.91263813, 4.17261791],
        atol=1e-6,
    )


def test_black_image_and_empty_mask_filters(tmp_path):
    black = tmp_path / "black.jpg"
    valid = tmp_path / "valid.jpg"
    empty_mask = tmp_path / "empty.png"
    valid_mask = tmp_path / "valid_mask.png"

    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(black)
    Image.fromarray(np.full((8, 8, 3), 7, dtype=np.uint8)).save(valid)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(empty_mask)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(mask).save(valid_mask)

    assert image_is_black(black)
    assert not image_is_black(valid)
    assert mask_is_empty(empty_mask)
    assert not mask_is_empty(valid_mask)


def test_colmap_feature_mask_uses_image_name_plus_png(tmp_path):
    src = tmp_path / "frame000001.png"
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    Image.fromarray(mask).save(src)

    dst = save_colmap_feature_mask(src, tmp_path / "colmap_masks", "frame000001.jpg")

    assert dst == tmp_path / "colmap_masks" / "frame000001.jpg.png"
    assert np.asarray(Image.open(dst).convert("L")).max() == 255
