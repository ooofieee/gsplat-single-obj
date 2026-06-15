from pathlib import Path

import numpy as np
from PIL import Image

from scripts.sam3_segment_video_dataset import (
    choose_anchor_mask,
    choose_anchor_object,
    mask_iou,
    mask_to_xywh,
    pick_output_mask,
    prompt_points_from_mask,
    write_contact_sheet,
    write_mask_and_masked_image,
)


def test_mask_to_xywh_clips_padding_and_normalizes():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 3:8] = True

    assert mask_to_xywh(mask, padding=1, relative=False) == [2.0, 1.0, 7.0, 5.0]
    np.testing.assert_allclose(
        mask_to_xywh(mask, padding=1, relative=True),
        [0.1, 0.1, 0.35, 0.5],
    )


def test_prompt_points_from_mask_uses_foreground_centroid():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:6, 4:10] = True

    points, labels = prompt_points_from_mask(mask, relative=False)

    assert labels == [1]
    assert points == [[6.5, 3.5]]


def test_prompt_points_from_mask_can_sample_multiple_foreground_points():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 3:18] = True

    points, labels = prompt_points_from_mask(mask, relative=False, num_points=5)

    assert labels == [1, 1, 1, 1, 1]
    assert len(points) == 5
    assert len({tuple(p) for p in points}) == 5
    for x, y in points:
        assert mask[int(round(y)), int(round(x))]


def test_choose_anchor_mask_uses_largest_nonempty_seed(tmp_path):
    small = tmp_path / "frame000001.png"
    large = tmp_path / "frame000002.png"
    empty = tmp_path / "frame000003.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(empty)
    a = np.zeros((8, 8), dtype=np.uint8)
    a[1:3, 1:3] = 255
    Image.fromarray(a).save(small)
    b = np.zeros((8, 8), dtype=np.uint8)
    b[1:6, 1:6] = 255
    Image.fromarray(b).save(large)

    chosen = choose_anchor_mask([small, large, empty])

    assert chosen == large


def test_choose_anchor_object_selects_mask_with_highest_seed_iou():
    seed = np.zeros((6, 6), dtype=bool)
    seed[1:4, 1:4] = True
    wrong = np.zeros((6, 6), dtype=bool)
    wrong[3:6, 3:6] = True
    right = np.zeros((6, 6), dtype=bool)
    right[1:4, 1:4] = True
    outputs = {
        "out_obj_ids": np.array([10, 11]),
        "out_binary_masks": np.stack([wrong, right]),
    }

    obj_id, mask, iou = choose_anchor_object(outputs, seed)

    assert obj_id == 11
    assert mask.dtype == bool
    assert iou == 1.0


def test_pick_output_mask_returns_tracked_object_id():
    first = np.zeros((4, 4), dtype=bool)
    second = np.ones((4, 4), dtype=bool)
    outputs = {
        "out_obj_ids": np.array([3, 7]),
        "out_binary_masks": np.stack([first, second]),
    }

    picked = pick_output_mask(outputs, target_obj_id=7)

    assert picked is not None
    assert picked.all()


def test_pick_output_mask_falls_back_to_single_mask_without_object_ids():
    mask = np.ones((4, 4), dtype=bool)
    outputs = {"out_binary_masks": mask[None, ...]}

    picked = pick_output_mask(outputs, target_obj_id=7)

    assert picked is not None
    assert picked.all()


def test_mask_iou_handles_empty_union():
    empty = np.zeros((3, 3), dtype=bool)
    full = np.ones((3, 3), dtype=bool)

    assert mask_iou(empty, empty) == 1.0
    assert mask_iou(empty, full) == 0.0


def test_write_mask_and_masked_image_preserves_foreground(tmp_path):
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    mask_path, masked_path = write_mask_and_masked_image(
        image=image,
        mask=mask,
        stem="frame000001",
        mask_dir=tmp_path / "masks",
        masked_image_dir=tmp_path / "masked_images",
    )

    saved_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    saved_image = np.asarray(Image.open(masked_path).convert("RGB"))
    np.testing.assert_array_equal(saved_mask, mask)
    assert np.count_nonzero(saved_image[~mask]) == 0
    np.testing.assert_array_equal(saved_image[mask], image[mask])


def test_write_contact_sheet_removes_stale_file_when_no_rows(tmp_path):
    output_path = tmp_path / "warnings.png"
    output_path.write_bytes(b"stale")

    write_contact_sheet(output_path, [])

    assert not output_path.exists()
