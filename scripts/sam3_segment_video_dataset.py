#!/usr/bin/env python3
"""Segment a COLMAP image sequence with SAM3 video tracking.

This script fixes the per-frame text-prompt failure mode where SAM3 can switch
between repeated object instances. A seed mask directory, such as CO3D GT masks,
is used to choose and prompt one anchor frame. SAM3 then tracks that object
through the ordered image sequence and writes per-frame masks plus masked RGBs.

Example:
    conda activate sam3
    python scripts/sam3_segment_video_dataset.py \
        --scene-dir /home/ooofieee/co3d_data/couch_colmap_sfm \
        --seed-mask-dir /home/ooofieee/co3d_data/couch_colmap_sfm/masks \
        --output-dir /home/ooofieee/co3d_data/couch_colmap_video_sam3 \
        --prompt "couch" \
        --link-scene-assets
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CHECKPOINT = "/home/ooofieee/sam3/checkpoints/sam3.pt"


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy/torch-like values to a numpy array without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _binary_masks_from_outputs(outputs: dict[str, Any]) -> np.ndarray:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        masks = outputs.get("masks")
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)

    masks_np = _to_numpy(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 4 and masks_np.shape[-1] == 1:
        masks_np = masks_np[..., 0]
    if masks_np.ndim != 3:
        raise ValueError(f"Expected masks with 2 or 3 spatial dims, got {masks_np.shape}")
    return masks_np > 0


def _obj_ids_from_outputs(outputs: dict[str, Any], num_masks: int) -> list[int]:
    obj_ids = outputs.get("out_obj_ids")
    if obj_ids is None:
        obj_ids = outputs.get("obj_ids")
    if obj_ids is None:
        return list(range(num_masks))
    obj_ids_np = _to_numpy(obj_ids).reshape(-1)
    return [int(x) for x in obj_ids_np[:num_masks]]


def load_bool_mask(path: Path, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def resize_bool_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape == (size[1], size[0]):
        return mask.astype(bool)
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def mask_to_xywh(
    mask: np.ndarray, padding: int = 0, relative: bool = True
) -> list[float]:
    """Return an XYWH box for a binary mask, clipped to image bounds."""
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create a prompt box from an empty mask.")

    height, width = mask.shape
    x0 = max(int(xs.min()) - padding, 0)
    y0 = max(int(ys.min()) - padding, 0)
    x1 = min(int(xs.max()) + 1 + padding, width)
    y1 = min(int(ys.max()) + 1 + padding, height)

    box = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    if relative:
        box = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    return box


def prompt_points_from_mask(
    mask: np.ndarray, relative: bool = True, num_points: int = 1
) -> tuple[list[list[float]], list[int]]:
    """Sample positive point prompts spread across the foreground mask."""
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create prompt points from an empty mask.")
    height, width = mask.shape
    num_points = max(1, int(num_points))
    if num_points == 1:
        x = float(xs.mean())
        y = float(ys.mean())
        if relative:
            x /= width
            y /= height
        return [[x, y]], [1]

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    target_fracs = [
        (0.5, 0.5),
        (0.25, 0.35),
        (0.75, 0.35),
        (0.35, 0.65),
        (0.65, 0.65),
        (0.5, 0.25),
        (0.5, 0.75),
        (0.2, 0.5),
        (0.8, 0.5),
    ]
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    chosen: list[tuple[float, float]] = []
    used_pixels: set[tuple[int, int]] = set()
    for fx, fy in target_fracs:
        if len(chosen) >= num_points:
            break
        target = np.array([x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)], dtype=np.float32)
        order = np.argsort(np.sum((coords - target[None, :]) ** 2, axis=1))
        for idx in order:
            x = float(coords[idx, 0])
            y = float(coords[idx, 1])
            key = (int(round(x)), int(round(y)))
            if key not in used_pixels:
                used_pixels.add(key)
                chosen.append((x, y))
                break

    points = [[x, y] for x, y in chosen]
    if relative:
        points = [[x / width, y / height] for x, y in chosen]
    return points, [1] * len(points)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        b = resize_bool_mask(b, (a.shape[1], a.shape[0]))
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def choose_anchor_mask(mask_paths: Sequence[Path]) -> Path:
    """Choose the largest nonempty seed mask as the anchor prompt frame."""
    best_path: Optional[Path] = None
    best_area = 0
    for path in mask_paths:
        if not path.exists():
            continue
        area = int(load_bool_mask(path).sum())
        if area > best_area:
            best_path = path
            best_area = area
    if best_path is None:
        raise ValueError("No nonempty seed masks were found.")
    return best_path


def choose_anchor_object(
    outputs: dict[str, Any], seed_mask: np.ndarray
) -> tuple[int, np.ndarray, float]:
    """Pick the prompted object whose anchor output overlaps the seed mask most."""
    masks = _binary_masks_from_outputs(outputs)
    obj_ids = _obj_ids_from_outputs(outputs, len(masks))
    if len(masks) == 0:
        raise ValueError("SAM3 returned no masks for the anchor prompt.")

    best_idx = 0
    best_iou = -1.0
    for idx, mask in enumerate(masks):
        iou = mask_iou(mask, seed_mask)
        if iou > best_iou:
            best_idx = idx
            best_iou = iou
    return obj_ids[best_idx], masks[best_idx].astype(bool), float(best_iou)


def pick_output_mask(
    outputs: Optional[dict[str, Any]], target_obj_id: Optional[int]
) -> Optional[np.ndarray]:
    if not outputs:
        return None
    masks = _binary_masks_from_outputs(outputs)
    if len(masks) == 0:
        return None
    obj_ids = _obj_ids_from_outputs(outputs, len(masks))

    if target_obj_id is None:
        return masks[0].astype(bool) if len(masks) == 1 else None
    for obj_id, mask in zip(obj_ids, masks):
        if obj_id == target_obj_id:
            return mask.astype(bool)
    if len(masks) == 1:
        return masks[0].astype(bool)
    return None


def write_mask_and_masked_image(
    image: np.ndarray,
    mask: np.ndarray,
    stem: str,
    mask_dir: Path,
    masked_image_dir: Path,
) -> tuple[Path, Path]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    masked_image_dir.mkdir(parents=True, exist_ok=True)

    mask = resize_bool_mask(mask, (image.shape[1], image.shape[0]))
    mask_path = mask_dir / f"{stem}.png"
    masked_path = masked_image_dir / f"{stem}.png"

    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    masked = image.copy()
    masked[~mask] = 0
    Image.fromarray(masked).save(masked_path)
    return mask_path, masked_path


def iter_image_paths(image_dir: Path, extensions: Iterable[str]) -> list[Path]:
    ext_set = {e.lower() for e in extensions}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in ext_set)


def find_seed_mask(seed_mask_dir: Path, image_path: Path) -> Path:
    candidates = [
        seed_mask_dir / f"{image_path.stem}.png",
        seed_mask_dir / f"{image_path.name}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No seed mask found for {image_path.name} in {seed_mask_dir}")


def link_scene_assets(scene_dir: Path, output_dir: Path) -> None:
    """Link static COLMAP assets so output_dir is directly trainable."""
    for name in ["images", "images_4", "sparse", "colmap_masks"]:
        src = scene_dir / name
        dst = output_dir / name
        if not src.exists() or dst.exists():
            continue
        os.symlink(src, dst, target_is_directory=src.is_dir())


def quality_record(
    image_name: str,
    mask: np.ndarray,
    seed_mask: Optional[np.ndarray],
    previous_area_ratio: Optional[float],
    min_seed_iou: float,
    max_area_jump: float,
) -> dict[str, Any]:
    area_ratio = float(np.asarray(mask).astype(bool).mean())
    record: dict[str, Any] = {
        "image": image_name,
        "area_ratio": area_ratio,
        "warnings": [],
    }
    if seed_mask is not None:
        record["seed_iou"] = mask_iou(mask, seed_mask)
        if record["seed_iou"] < min_seed_iou:
            record["warnings"].append("low_seed_iou")
    if previous_area_ratio is not None:
        denom = max(previous_area_ratio, 1e-6)
        record["area_jump"] = float(abs(area_ratio - previous_area_ratio) / denom)
        if record["area_jump"] > max_area_jump:
            record["warnings"].append("large_area_jump")
    if area_ratio == 0.0:
        record["warnings"].append("empty_mask")
    return record


def write_contact_sheet(
    output_path: Path,
    rows: list[tuple[Path, np.ndarray, Optional[np.ndarray], dict[str, Any]]],
    max_rows: int = 24,
) -> None:
    if not rows:
        return
    thumb_w, thumb_h, label_h = 180, 135, 28
    rendered = []
    for image_path, mask, seed_mask, record in rows[:max_rows]:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        panels = []
        for label, panel_mask in [("sam3", mask), ("seed", seed_mask)]:
            canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
            base = Image.open(image_path).convert("RGB")
            base.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(base, ((thumb_w - base.width) // 2, label_h + (thumb_h - base.height) // 2))
            if panel_mask is not None:
                overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
                resized = resize_bool_mask(panel_mask, base.size)
                alpha = Image.fromarray((resized.astype(np.uint8) * 110))
                overlay.putalpha(alpha)
                canvas.paste(overlay.convert("RGB"), ((thumb_w - base.width) // 2, label_h + (thumb_h - base.height) // 2), overlay)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 4), f"{image_path.stem} {label}", fill=(0, 0, 0))
            panels.append(canvas)
        row = Image.new("RGB", (thumb_w * 2, thumb_h + label_h), "white")
        row.paste(panels[0], (0, 0))
        row.paste(panels[1], (thumb_w, 0))
        rendered.append(row)
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * len(rendered)), "white")
    for idx, row in enumerate(rendered):
        sheet.paste(row, (0, idx * (thumb_h + label_h)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    scene_dir = args.scene_dir
    image_dir = args.image_dir or (scene_dir / "images" if scene_dir else None)
    if image_dir is None:
        raise ValueError("Pass --image-dir or --scene-dir.")
    image_dir = Path(image_dir)
    output_dir = Path(args.output_dir)
    seed_mask_dir = Path(args.seed_mask_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.link_scene_assets and scene_dir is not None:
        link_scene_assets(Path(scene_dir), output_dir)

    image_paths = iter_image_paths(image_dir, args.ext.split(","))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    seed_paths = [find_seed_mask(seed_mask_dir, p) for p in image_paths]
    if args.anchor_frame:
        anchor_stem = Path(args.anchor_frame).stem
        matches = [i for i, p in enumerate(image_paths) if p.stem == anchor_stem]
        if not matches:
            raise ValueError(f"Anchor frame {args.anchor_frame} is not in {image_dir}")
        anchor_idx = matches[0]
        anchor_seed_path = seed_paths[anchor_idx]
    else:
        anchor_seed_path = choose_anchor_mask(seed_paths)
        anchor_idx = seed_paths.index(anchor_seed_path)

    anchor_image = Image.open(image_paths[anchor_idx]).convert("RGB")
    anchor_mask = load_bool_mask(anchor_seed_path, anchor_image.size)
    anchor_box = mask_to_xywh(anchor_mask, padding=args.seed_padding, relative=True)
    anchor_points, anchor_point_labels = prompt_points_from_mask(
        anchor_mask,
        relative=True,
        num_points=args.num_positive_points,
    )

    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        version=args.version,
        compile=args.compile,
        warm_up=False,
    )

    outputs_by_frame: dict[int, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "image_dir": str(image_dir),
        "seed_mask_dir": str(seed_mask_dir),
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "anchor_frame_index": anchor_idx,
        "anchor_frame": image_paths[anchor_idx].name,
        "anchor_seed_mask": str(anchor_seed_path),
        "anchor_box_xywh_relative": anchor_box,
        "anchor_points_relative": anchor_points,
        "frames": [],
    }

    session_id = None
    try:
        response = predictor.handle_request(
            dict(
                type="start_session",
                resource_path=str(image_dir),
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
        )
        session_id = response["session_id"]
        prompt_request = dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=anchor_idx,
            output_prob_thresh=args.output_prob_thresh,
            rel_coordinates=True,
        )
        if args.prompt_mode in {"text", "text-box"}:
            prompt_request["text"] = args.prompt
        if args.prompt_mode in {"box", "text-box"}:
            prompt_request["bounding_boxes"] = [anchor_box]
            prompt_request["bounding_box_labels"] = [1]
        if args.prompt_mode == "point":
            prompt_request["points"] = anchor_points
            prompt_request["point_labels"] = anchor_point_labels
            prompt_request["obj_id"] = args.obj_id
        anchor_response = predictor.handle_request(prompt_request)
        outputs_by_frame[anchor_idx] = anchor_response["outputs"]
        target_obj_id, _, anchor_iou = choose_anchor_object(
            anchor_response["outputs"], anchor_mask
        )
        report["target_obj_id"] = target_obj_id
        report["anchor_iou"] = anchor_iou

        stream_request = dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=args.propagation_direction,
            start_frame_index=anchor_idx,
            output_prob_thresh=args.output_prob_thresh,
        )
        for response in predictor.handle_stream_request(stream_request):
            if "outputs" in response:
                outputs_by_frame[int(response["frame_index"])] = response["outputs"]
    finally:
        if session_id is not None:
            predictor.handle_request(dict(type="close_session", session_id=session_id))
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()

    mask_dir = output_dir / "masks"
    masked_image_dir = output_dir / "masked_images"
    previous_area_ratio: Optional[float] = None
    bad_rows: list[tuple[Path, np.ndarray, Optional[np.ndarray], dict[str, Any]]] = []
    for idx, image_path in enumerate(image_paths):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = pick_output_mask(outputs_by_frame.get(idx), report["target_obj_id"])
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=bool)
        else:
            mask = resize_bool_mask(mask, (image.shape[1], image.shape[0]))
        seed_mask = load_bool_mask(seed_paths[idx], (image.shape[1], image.shape[0]))
        write_mask_and_masked_image(
            image=image,
            mask=mask,
            stem=image_path.stem,
            mask_dir=mask_dir,
            masked_image_dir=masked_image_dir,
        )
        record = quality_record(
            image_name=image_path.name,
            mask=mask,
            seed_mask=seed_mask,
            previous_area_ratio=previous_area_ratio,
            min_seed_iou=args.min_seed_iou,
            max_area_jump=args.max_area_jump,
        )
        previous_area_ratio = record["area_ratio"]
        report["frames"].append(record)
        if record["warnings"]:
            bad_rows.append((image_path, mask, seed_mask, record))

    seed_ious = [f["seed_iou"] for f in report["frames"] if "seed_iou" in f]
    warnings = [f for f in report["frames"] if f["warnings"]]
    report["summary"] = {
        "num_frames": len(image_paths),
        "num_outputs": len(outputs_by_frame),
        "mean_seed_iou": float(np.mean(seed_ious)) if seed_ious else None,
        "min_seed_iou": float(np.min(seed_ious)) if seed_ious else None,
        "num_warning_frames": len(warnings),
        "warning_frames": [f["image"] for f in warnings[:50]],
    }

    report_path = output_dir / "sam3_video_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_contact_sheet(output_dir / "sam3_video_warnings.png", bad_rows)
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GT-seeded SAM3 video segmentation")
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--seed-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="couch")
    parser.add_argument(
        "--prompt-mode",
        choices=["box", "point", "text", "text-box"],
        default="point",
        help=(
            "SAM3 prompt type for the anchor frame. The local SAM3 API does not "
            "allow point prompts to be combined with text or boxes."
        ),
    )
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--num-positive-points", type=int, default=5)
    parser.add_argument("--anchor-frame", type=str, default=None)
    parser.add_argument("--seed-padding", type=int, default=8)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--min-seed-iou", type=float, default=0.5)
    parser.add_argument("--max-area-jump", type=float, default=1.5)
    parser.add_argument("--propagation-direction", choices=["both", "forward", "backward"], default="both")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bpe-path", type=str, default=None)
    parser.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--link-scene-assets", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ext", type=str, default=".jpg,.jpeg,.png")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    report = run_segmentation(args)
    summary = report["summary"]
    print(
        "Done. "
        f"frames={summary['num_frames']} "
        f"mean_seed_iou={summary['mean_seed_iou']:.3f} "
        f"min_seed_iou={summary['min_seed_iou']:.3f} "
        f"warnings={summary['num_warning_frames']}"
    )
    print(f"Outputs: {report['output_dir']}")


if __name__ == "__main__":
    main()
