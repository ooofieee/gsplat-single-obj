#!/usr/bin/env python3
"""Standardize a CO3D sequence into the COLMAP layout used by gsplat.

The main path is intentionally COLMAP/SfM based:

    output/
      images/
      images_4/
      masks/
      sparse/0/{cameras.bin,images.bin,points3D.bin}

CO3D annotations are used to choose valid frames, copy foreground masks, and
provide a stable initial PINHOLE camera for COLMAP. The final camera poses and
points come from COLMAP so the result matches the format expected by
examples.simple_trainer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

PYTORCH3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class PreparedFrame:
    frame_number: int
    image_name: str
    source_image: str
    source_mask: str


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * z * x - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotmat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to COLMAP qvec order [qw, qx, qy, qz]."""
    R = np.asarray(rotmat, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qvec = np.array(
            [
                0.25 / s,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
            ],
            dtype=np.float64,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qvec = np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qvec = np.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qvec = np.array(
            [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def ndc_to_colmap_pinhole(
    focal_length: Sequence[float],
    principal_point: Sequence[float],
    width: int,
    height: int,
    intrinsics_format: str,
) -> tuple[float, float, float, float]:
    """Convert CO3D/PyTorch3D NDC intrinsics to COLMAP pixel intrinsics.

    PyTorch3D NDC uses +X left and +Y up. Screen/COLMAP pixels use +X right and
    +Y down from the top-left image corner, so principal point signs flip.
    """
    fl = np.asarray(focal_length, dtype=np.float64)
    pp = np.asarray(principal_point, dtype=np.float64)

    if intrinsics_format == "ndc_isotropic":
        scale_x = scale_y = min(width, height) / 2.0
    elif intrinsics_format == "ndc_norm_image_bounds":
        scale_x = width / 2.0
        scale_y = height / 2.0
    else:
        raise ValueError(f"Unsupported CO3D intrinsics_format: {intrinsics_format}")

    fx = float(fl[0] * scale_x)
    fy = float(fl[1] * scale_y)
    cx = float(width / 2.0 - pp[0] * scale_x)
    cy = float(height / 2.0 - pp[1] * scale_y)
    return fx, fy, cx, cy


def co3d_to_colmap_w2c(
    R_co3d: Sequence[Sequence[float]],
    T_co3d: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CO3D right-multiply W2C pose to COLMAP/OpenCV W2C.

    CO3D stores PyTorch3D transforms as row-vector world-to-camera:
        X_cam_p3d = X_world @ R + T

    COLMAP stores column-vector OpenCV world-to-camera:
        X_cam_cv = R_w2c @ X_world + tvec
    """
    R = np.asarray(R_co3d, dtype=np.float64)
    T = np.asarray(T_co3d, dtype=np.float64)
    R_w2c = PYTORCH3D_TO_OPENCV @ R.T
    tvec = PYTORCH3D_TO_OPENCV @ T
    return rotmat_to_qvec(R_w2c), tvec


def camera_center_from_co3d(
    R_co3d: Sequence[Sequence[float]],
    T_co3d: Sequence[float],
) -> np.ndarray:
    R = np.asarray(R_co3d, dtype=np.float64)
    T = np.asarray(T_co3d, dtype=np.float64)
    return -R @ T


def image_is_black(path: Path, threshold: float = 1e-3) -> bool:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return float(img.mean()) <= threshold


def mask_is_empty(path: Path, threshold: int = 0) -> bool:
    mask = np.asarray(Image.open(path).convert("L"))
    return int(mask.max()) <= threshold


def composite_image_with_mask(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    black = Image.new("RGB", image.size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.composite(image, black, mask).save(output_path)


def save_colmap_feature_mask(mask_path: Path, mask_root: Path, image_name: str) -> Path:
    """Save a COLMAP feature mask as <relative image path>.png."""
    mask_root.mkdir(parents=True, exist_ok=True)
    dst = mask_root / f"{image_name}.png"
    Image.open(mask_path).convert("L").save(dst)
    return dst


def resize_images(image_dir: Path, resized_dir: Path, factor: int) -> None:
    resized_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(image_dir.iterdir()):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        dst = resized_dir / f"{src.stem}.png"
        img = Image.open(src).convert("RGB")
        width, height = img.size
        img = img.resize((max(1, width // factor), max(1, height // factor)), Image.LANCZOS)
        img.save(dst)


def _load_co3d_frames(co3d_root: Path, category: str):
    from co3d.dataset.data_types import FrameAnnotation, load_dataclass_jgzip
    from typing import List

    return load_dataclass_jgzip(
        str(co3d_root / category / "frame_annotations.jgz"),
        List[FrameAnnotation],
    )


def _train_frame_numbers(co3d_root: Path, category: str, subset_name: str) -> set[int] | None:
    set_list = co3d_root / category / "set_lists" / f"set_lists_{subset_name}.json"
    if not set_list.exists():
        return None
    data = json.loads(set_list.read_text())
    return {int(item[1]) for item in data["train"]}


def collect_valid_frames(
    co3d_root: Path,
    category: str,
    sequence: str,
    subset_name: str,
) -> tuple[list, dict[str, int]]:
    train_numbers = _train_frame_numbers(co3d_root, category, subset_name)
    frames = [
        f
        for f in _load_co3d_frames(co3d_root, category)
        if f.sequence_name == sequence and (train_numbers is None or f.frame_number in train_numbers)
    ]
    frames.sort(key=lambda f: f.frame_number)

    valid = []
    stats = {
        "annotated": len(frames),
        "missing_image": 0,
        "missing_mask": 0,
        "black_image": 0,
        "empty_mask": 0,
    }
    for frame in frames:
        image_path = co3d_root / frame.image.path
        mask_path = co3d_root / frame.mask.path
        if not image_path.exists():
            stats["missing_image"] += 1
            continue
        if not mask_path.exists():
            stats["missing_mask"] += 1
            continue
        if image_is_black(image_path):
            stats["black_image"] += 1
            continue
        if mask_is_empty(mask_path):
            stats["empty_mask"] += 1
            continue
        valid.append(frame)
    stats["valid"] = len(valid)
    return valid, stats


def prepare_colmap_inputs(
    co3d_root: Path,
    category: str,
    sequence: str,
    subset_name: str,
    output_dir: Path,
    composite_images: bool = False,
) -> tuple[list[PreparedFrame], tuple[float, float, float, float], dict[str, int]]:
    frames, stats = collect_valid_frames(co3d_root, category, sequence, subset_name)
    if not frames:
        raise RuntimeError("No valid CO3D frames survived filtering.")

    image_intrinsics = []
    prepared = []
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    colmap_mask_dir = output_dir / "colmap_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    colmap_mask_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        height, width = frame.image.size
        intrinsics = ndc_to_colmap_pinhole(
            frame.viewpoint.focal_length,
            frame.viewpoint.principal_point,
            width,
            height,
            frame.viewpoint.intrinsics_format,
        )
        image_intrinsics.append(intrinsics)

        image_name = f"frame{frame.frame_number:06d}.jpg"
        mask_name = f"frame{frame.frame_number:06d}.png"
        src_image = co3d_root / frame.image.path
        src_mask = co3d_root / frame.mask.path
        if composite_images:
            composite_image_with_mask(src_image, src_mask, image_dir / image_name)
        else:
            shutil.copy2(src_image, image_dir / image_name)
        shutil.copy2(src_mask, mask_dir / mask_name)
        save_colmap_feature_mask(src_mask, colmap_mask_dir, image_name)
        prepared.append(
            PreparedFrame(
                frame_number=int(frame.frame_number),
                image_name=image_name,
                source_image=frame.image.path,
                source_mask=frame.mask.path,
            )
        )

    fx, fy, cx, cy = np.median(np.asarray(image_intrinsics), axis=0).tolist()
    return prepared, (fx, fy, cx, cy), stats


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_colmap_sfm(output_dir: Path, camera_params: Sequence[float]) -> None:
    database = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    params = ",".join(f"{value:.12g}" for value in camera_params)

    feature_cmd = [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(output_dir / "images"),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            params,
            "--SiftExtraction.use_gpu",
            "0",
    ]
    colmap_mask_dir = output_dir / "colmap_masks"
    if colmap_mask_dir.exists():
        feature_cmd.extend(["--ImageReader.mask_path", str(colmap_mask_dir)])
    run_command(feature_cmd)
    run_command(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.guided_matching",
            "1",
        ]
    )
    run_command(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(output_dir / "images"),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
        ]
    )


def validate_colmap_scene(scene_dir: Path, data_factor: int = 4) -> dict[str, object]:
    from pycolmap import SceneManager

    sparse_dir = scene_dir / "sparse" / "0"
    image_dir = scene_dir / "images"
    factor_dir = scene_dir / f"images_{data_factor}"

    missing = [str(path) for path in (sparse_dir, image_dir, factor_dir) if not path.exists()]
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / name).exists():
            missing.append(str(sparse_dir / name))
    if missing:
        raise RuntimeError("Missing required COLMAP dataset paths: " + ", ".join(missing))

    manager = SceneManager(str(sparse_dir))
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    image_names = sorted(image.name for image in manager.images.values())
    missing_images = [name for name in image_names if not (image_dir / name).exists()]
    factor_files = sorted(
        p.name for p in factor_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if missing_images:
        raise RuntimeError(f"{len(missing_images)} COLMAP images are missing from images/")
    if len(factor_files) != len(image_names):
        raise RuntimeError(
            f"images_{data_factor} has {len(factor_files)} files but COLMAP has {len(image_names)} images"
        )
    if len(manager.points3D) == 0:
        raise RuntimeError("COLMAP sparse model has no points3D.")

    return {
        "scene_dir": str(scene_dir),
        "cameras": len(manager.cameras),
        "images": len(manager.images),
        "points3D": int(len(manager.points3D)),
        "images_factor": len(factor_files),
        "camera_models": sorted({str(camera.camera_type) for camera in manager.cameras.values()}),
    }


def write_report(output_dir: Path, report: dict[str, object]) -> None:
    (output_dir / "standardization_report.json").write_text(json.dumps(report, indent=2))


def build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, camera_params, stats = prepare_colmap_inputs(
        args.co3d_root,
        args.category,
        args.sequence,
        args.subset_name,
        output_dir,
        args.composite_images,
    )
    resize_images(output_dir / "images", output_dir / f"images_{args.data_factor}", args.data_factor)

    if args.run_colmap:
        run_colmap_sfm(output_dir, camera_params)
    elif not (output_dir / "sparse" / "0").exists():
        print("Prepared images/masks only. Re-run with --run-colmap to create sparse/0.")

    report: dict[str, object] = {
        "co3d_root": str(args.co3d_root),
        "category": args.category,
        "sequence": args.sequence,
        "subset_name": args.subset_name,
        "valid_frames": len(frames),
        "filter_stats": stats,
        "shared_pinhole_camera": {
            "fx": camera_params[0],
            "fy": camera_params[1],
            "cx": camera_params[2],
            "cy": camera_params[3],
        },
        "prepared_frames": [frame.__dict__ for frame in frames],
    }
    if (output_dir / "sparse" / "0").exists():
        report["validation"] = validate_colmap_scene(output_dir, args.data_factor)
    write_report(output_dir, report)
    print(json.dumps({k: report[k] for k in report if k != "prepared_frames"}, indent=2))


def validate(args: argparse.Namespace) -> None:
    report = validate_colmap_scene(args.scene_dir, args.data_factor)
    print(json.dumps(report, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--co3d-root", type=Path, default=Path("/home/ooofieee/co3d_data"))
    build_parser.add_argument("--category", default="couch")
    build_parser.add_argument("--sequence", default="617_99945_199053")
    build_parser.add_argument("--subset-name", default="manyview_test_0")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--data-factor", type=int, default=4)
    build_parser.add_argument("--run-colmap", action="store_true")
    build_parser.add_argument("--overwrite", action="store_true")
    build_parser.add_argument(
        "--composite-images",
        action="store_true",
        help="Write black-background masked RGB images. Default keeps original RGB and only masks COLMAP features.",
    )
    build_parser.set_defaults(func=build)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--scene-dir", type=Path, required=True)
    validate_parser.add_argument("--data-factor", type=int, default=4)
    validate_parser.set_defaults(func=validate)

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
