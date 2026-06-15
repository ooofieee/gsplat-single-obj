#!/usr/bin/env python3
"""
Video → COLMAP → 3DGS pipeline.

Steps:
  1. Extract --num-frames uniformly sampled frames from the video.
  2. Run COLMAP SfM (feature extraction + sequential matching + mapping).
  3. Generate downsampled images (images_N/) for training.
  4. Train with examples.simple_trainer (DefaultStrategy or MCMC).

Usage:
  conda activate gsplat
  cd /home/ooofieee/gsplat
  python scripts/video_to_3dgs.py /path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_video_frame_count(video_path: Path) -> int:
    """Return total frame count of the first video stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]

    nb_frames = stream.get("nb_frames", "N/A")
    if nb_frames not in ("N/A", "", None):
        return int(nb_frames)

    # Fallback: duration * fps
    fps = float(Fraction(stream["r_frame_rate"]))
    duration = float(stream["duration"])
    return int(fps * duration)


def extract_frames(video_path: Path, images_dir: Path, num_frames: int) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Counting frames in {video_path} ...")
    total = get_video_frame_count(video_path)
    print(f"Total frames: {total}  →  extracting {num_frames} uniformly")

    # Evenly-spaced 0-based frame indices
    indices = np.round(np.linspace(0, total - 1, num_frames)).astype(int)
    select_expr = "+".join(f"eq(n,{idx})" for idx in indices)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"select='{select_expr}'",
        "-vsync", "vfr",
        "-q:v", "2",          # JPEG quality 2 = near-lossless
        str(images_dir / "frame%04d.jpg"),
    ]
    _run(cmd)

    extracted = len(list(images_dir.glob("frame*.jpg")))
    print(f"Extracted {extracted} frames to {images_dir}")
    if extracted != num_frames:
        print(f"  Warning: expected {num_frames}, got {extracted}.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: COLMAP SfM
# ─────────────────────────────────────────────────────────────────────────────

def run_colmap(scene_dir: Path, use_gpu: bool = True) -> None:
    database = scene_dir / "database.db"
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu = "1" if use_gpu else "0"

    # Feature extraction — SIMPLE_RADIAL allows one radial distortion coeff,
    # which handles mild lens distortion common in phone/action-cam footage.
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu,
    ])

    # Sequential matcher covers temporally adjacent frames.
    # Exhaustive matcher then covers all remaining pairs — needed for a 360°
    # orbit so that frame 1 and frame 75 (opposite side) can be matched.
    # For 150 frames exhaustive is ~11 k pairs, fast enough.
    _run([
        "colmap", "sequential_matcher",
        "--database_path", str(database),
        "--SiftMatching.use_gpu", gpu,
        "--SequentialMatching.overlap", "15",
    ])
    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database),
        "--SiftMatching.use_gpu", gpu,
        "--SiftMatching.guided_matching", "1",
    ])

    _run([
        "colmap", "mapper",
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        # Lower these from defaults to accept smaller-baseline initializations
        "--Mapper.init_min_num_inliers", "15",
        "--Mapper.init_min_tri_angle", "4",
        "--Mapper.ba_local_max_refinements", "3",
    ])

    _ensure_best_model_at_zero(sparse_dir)


def _ensure_best_model_at_zero(sparse_dir: Path) -> None:
    """Move the COLMAP model with most registered images to sparse/0."""
    import pycolmap  # noqa: F401  (just to give a better error if missing)

    models = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not models:
        raise RuntimeError("COLMAP mapper produced no models in sparse/.")

    def image_count(model_dir: Path) -> int:
        images_bin = model_dir / "images.bin"
        if not images_bin.exists():
            return 0
        mgr = pycolmap.SceneManager(str(model_dir))
        mgr.load_images()
        return len(mgr.images)

    best = max(models, key=image_count)
    target = sparse_dir / "0"
    if best != target:
        print(f"Best COLMAP model: {best.name} → renaming to sparse/0")
        if target.exists():
            shutil.rmtree(target)
        best.rename(target)

    # Print summary stats
    mgr = pycolmap.SceneManager(str(target))
    mgr.load_cameras()
    mgr.load_images()
    mgr.load_points3D()
    print(
        f"COLMAP sparse/0: {len(mgr.cameras)} camera(s), "
        f"{len(mgr.images)} images, "
        f"{len(mgr.points3D)} points3D"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Downsampled images
# ─────────────────────────────────────────────────────────────────────────────

def generate_downsampled(scene_dir: Path, factor: int) -> None:
    from PIL import Image  # type: ignore

    images_dir = scene_dir / "images"
    factor_dir = scene_dir / f"images_{factor}"
    factor_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    skipped = 0
    for img_path in imgs:
        out_path = factor_dir / (img_path.stem + ".jpg")
        if out_path.exists():
            skipped += 1
            continue
        img = Image.open(img_path)
        new_w = img.width // factor
        new_h = img.height // factor
        img.resize((new_w, new_h), Image.LANCZOS).save(out_path, quality=95)

    print(
        f"Generated {factor_dir}: {len(imgs)} images at 1/{factor}× scale"
        + (f" ({skipped} already existed, skipped)" if skipped else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    scene_dir: Path,
    result_dir: Path,
    data_factor: int,
    max_steps: int,
    strategy: str,
    extra: list[str],
) -> None:
    gsplat_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, "-m", "examples.simple_trainer", strategy,
        "--data_dir", str(scene_dir),
        "--data_factor", str(data_factor),
        "--result_dir", str(result_dir),
        "--max_steps", str(max_steps),
        "--save_ply", "True",
        "--disable_viewer",
    ] + extra
    _run(cmd, cwd=gsplat_root)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    gsplat_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--scene-dir", type=Path, default=None,
        help="Scene directory for extracted frames + COLMAP "
             "(default: <gsplat_root>/data/<video_stem>)",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=None,
        help="Training output directory "
             "(default: <gsplat_root>/results/<video_stem>)",
    )
    parser.add_argument("--num-frames", type=int, default=150,
                        help="Number of frames to extract (default: 150)")
    parser.add_argument(
        "--data-factor", type=int, default=2,
        help="Downsampling factor for training images (default: 2). "
             "Use 4 if GPU memory is tight.",
    )
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument(
        "--strategy", choices=["default", "mcmc"], default="default",
        help="Densification strategy (default: default = original 3DGS paper)",
    )
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU for COLMAP (slower but avoids driver issues)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip frame extraction (reuse existing images/)")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP (reuse existing sparse/0/)")
    parser.add_argument("--skip-downsample", action="store_true",
                        help="Skip downsampled-image generation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (dry run through COLMAP only)")
    args, extra = parser.parse_known_args()
    # Strip leading '--' separator if present
    extra = [a for a in extra if a != "--"]

    stem = args.video.stem
    scene_dir = args.scene_dir or (gsplat_root / "data" / stem)
    result_dir = args.result_dir or (gsplat_root / "results" / stem)

    print(f"\n{'='*60}")
    print(f" Video → 3DGS: {args.video.name}")
    print(f"{'='*60}")
    print(f"  scene_dir  : {scene_dir}")
    print(f"  result_dir : {result_dir}")
    print(f"  num_frames : {args.num_frames}")
    print(f"  data_factor: {args.data_factor}")
    print(f"  max_steps  : {args.max_steps}")
    print(f"  strategy   : {args.strategy}")
    print(f"{'='*60}\n")

    if not args.skip_extract:
        print("─── Step 1/4: Extract frames ───────────────────────────────")
        extract_frames(args.video, scene_dir / "images", args.num_frames)
        print()

    if not args.skip_colmap:
        print("─── Step 2/4: COLMAP SfM ───────────────────────────────────")
        run_colmap(scene_dir, use_gpu=not args.no_gpu)
        print()

    if not args.skip_downsample:
        print(f"─── Step 3/4: Generate images_{args.data_factor}/ ─────────────────────")
        generate_downsampled(scene_dir, args.data_factor)
        print()

    if not args.skip_train:
        print("─── Step 4/4: Train simple_trainer ─────────────────────────")
        run_training(scene_dir, result_dir, args.data_factor, args.max_steps, args.strategy, extra)
        print()

    print(f"{'='*60}")
    print(f" Done!  Results → {result_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
