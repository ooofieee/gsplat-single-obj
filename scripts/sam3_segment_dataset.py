#!/usr/bin/env python3
"""Run SAM3 text-prompted segmentation on a COLMAP-format image directory.

Outputs:
    output_dir/masks/          — per-image binary masks (.png)
    output_dir/masked_images/  — RGB images with background zeroed out (.png)

Usage:
    conda activate sam3
    python scripts/sam3_segment_dataset.py \
        --image-dir /home/ooofieee/co3d_data/toytruck_colmap_sfm/images \
        --output-dir /home/ooofieee/co3d_data/toytruck_colmap_sfm_sam3 \
        --prompt "blue toytruck"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    parser = argparse.ArgumentParser(description="SAM3 text-prompted batch segmentation")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--ext", type=str, default=".jpg,.jpeg,.png")
    args = parser.parse_args()

    image_dir: Path = args.image_dir
    output_dir: Path = args.output_dir
    mask_dir = output_dir / "masks"
    masked_image_dir = output_dir / "masked_images"
    mask_dir.mkdir(parents=True, exist_ok=True)
    masked_image_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(args.ext.lower().split(","))

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir} (exts: {extensions})")

    print(f"Found {len(image_paths)} images.")

    # TF32 + bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Paths
    sam3_root = os.path.dirname(sam3.__file__)  # .../sam3/sam3
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    ckpt_path = "/home/ooofieee/sam3/checkpoints/sam3.pt"

    # Load SAM3 once
    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    print(f"Segmenting with prompt: '{args.prompt}'")
    for img_path in tqdm(image_paths, desc="SAM3"):
        out_stem = img_path.stem
        mask_path = mask_dir / f"{out_stem}.png"
        masked_img_path = masked_image_dir / f"{out_stem}.png"

        if mask_path.exists() and masked_img_path.exists():
            continue  # skip already processed

        image = Image.open(img_path).convert("RGB")

        # Run SAM3
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=args.prompt)

        # Extract best mask (highest score)
        masks = state["masks"]  # numpy array or tensor
        scores = state["scores"]

        if isinstance(masks, torch.Tensor):
            masks = masks.float().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.float().cpu().numpy()

        if masks is None or len(masks) == 0:
            print(f"  WARNING: No mask found for {img_path.name}, skipping")
            continue

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]  # [C, H, W] or [H, W]

        # Squeeze channel dim if present
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3:
            mask = mask[0]  # take first channel

        # Ensure binary mask
        if mask.dtype == bool:
            mask_bin = mask
        else:
            mask_bin = mask > 0.5

        # Save mask
        Image.fromarray((mask_bin * 255).astype(np.uint8)).save(mask_path)

        # Save masked image (black background)
        img_arr = np.array(image)
        img_arr[~mask_bin] = 0
        Image.fromarray(img_arr).save(masked_img_path)

    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
