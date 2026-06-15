# CO3D Couch Customized Trainer Design

## Goal

Create a copied trainer entry point for `/home/ooofieee/co3d_data/couch_colmap_rgbmask` that trains from SAM-masked RGB images and excludes mask-outside pixels from reconstruction loss.

## Dataset Behavior

The scene root remains `/home/ooofieee/co3d_data/couch_colmap_rgbmask` so the existing COLMAP parser can read `sparse/0` and original `images/*.jpg`. The customized trainer maps each COLMAP image name by stem:

- `images/frame000043.jpg` for COLMAP pose/name reference.
- `masked_images/frame000043.png` for the RGB training target.
- `masks/frame000043.png` for the per-image object mask.

When `data_factor > 1`, the customized dataset downsamples `masked_images` and `masks` at runtime to the same dimensions used by the parser, preserving camera intrinsics from the existing parser. Masks are converted to boolean with nonzero pixels as foreground.

## Trainer Behavior

`examples/customized_trainer.py` is copied from `examples/simple_trainer.py` and kept as the dedicated base for this object-only workflow. It does not modify CUDA kernels, rasterization, densification strategy, or shared core rendering.

The existing training mask plumbing is reused:

- `data["mask"]` is passed to `stage.render(..., masks=masks)`.
- L1 loss uses only foreground pixels.
- SSIM compares foreground-preserved images with mask-outside pixels zeroed.
- Rendered output outside the object mask remains zeroed.

Evaluation should also apply the object mask before metric/image comparisons so full-frame black background does not dominate reported quality.

## Defaults

The copied trainer defaults to:

- `data_dir="/home/ooofieee/co3d_data/couch_colmap_rgbmask"`
- `data_factor=4`
- `result_dir="results/couch_customized"`
- `disable_viewer=True`
- `load_exposure=False`

Users can still override config fields via the existing `tyro` CLI.

## Verification

Focused tests cover stem-based image/mask mapping, downsampled mask shape alignment, nonempty foreground masks, and default config values. A smoke import/CLI-level check verifies the new trainer module is usable without touching CUDA.
