import argparse
from pathlib import Path

import torch

from gsplat import export_splats
from scripts.train_poster_gaussian import SH_C0, build_splat_filter, viewer_aligned_export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--axis-mode", choices=["supersplat", "none"], default=None)
    parser.add_argument("--min-brightness", type=float, default=0.1)
    parser.add_argument("--min-alpha", type=float, default=0.08)
    parser.add_argument("--max-radius", type=float, default=1.5)
    args = parser.parse_args()

    model = torch.load(args.ckpt, map_location="cpu")
    axis_mode = args.axis_mode or model.get("axis_mode", "supersplat")

    means = model["means"].float()
    quats = torch.nn.functional.normalize(model["quats"].float(), dim=-1)
    log_scales = model.get("log_scales", None)
    if log_scales is None:
        log_scales = torch.log(model["scales"].float().clamp_min(1e-8))
    else:
        log_scales = log_scales.float()
    opacities = model["opacities"].float()
    colors = model["colors"].float().clamp(0.0, 1.0)

    export_means, export_quats = viewer_aligned_export(means, quats, axis_mode)
    mask = build_splat_filter(
        export_means,
        colors,
        opacities,
        min_brightness=args.min_brightness,
        min_alpha=args.min_alpha,
        max_radius=args.max_radius,
    )
    if not torch.any(mask):
        raise RuntimeError("filters removed all splats")

    sh0 = (colors.unsqueeze(1) - 0.5) / SH_C0
    shN = torch.empty((means.shape[0], 0, 3))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=export_means[mask].cuda(),
        scales=log_scales[mask].cuda(),
        quats=export_quats[mask].cuda(),
        opacities=opacities[mask].cuda(),
        sh0=sh0[mask].cuda(),
        shN=shN[mask].cuda(),
        format="ply",
        save_to=str(args.out),
    )
    print(f"saved {args.out} ({int(mask.sum())}/{len(mask)} splats)")


if __name__ == "__main__":
    main()
