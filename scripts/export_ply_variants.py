import argparse
from pathlib import Path

import torch

from gsplat import export_splats


SH_C0 = 0.28209479177387814


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def transform(means: torch.Tensor, quats: torch.Tensor, mode: str):
    dtype, device = means.dtype, means.device
    if mode == "raw":
        rot = torch.eye(3, device=device, dtype=dtype)
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    elif mode == "zup_to_yup":
        # Rotate -90 degrees around X: Z-up -> Y-up.
        rot = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        q = torch.tensor([2**0.5 / 2, -(2**0.5) / 2, 0.0, 0.0], device=device, dtype=dtype)
    elif mode == "zup_to_yup_flipz":
        rot = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        # This includes a reflection-like handedness flip for viewer triage; leave
        # splat local rotations unchanged because it is only for finding the right axis.
        q = None
    elif mode == "swap_xy_zup":
        rot = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            device=device,
            dtype=dtype,
        )
        q = None
    else:
        raise ValueError(mode)

    out_means = means @ rot.T
    out_means = out_means - out_means.median(dim=0).values
    radius = torch.quantile(torch.linalg.norm(out_means, dim=-1), 0.95).clamp_min(1e-6)
    out_means = out_means / radius

    out_quats = quats
    if q is not None:
        out_quats = quat_mul_wxyz(q.expand_as(quats), quats)
        out_quats = torch.nn.functional.normalize(out_quats, dim=-1)
    return out_means, out_quats


def export_one(out_path: Path, model: dict, mode: str, mask: torch.Tensor):
    means = model["means"][mask].cuda()
    log_scales = model["log_scales"][mask].cuda()
    quats = model["quats"][mask].cuda()
    opacities = model["opacities"][mask].cuda()
    colors = model["colors"][mask].cuda()
    means, quats = transform(means, quats, mode)
    sh0 = (colors.unsqueeze(1) - 0.5) / SH_C0
    shN = torch.empty((means.shape[0], 0, 3), device=means.device)
    export_splats(
        means=means,
        scales=log_scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(out_path),
    )
    print(f"saved {out_path} ({means.shape[0]} splats)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("results/poster_gaussian_v2/poster_gaussian_step2500.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/poster_gaussian_v2_variants"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = torch.load(args.ckpt, map_location="cpu")
    alpha = torch.sigmoid(model["opacities"])
    colors = model["colors"]
    brightness = colors.mean(dim=-1)
    warm_floor = (colors[:, 0] > colors[:, 1] + 0.06) & (colors[:, 1] > colors[:, 2] + 0.03)
    low_noise = alpha > 0.08
    chairish = low_noise & (brightness < 0.62) & ~warm_floor

    masks = {
        "full": torch.ones_like(alpha, dtype=torch.bool),
        "clean": low_noise,
        "chairish": chairish,
    }
    modes = ["raw", "zup_to_yup", "zup_to_yup_flipz", "swap_xy_zup"]
    for mask_name, mask in masks.items():
        for mode in modes:
            export_one(args.out_dir / f"poster_{mask_name}_{mode}.ply", model, mode, mask)


if __name__ == "__main__":
    main()
