import argparse
from pathlib import Path

import torch

from export_floor_xy import fit_floor_plane, quat_mul_wxyz, rotmat_to_quat_wxyz
from gsplat import export_splats


SH_C0 = 0.28209479177387814


def floor_xz_transform(
    means: torch.Tensor,
    quats: torch.Tensor,
    point: torch.Tensor,
    normal: torch.Tensor,
    flip_y: bool = False,
):
    y_axis = torch.nn.functional.normalize(normal, dim=0)
    if flip_y:
        y_axis = -y_axis
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=means.dtype)
    if torch.abs(torch.dot(ref, y_axis)) > 0.95:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=means.dtype)
    x_axis = ref - torch.dot(ref, y_axis) * y_axis
    x_axis = torch.nn.functional.normalize(x_axis, dim=0)
    z_axis = torch.cross(x_axis, y_axis, dim=0)
    z_axis = torch.nn.functional.normalize(z_axis, dim=0)

    # Rows map old coordinates into SuperSplat-style output coordinates:
    # X horizontal, Y up, Z horizontal.
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    out_means = (means - point) @ R.T
    out_means[:, 1] -= torch.median(out_means[:, 1])
    radius = torch.quantile(torch.linalg.norm(out_means[:, [0, 2]], dim=-1), 0.95).clamp_min(1e-6)
    out_means = out_means / radius

    q_align = rotmat_to_quat_wxyz(R)
    out_quats = quat_mul_wxyz(q_align.expand_as(quats), quats)
    out_quats = torch.nn.functional.normalize(out_quats, dim=-1)
    return out_means, out_quats, R, radius


def export_variant(
    out_path: Path,
    model: dict,
    mask: torch.Tensor,
    point: torch.Tensor,
    normal: torch.Tensor,
    flip_y: bool = False,
):
    means = model["means"][mask]
    quats = model["quats"][mask]
    means_out, quats_out, _, _ = floor_xz_transform(
        means, quats, point, normal, flip_y=flip_y
    )
    colors = model["colors"][mask].cuda()
    sh0 = (colors.unsqueeze(1) - 0.5) / SH_C0
    shN = torch.empty((means_out.shape[0], 0, 3), device="cuda")
    export_splats(
        means=means_out.cuda(),
        scales=model["log_scales"][mask].cuda(),
        quats=quats_out.cuda(),
        opacities=model["opacities"][mask].cuda(),
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(out_path),
    )
    print(f"saved {out_path} ({int(mask.sum())} splats)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("results/poster_gaussian_v2/poster_gaussian_step2500.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/poster_gaussian_floor_xz"))
    parser.add_argument("--flip-y", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(args.ckpt, map_location="cpu")
    alpha = torch.sigmoid(model["opacities"])
    colors = model["colors"]
    point, normal, plane_pts = fit_floor_plane(model["means"], colors, alpha)
    means_floor, _, R, radius = floor_xz_transform(
        model["means"], model["quats"], point, normal, flip_y=args.flip_y
    )
    floor_y = ((plane_pts - point) @ R[1]) / radius
    print("floor point", point.tolist())
    print("floor normal", normal.tolist())
    print("export rotation rows", R.tolist())
    print("floor y quantiles", torch.quantile(floor_y, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])).tolist())
    print("all y quantiles", torch.quantile(means_floor[:, 1], torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])).tolist())

    visible = alpha > 0.08
    warm_floor = (colors[:, 0] > colors[:, 1] + 0.06) & (colors[:, 1] > colors[:, 2] + 0.03)
    chairish = visible & (colors.mean(dim=1) < 0.62) & ~warm_floor
    suffix = "floor_xz_yup_flipped" if args.flip_y else "floor_xz_yup"
    export_variant(
        args.out_dir / f"poster_full_{suffix}.ply",
        model,
        torch.ones_like(alpha, dtype=torch.bool),
        point,
        normal,
        flip_y=args.flip_y,
    )
    export_variant(
        args.out_dir / f"poster_clean_{suffix}.ply",
        model,
        visible,
        point,
        normal,
        flip_y=args.flip_y,
    )
    export_variant(
        args.out_dir / f"poster_chairish_{suffix}.ply",
        model,
        chairish,
        point,
        normal,
        flip_y=args.flip_y,
    )


if __name__ == "__main__":
    main()
