import argparse
import random
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


def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        return torch.stack(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    diag = torch.diag(R)
    i = int(torch.argmax(diag).item())
    if i == 0:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = torch.stack(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )
    elif i == 1:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = torch.stack(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = torch.stack(
            [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]
        )
    return torch.nn.functional.normalize(q, dim=0)


def fit_floor_plane(means: torch.Tensor, colors: torch.Tensor, alpha: torch.Tensor):
    warm = (
        (colors[:, 0] > colors[:, 1] + 0.04)
        & (colors[:, 1] > colors[:, 2] + 0.02)
        & (alpha > 0.1)
        & (colors.mean(dim=1) > 0.22)
    )
    pts = means[warm]
    random.seed(0)
    torch.manual_seed(0)
    best = None
    threshold = 0.025
    for _ in range(4000):
        idx = torch.randint(0, len(pts), (3,))
        p = pts[idx]
        n = torch.cross(p[1] - p[0], p[2] - p[0], dim=0)
        norm = torch.linalg.norm(n)
        if norm < 1e-5:
            continue
        n = n / norm
        d = -(n @ p[0])
        inliers = torch.abs(pts @ n + d) < threshold
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, inliers)

    _, inliers = best
    plane_pts = pts[inliers]
    point = plane_pts.mean(dim=0)
    X = plane_pts - point
    cov = X.T @ X / X.shape[0]
    _, evecs = torch.linalg.eigh(cov)
    normal = evecs[:, 0]
    signed = (means - point) @ normal
    if torch.quantile(signed, 0.8) < 0:
        normal = -normal
    return point, normal, plane_pts


def floor_xy_transform(means: torch.Tensor, quats: torch.Tensor, point: torch.Tensor, normal: torch.Tensor):
    z_axis = torch.nn.functional.normalize(normal, dim=0)
    ref = torch.tensor([0.0, 1.0, 0.0], dtype=means.dtype)
    if torch.abs(torch.dot(ref, z_axis)) > 0.95:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=means.dtype)
    x_axis = ref - torch.dot(ref, z_axis) * z_axis
    x_axis = torch.nn.functional.normalize(x_axis, dim=0)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = torch.nn.functional.normalize(y_axis, dim=0)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    out_means = (means - point) @ R.T
    out_means[:, 2] -= torch.median(out_means[:, 2])
    radius = torch.quantile(torch.linalg.norm(out_means[:, :2], dim=-1), 0.95).clamp_min(1e-6)
    out_means = out_means / radius
    q_align = rotmat_to_quat_wxyz(R)
    out_quats = quat_mul_wxyz(q_align.expand_as(quats), quats)
    out_quats = torch.nn.functional.normalize(out_quats, dim=-1)
    return out_means, out_quats, R, radius


def export_variant(out_path: Path, model: dict, mask: torch.Tensor, point: torch.Tensor, normal: torch.Tensor):
    means = model["means"][mask]
    quats = model["quats"][mask]
    means_out, quats_out, _, _ = floor_xy_transform(means, quats, point, normal)
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
    parser.add_argument("--out-dir", type=Path, default=Path("results/poster_gaussian_floor_xy"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(args.ckpt, map_location="cpu")
    alpha = torch.sigmoid(model["opacities"])
    colors = model["colors"]
    point, normal, plane_pts = fit_floor_plane(model["means"], colors, alpha)
    means_floor, _, R, radius = floor_xy_transform(model["means"], model["quats"], point, normal)
    floor_z = ((plane_pts - point) @ R[2]) / radius
    print("floor point", point.tolist())
    print("floor normal", normal.tolist())
    print("export rotation rows", R.tolist())
    print("floor z quantiles", torch.quantile(floor_z, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])).tolist())
    print("all z quantiles", torch.quantile(means_floor[:, 2], torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])).tolist())

    visible = alpha > 0.08
    warm_floor = (colors[:, 0] > colors[:, 1] + 0.06) & (colors[:, 1] > colors[:, 2] + 0.03)
    chairish = visible & (colors.mean(dim=1) < 0.62) & ~warm_floor
    export_variant(args.out_dir / "poster_full_floor_xy.ply", model, torch.ones_like(alpha, dtype=torch.bool), point, normal)
    export_variant(args.out_dir / "poster_clean_floor_xy.ply", model, visible, point, normal)
    export_variant(args.out_dir / "poster_chairish_floor_xy.ply", model, chairish, point, normal)


if __name__ == "__main__":
    main()
