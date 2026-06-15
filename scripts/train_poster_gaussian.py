import argparse
import json
import math
import random
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gsplat import export_splats
from gsplat.rendering import rasterization


SH_C0 = 0.28209479177387814


def logit(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# COLMAP binary readers (for couch / CO3D→COLMAP datasets)
# ---------------------------------------------------------------------------

def read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


def read_colmap_cameras_binary(path: Path) -> dict:
    """Read COLMAP cameras.bin → dict[camera_id] = {model, width, height, params}."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(f, 24, "IiQQ")
            num_params = {0: 0, 1: 4, 2: 4, 3: 4, 4: 5, 5: 9}.get(model_id, 0)
            params = list(read_next_bytes(f, 8 * num_params, "d" * num_params))
            cameras[camera_id] = dict(model=model_id, width=width, height=height, params=params)
    return cameras


def read_colmap_images_binary(path: Path) -> dict:
    """Read COLMAP images.bin → dict[image_id] = {qvec, tvec, camera_id, name}."""
    images = {}
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "I")[0]
            qw, qx, qy, qz = read_next_bytes(f, 32, "dddd")
            tx, ty, tz = read_next_bytes(f, 24, "ddd")
            qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            camera_id = read_next_bytes(f, 4, "I")[0]
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\0":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8")
            num_points2d = read_next_bytes(f, 8, "Q")[0]
            f.seek(24 * num_points2d, 1)  # skip 2D points
            images[image_id] = dict(qvec=qvec, tvec=tvec, camera_id=camera_id, name=name)
    return images


def read_colmap_points3D_binary(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin → (points_xyz, colors_rgb_0_1)."""
    points_list = []
    colors_list = []
    with open(path, "rb") as f:
        num_points = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_points):
            pt_id = read_next_bytes(f, 8, "Q")[0]
            x, y, z = read_next_bytes(f, 24, "ddd")
            r, g, b = read_next_bytes(f, 3, "BBB")
            error = read_next_bytes(f, 8, "d")[0]
            track_len = read_next_bytes(f, 8, "Q")[0]
            f.seek(8 * track_len, 1)
            points_list.append([x, y, z])
            colors_list.append([r, g, b])

    points = np.asarray(points_list, dtype=np.float32)
    colors = np.asarray(colors_list, dtype=np.float32) / 255.0
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        ids = rng.choice(len(points), size=max_points, replace=False)
        points = points[ids]
        colors = colors[ids]
    return points, colors


def quat_wxyz_to_rotmat(qvec):
    """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def load_colmap_frames(data_dir: Path, factor: int, max_frames: int):
    """Load frames from COLMAP binary format.

    Returns: frames, K, radial, tangential (same signature as load_frames)
    frames: list of (image_path, c2w_matrix) where c2w is world-to-camera inverted
    """
    sparse_dir = data_dir / "sparse" / "0"
    cameras = read_colmap_cameras_binary(sparse_dir / "cameras.bin")
    images = read_colmap_images_binary(sparse_dir / "images.bin")
    image_dir = data_dir / f"images_{factor}"
    colmap_image_dir = data_dir / "images"

    frames = []
    camera_params = {}  # camera_id → (fx, fy, cx, cy, width, height)

    for cam_id, cam in cameras.items():
        if cam["model"] == 1:  # PINHOLE
            fx, fy, cx, cy = cam["params"]
            camera_params[cam_id] = (fx, fy, cx, cy, cam["width"], cam["height"])

    # Build mapping from COLMAP filenames → downsampled filenames
    # COLMAP images are .jpg, downsampled images_4 are .png
    colmap_files = sorted([f.name for f in colmap_image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    image_files = sorted([f.name for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    colmap_to_image = dict(zip(colmap_files, image_files))

    for img_id, img in sorted(images.items()):
        cam_id = img["camera_id"]
        if cam_id not in camera_params:
            continue
        colmap_name = img["name"]  # e.g., "frame000001.jpg"
        if colmap_name not in colmap_to_image:
            continue
        image_name = colmap_to_image[colmap_name]
        image_path = image_dir / image_name
        if not image_path.exists():
            continue

        # COLMAP w2c → c2w
        R_w2c = quat_wxyz_to_rotmat(img["qvec"])
        tvec = img["tvec"].astype(np.float32)
        R_c2w = R_w2c.T
        T_c2w = -R_c2w @ tvec
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = T_c2w
        frames.append((image_path, c2w))

    frames = sorted(frames, key=lambda x: x[0].name)
    if max_frames and len(frames) > max_frames:
        ids = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in ids]

    # Use the first camera's intrinsics (or most common) for the shared K
    if not camera_params:
        raise RuntimeError("No valid cameras found in COLMAP")
    cam0 = list(camera_params.values())[0]
    fx, fy, cx, cy, full_w, full_h = cam0
    fx = fx / factor
    fy = fy / factor
    cx = cx / factor
    cy = cy / factor
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    radial = torch.zeros(4)
    tangential = torch.zeros(2)
    return frames, K, radial, tangential


# ---------------------------------------------------------------------------
# ASCII PLY reader (original transforms.json path)
# ---------------------------------------------------------------------------

def read_ascii_pointcloud(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    rows = []
    rgbs = []
    for line in lines[end + 1 :]:
        if not line:
            continue
        vals = line.split()
        rows.append([float(vals[0]), float(vals[1]), float(vals[2])])
        rgbs.append([int(vals[3]), int(vals[4]), int(vals[5])])

    points = np.asarray(rows, dtype=np.float32)
    colors = np.asarray(rgbs, dtype=np.float32) / 255.0
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        ids = rng.choice(len(points), size=max_points, replace=False)
        points = points[ids]
        colors = colors[ids]
    return points, colors


# ---------------------------------------------------------------------------
# Scene normalization
# ---------------------------------------------------------------------------

def normalize_scene(
    points: np.ndarray, frames: list[tuple[Path, np.ndarray]]
) -> tuple[np.ndarray, list[tuple[Path, np.ndarray]], np.ndarray, float]:
    center = np.median(points, axis=0).astype(np.float32)
    centered = points - center[None, :]
    radius = np.percentile(np.linalg.norm(centered, axis=-1), 95).astype(np.float32)
    scale = float(max(radius, 1e-6))

    norm_points = centered / scale
    norm_frames = []
    for image_path, c2w in frames:
        c2w = c2w.copy()
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale
        norm_frames.append((image_path, c2w))
    return norm_points.astype(np.float32), norm_frames, center, scale


def build_splat_filter(
    means: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor | None = None,
    *,
    min_brightness: float = 0.0,
    min_alpha: float = 0.0,
    max_radius: float = 0.0,
) -> torch.Tensor:
    """Build a mask for removing dark, transparent, or far-away splats."""
    mask = torch.ones(means.shape[0], dtype=torch.bool, device=means.device)
    if min_brightness > 0.0:
        mask &= colors.mean(dim=-1) >= min_brightness
    if min_alpha > 0.0:
        if opacities is None:
            raise ValueError("min_alpha requires opacities")
        mask &= torch.sigmoid(opacities) >= min_alpha
    if max_radius > 0.0:
        center = means.median(dim=0).values
        radius = torch.linalg.norm(means - center, dim=-1)
        mask &= radius <= max_radius
    return mask


# ---------------------------------------------------------------------------
# Viewer export helpers
# ---------------------------------------------------------------------------

def viewer_aligned_export(
    means: torch.Tensor,
    quats: torch.Tensor,
    axis_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if axis_mode == "none":
        return means, quats
    if axis_mode == "supersplat":
        rot = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            device=means.device,
            dtype=means.dtype,
        )
        aligned = means @ rot.T
        aligned = aligned - aligned.median(dim=0).values
        q_align = torch.tensor(
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
            device=means.device,
            dtype=means.dtype,
        )
        q = quat_mul_xyzw(q_align.expand_as(quats), quats)
        return aligned, torch.nn.functional.normalize(q, dim=-1)
    raise ValueError(f"Unknown axis mode: {axis_mode}")


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Original transforms.json frame loader (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_frames(data_dir: Path, factor: int, max_frames: int):
    meta = json.loads((data_dir / "transforms.json").read_text())
    image_dir = data_dir / f"images_{factor}"
    frames = []
    for frame in meta["frames"]:
        image_name = Path(frame["file_path"]).name
        image_path = image_dir / image_name
        if image_path.exists():
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            frames.append((image_path, c2w))
    frames = sorted(frames, key=lambda x: x[0].name)
    if max_frames and len(frames) > max_frames:
        ids = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in ids]

    fl_x = float(meta["fl_x"]) / factor
    fl_y = float(meta["fl_y"]) / factor
    cx = float(meta["cx"]) / factor
    cy = float(meta["cy"]) / factor
    K = torch.tensor([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]])
    radial = torch.tensor(
        [
            float(meta.get("k1", 0.0)),
            float(meta.get("k2", 0.0)),
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )
    tangential = torch.tensor(
        [float(meta.get("p1", 0.0)), float(meta.get("p2", 0.0))],
        dtype=torch.float32,
    )
    return frames, K, radial, tangential


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def image_to_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def load_batch(
    frames: list[tuple[Path, np.ndarray]],
    width: int,
    height: int,
    batch_size: int,
    device: torch.device,
):
    batch = random.sample(frames, k=min(batch_size, len(frames)))
    images = []
    viewmats = []
    names = []
    for image_path, c2w_np in batch:
        images.append(image_to_tensor(image_path, (width, height)))
        viewmats.append(torch.linalg.inv(torch.from_numpy(c2w_np)))
        names.append(image_path.name)
    return (
        torch.stack(images, dim=0).to(device),
        torch.stack(viewmats, dim=0).to(device),
        names,
    )


def detect_format(data_dir: Path) -> str:
    """Auto-detect dataset format: 'colmap' or 'transforms'."""
    if (data_dir / "sparse" / "0" / "cameras.bin").exists():
        return "colmap"
    if (data_dir / "transforms.json").exists():
        return "transforms"
    raise RuntimeError(
        f"Cannot detect dataset format in {data_dir}. "
        "Expected sparse/0/cameras.bin (COLMAP) or transforms.json."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/nerfstudio_poster_hf/poster"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/poster_gaussian"))
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--point-repeats", type=int, default=1)
    parser.add_argument("--repeat-jitter", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=135)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--init-opacity", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--axis-mode", choices=["supersplat", "none"], default="supersplat")
    parser.add_argument("--use-distortion", action="store_true")
    parser.add_argument(
        "--init-min-brightness",
        type=float,
        default=0.0,
        help="Drop initial SfM points whose RGB mean is below this value in [0, 1].",
    )
    parser.add_argument(
        "--export-min-brightness",
        type=float,
        default=0.0,
        help="Drop exported splats whose RGB mean is below this value in [0, 1].",
    )
    parser.add_argument(
        "--export-min-alpha",
        type=float,
        default=0.0,
        help="Drop exported splats whose sigmoid opacity is below this value.",
    )
    parser.add_argument(
        "--export-max-radius",
        type=float,
        default=0.0,
        help="Drop exported splats farther than this normalized radius from the median center.",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda")
    args.result_dir.mkdir(parents=True, exist_ok=True)

    fmt = detect_format(args.data_dir)
    print(f"Detected format: {fmt}")

    if fmt == "colmap":
        sparse_dir = args.data_dir / "sparse" / "0"
        points, point_colors = read_colmap_points3D_binary(
            sparse_dir / "points3D.bin", args.max_points
        )
        frames, K_cpu, radial_cpu, tangential_cpu = load_colmap_frames(
            args.data_dir, args.factor, args.max_frames
        )
    else:
        points, point_colors = read_ascii_pointcloud(
            args.data_dir / "sparse_pc.ply", args.max_points
        )
        frames, K_cpu, radial_cpu, tangential_cpu = load_frames(
            args.data_dir, args.factor, args.max_frames
        )

    if not frames:
        raise RuntimeError(f"No images found under {args.data_dir / f'images_{args.factor}'}")

    if args.init_min_brightness > 0.0:
        init_mask = build_splat_filter(
            torch.from_numpy(points),
            torch.from_numpy(point_colors),
            min_brightness=args.init_min_brightness,
        ).numpy()
        before = len(points)
        points = points[init_mask]
        point_colors = point_colors[init_mask]
        if len(points) == 0:
            raise RuntimeError("--init-min-brightness removed all initial points")
        print(
            f"init filter kept {len(points)}/{before} points "
            f"(min_brightness={args.init_min_brightness})"
        )

    points, frames, scene_center, scene_scale = normalize_scene(points, frames)
    if args.point_repeats > 1:
        rng = np.random.default_rng(123)
        base_points = points
        base_colors = point_colors
        point_chunks = [base_points]
        color_chunks = [base_colors]
        for _ in range(args.point_repeats - 1):
            jitter = rng.normal(0.0, args.repeat_jitter, size=base_points.shape).astype(np.float32)
            point_chunks.append(base_points + jitter)
            color_chunks.append(base_colors)
        points = np.concatenate(point_chunks, axis=0)
        point_colors = np.concatenate(color_chunks, axis=0)

    # Keep the native aspect ratio from images_* unless explicitly overridden.
    sample = Image.open(frames[0][0])
    native_w, native_h = sample.size
    width = args.width or native_w
    height = args.height or native_h
    scale_x = width / native_w
    scale_y = height / native_h
    K_cpu = K_cpu.clone()
    K_cpu[0, :] *= scale_x
    K_cpu[1, :] *= scale_y
    K = K_cpu.to(device)
    radial = radial_cpu[None].to(device)
    tangential = tangential_cpu[None].to(device)
    bg_value = 1.0 if args.background == "white" else 0.0
    background = torch.full((1, 3), bg_value, device=device)

    means = torch.nn.Parameter(torch.from_numpy(points).to(device))
    colors = torch.nn.Parameter(logit(torch.from_numpy(point_colors).to(device)))
    opacities = torch.nn.Parameter(
        torch.full((len(points),), float(logit(torch.tensor(args.init_opacity))), device=device)
    )
    log_scales = torch.nn.Parameter(
        torch.full((len(points), 3), math.log(args.init_scale), device=device)
    )
    quats = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(points), 1)
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": args.lr * 0.08},
            {"params": [colors], "lr": args.lr},
            {"params": [opacities], "lr": args.lr},
            {"params": [log_scales], "lr": args.lr * 0.5},
            {"params": [quats], "lr": args.lr * 0.2},
        ]
    )

    print(f"frames={len(frames)} points={len(points)} size={width}x{height}")
    last_loss = None
    for step in range(1, args.steps + 1):
        gt, w2c, image_names = load_batch(frames, width, height, args.batch_size, device)
        scales = torch.exp(log_scales).clamp(1e-4, 1.0)
        rgbs = torch.sigmoid(colors)
        raster_kwargs = {"backgrounds": background.expand(w2c.shape[0], 3)}
        if args.use_distortion:
            raster_kwargs.update(
                radial_coeffs=radial.expand(w2c.shape[0], -1),
                tangential_coeffs=tangential.expand(w2c.shape[0], -1),
                with_ut=True,
            )
        renders, alphas, _ = rasterization(
            means,
            torch.nn.functional.normalize(quats, dim=-1),
            scales,
            torch.sigmoid(opacities),
            rgbs,
            w2c,
            K[None].expand(w2c.shape[0], -1, -1),
            width,
            height,
            packed=False,
            **raster_kwargs,
        )
        pred = renders
        photometric = torch.mean(torch.abs(pred - gt))
        coverage = torch.mean((1.0 - alphas) ** 2)
        loss = photometric + 0.005 * coverage
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step == 1 or step % 50 == 0:
            print(f"step {step:04d}/{args.steps} loss={last_loss:.6f} images={','.join(image_names[:3])}")

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = args.result_dir / f"poster_gaussian_step{step}.pt"
            ply_path = args.result_dir / f"poster_gaussian_step{step}.ply"
            with torch.no_grad():
                model = {
                    "means": means.detach().cpu(),
                    "scales": torch.exp(log_scales.detach()).cpu(),
                    "log_scales": log_scales.detach().cpu(),
                    "quats": torch.nn.functional.normalize(quats.detach(), dim=-1).cpu(),
                    "opacities": opacities.detach().cpu(),
                    "colors": torch.sigmoid(colors.detach()).cpu(),
                    "step": step,
                    "loss": last_loss,
                    "width": width,
                    "height": height,
                    "K": K.detach().cpu(),
                    "radial": radial.detach().cpu(),
                    "tangential": tangential.detach().cpu(),
                    "scene_center": torch.from_numpy(scene_center),
                    "scene_scale": scene_scale,
                    "axis_mode": args.axis_mode,
                    "frames_used": [str(p) for p, _ in frames],
                }
                torch.save(model, ckpt_path)
                preview = (pred[0].detach().clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                gt_preview = (gt[0].detach().clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(preview).save(args.result_dir / f"render_step{step}.png")
                Image.fromarray(gt_preview).save(args.result_dir / f"target_step{step}.png")
                sh0 = (torch.sigmoid(colors.detach()).unsqueeze(1) - 0.5) / SH_C0
                shN = torch.empty((means.shape[0], 0, 3), device=device)
                export_means, export_quats = viewer_aligned_export(
                    means.detach(),
                    torch.nn.functional.normalize(quats.detach(), dim=-1),
                    args.axis_mode,
                )
                export_colors = torch.sigmoid(colors.detach())
                export_opacities = opacities.detach()
                export_log_scales = log_scales.detach()
                export_mask = build_splat_filter(
                    export_means,
                    export_colors,
                    export_opacities,
                    min_brightness=args.export_min_brightness,
                    min_alpha=args.export_min_alpha,
                    max_radius=args.export_max_radius,
                )
                if not torch.any(export_mask):
                    raise RuntimeError("export filters removed all splats")
                if not torch.all(export_mask):
                    print(
                        f"export filter kept {int(export_mask.sum())}/{len(export_mask)} splats"
                    )
                export_splats(
                    means=export_means[export_mask],
                    scales=export_log_scales[export_mask],
                    quats=export_quats[export_mask],
                    opacities=export_opacities[export_mask],
                    sh0=sh0[export_mask],
                    shN=shN[export_mask],
                    format="ply",
                    save_to=str(ply_path),
                )
                print(f"saved {ckpt_path}")
                print(f"saved {ply_path}")


if __name__ == "__main__":
    main()
