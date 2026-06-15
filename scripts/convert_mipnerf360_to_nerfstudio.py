import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * z * x - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, dict]:
    camera_models = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
    }
    cameras = {}
    with path.open("rb") as fid:
        (num_cameras,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = camera_models[model_id]
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path: Path) -> dict[int, dict]:
    images = {}
    with path.open("rb") as fid:
        (num_images,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"))
            tvec = np.array(read_next_bytes(fid, 24, "ddd"))
            camera_id = read_next_bytes(fid, 4, "i")[0]
            name = b""
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name += char
            (num_points2d,) = read_next_bytes(fid, 8, "Q")
            fid.seek(num_points2d * 24, 1)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
            }
    return images


def read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyzs = []
    rgbs = []
    with path.open("rb") as fid:
        (num_points,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_points):
            fid.seek(8, 1)
            xyzs.append(read_next_bytes(fid, 24, "ddd"))
            rgbs.append(read_next_bytes(fid, 3, "BBB"))
            fid.seek(8, 1)
            (track_len,) = read_next_bytes(fid, 8, "Q")
            fid.seek(track_len * 8, 1)
    return np.asarray(xyzs, dtype=np.float32), np.asarray(rgbs, dtype=np.uint8)


def read_cameras_text(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": np.asarray([float(v) for v in parts[4:]], dtype=np.float64),
        }
    return cameras


def read_images_text(path: Path) -> dict[int, dict]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    images = {}
    for line in lines[0::2]:
        parts = line.split()
        image_id = int(parts[0])
        images[image_id] = {
            "qvec": np.asarray([float(v) for v in parts[1:5]], dtype=np.float64),
            "tvec": np.asarray([float(v) for v in parts[5:8]], dtype=np.float64),
            "camera_id": int(parts[8]),
            "name": parts[9],
        }
    return images


def read_points3d_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyzs = []
    rgbs = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyzs.append([float(v) for v in parts[1:4]])
        rgbs.append([int(v) for v in parts[4:7]])
    return np.asarray(xyzs, dtype=np.float32), np.asarray(rgbs, dtype=np.uint8)


def load_colmap_model(sparse_dir: Path):
    if (sparse_dir / "cameras.bin").exists():
        cameras = read_cameras_binary(sparse_dir / "cameras.bin")
        images = read_images_binary(sparse_dir / "images.bin")
        points, colors = read_points3d_binary(sparse_dir / "points3D.bin")
    else:
        cameras = read_cameras_text(sparse_dir / "cameras.txt")
        images = read_images_text(sparse_dir / "images.txt")
        points, colors = read_points3d_text(sparse_dir / "points3D.txt")
    return cameras, images, points, colors


def intrinsics(camera: dict):
    model = camera["model"]
    params = camera["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    raise ValueError(f"Unsupported COLMAP camera model: {model}")


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def convert_images(src_dir: Path, dst_dir: Path, factor: int | None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for path in paths:
        out = dst_dir / (path.stem + ".png")
        if out.exists():
            continue
        img = Image.open(path).convert("RGB")
        if factor is not None:
            w, h = img.size
            img = img.resize((max(1, w // factor), max(1, h // factor)), Image.LANCZOS)
        img.save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=8)
    args = parser.parse_args()

    sparse_dir = args.scene_dir / "sparse" / "0"
    image_dir = args.scene_dir / "images"
    existing_factor_dir = args.scene_dir / f"images_{args.factor}"
    if not sparse_dir.exists():
        raise RuntimeError(f"Missing COLMAP sparse model: {sparse_dir}")
    if not image_dir.exists() and not existing_factor_dir.exists():
        raise RuntimeError(f"Missing image directory: {image_dir} or {existing_factor_dir}")

    cameras, images, points, colors = load_colmap_model(sparse_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if existing_factor_dir.exists():
        convert_images(existing_factor_dir, args.out_dir / f"images_{args.factor}", None)
    else:
        convert_images(image_dir, args.out_dir / f"images_{args.factor}", args.factor)
    write_ascii_ply(args.out_dir / "sparse_pc.ply", points, colors)

    first_cam = cameras[next(iter(cameras))]
    fl_x, fl_y, cx, cy = intrinsics(first_cam)
    frames = []
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    for image in sorted(images.values(), key=lambda item: item["name"]):
        R = qvec_to_rotmat(image["qvec"])
        t = image["tvec"]
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w_opencv = np.linalg.inv(w2c)
        c2w_opengl = c2w_opencv @ opencv_to_opengl
        out_name = Path(image["name"]).stem + ".png"
        frames.append(
            {
                "file_path": f"./images_{args.factor}/{out_name}",
                "transform_matrix": c2w_opengl.tolist(),
            }
        )

    meta = {
        "camera_model": "OPENCV",
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(first_cam["width"]),
        "h": int(first_cam["height"]),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }
    (args.out_dir / "transforms.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir}")
    print(f"frames={len(frames)} points={len(points)} factor={args.factor}")


if __name__ == "__main__":
    main()
