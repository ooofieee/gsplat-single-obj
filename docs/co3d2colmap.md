# CO3D couch 标准化为 COLMAP

## 结论

主训练器 `examples.simple_trainer` 的标准输入不是 CO3D 原始目录，也不是 NerfStudio `transforms.json`，而是 COLMAP 模型：

```text
scene/
  images/
  images_4/
  masks/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

当前 couch 的标准数据集是：

```text
/home/ooofieee/co3d_data/couch_colmap_sfm
```

不要再优先使用 `/home/ooofieee/co3d_data/couch_colmap`。那份数据虽然能被 `pycolmap` 读取，但点云是 mask-guided synthetic initialization，不是真正的 COLMAP/SfM sparse reconstruction。

## 已确认的问题

CO3D couch 源目录：

```text
/home/ooofieee/co3d_data/couch/617_99945_199053
```

源数据是标准 CO3D 结构：

- `images/`: 202 张
- `masks/`: 202 张
- `depths/`: 202 张，但本地 depth 对训练初始化不可直接当 SfM 点云使用
- `frame_annotations.jgz`: 202 条
- `sequence_annotations.jgz`: 1 条，`point_cloud=None`
- `set_lists_manyview_test_0.json`: `train=102`, `val=100`, `test=100`

CO3D 标注里的相机不是 COLMAP 格式：

```text
ViewpointAnnotation:
  R, T              # PyTorch3D right-multiply W2C: X_cam = X_world @ R + T
  focal_length
  principal_point
  intrinsics_format # couch 为 ndc_isotropic
```

关键数学约定：

```text
S = diag(-1, -1, 1)
R_colmap = S @ R_co3d.T
t_colmap = S @ T_co3d
camera_center_world = -R_co3d @ T_co3d
```

CO3D/PyTorch3D NDC principal point 到 COLMAP pixel principal point 需要符号翻转：

```text
ndc_isotropic:
  scale = min(width, height) / 2
  fx = focal_x * scale
  fy = focal_y * scale
  cx = width  / 2 - pp_x * scale
  cy = height / 2 - pp_y * scale
```

## 重建标准数据集

脚本：

```text
scripts/standardize_co3d_colmap.py
```

从 CO3D 官方结构重建 couch 标准 COLMAP/SfM 数据集：

```bash
conda activate gsplat
cd /home/ooofieee/gsplat

PYTHONPATH=.:examples:/home/ooofieee/co3d \
python scripts/standardize_co3d_colmap.py build \
  --co3d-root /home/ooofieee/co3d_data \
  --category couch \
  --sequence 617_99945_199053 \
  --subset-name manyview_test_0 \
  --output-dir /home/ooofieee/co3d_data/couch_colmap_sfm_new \
  --data-factor 4 \
  --run-colmap
```

脚本会：

1. 读取 `frame_annotations.jgz`
2. 只取 official set list 的 train split
3. 过滤黑帧和空 mask
4. 用 mask 把前景合成到黑背景，写入 `images/`
5. 同步写入 `masks/`
6. 生成 `images_4/`
7. 用 COLMAP `feature_extractor + exhaustive_matcher + mapper` 重建 `sparse/0`
8. 写 `standardization_report.json`

## 验证命令

```bash
conda activate gsplat
cd /home/ooofieee/gsplat

PYTHONPATH=.:examples python scripts/standardize_co3d_colmap.py validate \
  --scene-dir /home/ooofieee/co3d_data/couch_colmap_sfm \
  --data-factor 4
```

当前结果：

```text
cameras: 1
images: 102
points3D: 17735
images_factor: 102
```

COLMAP analyzer：

```text
Cameras: 1
Registered images: 102 / 102
Points: 17735
Observations: 111225
Mean track length: 6.27
Mean reprojection error: 1.02 px
```

gsplat Parser 验证：

```text
[Parser] 102 images, taken by 1 cameras.
points: (17735, 3)
train: 89
val: 13
sample image shape: (268, 477, 3)
```

## 训练命令

```bash
conda activate gsplat
cd /home/ooofieee/gsplat
export PYTHONPATH=.:examples

python -m examples.simple_trainer default \
  --data_dir /home/ooofieee/co3d_data/couch_colmap_sfm \
  --data_factor 4 \
  --result_dir /home/ooofieee/gsplat/results/couch_sfm \
  --max_steps 30000 \
  --save_ply True \
  --disable_viewer
```
