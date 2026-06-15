# 3D Gaussian Splatting 训练流程详解

> 基于 `examples/simple_trainer.py`，以 COLMAP 数据格式（默认）为例。

---

## 1. 概览

训练流程分为以下几个阶段：

```
数据加载 → 高斯初始化 → 训练循环 → 评估 & 渲染
              ↓
         (Densification / 后处理)
```

---

## 2. 数据加载 (Parser & Dataset)

### 2.1 输入目录结构（COLMAP 格式）

```
data_dir/
├── sparse/0/              # COLMAP SfM 稀疏重建输出
│   ├── cameras.bin        # 相机内参
│   ├── images.bin         # 每张图的位姿 (world-to-camera)
│   └── points3D.bin       # 3D 稀疏点云 + RGB 颜色
├── images/                # 原始分辨率图片
└── images_N/              # (可选) N 倍下采样图片
```

### 2.2 Parser 解析过程 (`examples/datasets/colmap.py`)

1. **读取 COLMAP 稀疏数据**: 使用 `pycolmap.SceneManager` 加载 `sparse/0/` 下的 `cameras.bin`, `images.bin`, `points3D.bin`
2. **提取相机参数**:
   - 每张图的 `camtoworld` 矩阵 (4×4)
   - 每个相机的内参矩阵 `K` (3×3)，包含 `fx, fy, cx, cy`
   - 支持多种 COLMAP 相机模型: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE`
3. **去畸变**: 如果有畸变参数，用 OpenCV 计算 undistort map，更新 K 矩阵
4. **提取 3D 点云**: `points` (N×3) 和 `points_rgb` (N×3, uint8)，后续作为高斯初始值
5. **世界空间归一化** (`normalize_world_space=True`):
   - 相似变换使相机围绕原点
   - PCA 对齐主轴
   - 如果大部分点在上方则上下翻转（z 轴翻转）
6. **缩放因子**: 如果 `data_factor > 1`，图片和 K 矩阵按比例缩放
7. **数据集分割**: 每 `test_every` 张图取 1 张作为验证集，其余为训练集
8. **可选的 EXIF 加载**: 从原始图片读取曝光值

### 2.3 Dataset 每个 batch 返回

```python
{
    "K":           Tensor[1, 3, 3],    # 相机内参矩阵
    "camtoworld":  Tensor[1, 4, 4],    # 相机到世界变换
    "image":       Tensor[1, H, W, 3], # RGB 图片 (uint8, 0-255)
    "image_id":    Tensor[1],          # 图片在数据集中的索引
    "camera_idx":  Tensor[1],          # 0-based 连续相机索引
    "mask":        Tensor[1, H, W],    # (可选) 鱼眼 ROI mask
    "exposure":    Tensor[1],          # (可选) EXIF 曝光值 (去均值)
    "points":      Tensor[1, M, 2],    # (可选) 投影点坐标 (depth loss 用)
    "depths":      Tensor[1, M],       # (可选) 投影点深度
}
```

---

## 3. 高斯初始化 (`create_splats_with_optimizers`)

### 3.1 初始化方式

| `init_type` | 位置来源 | 颜色来源 |
|-------------|---------|---------|
| `"sfm"` (默认) | COLMAP 稀疏点云 | COLMAP 点云 RGB |
| `"lidar"` | LiDAR/雷达点云 | 点云颜色或灰色 |
| `"random"` | 在 `init_extent × scene_scale` 范围内随机生成 `init_num_pts` 个点 | 随机 RGB |

### 3.2 参数初始化

1. **means** (位置): 直接使用点云坐标，DDP 时按 rank 均匀分配
2. **scales** (尺度): 用 KNN (k=4) 计算每个点到最近 3 个邻居的平均距离作为初始尺度，取 log 后存入
3. **quats** (旋转): 随机初始化 (N×4)
4. **opacities** (不透明度): 初始化为 `logit(init_opa)`，默认 `init_opa=0.1`
5. **sh0** (球谐系数第 0 阶/基色): 用 RGB 转换到 SH 空间
6. **shN** (球谐高阶系数): 初始化为 0

### 3.3 优化器

- 默认使用 **Adam**（支持 `SparseAdam` 和 `SelectiveAdam` 变体）
- 学习率按 `sqrt(batch_size × world_size)` 缩放
- betas 根据 batch size 调整

各参数的学习率（默认值）:

| 参数 | 学习率 | 说明 |
|------|--------|------|
| means | 1.6e-4 × scene_scale | 有 ExponentialLR schedule，最终衰减到 0.01× |
| scales | 5e-3 | |
| quats | 1e-3 | |
| opacities | 5e-2 | |
| sh0 | 2.5e-3 | |
| shN | 2.5e-3 / 20 | 高阶 SH 学得更慢 |

---

## 4. 训练循环 (`Runner.train()`)

### 4.1 主循环结构

```
for step in range(max_steps):
    1. 获取一个 batch 数据
    2. 可选: 相机位姿优化 / 加噪
    3. SH 阶数调度
    4. 前向渲染 (rasterization)
    5. 计算 loss
    6. 反向传播
    7. 优化器 step + scheduler step
    8. Densification / Pruning (strategy)
    9. 定期评估 & 保存
```

### 4.2 前向渲染 (Rasterization)

调用 `gsplat.rendering.rasterization()`:

```
输入:
  means[N, 3]      高斯中心位置
  quats[N, 4]      旋转四元数（内部自动归一化）
  scales[N, 3]     exp 后的尺度
  opacities[N,]     sigmoid 后的不透明度
  colors[N, K, 3]  SH 系数拼接 (sh0 + shN)
  viewmats[C, 4, 4]  world-to-camera 矩阵
  Ks[C, 3, 3]       相机内参

输出:
  render_colors[C, H, W, 3]  渲染的 RGB
  render_alphas[C, H, W, 1]  渲染的 alpha
  info (dict):
    - "means2d": 2D 投影均值 (用于 densification 梯度)
    - "radii": 高斯在屏幕上的半径
    - "gaussian_ids": 可见高斯的 ID (packed 模式)
```

关键参数:
- `camera_model`: 默认 `"pinhole"`，NCore 数据可切换为 `"ftheta"` 等
- `rasterize_mode`: `"classic"` 或 `"antialiased"`
- `packed`: 减少内存但略慢
- `absgrad`: 是否用绝对梯度 (配合 `DefaultStrategy(absgrad=True)`)

### 4.3 可选模块

#### 外观优化 (Appearance Optimization, `--app_opt`)
- 将 `sh0/shN` 替换为 `features[N, feature_dim] + colors[N, 3]`
- 通过 MLP 将 feature + 视角方向 + embedding 映射为 RGB
- 目的: 补偿不同图片间的曝光/白平衡差异

#### 相机位姿优化 (Pose Optimization, `--pose_opt`)
- 对每张图的 camtoworld 施加可学习的微小修正
- 适用于 COLMAP 位姿不够精确的场景

#### 后处理 (Post-processing)
- **Bilateral Grid**: 学习一个 3D 网格对渲染结果做仿射颜色校正
- **PPISP**: 学习 per-frame 像素级 ISP 校正

### 4.4 Loss 计算

```python
L1_loss   = |render - gt|₁           # 主 loss
SSIM_loss = 1 - SSIM(render, gt)     # 结构相似度
loss = (1 - ssim_lambda) * L1 + ssim_lambda * SSIM  # ssim_lambda 默认 0.2
```

可选 loss:
| Loss | 权重 | 说明 |
|------|------|------|
| Depth L1 | `depth_lambda` (1e-2) | 在视差空间计算，将 SfM 点投影监督 |
| Opacity Reg | `opacity_reg` | `-log(1 - |opacity|)` 正则化 |
| Scale Reg | `scale_reg` | `-log(1 - |scale|)` 正则化 |

若 `random_bkgd=True`: 渲染颜色与随机背景按 alpha 混合后再算 loss，防止透明高斯。

### 4.5 SH 阶数调度

```python
sh_degree_to_use = min(step // sh_degree_interval, sh_degree)
```

- 默认每 1000 步提升一阶，从 0 阶到 3 阶
- 低阶 SH 先收敛，高阶逐步加入，稳定训练

### 4.6 学习率调度

- **means**: ExponentialLR，gamma = 0.01^(1/max_steps)，即最终衰减到初始的 1%
- **pose_opt**: 同上
- **bilateral_grid**: Linear warmup (1000步) + Exponential decay

---

## 5. Densification 策略

### 5.1 DefaultStrategy (原始 3DGS 论文)

```
每 refine_every (100) 步执行一次，在 refine_start_iter (500) 到 refine_stop_iter (15000) 之间:
```

#### 步骤:

1. **累积梯度**: 每步将 2D 投影均值 (`means2d`) 的梯度范数累加到 `grad2d` 状态中，记录可见次数 `count`
2. **Grow (增长)**:
   - `grad2d / count > grow_grad2d (0.0002)` 的高斯被选中
   - 小尺度 (≤ grow_scale3d × scene_scale) → **Duplicate** (复制)
   - 大尺度 (> grow_scale3d × scene_scale) → **Split** (分裂成 2 个)
3. **Prune (剪枝)**:
   - 不透明度 < prune_opa (0.005) → 移除
   - 尺度 > prune_scale3d (0.1) × scene_scale → 移除 (防止浮游物)
4. **Opacity Reset**: 每 `reset_every` (3000) 步，将所有不透明度重置为 `prune_opa × 2`
5. 重置累积状态 `grad2d` 和 `count`

若 `absgrad=True` (AbsGS 论文): 使用绝对梯度替代平均梯度，通常结果更好。

### 5.2 MCMCStrategy (MCMC 论文)

```
每 refine_every 步，从 refine_start_iter 到 refine_stop_iter:
```

- 基于 MCMC 采样的 densification
- 添加噪声注入机制（在 `noise_injection_stop_iter` 之前）
- 不需要 opacity reset
- 配合 `opacity_reg` 和 `scale_reg` 使用（默认 0.01）
- 需要 `antialiased=True` 和 `packed=True` 以获得最佳效果

---

## 6. 分布式训练 (DDP)

- 使用 `torch.nn.parallel.DistributedDataParallel`
- 高斯在初始化时按 rank 均匀分配
- 学习率乘以 `sqrt(world_size)`，步数除以 `steps_scaler`
- Viewer 在多 GPU 模式下自动禁用
- 部分功能不支持多 GPU（如后处理、batch_size > 1）

---

## 7. 评估 & 保存

### 7.1 评估 (`eval_steps`)

在验证集上计算:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity (可选 alex/vgg)
- **cc_PSNR/cc_SSIM/cc_LPIPS**: 颜色校正后的指标 (可选，affine 或 quadratic 校正)

渲染图保存到 `{result_dir}/renders/val_step{step}_{idx}.png`

### 7.2 轨迹渲染 (`render_traj_path`)

生成新视角视频:
- `"interp"`: 在训练视角间插值
- `"ellipse"`: 椭圆轨迹
- `"spiral"`: 螺旋轨迹
- `"raw"`: 直接用原始位姿

视频保存到 `{result_dir}/videos/traj_{step}.mp4`

### 7.3 Checkpoint (`save_steps`, `ply_steps`)

- **Checkpoint** (`.pt`): 包含 splats state_dict、步数、优化器状态等
- **PLY**: 用 `export_splats()` 导出标准 3DGS PLY 格式

---

## 8. 输出目录结构

```
results/garden/
├── cfg.yml              # 配置备份
├── ckpts/               # 模型 checkpoint
│   └── ckpt_6999_rank0.pt
├── stats/               # 评估指标 JSON
│   ├── train_step6999_rank0.json
│   └── val_step6999.json
├── renders/             # 渲染图片
│   ├── val_step6999_0000.png
│   └── ...
├── ply/                 # PLY 点云导出
│   └── point_cloud_6999.ply
├── videos/              # 轨迹视频
│   └── traj_6999.mp4
├── tb/                  # TensorBoard 日志
└── compression/         # (可选) 压缩结果
```

---

## 9. 完整参数速查

### 数据
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `data/360_v2/garden` | COLMAP 数据目录 |
| `data_factor` | `4` | 下采样倍率 |
| `data_type` | `colmap` | 数据格式: `colmap` / `ncore` |
| `test_every` | `8` | 每 N 张取 1 张做验证 |
| `normalize_world_space` | `True` | 世界空间归一化 |
| `load_exposure` | `True` | 加载 EXIF 曝光数据 |

### 训练
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | `30000` | 总训练步数 |
| `batch_size` | `1` | 批次大小 |
| `steps_scaler` | `1.0` | 步数缩放因子 |
| `ssim_lambda` | `0.2` | SSIM loss 权重 |
| `random_bkgd` | `False` | 随机背景增强 |

### 高斯初始化
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `init_type` | `sfm` | 初始化方式: `sfm` / `random` / `lidar` |
| `init_num_pts` | `100000` | 随机初始化时的点数 |
| `init_extent` | `3.0` | 随机初始化范围 (× scene_scale) |
| `init_opa` | `0.1` | 初始不透明度 |
| `init_scale` | `1.0` | 初始尺度倍数 |
| `sh_degree` | `3` | SH 最大阶数 |
| `sh_degree_interval` | `1000` | 每多少步升一阶 SH |

### 渲染
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `near_plane` | `0.01` | 近裁剪面 |
| `far_plane` | `1e10` | 远裁剪面 |
| `packed` | `False` | Packed 模式 (省内存) |
| `antialiased` | `False` | 抗锯齿渲染 |
| `camera_model` | `pinhole` | 相机模型 |

### 执行
```bash
# 单 GPU (DefaultStrategy)
python -m examples.simple_trainer default --data_dir data/360_v2/garden

# 单 GPU (MCMCStrategy)
python -m examples.simple_trainer mcmc --data_dir data/360_v2/garden

# 4 GPU 分布式
CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25
```
