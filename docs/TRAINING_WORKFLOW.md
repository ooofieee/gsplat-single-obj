# 3DGS 训练操作流程

> 从零开始，用 gsplat 训练一个 3D Gaussian Splatting 模型的完整步骤。

---

## 1. 数据集准备

### 使用现成数据集（已有相机位姿和点云）

如果你已经通过 CO3D→COLMAP/SfM 标准化得到了带位姿和真实稀疏点云的数据集，目录结构如下：

```
couch_colmap_sfm/
├── images/          # mask-composited RGB 图像 (102 张)
│   ├── frame000001.jpg
│   └── ...
├── images_4/        # 4× 下采样图像 (102 张)
├── masks/           # (可选) 实例分割掩码
│   ├── frame000001.png
│   └── ...
└── sparse/0/
    ├── cameras.bin    # COLMAP PINHOLE 相机
    ├── images.bin     # COLMAP 注册图像位姿
    └── points3D.bin   # COLMAP SfM 稀疏点云
```

直接指定 `--data_dir` 即可开始训练。

当前 couch 的标准产物是：

```bash
/home/ooofieee/co3d_data/couch_colmap_sfm
```

验证命令：

```bash
PYTHONPATH=.:examples python scripts/standardize_co3d_colmap.py validate \
    --scene-dir /home/ooofieee/co3d_data/couch_colmap_sfm \
    --data-factor 4
```

---

## 2. 训练

### 2.1 环境激活 & 下采样图片生成

```bash
# 1. 激活 gsplat conda 环境
conda activate gsplat

# 2. 进入仓库根目录
cd /home/ooofieee/gsplat

# 3. 标准 couch SFM 数据集已经包含 images_4
PYTHONPATH=.:examples python scripts/standardize_co3d_colmap.py validate \
    --scene-dir /home/ooofieee/co3d_data/couch_colmap_sfm \
    --data-factor 4
```

> 数据集目录结构：
> ```
> couch_colmap_sfm/
> ├── images/          # 原始图片 (102 张 .jpg)
> ├── images_4/        # 4× 下采样 (.png), 训练实际加载此目录
> └── sparse/0/        # COLMAP 二进制文件
> ```

### 2.2 基础训练命令

```bash
# 在仓库根目录运行
# DefaultStrategy (原始 3DGS 论文方法)
python -m examples.simple_trainer default \
    --data_dir /home/ooofieee/co3d_data/couch_colmap_sfm \
    --data_factor 4 \
    --result_dir /home/ooofieee/gsplat/results/couch_sfm \
    --max_steps 30000 \
    --save_ply True

# MCMCStrategy (MCMC 方法，通常效果更好)
python -m examples.simple_trainer mcmc \
    --data_dir /home/ooofieee/co3d_data/couch_colmap_sfm \
    --data_factor 4 \
    --result_dir /home/ooofieee/gsplat/results/couch_sfm_mcmc \
    --max_steps 30000 \
    --save_ply True
```

### 2.3 关键参数说明

> **注意**: 对于 couch 等 CO3D 单序列数据，`--batch_size` 必须为 1（多相机场景要求）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data/360_v2/garden` | COLMAP 数据目录路径 |
| `--data_factor` | `4` | 图片下采样倍率。越大显存占用越小，细节越少 |
| `--result_dir` | `results/garden` | 结果输出目录 |
| `--max_steps` | `30000` | 总训练步数 |
| `--batch_size` | `1` | 每步训练几张图（多相机场景必须为 1） |
| `--ssim_lambda` | `0.2` | SSIM loss 权重 |
| `--sh_degree` | `3` | 球谐函数最大阶数（越高细节越多，计算越慢） |

### 2.4 显存不足时

```bash
# 方法 1: 增大下采样倍率
--data_factor 8

# 方法 2: 使用 packed 模式（省显存，略慢）
--packed True

# 方法 3: 减少 SH 阶数
--sh_degree 1
```

### 2.5 多 GPU 分布式训练

```bash
# 4 卡训练，总步数缩小 4 倍
CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default \
    --data_dir data/360_v2/garden \
    --steps_scaler 0.25
```

---

## 3. 训练过程监控

### 3.1 命令行输出

训练时 tqdm 进度条会实时显示：

```
loss=0.045| sh degree=2| : 100%|████████| 29999/30000
```

- `loss`: 当前总 loss
- `sh degree`: 当前使用的 SH 阶数（从 0 逐步升到 3）

### 3.2 ViSER 可视化 (默认开启)

训练开始时自动在 `http://localhost:8080` 启动一个 Web 可视化界面，可以：
- 实时查看训练渲染结果
- 切换 RGB / 深度 / Alpha 显示模式
- 暂停/继续训练
- 调整渲染参数

如果不需要，加 `--disable_viewer` 关闭。

### 3.3 TensorBoard

```bash
tensorboard --logdir results/garden/tb
```

记录的指标:
- `train/loss`, `train/l1loss`, `train/ssimloss`
- `train/num_GS` (高斯数量变化)
- `train/mem` (显存占用)

---

## 4. 查看结果

训练完成后，`result_dir` 下的目录结构：

```
results/garden/
├── cfg.yml                    # 训练配置备份
├── ckpts/                     # 模型 checkpoint (.pt)
│   ├── ckpt_6999_rank0.pt
│   └── ckpt_29999_rank0.pt
├── ply/                       # PLY 点云 (可在其他 viewer 打开)
│   └── point_cloud_29999.ply
├── renders/                   # 验证集渲染对比图
│   ├── val_step6999_0000.png  # 左边 GT，右边 渲染
│   └── ...
├── stats/                     # 评估指标 JSON
│   ├── val_step6999.json
│   └── val_step29999.json
├── videos/                    # 新视角渲染视频
│   └── traj_29999.mp4
└── tb/                        # TensorBoard 日志
```

### 4.1 评估指标

在 `stats/val_step29999.json` 中可以查看：

```json
{
  "psnr": 27.5,
  "ssim": 0.85,
  "lpips": 0.12,
  "num_GS": 1500000
}
```

- **PSNR** > 25 dB 算可以，> 30 dB 算好
- **SSIM** 越接近 1 越好
- **LPIPS** 越接近 0 越好

---

## 5. 仅评估 / 渲染 (不重新训练)

如果有已保存的 checkpoint：

```bash
python -m examples.simple_trainer default \
    --data_dir data/360_v2/garden \
    --result_dir results/garden_eval \
    --ckpt results/garden/ckpts/ckpt_29999_rank0.pt
```

这会跳过训练，直接运行评估和轨迹渲染。

---

## 6. 常见问题

### Q: 训练到一半 Gaussians 数量爆炸/归零？
- DefaultStrategy: 调大 `--strategy.grow_grad2d` (如 0.0005)，或调大 `--strategy.prune_opa` (如 0.01)
- MCMCStrategy: 增大 `--opacity_reg` 和 `--scale_reg`

### Q: 渲染结果模糊？
- 减小 `--data_factor`（用更高分辨率图片）
- 增大 `--sh_degree`
- 增加 `--max_steps`

### Q: 显存不够？
- 增大 `--data_factor`（4→8）
- 用 `--packed True`
- 减小 `--sh_degree`

---

## 7. 2DGS 训练

2D Gaussian Splatting 使用 `simple_trainer_2dgs.py`：

```bash
python -m examples.simple_trainer_2dgs \
    --data_dir data/360_v2/garden \
    --data_factor 4 \
    --result_dir results/garden_2dgs
```

2DGS 适用于表面重建，额外输出 normal map 和 distortion map。
