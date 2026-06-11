# Research Flow & History — Object-Centric 3D Gaussian Splatting

> 科研目标：从 CO3D 视频序列中抠出单个物体，用 3D Gaussian Splatting 重建高质量物体模型。
> 核心思路：SAM3 前景分割 → COLMAP SfM → Mask-aware 3DGS 训练。

---

## 1. 整体流程概览

```
CO3D 原始数据
  │  (images/, masks/, frame_annotations.jgz)
  │  只有 images 和 masks 有用
  ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: SAM3 Prompt Segmentation    ← 当前阶段      │
│  (在 COLMAP 之前)                                    │
│  - 用 SAM3 对物体做 prompt-based 前景分割             │
│  - 生成 masked_images/ + masks/                      │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: COLMAP SfM 重建                             │
│  scripts/standardize_co3d_colmap.py                   │
│  - 读取 CO3D frame_annotations                       │
│  - 转换 intrinsics (NDC → PINHOLE)                   │
│  - 过滤无效帧                                        │
│  - 运行 COLMAP feature_extractor + matcher + mapper   │
│  - 输出: images/ + masks/ + sparse/0/                │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Mask-aware 3DGS 训练                        │
│  examples/customized_trainer.py                       │
│  - 基于 simple_trainer                               │
│  - MaskedObjectDataset: 加载 masked_images + masks   │
│  - 用 mask 剔除背景像素的 L1 / SSIM loss              │
│  - 渲染输出 mask 外区域为零                           │
│  - 当前策略: 简单删除 mask 以外的 Gaussians           │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: 导出 & 后处理                                │
│  scripts/export_clean_poster_gaussian.py              │
│  - PLY 导出 + 过滤 (brightness, alpha, radius)       │
│  scripts/train_poster_gaussian.py                     │
│  - 后处理训练/fine-tuning                            │
└─────────────────────────────────────────────────────┘
```

---

## 2. 数据资产清单

### 2.1 CO3D 原始数据

| 路径 | 说明 |
|------|------|
| `/home/ooofieee/co3d_data/couch/617_99945_199053/` | CO3D couch 原始序列 (202 帧) |
| `  ├── images/` | 原始 RGB (202 张) |
| `  ├── masks/` | CO3D ground-truth 分割 mask (202 张) |
| `  ├── depths/` | CO3D depth maps (202 张) |
| `  ├── frame_annotations.jgz` | 帧级标注 (R, T, focal_length, principal_point) |
| `  ├── sequence_annotations.jgz` | 序列级标注 (point_cloud=None) |
| `  └── set_lists_manyview_test_0.json` | train=102, val=100, test=100 |

**关键发现**: CO3D 标注中的相机模型是 PyTorch3D NDC 格式，不是 COLMAP 格式。depth 也不能直接当 SfM 点云用。

### 2.2 标准化后的 COLMAP 数据集

| 路径 | 说明 | 用途 |
|------|------|------|
| `/home/ooofieee/co3d_data/couch_colmap_sfm` | 标准 COLMAP SfM 数据集 | simple_trainer 训练 |
| `  ├── images/` | mask-composited RGB (102 张, black bg) | |
| `  ├── images_4/` | 4× 下采样 | |
| `  ├── masks/` | CO3D GT masks | |
| `  ├── colmap_masks/` | COLMAP 特征提取 mask | |
| `  └── sparse/0/` | COLMAP SfM 输出 (1 camera, 102 images, 17735 points) | |
| `/home/ooofieee/co3d_data/couch_colmap_rgbmask` | SAM 分割后的数据集 | **customized_trainer 训练** |
| `  ├── images/` | 原始 RGB | |
| `  ├── masked_images/` | SAM3 抠像结果 (黑背景) | |
| `  ├── masks/` | SAM3 分割 mask | |
| `  └── sparse/0/` | COLMAP SfM (共享或独立重建) | |

### 2.3 训练产出

| 路径 | 说明 |
|------|------|
| `/home/ooofieee/gsplat/results/couch_customized/` | customized_trainer 默认输出目录 |
| `/home/ooofieee/gsplat/results/couch_sfm/` | simple_trainer SfM 基线训练结果 |
| `  ├── ckpts/` | 模型 checkpoint (.pt) |
| `  ├── ply/` | PLY 点云导出 |
| `  ├── renders/` | 验证集渲染对比图 |
| `  ├── stats/` | PSNR/SSIM/LPIPS 指标 JSON |
| `  ├── videos/` | 新视角渲染视频 |
| `  └── tb/` | TensorBoard 日志 |

---

## 3. 关键技术决策记录

### 3.1 为什么不用 CO3D 的 depth 和 point_cloud？

- `sequence_annotations.jgz` 中 `point_cloud=None`
- CO3D depth 是 per-frame 估计值，不是 SfM 稀疏重建的三维点
- COLMAP SfM 的 points3D 才是可复用的稀疏点云，适合初始化 Gaussians

### 3.2 坐标系转换: PyTorch3D → COLMAP

```
S = diag(-1, -1, 1)            # PyTorch3D → OpenCV 的符号矩阵
R_colmap = S @ R_co3d.T        # 旋转: 转置 + 轴翻转
t_colmap = S @ T_co3d          # 平移: 轴翻转
camera_center = -R_co3d @ T_co3d
```

NDC principal point → COLMAP pixel:
```
scale = min(width, height) / 2
fx = focal_x * scale
fy = focal_y * scale
cx = width/2  - pp_x * scale   # 符号翻转!
cy = height/2 - pp_y * scale
```

### 3.3 训练防泄漏策略

- 只使用 `set_lists` 中的 train split，验证集不参与 SfM 和训练
- 过滤黑帧 (mean ≤ 0.001) 和空 mask (max = 0)
- 原始数据和 SAM 分割后，SfM 使用的 camera 可能不同

### 3.4 为什么是"简单删除 mask 外高斯"的策略？

当前 customized_trainer 的核心思路是 **训练阶段的 loss masking**，而非显式的几何剔除:
1. 渲染颜色在 mask 外区域被置零 (`render_colors[~masks] = 0`)
2. L1 loss 只计算 mask 内像素 (`l1_loss(colors[masks], pixels[masks])`)
3. SSIM loss 将 mask 外区域零化后计算
4. mask 外的高斯不会被直接删除，但由于背景区域没有梯度信号，它们会自然萎缩或被 densification 的 prune 步骤清理

这是最简单直接的方法，缺点是可能留下一些不可见的浮游高斯。

---

## 4. 代码文件索引

### 4.1 数据预处理

| 文件 | 功能 |
|------|------|
| `scripts/standardize_co3d_colmap.py` | CO3D → COLMAP 标准化 (build + validate) |
| `tests/test_standardize_co3d_colmap.py` | 标准化脚本的测试 |

### 4.2 训练器

| 文件 | 功能 |
|------|------|
| `examples/simple_trainer.py` | gsplat 官方训练器 (全场景 3DGS) |
| `examples/customized_trainer.py` | **自定义训练器** — 继承 simple_trainer，添加 `MaskedObjectDataset`，支持 mask-aware loss |
| `tests/test_customized_trainer.py` | customized_trainer 的测试 |

关键区别 (`MaskedObjectDataset` vs 原始 `Dataset`):
- 从 `masked_images/<stem>.png` 加载 RGB (SAM 抠像结果)
- 从 `masks/<stem>.png` 加载 object mask
- 返回 `data["mask"]` 用于 loss 计算
- 支持 patch_size 的 mask 同步裁剪

> **PLY 导出默认值差异**: `customized_trainer.py` 默认 `save_ply=True`，训练结束自动导出点云；`simple_trainer.py` 默认 `save_ply=False`，需要手动传 `--save_ply True`。

### 4.3 后处理

| 文件 | 功能 |
|------|------|
| `scripts/sam3_segment_dataset.py` | SAM3 text-prompt 批量分割脚本 |
| `scripts/train_poster_gaussian.py` | 后处理高斯训练/fine-tuning |
| `scripts/export_clean_poster_gaussian.py` | 过滤导出干净的 PLY (brightness/alpha/radius 阈值) |
| `scripts/export_ply_variants.py` | PLY 导出变体 |
| `scripts/export_floor_xy.py` | 基于 floor 平面过滤导出 |
| `scripts/export_floor_xz.py` | 同上 (不同平面方向) |
| `tests/test_train_poster_gaussian_filters.py` | poster Gaussian 过滤的测试 |

### 4.4 文档

| 文件 | 说明 |
|------|------|
| `docs/co3d2colmap.md` | CO3D → COLMAP 转换详解 |
| `docs/TRAINING_PIPELINE.md` | gsplat 训练流程技术文档 |
| `docs/TRAINING_WORKFLOW.md` | 从零开始训练的操作手册 |
| `docs/RESEARCH_FLOW.md` | **本文档 — 科研流程与历史** |
| `docs/superpowers/specs/2026-06-07-co3d-couch-customized-trainer-design.md` | customized_trainer 设计规格 |
| `docs/superpowers/plans/2026-06-07-co3d-couch-customized-trainer.md` | customized_trainer 实现计划 |

---

## 5. CO3D 稠密点云作为初始化

### 5.1 背景

CO3D 部分序列自带稠密点云（`{sequence}/pointcloud.ply`，从 depth fusion 生成），点数通常是 SfM 稀疏点的 50-100 倍。例如：

| 场景 | SfM 稀疏点 | CO3D 稠密点 | 倍数 |
|------|----------|-----------|------|
| toytruck (190) | 6,207 | 620,465 | 100× |
| book (119) | 12,876 | ? | ? |

### 5.2 坐标系对齐问题

CO3D 点云在 CO3D 的 PyTorch3D 世界坐标系下，而 COLMAP SfM 输出在 COLMAP 自建的世界坐标系下。两者需要一个相似变换才能对齐。

**解决方式**：用相机中心（camera centers）做 Procrustes 对齐。

```
1. 从 CO3D frame_annotations 提取 camera center: -R @ T
2. 从 COLMAP sparse/0 images.bin 提取 camera center: -R.T @ t
3. 匹配同帧号的两组中心点
4. 计算相似变换: X_colmap = scale × R @ X_co3d + t
5. 将 CO3D 点云变换到 COLMAP 空间
6. 再套 parser.transform 进入归一化训练空间
```

**关键要求**：COLMAP SfM 和 CO3D 点云必须来自**同一条 CO3D 序列**，否则对齐残差会很大（~5 vs ~0.1）。

### 5.3 过密问题与降采样

62 万点直接初始化会导致 62 万 Gaussians，训练极慢且膨胀失控。解决方案：
- 随机降采样到 30,000 点（`co3d_pc_max_points`）
- 配合抑制 densification 的策略参数

### 5.4 Densification 策略调参

DefaultStrategy 的关键参数和针对稠密点云初始化的调参：

| 参数 | 默认 | 稠密初始化调法 | 原理 |
|------|------|-------------|------|
| `reset_every` | 3,000 | **999999** (禁用) | 避免周期性 opacity reset 让无用高斯无法衰减 |
| `grow_grad2d` | 0.0002 | **0.0008~0.001** | 提高 split/clone 门槛，只有真正需要的地方才增长 |
| `mask_densify_erode_px` | 0 | **5** | 用内缩后的 mask 抑制边缘/外部高斯触发 split/clone |
| `mask_densify_min_ratio` | 1.0 | **1.0** | 当前 batch 中必须全部落在 eroded mask 内才允许进入 densification 统计 |

`reset_every` 是核心问题：它每 3000 步把所有高斯 opacity 拉回 1%，导致前景区域外的低不透明度高斯无法自然衰减到 prune 阈值以下。mask-only 训练前景只占 ~6% 像素，这个效应尤其严重。

### 5.5 Edge-aware densification gate

2026-06-10 新增关键修复：**mask-aware loss 本身不足以阻止物体轮廓附近的 densification**。`DefaultStrategy` 的 grow/split 只看 `means2d.grad > grow_grad2d`，而物体边缘通常正是屏幕空间梯度最大的区域，因此会在轮廓附近不断 split/clone，形成单体周围的尖刺。

实现位置：`examples/customized_trainer.py`

Config 新增参数：
- `mask_densify_erode_px: int = 0` — 0 表示关闭；推荐 toytruck 使用 5
- `mask_densify_min_ratio: float = 1.0` — 当前 batch 视角中落在 eroded mask 内的最低比例
- `mask_densify_from_iter: int = 0`
- `mask_densify_until_iter: int = 15_000`
- `mask_densify_verbose: bool = False`

训练循环中的时序：

```
forward → masked loss → loss.backward()
  → _apply_mask_densification_gate()
  → optimizer.step()
  → DefaultStrategy.step_post_backward()
```

gate 的核心逻辑：
1. 对当前训练 mask 做 erosion，得到更保守的物体内部区域。
2. 将当前所有 Gaussian center 投影到当前 batch 视角。
3. 对投影到 eroded mask 外、原始 mask 边缘、画面外、或相机后的 Gaussian，将 `info["means2d"].grad` 置零。
4. 只改变 densification 统计所用的 screen-space gradient，不改变 photometric loss，也不直接改 Gaussian 参数梯度。

这相当于：**边缘仍参与颜色监督，但不再作为 split/clone 的种子区域**。目前只对 `DefaultStrategy` + unpacked mode 生效；MCMC 和 packed mode 暂不启用。

配套测试：
- `tests/test_customized_trainer.py::test_erode_densify_mask_falls_back_when_erosion_clears_mask`
- `tests/test_customized_trainer.py::test_mask_densification_gate_zeroes_gradients_outside_eroded_mask`

验证命令：

```bash
conda run -n gsplat env PYTHONPATH=.:examples pytest tests/test_customized_trainer.py -q
# 4 passed
```

### 5.6 实现路径

`customized_trainer.py` Config 新增参数：
- `co3d_pc_path: Optional[str]` — CO3D PLY 点云路径
- `co3d_pc_max_points: int = 30_000` — 降采样后的最大点数
- `mask_densify_erode_px: int = 0` — edge-aware densification gate 的 erosion 半径
- `mask_densify_min_ratio: float = 1.0` — 保留 densification 统计的 mask 内投影比例阈值

Runner.__init__ 在 parser 创建后自动加载、对齐、降采样，替换 `parser.points`。

---

## 6. 实验结果

### 6.1 toytruck — 完整对比

| 指标 | SfM (baseline) | CO3D 620K raw | CO3D 30K (坐标未对齐) | CO3D 30K 对齐 | **CO3D 30K + edge gate** |
|------|:--:|:--:|:--:|:--:|:--:|
| 初始化点 | 6,207 | 620,465 | 30,000 | 30,000 | 30,000 |
| 最终 GS | 150,632 | 542K (stuck) | 62,698 | 53,837 | **40,321** |
| 膨胀倍率 | 24× | - | 2.1× | 1.8× | **1.34×** |
| PSNR | 40.56 | 34.3 (@7K) | 37.91 | 40.90 | **41.01** |
| SSIM | 0.993 | 0.977 (@7K) | 0.989 | 0.994 | **0.9941** |
| LPIPS | 0.005 | 0.023 (@7K) | 0.010 | 0.0045 | **0.00439** |

CO3D 30K 对齐版：三项指标全部超过 SfM 基线，GS 数量仅 36%。

CO3D 30K + edge gate 版：在保持/略提升指标的同时，将最终 GS 从 53,837 降到 40,321，说明边缘 densification gate 有效抑制了不必要的 split/clone。结果目录：

```
/home/ooofieee/gsplat/results/toytruck_edgegate_co3dpc
```

### 6.2 book — 结果

| 指标 | 数值 |
|------|------|
| 初始化 | 30,000 (CO3D PC) |
| 最终 GS | 151,036 |
| 膨胀倍率 | 5.0× |
| PSNR | 39.75 |
| SSIM | 0.995 |
| LPIPS | 0.0068 |

book 的 SSIM 0.995 略超 toytruck，但膨胀 5× 偏高，`grow_grad2d` 可能需进一步调大。

---

## 7. 当前状态 & 进度

### 已完成 ✓

- [x] CO3D 数据探索：确定只有 images 和 masks 可用
- [x] COLMAP SfM 标准化流水线 (`standardize_co3d_colmap.py`)
- [x] `couch_colmap_sfm` 数据集构建与验证 (102 images, 17735 SfM points)
- [x] `simple_trainer` 基线训练流程跑通
- [x] SAM3 prompt 分割 (在 COLMAP 之前完成)
- [x] `couch_colmap_rgbmask` 数据集准备
- [x] `customized_trainer.py` 实现 (`MaskedObjectDataset` + mask-aware loss)
- [x] `customized_trainer` 测试通过
- [x] COLMAP SfM GPU/CPU 模式确认（头服务器无 OpenGL，只能用 CPU）
- [x] CO3D 稠密点云发现与分析（toytruck 620K 点，book 119 有 PC）
- [x] CO3D → COLMAP 坐标系 Procrustes 对齐 (camera centers 匹配)
- [x] `customized_trainer` 新增 `co3d_pc_path` + `co3d_pc_max_points` 参数
- [x] Densification 策略分析：`reset_every` 是造成稠密初始化膨胀的主因
- [x] toytruck: SAM3 "blue toytruck" + CO3D 30K PC 对齐训练 → PSNR 40.90, GS 54K
- [x] toytruck: edge-aware densification gate (`mask_densify_erode_px=5`) → PSNR 41.01, GS 40K
- [x] book: SAM3 "book" + CO3D 30K PC 对齐训练 → PSNR 39.75, SSIM 0.995
- [x] `scripts/sam3_segment_dataset.py` SAM3 批量分割脚本
- [x] PLY 导出与过滤脚本

### 进行中 🔄

- [ ] toytruck 和 book 的定性渲染对比（vs SfM 基线）
- [ ] toytruck edge-gate PLY 定性检查：重点看物体周围尖刺是否消失
- [ ] book densification 参数微调（膨胀 5× 偏高）
- [ ] SAM3 分割质量的系统评估 (不同 prompt 策略的效果对比)

### 待探索 📋

- [ ] 多种 SAM3 prompt 策略对比 (point, box, text)
- [ ] CO3D 点云 voxel-grid 降采样（替代随机降采样，改善覆盖均匀性）
- [ ] 训练时动态 mask 更新 (morphological dilation/erosion, CRF refinement)
- [ ] 3D 层面的显式高斯剔除 (ray-Gaussian intersection + mask consistency)
- [ ] **Densification 后 mask-贡献审计** — edge gate 已经抑制边缘 split/clone，但 split 后仍可能存在少量 mask 外残留。方案：每个 refine_every 步 densification 后采样 K 个训练视角渲染，对 mask 内像素 alpha 贡献 < 阈值的高斯，主动将 opacity 压到 prune_opa 以下，让下次 prune 回收。与调参组合使用。
- [ ] 多物体场景的分离重建
- [ ] 其他 CO3D 类别泛化测试
- [ ] 与 2DGS (simple_trainer_2dgs) 的对比实验
- [ ] 评估指标设计 (foreground-only PSNR/SSIM vs full image)

---

## 6. 常用命令速查

### 6.1 COLMAP 数据集构建

```bash
conda activate gsplat
cd /home/ooofieee/gsplat

# 从 CO3D 原始数据构建 COLMAP 数据集
PYTHONPATH=.:examples:/home/ooofieee/co3d \
python scripts/standardize_co3d_colmap.py build \
  --co3d-root /home/ooofieee/co3d_data \
  --category couch \
  --sequence 617_99945_199053 \
  --output-dir /home/ooofieee/co3d_data/couch_colmap_sfm_new \
  --data-factor 4 \
  --run-colmap

# 验证数据集
PYTHONPATH=.:examples python scripts/standardize_co3d_colmap.py validate \
  --scene-dir /home/ooofieee/co3d_data/couch_colmap_sfm \
  --data-factor 4
```

### 6.2 训练

```bash
# Standard trainer (全场景基线)
python -m examples.simple_trainer default \
  --data_dir /home/ooofieee/co3d_data/couch_colmap_sfm \
  --data_factor 4 \
  --result_dir results/couch_sfm \
  --max_steps 30000 --disable_viewer

# Customized trainer (mask-aware, 抠像物体)
python -m examples.customized_trainer default \
  --data_dir /home/ooofieee/co3d_data/couch_colmap_rgbmask \
  --data_factor 4 \
  --result_dir results/couch_customized \
  --max_steps 30000

# MCMC strategy (通常效果更好)
python -m examples.customized_trainer mcmc \
  --data_dir /home/ooofieee/co3d_data/couch_colmap_rgbmask \
  --data_factor 4 \
  --result_dir results/couch_customized_mcmc \
  --max_steps 30000

# Toytruck: CO3D dense point cloud + edge-aware densification gate
python -m examples.customized_trainer default \
  --data_dir /home/ooofieee/co3d_data/toytruck_colmap_sfm_sam3 \
  --data_factor 4 \
  --result_dir results/toytruck_edgegate_co3dpc \
  --max_steps 30000 \
  --disable_viewer \
  --save_ply \
  --co3d_pc_path /home/ooofieee/co3d_data/toytruck/190_20494_39385/pointcloud.ply \
  --co3d_pc_max_points 30000 \
  --mask_densify_erode_px 5 \
  --mask_densify_min_ratio 1.0 \
  --strategy.grow_grad2d 0.0008 \
  --strategy.reset_every 999999
```

### 6.3 导出

```bash
# 过滤导出干净 PLY
python scripts/export_clean_poster_gaussian.py \
  --ckpt results/couch_customized/ckpts/ckpt_29999_rank0.pt \
  --out results/couch_customized/clean.ply \
  --min-brightness 0.1 --min-alpha 0.08 --max-radius 1.5
```

### 6.4 测试

```bash
conda activate gsplat
PYTHONPATH=.:examples pytest tests/test_customized_trainer.py -q
PYTHONPATH=.:examples pytest tests/test_standardize_co3d_colmap.py -q
```

---

## 7. 时间线

| 日期 | 事件 |
|------|------|
| 2026-06 初 | CO3D 数据集探索，发现只有 images/masks 可用 |
| 2026-06-04 | 标准化流水线 `standardize_co3d_colmap.py` 完成，COLMAP SfM 跑通 |
| 2026-06-06 | `couch_colmap_sfm` 构建与验证，simple_trainer 基线训练跑通 |
| 2026-06-06 | SAM3 prompt 分割实验起步 |
| 2026-06-07 | `couch_colmap_rgbmask` 数据集准备 |
| 2026-06-07 | `customized_trainer.py` 设计与实现，测试通过 |
| 2026-06-09 | 科研流程文档整理 |
| 2026-06-09 | toytruck COLMAP SfM 构建 + SAM3 "blue toytruck" 分割 |
| 2026-06-09 | toytruck simple_trainer 基线 → PSNR 27.44, GS 405K |
| 2026-06-09 | toytruck customized_trainer (SfM) → PSNR 40.56, GS 151K |
| 2026-06-09 | 发现 CO3D 稠密点云 (620K 点)，首次尝试坐标系未对齐 + 过密 |
| 2026-06-09 | Densification 分析：`reset_every` 是稠密初始化 GS 不衰减的主因 |
| 2026-06-09 | CO3D→COLMAP Procrustes 对齐修复 |
| 2026-06-09 | `customized_trainer` 新增 `co3d_pc_path` + `co3d_pc_max_points` + 自动对齐 |
| 2026-06-09 | toytruck CO3D 30K 对齐 → PSNR 40.90, SSIM 0.994, GS 54K (全面超 SfM) |
| 2026-06-09 | book 序列 119 SfM 重建 + SAM3 "book" + CO3D PC → PSNR 39.75, SSIM 0.995 |
| 2026-06-10 | 发现单体重建周围仍有尖刺，定位为边缘高梯度触发 densification |
| 2026-06-10 | 新增 edge-aware densification gate：用 eroded mask 将边缘/外部高斯的 `means2d.grad` 置零 |
| 2026-06-10 | toytruck edge gate 训练完成 → PSNR 41.01, SSIM 0.9941, LPIPS 0.00439, GS 40,321 |

---

## 8. 已知问题 & 注意事项

1. **COLMAP 特征提取用 CPU**: `--SiftExtraction.use_gpu 0`，速度较慢但稳定
2. **single_camera=1**: 所有帧共享同一 PINHOLE 内参 (取中位数)，CO3D gt 内参不用
3. **ba_refine 关闭**: focal length 和 principal point 在 bundle adjustment 中固定
4. **couch_colmap vs couch_colmap_sfm**: 旧版 `couch_colmap` 的点云是 mask-guided synthetic 的，不是真正的 SfM 重建，已弃用
5. **batch_size=1 限制**: 多相机数据集强制 batch_size=1
6. **mask 目录结构**: COLMAP 特征提取 mask 放 `colmap_masks/<image_name>.png`，训练用 mask 放 `masks/<stem>.png`
7. **COLMAP GPU 不可用**: 头服务器无显示 + COLMAP OpenGL 依赖，SIFT 特征提取只能用 CPU
8. **CO3D 点云必须同序列**: COLMAP SfM 和 CO3D 点云必须来自同一条 CO3D 序列，否则相机轨迹不同，Procrustes 对齐残差会达到 5+（同序列仅 0.06-0.29）
9. **tyro bool 参数**: `--save_ply` 不需要跟 `True`，直接 `--save_ply` 即可；`--save_ply True` 中 `True` 被当成额外参数报错
10. **edge gate 只抑制 densification，不改 loss**: `mask_densify_erode_px` 只会影响 `DefaultStrategy` 使用的 `means2d.grad` 统计，不会把边缘像素从 photometric loss 中移除。
11. **packed mode 暂不支持 edge gate**: 当前实现中 `cfg.packed=True` 时 `_apply_mask_densification_gate()` 会直接跳过。
