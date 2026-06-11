# Mask Edge Densification Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent mask-boundary and out-of-mask Gaussians from triggering DefaultStrategy split/clone densification in `examples/customized_trainer.py`.

**Architecture:** Keep mask-aware photometric loss unchanged, but gate densification statistics before `DefaultStrategy.step_post_backward()` reads `info["means2d"].grad`. The gate projects current Gaussian centers into the current training view, checks an eroded object mask, and zeros screen-space densification gradients for Gaussians outside that safe interior region. This suppresses edge spikes without removing useful color gradients from ordinary optimizer updates.

**Tech Stack:** Python 3.10, PyTorch, OpenCV (`cv2.erode`), pytest, gsplat `DefaultStrategy`.

---

## File Structure

- Modify `examples/customized_trainer.py`
  - Add config knobs for densification masking.
  - Add helper methods on `Runner`:
    - `_erode_densify_mask(mask: Tensor, erode_px: int) -> Tensor`
    - `_densification_keep_mask(camtoworlds, Ks, masks, width, height) -> Tensor`
    - `_apply_mask_densification_gate(info, camtoworlds, Ks, masks, width, height) -> int`
  - Call `_apply_mask_densification_gate()` after `loss.backward()` and before optimizer/strategy post-backward steps.
- Modify `tests/test_customized_trainer.py`
  - Add tests for mask erosion and gradient gating behavior using a tiny `Runner` instance created via `Runner.__new__`.

## Behavioral Requirements

- Default behavior must stay unchanged: edge-aware densification gate is disabled by default.
- Gate applies only to `DefaultStrategy`; MCMC remains unchanged.
- Gate uses only the current training view/batch, not a slow multi-view audit.
- Gate must support unpacked `info["means2d"]` shaped `[C, N, 2]`.
- Packed mode may be explicitly skipped in this first implementation: if `cfg.packed` is true, return without modifying gradients.
- Gate must not alter photometric loss or optimizer gradients for Gaussian parameters directly; it only changes `info["means2d"].grad`, which `DefaultStrategy` uses for densification statistics.
- If erosion removes the entire mask, fall back to the original mask so training does not disable all densification on thin objects.

---

### Task 1: Add Failing Tests For Mask Erosion And Densification Gradient Gate

**Files:**
- Modify: `tests/test_customized_trainer.py`

- [ ] **Step 1: Add test imports**

Add `from types import SimpleNamespace` after the existing imports:

```python
from pathlib import Path
from types import SimpleNamespace

import torch
```

- [ ] **Step 2: Add mask erosion test**

Append this test to `tests/test_customized_trainer.py`:

```python
def test_erode_densify_mask_falls_back_when_erosion_clears_mask():
    from examples.customized_trainer import Runner

    runner = Runner.__new__(Runner)
    runner.device = torch.device("cpu")

    mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    mask[:, 2, 2] = True

    eroded = runner._erode_densify_mask(mask, erode_px=2)

    assert torch.equal(eroded, mask)
```

- [ ] **Step 3: Add densification gradient gate test**

Append this test to `tests/test_customized_trainer.py`:

```python
def test_mask_densification_gate_zeroes_gradients_outside_eroded_mask():
    from gsplat.strategy import DefaultStrategy
    from examples.customized_trainer import Runner

    runner = Runner.__new__(Runner)
    runner.device = torch.device("cpu")
    runner.world_rank = 0
    runner.cfg = SimpleNamespace(
        mask_densify_erode_px=1,
        mask_densify_min_ratio=1.0,
        mask_densify_from_iter=0,
        mask_densify_until_iter=999999,
        mask_densify_verbose=False,
        packed=False,
        strategy=DefaultStrategy(),
    )
    runner.splats = {
        "means": torch.nn.Parameter(
            torch.tensor(
                [
                    [2.0, 2.0, 1.0],  # safely inside eroded mask
                    [1.0, 1.0, 1.0],  # on original mask edge, removed by erosion
                    [0.0, 0.0, 1.0],  # outside mask
                    [2.0, 2.0, -1.0], # behind camera
                ],
                dtype=torch.float32,
            )
        )
    }

    means2d = torch.zeros(1, 4, 2, dtype=torch.float32, requires_grad=True)
    means2d_grad = torch.ones_like(means2d)
    info = {"means2d": means2d}
    means2d.backward(means2d_grad)

    camtoworlds = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    Ks = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    mask[:, 1:4, 1:4] = True

    suppressed = runner._apply_mask_densification_gate(
        info=info,
        camtoworlds=camtoworlds,
        Ks=Ks,
        masks=mask,
        width=5,
        height=5,
        step=100,
    )

    assert suppressed == 3
    assert torch.equal(
        info["means2d"].grad,
        torch.tensor(
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
    )
```

- [ ] **Step 4: Run tests and verify failure**

Run:

```bash
conda run -n gsplat env PYTHONPATH=.:examples pytest tests/test_customized_trainer.py::test_erode_densify_mask_falls_back_when_erosion_clears_mask tests/test_customized_trainer.py::test_mask_densification_gate_zeroes_gradients_outside_eroded_mask -q
```

Expected: both new tests fail because `Runner._erode_densify_mask` and `Runner._apply_mask_densification_gate` do not exist yet.

---

### Task 2: Add Config Knobs

**Files:**
- Modify: `examples/customized_trainer.py`

- [ ] **Step 1: Add fields to `Config` after `mask_audit_ratio`**

Add this exact block:

```python
    # Edge-aware densification gate. This does not change the photometric loss;
    # it only prevents mask-edge/out-of-mask Gaussians from entering split/clone
    # statistics in DefaultStrategy.
    mask_densify_erode_px: int = 0        # 0 = disabled
    mask_densify_min_ratio: float = 1.0   # fraction of current views inside eroded mask
    mask_densify_from_iter: int = 0
    mask_densify_until_iter: int = 15_000
    mask_densify_verbose: bool = False
```

- [ ] **Step 2: Run default config test**

Run:

```bash
conda run -n gsplat env PYTHONPATH=.:examples pytest tests/test_customized_trainer.py::test_customized_config_defaults_to_couch_masked_scene -q
```

Expected: PASS.

---

### Task 3: Implement Mask Erosion Helper

**Files:**
- Modify: `examples/customized_trainer.py`

- [ ] **Step 1: Add `_erode_densify_mask` before `_mask_contribution_audit`**

Insert this method inside `Runner`, immediately before `@torch.no_grad() def _mask_contribution_audit`:

```python
    @staticmethod
    def _erode_densify_mask(mask: Tensor, erode_px: int) -> Tensor:
        """Return a mask interior for densification gating.

        If erosion removes the whole object for a view, keep the original mask
        for that view so thin objects still allow some densification.
        """
        if erode_px <= 0:
            return mask

        mask_cpu = mask.detach().to("cpu").bool()
        if mask_cpu.ndim == 2:
            mask_cpu = mask_cpu.unsqueeze(0)

        kernel_size = erode_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded_views = []
        for view_mask in mask_cpu:
            src = view_mask.numpy().astype(np.uint8)
            eroded = cv2.erode(src, kernel, iterations=1).astype(bool)
            if not eroded.any() and src.any():
                eroded = src.astype(bool)
            eroded_views.append(torch.from_numpy(eroded))

        eroded_mask = torch.stack(eroded_views, dim=0).to(
            device=mask.device, dtype=torch.bool
        )
        if mask.ndim == 2:
            eroded_mask = eroded_mask.squeeze(0)
        return eroded_mask
```

- [ ] **Step 2: Run erosion test**

Run:

```bash
conda run -n gsplat env PYTHONPATH=.:examples pytest tests/test_customized_trainer.py::test_erode_densify_mask_falls_back_when_erosion_clears_mask -q
```

Expected: PASS.

---

### Task 4: Implement Densification Keep Mask And Gradient Gate

**Files:**
- Modify: `examples/customized_trainer.py`

- [ ] **Step 1: Add helper methods after `_erode_densify_mask`**

Insert these methods inside `Runner`, immediately after `_erode_densify_mask`:

```python
    @torch.no_grad()
    def _densification_keep_mask(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        masks: Tensor,
        width: int,
        height: int,
    ) -> Tensor:
        """Return per-Gaussian keep mask for DefaultStrategy densification stats."""
        means = self.splats["means"]
        n_gaussians = means.shape[0]
        safe_masks = self._erode_densify_mask(
            masks.bool(), self.cfg.mask_densify_erode_px
        )
        if safe_masks.ndim == 2:
            safe_masks = safe_masks.unsqueeze(0)

        hit_count = torch.zeros(n_gaussians, device=means.device)
        vis_count = torch.zeros(n_gaussians, device=means.device)

        for view_idx in range(camtoworlds.shape[0]):
            c2w = camtoworlds[view_idx]
            K_mat = Ks[view_idx]
            mask = safe_masks[view_idx]
            w2c = torch.linalg.inv(c2w)
            pts_cam = means @ w2c[:3, :3].T + w2c[:3, 3]
            depths = pts_cam[:, 2]

            pts_img = pts_cam @ K_mat.T
            z = pts_img[:, 2].clamp(min=1e-10)
            u = pts_img[:, 0] / z
            v = pts_img[:, 1] / z

            valid = (depths > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            vis_count[valid] += 1.0

            u_valid = u[valid].long().clamp(0, width - 1)
            v_valid = v[valid].long().clamp(0, height - 1)
            hit_count[valid] += mask[v_valid, u_valid].float()

        hit_ratio = torch.where(
            vis_count > 0,
            hit_count / vis_count.clamp(min=1.0),
            torch.zeros_like(vis_count),
        )
        return (vis_count > 0) & (hit_ratio >= self.cfg.mask_densify_min_ratio)

    @torch.no_grad()
    def _apply_mask_densification_gate(
        self,
        info: Dict,
        camtoworlds: Tensor,
        Ks: Tensor,
        masks: Optional[Tensor],
        width: int,
        height: int,
        step: int,
    ) -> int:
        """Zero DefaultStrategy screen-space gradients outside eroded masks."""
        cfg = self.cfg
        if masks is None:
            return 0
        if not isinstance(cfg.strategy, DefaultStrategy):
            return 0
        if cfg.mask_densify_erode_px <= 0:
            return 0
        if step < cfg.mask_densify_from_iter or step >= cfg.mask_densify_until_iter:
            return 0
        if getattr(cfg, "packed", False):
            return 0

        means2d = info.get(cfg.strategy.key_for_gradient)
        if means2d is None or means2d.grad is None:
            return 0

        keep = self._densification_keep_mask(
            camtoworlds=camtoworlds,
            Ks=Ks,
            masks=masks,
            width=width,
            height=height,
        )
        suppress = ~keep
        n_suppress = int(suppress.sum().item())
        if n_suppress == 0:
            return 0

        grad = means2d.grad
        if grad.ndim == 3:
            grad[:, suppress, :] = 0
        elif grad.ndim == 2:
            grad[suppress, :] = 0
        else:
            raise ValueError(f"Unexpected means2d grad shape: {tuple(grad.shape)}")

        if cfg.mask_densify_verbose and self.world_rank == 0:
            print(
                f"[MaskDensify] step={step}: suppressed {n_suppress}/"
                f"{keep.numel()} Gaussians from densification stats"
            )
        return n_suppress
```

- [ ] **Step 2: Run gradient gate test**

Run:

```bash
conda run -n gsplat env PYTHONPATH=.:examples pytest tests/test_customized_trainer.py::test_mask_densification_gate_zeroes_gradients_outside_eroded_mask -q
```

Expected: PASS.

---

### Task 5: Wire Gate Into Training Loop

**Files:**
- Modify: `examples/customized_trainer.py`

- [ ] **Step 1: Call the gate immediately after `loss.backward()`**

Find:

```python
            loss.backward()

            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
```

Replace it with:

```python
            loss.backward()
            n_densify_suppressed = self._apply_mask_densification_gate(
                info=info,
                camtoworlds=camtoworlds,
                Ks=Ks,
                masks=masks,
                width=width,
                height=height,
                step=step,
            )

            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if n_densify_suppressed > 0:
                desc += f"mask densify suppressed={n_densify_suppressed}| "
```

- [ ] **Step 2: Add TensorBoard scalar near existing train scalars**

Find:

```python
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
```

Replace with:

```python
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar(
                    "train/mask_densify_suppressed", n_densify_suppressed, step
                )
                self.writer.add_scalar("train/mem", mem, step)
```

- [ ] **Step 3: Run full customized trainer tests**

Run:

```bash
conda run -n gsplat env PYTHONPATH=.:examples pytest tests/test_customized_trainer.py -q
```

Expected: PASS.

---

### Task 6: Smoke Test CLI Parsing

**Files:**
- No source changes unless this fails.

- [ ] **Step 1: Check help includes new knobs**

Run:

```bash
conda run -n gsplat env PYTHONPATH=.:examples python -m examples.customized_trainer default --help | rg "mask-densify"
```

Expected output includes:

```text
--mask-densify-erode-px
--mask-densify-min-ratio
--mask-densify-from-iter
--mask-densify-until-iter
--mask-densify-verbose
```

- [ ] **Step 2: Run a tiny training smoke if dataset exists**

Run:

```bash
if [ -d /home/ooofieee/co3d_data/toytruck_colmap_sfm_sam3 ]; then
  conda run -n gsplat env PYTHONPATH=.:examples python -m examples.customized_trainer default \
    --data_dir /home/ooofieee/co3d_data/toytruck_colmap_sfm_sam3 \
    --data_factor 8 \
    --result_dir results/toytruck_mask_densify_gate_smoke \
    --max_steps 20 \
    --disable_viewer \
    --save_ply \
    --mask_densify_erode_px 5 \
    --mask_densify_min_ratio 1.0 \
    --mask_densify_until_iter 15000
fi
```

Expected: command completes without an exception. If it runs, report the full results path:

```text
/home/ooofieee/gsplat/results/toytruck_mask_densify_gate_smoke
```

---

## Suggested Real Training Command

After implementation passes tests, run the real experiment with:

```bash
conda run -n gsplat env PYTHONPATH=.:examples python -m examples.customized_trainer default \
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
  --mask_densify_until_iter 15000 \
  --strategy.grow_grad2d 0.0008 \
  --strategy.reset_every 999999
```

Full results path for that run:

```text
/home/ooofieee/gsplat/results/toytruck_edgegate_co3dpc
```

## Self-Review

- Spec coverage: The plan addresses the suspected root cause by gating densification gradients at mask edges while leaving loss untouched.
- Placeholder scan: No TBD/TODO placeholders are present.
- Type consistency: Config names are snake_case in Python and map to tyro CLI flags; helper signatures use `Tensor`, `Optional[Tensor]`, and `Dict`, which are already imported in `examples/customized_trainer.py`.
