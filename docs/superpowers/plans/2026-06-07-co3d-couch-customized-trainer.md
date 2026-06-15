# CO3D Couch Customized Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a copied `examples/customized_trainer.py` that trains CO3D couch from `masked_images` and `masks`.

**Architecture:** Keep the stock trainer architecture and core gsplat rendering unchanged. Add a small per-image masked COLMAP dataset adapter inside the copied trainer, then wire `Runner` to use it for COLMAP data.

**Tech Stack:** Python 3.10, PyTorch, pycolmap, imageio/PIL, tyro, pytest.

---

### Task 1: Tests For Masked Dataset Defaults And Mapping

**Files:**
- Create: `tests/test_customized_trainer.py`
- Create: `examples/customized_trainer.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `CustomizedConfig`, instantiate a parser/dataset on `/home/ooofieee/co3d_data/couch_colmap_rgbmask`, and assert the returned image and mask shapes match at `data_factor=4`.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/ooofieee/anaconda3/etc/profile.d/conda.sh && conda activate gsplat && PYTHONPATH=.:examples pytest tests/test_customized_trainer.py -q`

Expected: FAIL because `examples.customized_trainer` does not exist.

### Task 2: Copied Trainer And Dataset Adapter

**Files:**
- Create: `examples/customized_trainer.py`

- [ ] **Step 1: Copy `examples/simple_trainer.py`**

Use `cp examples/simple_trainer.py examples/customized_trainer.py` as a mechanical copy before patching behavior.

- [ ] **Step 2: Add `MaskedObjectDataset`**

Subclass/import the stock COLMAP `Dataset`, load RGB from `masked_images/<stem>.png`, load mask from `masks/<stem>.png`, resize both to the parser image size, crop both when `patch_size` is enabled, and return `data["mask"]`.

- [ ] **Step 3: Add customized config defaults**

Rename the dataclass to `CustomizedConfig`, set the couch data/result defaults, and disable exposure/viewer by default.

- [ ] **Step 4: Wire `Runner` COLMAP path**

Use the stock `Parser`, but instantiate `MaskedObjectDataset` for train and val when `data_type == "colmap"`.

### Task 3: Verification

**Files:**
- Modify: `tests/test_customized_trainer.py`
- Modify: `examples/customized_trainer.py`

- [ ] **Step 1: Run focused tests**

Run: `source /home/ooofieee/anaconda3/etc/profile.d/conda.sh && conda activate gsplat && PYTHONPATH=.:examples pytest tests/test_customized_trainer.py -q`

Expected: PASS.

- [ ] **Step 2: Run syntax/import check**

Run: `source /home/ooofieee/anaconda3/etc/profile.d/conda.sh && conda activate gsplat && PYTHONPATH=.:examples python -m py_compile examples/customized_trainer.py`

Expected: PASS.
