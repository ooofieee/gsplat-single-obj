from pathlib import Path
from types import SimpleNamespace

import torch


DATA_DIR = Path("/home/ooofieee/co3d_data/couch_colmap_rgbmask")


def test_customized_config_defaults_to_couch_masked_scene():
    from examples.customized_trainer import CustomizedConfig

    cfg = CustomizedConfig()

    assert cfg.data_dir == str(DATA_DIR)
    assert cfg.data_factor == 4
    assert cfg.result_dir == "results/couch_customized"
    assert cfg.disable_viewer is True
    assert cfg.load_exposure is False
    assert cfg.save_ply is True


def test_masked_object_dataset_loads_masked_rgb_and_foreground_mask():
    from datasets.colmap import Parser
    from examples.customized_trainer import MaskedObjectDataset

    parser = Parser(
        data_dir=str(DATA_DIR),
        factor=4,
        normalize=True,
        test_every=8,
        load_exposure=False,
    )
    dataset = MaskedObjectDataset(parser, split="train")

    sample = dataset[0]

    image = sample["image"]
    mask = sample["mask"]

    assert image.shape[:2] == mask.shape
    assert image.shape[-1] == 3
    assert mask.dtype == torch.bool
    assert mask.any()
    assert (~mask).any()
    assert torch.count_nonzero(image[~mask]) == 0


def test_erode_densify_mask_falls_back_when_erosion_clears_mask():
    from examples.customized_trainer import Runner

    runner = Runner.__new__(Runner)
    runner.device = torch.device("cpu")

    mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    mask[:, 2, 2] = True

    eroded = runner._erode_densify_mask(mask, erode_px=2)

    assert torch.equal(eroded, mask)


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
