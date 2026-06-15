import torch

from scripts.train_poster_gaussian import build_splat_filter


def test_build_splat_filter_removes_dark_low_alpha_and_far_outliers():
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]
    )
    colors = torch.tensor(
        [
            [0.4, 0.4, 0.4],
            [0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.01, 0.01, 0.01],
            [0.7, 0.7, 0.7],
        ]
    )
    opacities = torch.logit(torch.tensor([0.4, 0.2, 0.03, 0.9, 0.8, 0.9]))

    mask = build_splat_filter(
        means,
        colors,
        opacities,
        min_brightness=0.1,
        min_alpha=0.08,
        max_radius=1.0,
    )

    assert mask.tolist() == [True, True, False, True, False, False]
