from __future__ import annotations

import numpy as np
import pytest
import torch

from etsr.evaluation.predictive_phase1_audit import _linear_cka, _region_pool, _ridge_probe


def test_region_pool_separates_sample_specific_active_and_tail_steps():
    sequence = torch.tensor(
        [
            [[[1.0]], [[10.0]]],
            [[[3.0]], [[20.0]]],
            [[[5.0]], [[30.0]]],
            [[[7.0]], [[40.0]]],
        ]
    ).unsqueeze(2)
    endpoints = torch.tensor([2, 3])

    active = _region_pool(sequence, endpoints, "active")
    tail = _region_pool(sequence, endpoints, "tail")

    torch.testing.assert_close(active[:, 0], torch.tensor([2.0, 20.0]))
    torch.testing.assert_close(tail[:, 0], torch.tensor([6.0, 40.0]))


def test_ridge_probe_and_cka_recover_clear_linear_information():
    train_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    train_targets = np.asarray([0, 0, 1, 1])
    holdout_features = np.asarray([[-1.5], [1.5]])
    holdout_targets = np.asarray([0, 1])

    result = _ridge_probe(
        train_features,
        train_targets,
        holdout_features,
        holdout_targets,
        num_classes=2,
        ridge=1e-3,
    )

    assert result["accuracy"] == 1.0
    assert _linear_cka(torch.tensor(train_features), torch.tensor(3 * train_features)) == pytest.approx(
        1.0
    )
