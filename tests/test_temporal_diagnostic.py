import pytest
import torch
from torch import nn

from etsr.evaluation.temporal_diagnostic import (
    deterministic_prefix_sum,
    gather_temporal_values,
    temporal_readout_logits,
)


def test_prefix_sum_uses_fixed_order_and_matches_reference():
    values = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])

    result = deterministic_prefix_sum(values)

    assert torch.equal(result, torch.tensor([[1.0, 2.0], [4.0, 1.0], [4.5, 5.0]]))
    with pytest.raises(ValueError, match="non-empty time"):
        deterministic_prefix_sum(torch.empty(0, 2))


def test_temporal_readout_decomposes_prefix_normalization_from_tail_activity():
    encoded = torch.tensor([1.0, 3.0, 0.0, 4.0]).reshape(4, 1, 1, 1, 1)
    head = nn.Linear(1, 1)
    with torch.no_grad():
        head.weight.fill_(2.0)
        head.bias.fill_(0.5)

    logits = temporal_readout_logits(encoded, head)

    assert logits["prefix_mean"].flatten().tolist() == pytest.approx([2.5, 4.5, 3.1666667, 4.5])
    assert logits["fixed_horizon_denominator"].flatten().tolist() == pytest.approx(
        [1.0, 2.5, 2.5, 4.5]
    )
    assert torch.equal(logits["prefix_mean"][-1], logits["fixed_horizon_denominator"][-1])


def test_gather_temporal_values_selects_one_causal_step_per_sample():
    values = torch.tensor(
        [
            [[10.0, 11.0], [20.0, 21.0]],
            [[12.0, 13.0], [22.0, 23.0]],
            [[14.0, 15.0], [24.0, 25.0]],
        ]
    )

    selected = gather_temporal_values(values, torch.tensor([1, 3]))

    assert selected.tolist() == [[10.0, 11.0], [24.0, 25.0]]
    with pytest.raises(ValueError, match="fit the time axis"):
        gather_temporal_values(values, torch.tensor([0, 2]))
