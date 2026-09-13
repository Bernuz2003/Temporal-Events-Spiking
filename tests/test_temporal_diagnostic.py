import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from etsr import runner
from etsr.config import load_config
from etsr.data.common import DatasetBundle
from etsr.evaluation.temporal_diagnostic import (
    _set_disabled_tcap_delays,
    _tcap_modules,
    deterministic_prefix_sum,
    diagnose_tcap_taps,
    gather_temporal_values,
    temporal_readout_logits,
)
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.temporal import CausalTemporalChannelMixer


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


def test_tcap_delay_intervention_zeros_full_matrices_and_restores_weights():
    model = nn.Sequential(
        CausalTemporalChannelMixer(2, delays=(1, 2, 4)),
        CausalTemporalChannelMixer(2, delays=(1, 2, 4)),
    )
    modules = _tcap_modules(model)
    with torch.no_grad():
        for index, (_, module) in enumerate(modules, start=1):
            module.weight.copy_(
                torch.arange(module.weight.numel()).reshape_as(module.weight) + index
            )
    originals = {name: module.weight.detach().clone() for name, module in modules}

    _set_disabled_tcap_delays(modules, originals, (2,))
    for name, module in modules:
        assert torch.count_nonzero(module.weight[1]) == 0
        assert torch.equal(module.weight[0], originals[name][0])
        assert torch.equal(module.weight[2], originals[name][2])

    _set_disabled_tcap_delays(modules, originals, ())
    for name, module in modules:
        assert torch.equal(module.weight, originals[name])
    with pytest.raises(ValueError, match="absent from the checkpoint"):
        _set_disabled_tcap_delays(modules, originals, (8,))


def test_tcap_diagnostic_writes_paired_checkpoint_only_ablation(tmp_path, monkeypatch):
    model = MiniQKFormer(
        2,
        2,
        embed_dim=16,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2, 4),
    )
    modules = _tcap_modules(model)
    with torch.no_grad():
        for _, module in modules:
            module.weight.fill_(0.01)
    originals = {name: module.weight.detach().clone() for name, module in modules}

    frames = torch.rand(4, 4, 2, 16, 16)
    targets = torch.tensor([0, 1, 0, 1])
    indices = torch.arange(4)
    dataset = TensorDataset(frames, targets, indices)
    dataset.targets = targets.tolist()
    bundle = DatasetBundle(dataset, dataset, None, ["a", "b"])
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint-stub")
    checkpoint = {
        "epoch": 7,
        "score": 0.5,
        "num_classes": 2,
        "config": {"runtime": {"git_commit": "training-commit"}},
    }
    monkeypatch.setattr(
        runner,
        "_load_checkpoint_context",
        lambda _config, _path: (
            checkpoint_path,
            checkpoint,
            bundle,
            model,
            torch.device("cpu"),
            {"dataset_index_sha256": "index"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_dvslip_group_metrics",
        lambda _config, _classes, _confusion: {
            "metrics": {
                "Acc1": {"accuracy": 0.5, "samples": 2, "class_count": 1},
                "Acc2": {"accuracy": 0.5, "samples": 2, "class_count": 1},
            }
        },
    )
    config = load_config("configs/dvslip_f_temporal_capacity.yaml")
    config["dataset"].update({"batch_size": 2, "num_workers": 0, "pin_memory": False})
    config["representation"].update({"window_us": 200_000, "bin_width_us": 50_000})
    config["evaluation"]["absolute_prefix_times_us"] = [50_000, 100_000, 150_000, 200_000]

    summary = diagnose_tcap_taps(config, checkpoint_path, tmp_path / "diagnostic")

    assert summary["samples"] == 4
    assert summary["mixer_count"] == 2
    assert set(summary["conditions"]) == {
        "intact",
        "without_delay_1",
        "without_delay_2",
        "without_delay_4",
        "without_all_history",
    }
    assert set(summary["ablation_effects"]) == set(summary["conditions"]) - {"intact"}
    assert (tmp_path / "diagnostic" / "tcap_ablation_predictions.csv").is_file()
    assert (tmp_path / "diagnostic" / "tcap_tap_norms.csv").is_file()
    for name, module in modules:
        assert torch.equal(module.weight, originals[name])
