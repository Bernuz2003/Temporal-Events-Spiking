from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from etsr.evaluation import innovation_probe as probe


def test_structured_controls_match_initial_graph_and_parameter_budget():
    models = probe._initial_probes(100, 42)
    counts = probe._parameter_counts(models)
    assert {counts[name]["total"] for name in probe.PRIMARY_MODES} == {255_716}
    assert counts["b"]["classification_trainable"] == 255_716
    assert counts["c"]["classification_trainable"] == 91_492
    assert counts["b_rand"]["classification_trainable"] == 91_492
    for key, value in models["b"].state_dict().items():
        torch.testing.assert_close(value, models["c"].state_dict()[key])
        torch.testing.assert_close(value, models["b_rand"].state_dict()[key])

    z = torch.randn(2, 40, 128)
    with torch.no_grad():
        torch.testing.assert_close(models["b"](z), models["c"](z))
        torch.testing.assert_close(models["b"](z), models["b_rand"](z))


def test_history_is_causal_and_anchor_matches_mean_readout():
    z = torch.arange(10.0).reshape(1, 10, 1)
    history = probe.causal_history(z)
    assert history[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert history[0, 8].tolist() == [7.0, 6.0, 4.0, 0.0]
    changed = z.clone()
    changed[:, 7:] += 100
    torch.testing.assert_close(probe.causal_history(z)[:, :7], probe.causal_history(changed)[:, :7])

    head = nn.Linear(128, 100)
    states = torch.randn(3, 40, 128)
    torch.testing.assert_close(head(states).mean(1), head(states.mean(1)))


def test_gate_uses_seed_variation_and_sample_bootstrap_is_conditional():
    assert probe._seed_gate([0.02, 0.03, 0.025, 0.02, 0.03])["pass"]
    assert not probe._seed_gate([0.05, 0.05, 0.05, 0.05, -0.05])["pass"]
    assert not probe._seed_gate([0.001, 0.001, 0.001, 0.001, -0.01])["pass"]
    target = np.array([0, 0, 1, 1])
    left = np.array([0, 0, 1, 1])
    right = np.array([1, 0, 0, 1])
    result = probe._paired_sample_bootstrap(target, left, right, 2, 42, repetitions=20)
    assert result["conditional_on_fitted_seed_pair"]
    assert result["ci95_low"] >= 0
    assert probe._mcnemar_exact(target, left, right)["two_sided_p"] == pytest.approx(0.5)


def test_q_skill_detects_a_constant_prediction_even_with_nonzero_rms():
    class ConstantPredictor(nn.Module):
        def forward(self, history):
            return torch.ones(*history.shape[:-1], 128, device=history.device)

    z = torch.randn(3, 40, 128)
    skill = probe._q_skill(ConstantPredictor(), z, torch.full((3,), 25), torch.device("cpu"))
    assert skill["active"]["prediction_rms_over_target_rms"] > 0
    assert skill["active"]["prediction_std_over_target_std"] == pytest.approx(0.0, abs=1e-7)


def test_fit_and_validation_are_separate_and_validation_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "Q_EPOCHS", 1)
    monkeypatch.setattr(probe, "HEAD_EPOCHS", 1)
    monkeypatch.setattr(probe, "BATCH_SIZE", 32)
    monkeypatch.setattr(probe, "_paired_sample_bootstrap", lambda *args, **kwargs: {"ci95_low": 0.0})
    monkeypatch.setattr(probe, "_group_summary", lambda *args: {})
    monkeypatch.setattr(
        probe,
        "_q_skill",
        lambda *args: {
            "active": {
                "skill_vs_persistence": 0.2,
                "skill_vs_recent_mean": 0.2,
                "prediction_rms_over_target_rms": 0.5,
                "prediction_std_over_target_std": 0.5,
            }
        },
    )

    class FrozenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(128, 100)

    model = FrozenModel()
    monkeypatch.setattr(probe, "build_model", lambda *args: model)
    monkeypatch.setattr(
        probe,
        "load_model_state",
        lambda *args: {
            "epoch": 1,
            "score": 0.1,
            "config": {"experiment": {"name": "dvslip_predictive_dynamic_tcap", "seed": 42}},
        },
    )
    bundle = SimpleNamespace(train=object(), validation=object(), classes=[str(i) for i in range(100)])
    monkeypatch.setattr(probe, "build_dataset_bundle", lambda config: bundle)
    reads = []

    def features(_model, partition, _config, _device):
        reads.append(partition)
        generator = torch.Generator().manual_seed(1 if partition is bundle.train else 2)
        n = 12 if partition is bundle.train else 6
        return {
            "z": torch.randn(n, 40, 128, generator=generator),
            "target": torch.arange(n) % 3,
            "index": torch.arange(n),
            "endpoint": torch.full((n,), 25),
        }

    monkeypatch.setattr(probe, "_features_for_partition", features)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic")
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    config = {
        "experiment": {"name": "dvslip_predictive_dynamic_tcap", "seed": 42},
        "dataset": {"name": "dvslip", "split_manifest": str(split)},
        "model": {
            "temporal_channel_mixer_dynamic_routing": True,
            "temporal_channel_mixer_routing_stages": [2],
            "temporal_channel_mixer_routing_parameterization": "amplitude_allocation",
        },
        "augmentation": {},
    }
    fitted = probe.fit_innovation_probe(config, checkpoint, tmp_path / "fit")
    assert fitted["development_validation_opened"] is False
    assert reads == [bundle.train]
    evaluated = probe.evaluate_innovation_probe(
        config, checkpoint, tmp_path / "fit", tmp_path / "evaluation"
    )
    assert evaluated["validation_read_count"] == 1
    assert reads == [bundle.train, bundle.validation]
    with pytest.raises(FileExistsError):
        probe.evaluate_innovation_probe(config, checkpoint, tmp_path / "fit", tmp_path / "evaluation")
