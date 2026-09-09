import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from etsr import runner, workflows
from etsr.config import load_config, save_config
from etsr.data.common import DatasetBundle
from etsr.training.gates import overfit_gate


def healthy_row(**overrides):
    return {
        "train_loss": 1.0, "validation_loss": 1.0, "validation_accuracy": 1.0,
        "gradient_norm_mean": 0.5, "gradient_nonfinite_fraction": 0.0,
        "amp_overflow_fraction": 0.0, **overrides,
    }


def test_overfit_gate_requires_stable_final_performance_and_finite_history():
    rows = [healthy_row() for _ in range(5)]
    assert overfit_gate(rows)["passed"]
    assert not overfit_gate(rows[:4])["passed"]
    assert not overfit_gate([*rows, healthy_row(validation_accuracy=0.5)])["passed"]
    assert not overfit_gate([healthy_row(gradient_norm_mean=float("nan")), *rows])["passed"]
    assert not overfit_gate([healthy_row(amp_overflow_fraction=0.1), *rows])["passed"]


@pytest.mark.parametrize("passed", [False, True])
def test_candidate_failure_blocks_full_run_and_success_profiles_fresh_config(tmp_path, monkeypatch, passed):
    config = load_config("configs/dvslip_gated_v2.yaml")
    original = copy.deepcopy(config)
    calls = []

    def train(candidate):
        calls.append(copy.deepcopy(candidate))
        directory = tmp_path / str(len(calls))
        directory.mkdir()
        save_config(candidate, directory / "config_resolved.yaml")
        # Simulate runner mutation to detect accidental reuse of diagnostic training settings.
        candidate["augmentation"]["horizontal_flip_probability"] = 0.0
        return {"artifact_dir": str(directory), "checkpoint": str(directory / "best.pt"),
                "overfit_gate": {"passed": passed}}

    profiles = []
    monkeypatch.setattr(workflows, "train_experiment", train)
    monkeypatch.setattr(workflows, "profile_checkpoint", lambda *args: profiles.append(args))
    if passed:
        result = workflows.run_candidate(config)
        assert result["status"] == "complete"
        assert len(calls) == 2 and len(profiles) == 1
        assert calls[1] == original
        assert profiles[0][0]["training"] == original["training"]
    else:
        with pytest.raises(RuntimeError, match="no full training"):
            workflows.run_candidate(config)
        assert len(calls) == 1 and not profiles
    assert config == original
    assert calls[0]["training"]["overfit"]["stop_on_pass"]


def test_candidate_rejects_simultaneous_training_recipe_change():
    config = load_config("configs/dvslip_f.yaml")
    config["training"]["gradient_clip_norm"] = 2.0
    with pytest.raises(ValueError, match="preserve baseline training"):
        workflows.run_candidate(config)


def test_candidate_accepts_only_the_registered_phase_representation_change(monkeypatch):
    config = load_config("configs/dvslip_b_phase_e1.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["model"]["embed_dim"] = 192
    with pytest.raises(ValueError, match="may change only phase encoding"):
        workflows.run_candidate(config)


@pytest.mark.parametrize("filename", ["dvslip_f_tbr.yaml", "dvslip_f_spike_tbr_lif.yaml"])
def test_candidate_accepts_only_preregistered_tbr_settings(filename, monkeypatch):
    config = load_config(f"configs/{filename}")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["representation"]["bits"] = 4
    config["representation"]["micro_bin_width_us"] = 12_500
    with pytest.raises(ValueError, match="fixed to the preregistered"):
        workflows.run_candidate(config)


def test_candidate_accepts_only_preregistered_multigranular_lite(monkeypatch):
    config = load_config("configs/dvslip_f_multigranular_lite.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["model"]["multigranular_fine_channels"] = 32
    with pytest.raises(ValueError, match="fixed to its preregistered design"):
        workflows.run_candidate(config)


def test_runner_overfit_early_stops_and_records_actual_subset(tmp_path, monkeypatch):
    frames = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    targets = torch.tensor([0, 0, 1, 1])
    dataset = TensorDataset(frames, targets, torch.arange(4))
    dataset.targets = targets.tolist()
    bundle = DatasetBundle(dataset, dataset, None, ["a", "b"])
    monkeypatch.setattr(runner, "build_dataset_bundle", lambda config: bundle)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def model_factory(config, classes):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[8., -8.], [-8., 8.]]))
        return model

    monkeypatch.setattr(runner, "build_model", model_factory)
    config = load_config("configs/dvslip_e0.yaml")
    config["experiment"].update({"artifact_root": str(tmp_path / "artifacts"),
                                 "checkpoint_root": str(tmp_path / "checkpoints")})
    config.pop("evaluation")
    config["dataset"]["num_workers"] = 0
    config["training"]["epochs"] = 20
    config["training"]["overfit"] = {"class_count": 2, "samples_per_class": 2, "stop_on_pass": True}
    summary = runner.train_experiment(config)
    assert summary["completed_epochs"] == 5
    assert summary["overfit_gate"]["passed"]
    resolved = load_config(Path(summary["artifact_dir"]) / "config_resolved.yaml")
    assert resolved["runtime"]["overfit_train_indices"] == [0, 1, 2, 3]
    assert resolved["training"]["amp"] is False
    assert resolved["augmentation"]["horizontal_flip_probability"] == 0


def test_backfill_uses_resolved_run_config_preserves_legacy_and_reports_missing(tmp_path, monkeypatch):
    artifacts, checkpoints = tmp_path / "artifacts", tmp_path / "checkpoints"
    for run in ("existing", "missing", "overfit"):
        directory = artifacts / run
        directory.mkdir(parents=True)
        config = load_config("configs/dvslip_gated_v2.yaml")
        if run == "overfit":
            config["training"]["overfit"] = {"class_count": 16, "samples_per_class": 4}
        save_config(config, directory / "config_resolved.yaml")
        (directory / "summary.json").write_text("{}")
        (directory / "hardware_profile.json").write_text("legacy")
    checkpoint = checkpoints / "existing" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    calls = []

    def profile(config, path, output, **kwargs):
        calls.append((config, path, output))
        return {"checkpoint_sha256": "hash", "sampling": {}}

    monkeypatch.setattr(workflows, "profile_checkpoint", profile)
    report = workflows.profile_completed_runs(artifacts, checkpoints)
    assert not report["complete"]
    assert [row["status"] for row in report["runs"]] == ["complete", "missing_checkpoint"]
    assert calls[0][0]["model"]["gated_initial_memory_steps"] == 20
    assert calls[0][1] == checkpoint
    assert calls[0][2].name == "hardware_profile_v4.json"
    assert (artifacts / "existing" / "hardware_profile.json").read_text() == "legacy"


def test_checkpoint_profile_rejects_resolved_config_from_another_run(tmp_path, monkeypatch):
    config = load_config("configs/dvslip_e0.yaml")
    config["runtime"] = {"run_id": "requested", "classes": ["a", "b"]}
    saved_config = copy.deepcopy(config)
    saved_config["runtime"]["run_id"] = "different"
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {}, "epoch": 1, "score": 0.5,
                "config": saved_config, "num_classes": 2}, checkpoint)
    monkeypatch.setattr(runner, "build_dataset_bundle", lambda cfg: DatasetBundle(
        [], [], None, ["a", "b"]))
    with pytest.raises(ValueError, match="different runs"):
        runner.profile_checkpoint(config, checkpoint, tmp_path / "profile.json")


def test_smilies_foreground_forwards_gpu_and_arguments_through_cleanenv(tmp_path):
    fake = tmp_path / "singularity"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/smilies/run_command.sh"
    result = subprocess.run(
        ["bash", str(script), "--foreground", "candidate", "--config", "config with spaces.yaml"],
        env={**os.environ, "SINGULARITY": str(fake), "CUDA_VISIBLE_DEVICES": "GPU-example"},
        capture_output=True, text=True, check=True,
    )
    args = result.stdout.splitlines()
    assert "--cleanenv" in args
    assert "CUDA_VISIBLE_DEVICES=GPU-example" in args
    assert args[-4:] == ["etsr.cli", "candidate", "--config", "config with spaces.yaml"]
    assert args.index("CUDA_VISIBLE_DEVICES=GPU-example") > args.index("env")
    json.dumps(args)
