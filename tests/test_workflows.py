import copy
import csv
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
from etsr.models.factory import build_model
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.training.gates import overfit_gate
from etsr.utils.io import append_csv


def healthy_row(**overrides):
    return {
        "train_loss": 1.0, "validation_loss": 1.0, "validation_accuracy": 1.0,
        "gradient_norm_mean": 0.5, "gradient_nonfinite_fraction": 0.0,
        "amp_overflow_fraction": 0.0, **overrides,
    }


def test_append_csv_aligns_reordered_rows_and_expands_the_schema(tmp_path):
    output = tmp_path / "history.csv"
    append_csv({"epoch": 1, "first": 10, "second": 20}, output)
    append_csv({"epoch": 2, "second": 200, "first": 100}, output)
    append_csv({"epoch": 3, "first": 1000, "third": 3000}, output)

    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["epoch", "first", "second", "third"]
    assert rows == [
        {"epoch": "1", "first": "10", "second": "20", "third": ""},
        {"epoch": "2", "first": "100", "second": "200", "third": ""},
        {"epoch": "3", "first": "1000", "second": "", "third": "3000"},
    ]


def test_delay_trajectory_records_continuous_motion_before_hard_change():
    mixer = CausalTemporalChannelMixer(2, delays=(1, 2, 4, 8), learnable_delays=True)
    with torch.no_grad():
        mixer.delay_centers[1].copy_(torch.tensor([2.2, 2.8]))
    rows = runner._delay_trajectory_rows({"test_mixer": mixer}, epoch=7)
    assert len(rows) == 4
    assert rows[1]["mean_center"] == pytest.approx(2.5)
    assert rows[1]["mean_absolute_shift_bins"] == pytest.approx(0.5)
    assert rows[1]["fraction_hard_changed"] == pytest.approx(0.5)
    assert rows[1]["module"] == "test_mixer"


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


def test_candidate_accepts_only_preregistered_multigranular_capacity(monkeypatch):
    config = load_config("configs/dvslip_f_multigranular_capacity.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["representation"]["fine_spatial_stride"] = 8
    with pytest.raises(ValueError, match="fixed to its preregistered design"):
        workflows.run_candidate(config)


def test_candidate_accepts_preregistered_multigranular_temporal_capacity(monkeypatch):
    config = load_config("configs/dvslip_f_multigranular_temporal_capacity.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["model"]["temporal_channel_mixer_delays"] = [1, 3, 5]
    with pytest.raises(ValueError, match="fixed to its preregistered design"):
        workflows.run_candidate(config)


def test_candidate_accepts_only_preregistered_stage1_depthwise_probe(monkeypatch):
    config = load_config("configs/dvslip_f_tcap_stage1_dwc3.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["model"]["stage1_depthwise_kernel_size"] = 5
    with pytest.raises(ValueError, match="fixed to stage1 depthwise convolution 3x3"):
        workflows.run_candidate(config)


def test_candidate_accepts_only_preregistered_tcap_d8_probe(monkeypatch):
    config = load_config("configs/dvslip_f_temporal_capacity_d8.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["model"]["temporal_channel_mixer_delays"] = [1, 2, 4, 8, 16]
    with pytest.raises(ValueError, match="may add only delay 8"):
        workflows.run_candidate(config)


def test_candidate_accepts_only_registered_dwc3_d8_combination(monkeypatch):
    config = load_config("configs/dvslip_f_tcap_stage1_dwc3_d8.yaml")
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("gate reached")),
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)

    config["model"]["stage1_depthwise_kernel_size"] = 5
    with pytest.raises(ValueError, match="combine only the two registered changes"):
        workflows.run_candidate(config)


def test_candidate_learnable_delays_changes_only_delay_locations(monkeypatch):
    config = load_config("configs/dvslip_f_tcap_stage1_dwc3_learnable_delays.yaml")
    calls = []

    def stop_at_gate(gate):
        calls.append(gate)
        raise RuntimeError("gate reached")

    monkeypatch.setattr(
        workflows,
        "train_experiment",
        stop_at_gate,
    )
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)
    assert calls[0]["training"]["epochs"] == 500
    assert calls[0]["training"]["delay_anneal_epochs"] == 128

    config["model"]["temporal_channel_mixer_delays"] = [1, 2, 4, 9]
    with pytest.raises(ValueError, match="learn only the four existing TCAP taps"):
        workflows.run_candidate(config)


def test_candidate_accepts_only_preregistered_dvsgesture_transfer(monkeypatch):
    config = load_config("configs/dvsgesture_f_tcap_stage1_dwc3_d8.yaml")
    calls = []

    def stop_at_gate(candidate):
        calls.append(candidate)
        raise RuntimeError("gate reached")

    monkeypatch.setattr(workflows, "train_experiment", stop_at_gate)
    with pytest.raises(RuntimeError, match="gate reached"):
        workflows.run_candidate(config)
    assert calls[0]["training"]["overfit"]["class_count"] == 11

    config["model"]["temporal_channel_mixer_delays"] = [1, 2, 4]
    with pytest.raises(ValueError, match=r"fixed to the frozen DWC3\+TCAP-d8 topology"):
        workflows.run_candidate(config)


def test_replication_changes_only_seed_and_profiles_best(tmp_path, monkeypatch):
    config = load_config("configs/dvslip_f_temporal_capacity.yaml")
    original = copy.deepcopy(config)
    calls = []

    def train(replication):
        calls.append(copy.deepcopy(replication))
        directory = tmp_path / "replication"
        directory.mkdir()
        save_config(replication, directory / "config_resolved.yaml")
        return {"artifact_dir": str(directory), "checkpoint": str(tmp_path / "best.pt")}

    profiles = []
    monkeypatch.setattr(workflows, "train_experiment", train)
    monkeypatch.setattr(workflows, "profile_checkpoint", lambda *args: profiles.append(args))

    result = workflows.run_replication(config, 43)

    assert config == original
    assert calls[0] == {**original, "experiment": {**original["experiment"], "seed": 43}}
    assert len(profiles) == 1
    assert result["status"] == "complete"
    assert result["seed"] == 43


@pytest.mark.parametrize(
    "config_path,expected_family",
    [
        (
            "configs/dvslip_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml",
            "temporal_mask",
        ),
        (
            "configs/dvslip_f_tcap_stage1_dwc3_d8_spatial_erasing.yaml",
            "spatial_erasing",
        ),
        (
            "configs/dvsgesture_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml",
            "temporal_mask",
        ),
        (
            "configs/dvsgesture_f_tcap_stage1_dwc3_d8_event_mix.yaml",
            "event_mix",
        ),
    ],
)
def test_supervised_refinement_accepts_single_augmentation_configs(
    tmp_path, monkeypatch, config_path, expected_family
):
    config = load_config(config_path)
    reference_name = load_config(config["refinement"]["reference_config"])["experiment"]["name"]
    original = copy.deepcopy(config)
    calls = []

    def train(refinement):
        calls.append(copy.deepcopy(refinement))
        directory = tmp_path / expected_family
        directory.mkdir()
        save_config(refinement, directory / "config_resolved.yaml")
        return {"artifact_dir": str(directory), "checkpoint": str(directory / "best.pt")}

    profiles = []
    monkeypatch.setattr(workflows, "train_experiment", train)
    monkeypatch.setattr(workflows, "profile_checkpoint", lambda *args: profiles.append(args))

    result = workflows.run_supervised_refinement(config)

    assert config == original
    assert calls == [original]
    assert len(profiles) == 1
    assert result["reference_experiment"] == reference_name
    assert result["refinement"] == original["refinement"]
    assert result["changed_augmentation_families"] == (expected_family,)
    assert result["status"] == "complete"

    config["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="recipe identifier"):
        workflows.run_supervised_refinement(config)

    if expected_family == "temporal_mask":
        two_families = copy.deepcopy(original)
        two_families["augmentation"].update(
            {"spatial_erasing_count": 1, "spatial_erasing_max_pixels": 8}
        )
        with pytest.raises(ValueError, match="exactly one family"):
            workflows.run_supervised_refinement(two_families)


def test_supervised_refinement_workflow_is_dataset_independent(tmp_path, monkeypatch):
    config = load_config("configs/dvsgesture_f_tcap_stage1_dwc3_d8.yaml")
    config["experiment"]["name"] += "_temporal_maskout"
    config["training"]["recipe_id"] += "_tm4x5"
    config["augmentation"].update(
        {"temporal_mask_count": 4, "temporal_mask_max_steps": 5}
    )
    config["refinement"] = {
        "kind": "augmentation",
        "stage": "single",
        "reference_config": "configs/dvsgesture_f_tcap_stage1_dwc3_d8.yaml",
    }

    def train(refinement):
        directory = tmp_path / "dvsgesture_refinement"
        directory.mkdir()
        save_config(refinement, directory / "config_resolved.yaml")
        return {"artifact_dir": str(directory), "checkpoint": str(directory / "best.pt")}

    monkeypatch.setattr(workflows, "train_experiment", train)
    monkeypatch.setattr(workflows, "profile_checkpoint", lambda *_args: None)

    result = workflows.run_supervised_refinement(config)

    assert result["reference_experiment"] == "dvsgesture_f_tcap_stage1_dwc3_d8"
    assert result["changed_augmentation_families"] == ("temporal_mask",)


def test_supervised_refinement_accepts_only_declared_multi_family_combinations(monkeypatch):
    config = load_config("configs/dvslip_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml")
    config["experiment"]["name"] = "dvslip_f_tcap_stage1_dwc3_d8_tm8x4_se4x20"
    config["training"]["recipe_id"] = "dvslip_e0_128_tm8x4_se4x20"
    config["augmentation"].update(
        {"spatial_erasing_count": 4, "spatial_erasing_max_pixels": 20}
    )
    config["refinement"]["stage"] = "combination"
    monkeypatch.setattr(
        workflows,
        "train_experiment",
        lambda _config: (_ for _ in ()).throw(RuntimeError("training reached")),
    )

    with pytest.raises(RuntimeError, match="training reached"):
        workflows.run_supervised_refinement(config)

    config["refinement"]["stage"] = "single"
    with pytest.raises(ValueError, match="exactly one family"):
        workflows.run_supervised_refinement(config)

    config["refinement"]["stage"] = "combination"
    config["augmentation"]["horizontal_flip_probability"] = 0.0
    with pytest.raises(ValueError, match="exactly two families"):
        workflows.run_supervised_refinement(config)


def _failing_gate(tmp_path, calls):
    def train(candidate):
        calls.append(copy.deepcopy(candidate))
        directory = tmp_path / "gate"
        directory.mkdir(exist_ok=True)
        return {
            "artifact_dir": str(directory),
            "checkpoint": str(directory / "best.pt"),
            "overfit_gate": {"passed": False},
        }

    return train


def _bypass_audit_and_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "etsr.evaluation.predictive_diagnostic.run_predictive_preflight",
        lambda _config, _output: {"passed": True},
    )
    monkeypatch.setattr(
        workflows,
        "validate_predictive_training_authorization",
        lambda _predictive, _continuation: tmp_path / "audit.json",
    )


def test_predictive_continuation_failure_blocks_the_full_run(tmp_path, monkeypatch):
    config = load_config("configs/dvslip_predictive_r0.yaml")
    calls = []
    _bypass_audit_and_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(workflows, "train_experiment", _failing_gate(tmp_path, calls))
    with pytest.raises(RuntimeError, match="no full training"):
        workflows.run_predictive_continuation(config)
    assert len(calls) == 1
    assert calls[0]["training"]["overfit"]["stop_on_pass"]
    assert calls[0]["training"]["epochs"] == 50


def test_predictive_scratch_failure_blocks_the_full_run(tmp_path, monkeypatch):
    config = load_config("configs/dvslip_predictive_s1.yaml")
    calls = []
    _bypass_audit_and_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(workflows, "train_experiment", _failing_gate(tmp_path, calls))
    with pytest.raises(RuntimeError, match="no full training"):
        workflows.run_predictive_scratch(config)
    assert len(calls) == 1
    gate = calls[0]
    assert gate["training"]["epochs"] == 500
    assert gate["training"]["recipe_id"] == "dvslip_e0_128_overfit"
    assert gate["training"]["overfit"] == {
        "class_count": 16,
        "samples_per_class": 4,
        "stop_on_pass": True,
    }
    assert "continuation" not in gate


def test_predictive_continuation_rejects_recipe_and_representation_drift():
    config = load_config("configs/dvslip_predictive_r0.yaml")
    config["training"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="canonical training"):
        workflows.run_predictive_continuation(config)

    config = load_config("configs/dvslip_predictive_r0.yaml")
    config["representation"]["bin_width_us"] = 25_000
    with pytest.raises(ValueError, match="preregistered representation"):
        workflows.run_predictive_continuation(config)

    config = load_config("configs/dvslip_predictive_late_prefix.yaml")
    config["continuation"]["new_parameter_learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="canonical new_parameter_learning_rate"):
        workflows.run_predictive_continuation(config)

    config = load_config("configs/dvslip_predictive_s0.yaml")
    with pytest.raises(ValueError, match="continuation and predictive sections"):
        workflows.run_predictive_continuation(config)


def test_predictive_scratch_rejects_recipe_topology_and_regime_drift():
    config = load_config("configs/dvslip_predictive_s0.yaml")
    config["training"]["epochs"] = 64
    with pytest.raises(ValueError, match="frozen C0 training"):
        workflows.run_predictive_scratch(config)

    config = load_config("configs/dvslip_predictive_dynamic_tcap.yaml")
    config["model"]["temporal_channel_mixer_delays"] = [1, 2, 4]
    with pytest.raises(ValueError, match="only registered phase fields"):
        workflows.run_predictive_scratch(config)

    config = load_config("configs/dvslip_predictive_late_prefix.yaml")
    with pytest.raises(ValueError, match="must not declare a continuation"):
        workflows.run_predictive_scratch(config)

    config = load_config("configs/dvslip_predictive_s1.yaml")
    config["predictive"]["phase1_audit_report"] = "artifacts/other_audit.json"
    with pytest.raises(ValueError, match="canonical phase-1 audit report"):
        workflows.run_predictive_scratch(config)


def test_direct_training_cannot_bypass_predictive_phase1_gates(tmp_path):
    blocked = load_config("configs/dvslip_predictive_fine_future.yaml")
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        runner.train_experiment(blocked)

    for name in ("r0", "s0", "dynamic_tcap"):
        missing_audit = load_config(f"configs/dvslip_predictive_{name}.yaml")
        missing_audit["predictive"]["phase1_audit_report"] = str(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError, match="has not been produced"):
            runner.train_experiment(missing_audit)


def _write_phase1_audit(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "sections": [
                    "A1_gradient_authority",
                    "A2_discriminative_probes",
                    "A3_representation_movement",
                    "A4_tail_margin",
                ],
                "official_test_used": False,
                "checkpoints": {"c0": {"sha256": "0" * 64}},
            }
        )
    )
    return path


def test_runner_scratch_branch_calibrates_and_persists_auxiliary_authority(tmp_path, monkeypatch):
    torch.manual_seed(0)
    targets = torch.tensor([0, 0, 1, 1])
    dataset = TensorDataset(torch.rand(4, 6, 2, 32, 32), targets, torch.arange(4))
    dataset.targets = targets.tolist()
    bundle = DatasetBundle(dataset, dataset, None, ["a", "b"])
    monkeypatch.setattr(runner, "build_dataset_bundle", lambda config: bundle)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = load_config("configs/dvslip_predictive_s0.yaml")
    config["predictive"]["phase1_audit_report"] = str(_write_phase1_audit(tmp_path / "audit.json"))
    config["predictive"]["objective"]["ramp_epochs"] = 2
    config["experiment"].update(
        {
            "artifact_root": str(tmp_path / "artifacts"),
            "checkpoint_root": str(tmp_path / "checkpoints"),
        }
    )
    config.pop("evaluation")
    config["dataset"].update({"num_workers": 0, "batch_size": 2})
    config["training"].update({"epochs": 3, "warmup_epochs": 1})
    config["training"]["overfit"] = {
        "class_count": 2,
        "samples_per_class": 2,
        "stop_on_pass": False,
    }

    summary = runner.train_experiment(config)

    predictive = summary["predictive"]
    assert predictive["regime"] == "from_scratch"
    assert predictive["backbone_initialization"] == "c0_topology_same_seed"
    calibration = predictive["authority_calibration"]
    assert [record["epoch"] for record in calibration] == [0, 2, 3]
    assert calibration[0]["shared_ratio"] == pytest.approx(0.25, rel=1e-6)
    assert predictive["final_nominal_weight"] == calibration[-1]["weight"]

    artifact_dir = Path(summary["artifact_dir"])
    with (artifact_dir / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [row["selection_eligible"] for row in history] == ["False", "True", "True"]
    assert float(history[0]["authority_shared_ratio"]) == pytest.approx(0.25, rel=1e-6)
    assert all(row["auxiliary_nominal_weight"] for row in history)
    assert all(row["train_temporal_variation_stage1_active"] for row in history)
    assert summary["best_epoch"] >= 2

    state = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=False)
    assert state["objective_state"]["weight"] == predictive["final_nominal_weight"]
    deployment = load_config(artifact_dir / "deployment_config_resolved.yaml")
    assert "predictive" not in deployment and "continuation" not in deployment
    assert "temporal_channel_mixer_predictive_auxiliary" not in deployment["model"]
    assert "temporal_channel_mixer_predictive_stages" not in deployment["model"]
    # The profiler rebuilds the deployed model from these two files: they must match strictly.
    deployed = torch.load(summary["deployment_checkpoint"], map_location="cpu", weights_only=False)
    build_model(deployment["model"], 2).load_state_dict(deployed["model"])


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
    assert summary["late_window"]["declared_size"] == 16
    assert summary["late_window"]["observed_size"] == 5
    assert summary["late_window"]["first_epoch"] == 1
    assert summary["late_window"]["last_epoch"] == 5
    with (Path(summary["artifact_dir"]) / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [row["selection_eligible"] for row in history] == ["True"] * 5


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
