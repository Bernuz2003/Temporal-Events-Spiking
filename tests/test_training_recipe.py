import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from etsr.cli import build_parser
from etsr.data.common import DatasetBundle, balanced_overfit_bundle
from etsr.runner import _checkpoint_evaluation_contract, _readout_metadata
from etsr.training.checkpointing import load_training_state, save_training_state
from etsr.training.engine import (
    _objective_gradient_diagnostics,
    evaluate,
    make_optimizer,
    make_scheduler,
    train_one_epoch,
)


class _DisabledScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


class _TimeStepRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed_steps = []

    def forward(self, frames):
        self.observed_steps.append(frames.shape[1])
        return torch.zeros(frames.shape[0], 2)


class _TwoRateStepRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed_steps = []

    def forward(self, frames):
        self.observed_steps.append((frames["coarse"].shape[1], frames["fine"].shape[1]))
        return torch.zeros(frames["coarse"].shape[0], 2)


def test_nonfinite_fp32_gradient_fails_before_optimizer_mutation():
    model = nn.Linear(2, 2)
    original = model.weight.detach().clone()
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.tensor([0, 1]), torch.arange(2)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        train_one_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"),
                        _DisabledScaler(), False, 1.0, 1)
    assert torch.equal(original, model.weight)


def test_train_cli_accepts_generic_overfit_and_epoch_overrides():
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            "fixture.yaml",
            "--overfit",
            "16",
            "4",
            "--epochs",
            "50",
            "--resume",
            "last.pt",
            "--readout",
            "diagonal_gated",
            "--readout-time",
            "last_event",
            "--bin-width-us",
            "25000",
            "--no-cross-time",
            "--temporal-mask",
            "6",
            "8",
            "--spatial-erasing",
            "4",
            "20",
        ]
    )

    assert args.overfit == [16, 4]
    assert args.epochs == 50
    assert args.resume == "last.pt"
    assert args.readout == "diagonal_gated"
    assert args.readout_time == "last_event"
    assert args.bin_width_us == 25_000
    assert args.no_cross_time is True
    assert args.temporal_mask == [6, 8]
    assert args.spatial_erasing == [4, 20]


def test_refinement_cli_requires_an_explicit_config():
    args = build_parser().parse_args(
        [
            "refine",
            "--config",
            "configs/dvslip_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml",
        ]
    )

    assert args.command == "refine"
    assert args.config.endswith("dvslip_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml")


def test_readout_metadata_marks_last_event_as_a_causal_snapshot_policy():
    config = {"model": {"readout": "last", "readout_time": "last_event"}}

    assert _readout_metadata(config) == {
        "name": "last",
        "time": "last_event",
        "endpoint_knowledge": "none",
        "tail_policy": "latest_event_snapshot",
    }


def test_checkpoint_evaluation_cli_requires_explicit_inputs():
    args = build_parser().parse_args(
        [
            "evaluate-checkpoint",
            "--config",
            "configs/dvslip_e0.yaml",
            "--checkpoint",
            "checkpoints/run/best.pt",
            "--output",
            "artifacts/run/checkpoint_evaluation",
        ]
    )

    assert args.checkpoint.endswith("best.pt")
    assert args.output.endswith("checkpoint_evaluation")


def test_checkpoint_profile_cli_has_a_bounded_default_sample_count():
    args = build_parser().parse_args(
        [
            "profile-checkpoint",
            "--config",
            "configs/dvslip_e0.yaml",
            "--checkpoint",
            "checkpoints/run/best.pt",
            "--output",
            "artifacts/run/hardware_profile.json",
        ]
    )

    assert args.samples == 64


def test_temporal_diagnostic_cli_requires_selected_checkpoint_and_output():
    args = build_parser().parse_args(
        [
            "temporal-diagnostic",
            "--config",
            "artifacts/run/config_resolved.yaml",
            "--checkpoint",
            "checkpoints/run/best.pt",
            "--output",
            "artifacts/run/temporal_diagnostic",
        ]
    )

    assert args.config.endswith("config_resolved.yaml")
    assert args.checkpoint.endswith("best.pt")
    assert args.output.endswith("temporal_diagnostic")

    pair = build_parser().parse_args(
        [
            "temporal-diagnostic-pair",
            "--baseline-config",
            "artifacts/b/config_resolved.yaml",
            "--baseline-checkpoint",
            "checkpoints/b/best.pt",
            "--plif-config",
            "artifacts/p/config_resolved.yaml",
            "--plif-checkpoint",
            "checkpoints/p/best.pt",
            "--output",
            "artifacts/temporal_pair",
        ]
    )
    assert pair.baseline_checkpoint.endswith("best.pt")
    assert pair.plif_config.endswith("config_resolved.yaml")


def test_checkpoint_evaluation_contract_allows_only_metric_changes():
    common = {
        "dataset": {"name": "dvslip"},
        "representation": {"name": "count_frames_e0"},
        "augmentation": {"horizontal_flip_probability": 0.5},
        "model": {"name": "mini_qkformer", "embed_dim": 128},
        "training": {"recipe_id": "e0"},
    }
    checkpoint = {**common, "evaluation": {}}
    current = {**common, "evaluation": {"relative_prefix_fractions": [0.5, 1.0]}}

    assert _checkpoint_evaluation_contract(checkpoint) == _checkpoint_evaluation_contract(current)
    current["model"] = {**current["model"], "embed_dim": 192}
    assert _checkpoint_evaluation_contract(checkpoint) != _checkpoint_evaluation_contract(current)


def test_warmup_cosine_scheduler_reaches_base_and_minimum_rates():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = make_scheduler(
        optimizer,
        {
            "epochs": 8,
            "scheduler": "cosine",
            "warmup_epochs": 4,
            "warmup_start_factor": 0.01,
            "min_learning_rate": 1e-6,
        },
    )

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] == pytest.approx(1e-5)
    assert rates[4] == pytest.approx(1e-3)
    assert rates[-1] == pytest.approx(1e-6)


def test_optimizer_can_assign_a_distinct_lr_to_new_continuation_parameters():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    optimizer = make_optimizer(
        model,
        {"learning_rate": 1e-5, "weight_decay": 5e-4},
        new_parameter_names={"1.weight", "1.bias"},
        new_parameter_learning_rate=1e-4,
    )

    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups.keys() == {"inherited", "new"}
    assert groups["inherited"]["lr"] == pytest.approx(1e-5)
    assert groups["new"]["lr"] == pytest.approx(1e-4)
    assert sum(parameter.numel() for parameter in groups["new"]["params"]) == 8


def test_discriminative_lr_ratio_survives_the_entire_cosine_schedule():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    optimizer = make_optimizer(
        model,
        {"learning_rate": 1e-5, "weight_decay": 5e-4},
        new_parameter_names={"1.weight", "1.bias"},
        new_parameter_learning_rate=1e-4,
    )
    scheduler = make_scheduler(
        optimizer,
        {
            "epochs": 8,
            "learning_rate": 1e-5,
            "warmup_epochs": 2,
            "warmup_start_factor": 0.01,
            "min_learning_rate": 1e-6,
        },
    )
    for _ in range(9):
        rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        assert rates["new"] / rates["inherited"] == pytest.approx(10.0)
        optimizer.step()
        scheduler.step()


def test_objective_gradient_diagnostics_report_scale_and_shared_alignment():
    model = nn.Module()
    model.head = nn.Linear(2, 1, bias=False)
    inputs = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    outputs = model.head(inputs)
    classification = outputs.square().mean()
    auxiliary = (outputs - 1.0).square().mean()

    metrics = _objective_gradient_diagnostics(model, classification, auxiliary, 0.1)

    assert metrics["gradient_head_classification_norm"] > 0.0
    assert metrics["gradient_head_weighted_auxiliary_norm"] > 0.0
    assert metrics["gradient_head_ratio"] == pytest.approx(
        metrics["gradient_head_weighted_auxiliary_norm"]
        / metrics["gradient_head_classification_norm"]
    )
    assert -1.0 <= metrics["classification_auxiliary_gradient_cosine"] <= 1.0


def test_gradient_accumulation_steps_once_per_complete_or_final_group(monkeypatch):
    frames = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    targets = torch.tensor([0, 1, 0, 1, 0])
    loader = DataLoader(TensorDataset(frames, targets, torch.arange(5)), batch_size=1)
    model = nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer_steps = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)
    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        _DisabledScaler(),
        amp_enabled=False,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=2,
    )

    assert optimizer_steps == 3
    assert metrics["accuracy"] >= 0.0
    assert metrics["gradient_norm_mean"] is not None
    assert 0.0 <= metrics["gradient_clip_fraction"] <= 1.0
    assert metrics["gradient_nonfinite_fraction"] == 0.0
    assert metrics["amp_overflow_fraction"] == 0.0


def test_gradient_accumulation_weights_a_short_final_microbatch_by_sample():
    frames = torch.tensor([[0.0], [1.0], [2.0]])
    targets = torch.tensor([0, 1, 0])
    accumulated_model = nn.Linear(1, 2)
    full_batch_model = nn.Linear(1, 2)
    full_batch_model.load_state_dict(accumulated_model.state_dict())

    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.01)
    train_one_epoch(
        accumulated_model,
        DataLoader(TensorDataset(frames, targets, torch.arange(3)), batch_size=2),
        accumulated_optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        _DisabledScaler(),
        amp_enabled=False,
        gradient_clip_norm=None,
        gradient_accumulation_steps=2,
    )

    full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.01)
    full_batch_optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(full_batch_model(frames), targets).backward()
    full_batch_optimizer.step()

    for accumulated, full_batch in zip(
        accumulated_model.parameters(), full_batch_model.parameters(), strict=True
    ):
        assert torch.allclose(accumulated, full_batch)


def test_balanced_overfit_bundle_reuses_only_the_selected_train_samples():
    targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    dataset = TensorDataset(torch.arange(8), targets)
    dataset.targets = targets
    bundle = DatasetBundle(
        train=dataset,
        validation=TensorDataset(torch.tensor([99]), torch.tensor([2])),
        holdout=None,
        classes=["zero", "one", "two"],
    )

    overfit = balanced_overfit_bundle(bundle, class_count=2, samples_per_class=2)

    assert overfit.train is overfit.validation
    assert len(overfit.train) == 4
    assert [int(overfit.train[index][1]) for index in range(4)] == [0, 0, 1, 1]


def test_evaluate_limits_input_to_the_requested_temporal_prefix():
    model = _TimeStepRecorder()
    frames = torch.ones(3, 5, 2, 4, 4)
    targets = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(frames, targets, torch.arange(3)), batch_size=2)

    result, _ = evaluate(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        num_classes=2,
        prefix_steps=2,
    )

    assert result.samples == 3
    assert model.observed_steps == [2, 2]


def test_evaluate_slices_multigranular_prefixes_at_the_registered_clock_ratio():
    class TwoRateDataset(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {
                "coarse": torch.ones(5, 2, 4, 4),
                "fine": torch.ones(40, 2, 2, 2),
            }, index % 2, index

    model = _TwoRateStepRecorder()
    result, _ = evaluate(
        model,
        DataLoader(TwoRateDataset(), batch_size=2),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        num_classes=2,
        prefix_steps=2,
    )

    assert result.samples == 3
    assert model.observed_steps == [(2, 16), (2, 16)]


def test_evaluate_groups_sample_specific_prefixes_without_reordering_outputs():
    class PrefixSensitiveModel(nn.Module):
        def forward(self, frames):
            score = frames.sum(dim=(1, 2, 3, 4))
            return torch.stack((score, -score), dim=1)

    frames = torch.zeros(3, 4, 1, 1, 1)
    frames[0, :, 0, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    frames[1, :, 0, 0, 0] = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    frames[2, :, 0, 0, 0] = torch.tensor([1.0, -1.0, -1.0, -1.0])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(frames, targets, torch.arange(3)), batch_size=3)

    result, _ = evaluate(
        PrefixSensitiveModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        num_classes=2,
        prefix_steps=[1, 2, 4],
    )

    assert result.accuracy == 1.0


def test_last_checkpoint_restores_complete_epoch_boundary_state(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = _DisabledScaler()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_lr = optimizer.param_groups[0]["lr"]
    path = tmp_path / "last.pt"
    torch.manual_seed(123)

    save_training_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=3,
        best_score=0.5,
        best_epoch=2,
        config={"experiment": {"name": "fixture"}},
        num_classes=2,
        run_id="fixture__seed7",
        artifact_dir=tmp_path / "artifacts",
        peak_cuda_memory_bytes=123,
    )
    expected_random_value = torch.rand(())
    with torch.no_grad():
        model.weight.zero_()
    optimizer.param_groups[0]["lr"] = 9.0
    torch.manual_seed(999)

    checkpoint = load_training_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_epoch"] == 2
    assert checkpoint["peak_cuda_memory_bytes"] == 123
    assert optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)
    assert torch.equal(torch.rand(()), expected_random_value)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
