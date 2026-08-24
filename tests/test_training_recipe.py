import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from etsr.cli import build_parser
from etsr.data.common import DatasetBundle, balanced_overfit_bundle
from etsr.training.checkpointing import load_training_state, save_training_state
from etsr.training.engine import make_scheduler, train_one_epoch


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
        ]
    )

    assert args.overfit == [16, 4]
    assert args.epochs == 50
    assert args.resume == "last.pt"


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
