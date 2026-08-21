from __future__ import annotations

from copy import deepcopy

import pytest

from etsr.smoke import SmokeConfigError, run_smoke_test, validate_smoke_config


def _smoke_config(tmp_path=None):
    root = "." if tmp_path is None else str(tmp_path)
    return {
        "experiment": {
            "name": "smoke_synthetic",
            "seed": 7,
            "artifact_root": root,
            "checkpoint_root": root,
            "deterministic": True,
        },
        "dataset": {
            "name": "synthetic_temporal_order",
            "frames_number": 8,
            "image_size": 32,
            "num_classes": 4,
            "train_samples": 32,
            "validation_samples": 16,
            "test_samples": 16,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "name": "mini_qkformer",
            "in_channels": 2,
            "embed_dim": 32,
            "num_heads": 4,
            "mlp_ratio": 2.0,
        },
        "training": {"epochs": 1, "amp": False, "evaluate_holdout": True},
        "profiling": {"enabled": True, "max_batches": 1},
        "audit": {
            "perturbations": [{"name": "original"}, {"name": "reverse_time"}],
            "prefix_fractions": [0.5, 1.0],
        },
    }


def test_smoke_config_accepts_only_a_bounded_synthetic_run():
    config = _smoke_config()
    validate_smoke_config(config)

    invalid_cases = []
    for section, field, value in (
        ("dataset", "name", "dvsgc"),
        ("dataset", "train_samples", 65),
        ("dataset", "num_workers", 1),
        ("training", "epochs", 2),
        ("training", "amp", True),
        ("model", "embed_dim", 64),
        ("profiling", "max_batches", 2),
    ):
        invalid = deepcopy(config)
        invalid[section][field] = value
        invalid_cases.append(invalid)

    for invalid in invalid_cases:
        with pytest.raises(SmokeConfigError):
            validate_smoke_config(invalid)


def test_smoke_orchestration_writes_compact_success_artifact(tmp_path, monkeypatch):
    config = _smoke_config(tmp_path)
    config["_source_path"] = "/project/configs/smoke.yaml"
    artifact_dir = tmp_path / "smoke_run"
    artifact_dir.mkdir()

    def fake_train(received):
        assert received is config
        return {
            "run_id": "smoke_run",
            "artifact_dir": str(artifact_dir),
            "checkpoint": str(tmp_path / "best.pt"),
            "validation": {"accuracy": 0.5},
            "test": {"accuracy": 0.5},
        }

    def fake_audit(received, checkpoint):
        assert received is config
        assert checkpoint == str(tmp_path / "best.pt")
        return {"conditions": {"original": {}, "reverse_time": {}}}

    monkeypatch.setattr("etsr.smoke.train_experiment", fake_train)
    monkeypatch.setattr("etsr.smoke.run_temporal_audit", fake_audit)

    result = run_smoke_test(config)

    assert result["status"] == "passed"
    assert result["official_test_used"] is False
    assert result["audit_conditions"] == ["original", "reverse_time"]
    assert (artifact_dir / "smoke_summary.json").is_file()
