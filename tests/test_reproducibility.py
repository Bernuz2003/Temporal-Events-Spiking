import json
import os

import torch

from etsr.reproducibility import collect_environment, seed_everything


def test_environment_snapshot_is_json_serializable_and_privacy_bounded():
    snapshot = collect_environment(torch.device("cpu"))

    assert snapshot["schema_version"] == 1
    assert snapshot["python"]["version"]
    assert snapshot["torch_runtime"]["selected_device"] == "cpu"
    assert snapshot["torch_runtime"]["version"] == str(torch.__version__)
    assert "cublas_workspace_config" in snapshot["torch_runtime"]
    assert snapshot["privacy"]["full_process_environment_recorded"] is False
    assert "PATH" not in snapshot
    json.dumps(snapshot)


def test_deterministic_seed_configures_the_required_cublas_workspace(monkeypatch):
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    try:
        seed_everything(7, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
