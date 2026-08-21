import json

import torch

from etsr.reproducibility import collect_environment


def test_environment_snapshot_is_json_serializable_and_privacy_bounded():
    snapshot = collect_environment(torch.device("cpu"))

    assert snapshot["schema_version"] == 1
    assert snapshot["python"]["version"]
    assert snapshot["torch_runtime"]["selected_device"] == "cpu"
    assert snapshot["torch_runtime"]["version"] == str(torch.__version__)
    assert snapshot["privacy"]["full_process_environment_recorded"] is False
    assert "PATH" not in snapshot
    json.dumps(snapshot)
