import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from etsr.models.mini_qkformer import MiniQKFormer
from etsr.profiling import profile_model


def test_hardware_profile_separates_multivalued_mac_spike_ac_and_state():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=4,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
    )
    frames = torch.zeros(2, 4, 2, 32, 32)
    frames[0, 0, 0, 0, 0] = 2.0
    loader = DataLoader(
        TensorDataset(frames, torch.tensor([0, 1]), torch.arange(2)),
        batch_size=2,
    )

    profile = profile_model(model, loader, torch.device("cpu"), max_samples=1)

    assert profile["samples_profiled"] == 1
    first_conv = profile["layers"]["patch_embed1.main1.conv"]
    assert first_conv["multivalued_mac_potential"] == 4 * 8 * 32 * 32 * 2 * 3 * 3
    assert profile["operations_per_sample"]["binary_ac_potential"] > 0
    assert profile["operations_per_sample"]["attention_sop_potential"] > 0
    assert profile["operations_per_sample"]["sop_potential"] == pytest.approx(
        profile["operations_per_sample"]["binary_ac_potential"]
        + profile["operations_per_sample"]["attention_sop_potential"]
    )
    assert 0.0 <= profile["activity"]["global_firing_rate"] <= 1.0
    assert profile["state"]["persistent_state_elements"] > 0
    assert profile["state"]["runtime_persistent_state_bits"] == (
        32 * profile["state"]["persistent_state_elements"]
    )
    assert profile["state"]["writes_per_sample"] > profile["state"]["reads_per_sample"]
    assert profile["state"]["hardware_precision_bits"] is None


def test_hardware_profile_rejects_an_empty_sample_budget():
    model = MiniQKFormer(in_channels=2, num_classes=2, embed_dim=16, num_heads=4)
    frames = torch.zeros(1, 2, 2, 16, 16)
    loader = DataLoader(TensorDataset(frames, torch.tensor([0]), torch.arange(1)))

    with pytest.raises(ValueError, match="positive"):
        profile_model(model, loader, torch.device("cpu"), max_samples=0)
