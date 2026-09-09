import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from etsr.models.mini_qkformer import MiniQKFormer
from etsr.profiling import profile_model
from etsr.profiling.energy import horowitz_reference
from etsr.profiling.selection import profile_indices


def test_profile_binary_classification_is_independent_of_loader_batch_size():
    model = torch.nn.Linear(2, 2, bias=False)
    dataset = TensorDataset(
        torch.tensor([[0.0, 1.0], [2.0, 0.0]]), torch.tensor([0, 1]), torch.arange(2)
    )
    profiles = [
        profile_model(model, DataLoader(dataset, batch_size=batch), torch.device("cpu"), 2)
        for batch in (1, 2)
    ]
    assert profiles[0] == profiles[1]
    assert profiles[0]["operations_per_sample"]["multivalued_mac_potential"] == 2
    assert profiles[0]["operations_per_sample"]["binary_ac_potential"] == 2


def test_profile_selection_covers_classes_instead_of_sorted_prefix():
    targets = [label for label in range(100) for _ in range(10)]
    selected = profile_indices(targets, 64)
    assert len(selected) == len(set(selected)) == 64
    assert len({targets[index] for index in selected}) == 64
    assert selected == profile_indices(targets, 64)
    assert len(profile_indices(targets, 2000)) == len(targets)


def test_horowitz_units_and_fir_not_double_counted():
    ops = {
        "multivalued_mac_potential": 10,
        "binary_ac_potential": 20,
        "binary_ac_activity_estimate": 5,
        "attention_sop_potential": 3,
        "elementwise_add": 7,
        "elementwise_multiply": 4,
        "attention_scale_multiply": 2,
        "temporal_fir_multiply": 4,
        "temporal_fir_add": 7,
    }
    energy = horowitz_reference(ops)
    assert energy["covered_arithmetic_activity_proxy_uj_per_sample"] == pytest.approx(
        (4.6 * 10 + 0.9 * (5 + 3 + 7) + 3.7 * (4 + 2)) / 1e6
    )
    assert energy["total_hardware_energy_uj_per_sample"] is None


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


def test_hardware_profile_counts_pyramidal_pooling_and_temporal_fir_state():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=4,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_fir=True,
    )
    frames = torch.rand(1, 4, 2, 32, 32)
    loader = DataLoader(TensorDataset(frames, torch.tensor([0]), torch.arange(1)))

    profile = profile_model(model, loader, torch.device("cpu"), max_samples=1)

    assert profile["schema_version"] == 4
    assert profile["operations_per_sample"]["maxpool_comparison"] > 0
    assert profile["operations_per_sample"]["temporal_fir_multiply"] > 0
    assert profile["operations_per_sample"]["temporal_fir_add"] > 0
    first = profile["layers"]["patch_embed1.main4.temporal_fir"]
    second = profile["layers"]["patch_embed2.down.temporal_fir"]
    assert first["persistent_state_elements"] == 2 * 16 * 4 * 4
    assert second["persistent_state_elements"] == 4 * 32 * 2 * 2
    assert first["state_reads"] > first["state_writes"]


def test_hardware_profile_counts_temporal_capacity_and_plif_separately():
    frames = torch.rand(1, 4, 2, 32, 32)
    loader = DataLoader(TensorDataset(frames, torch.tensor([0]), torch.arange(1)))
    capacity = MiniQKFormer(
        2,
        4,
        embed_dim=32,
        num_heads=4,
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2, 4),
    )
    capacity_profile = profile_model(capacity, loader, torch.device("cpu"), 1)
    assert capacity_profile["operations_per_sample"]["temporal_channel_mixer_mac"] > 0
    first = capacity_profile["layers"]["patch_embed1.main4.temporal_channel_mixer"]
    second = capacity_profile["layers"]["patch_embed2.down.temporal_channel_mixer"]
    assert first["delays"] == [1, 2, 4]
    assert first["persistent_state_elements"] == 4 * 16 * 4 * 4
    assert second["persistent_state_elements"] == 4 * 32 * 2 * 2
    plif = MiniQKFormer(2, 4, embed_dim=32, num_heads=4, learnable_lif_tau=True)
    plif_profile = profile_model(plif, loader, torch.device("cpu"), 1)
    dynamics = plif_profile["neuron_dynamics"]
    assert dynamics["type"] == "per_channel_plif"
    assert dynamics["plif_parameter_elements"] > 0
    assert all(
        layer["effective_tau_mean"] == pytest.approx(2.0) for layer in dynamics["layers"].values()
    )
    assert (
        plif_profile["state"]["persistent_state_elements"]
        == (
            profile_model(
                MiniQKFormer(2, 4, embed_dim=32, num_heads=4), loader, torch.device("cpu"), 1
            )["state"]["persistent_state_elements"]
        )
    )


def test_hardware_profile_accepts_multigranular_batches_and_counts_reducer_state():
    class TwoRateDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                "coarse": torch.rand(4, 2, 32, 32),
                "fine": torch.rand(32, 2, 4, 4),
            }, 0, 0

    model = MiniQKFormer(
        2,
        4,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        multigranular=True,
        multigranular_fine_channels=4,
        multigranular_temporal_groups=16,
        multigranular_fusion="add",
        multigranular_micro_steps=8,
    )
    profile = profile_model(
        model,
        DataLoader(TwoRateDataset(), batch_size=1),
        torch.device("cpu"),
        1,
    )

    assert profile["samples_profiled"] == 1
    reducer = profile["layers"]["fine_temporal_branch.temporal_reduce"]
    assert reducer["binary_ac_potential"] > 0
    history = profile["layers"]["fine_temporal_branch"]
    assert history["persistent_state_elements"] == 7 * 16 * 4 * 4
    assert history["state_reads"] > 0


def test_hardware_profile_distinguishes_no_cross_time_from_gated_readout_state():
    frames = torch.zeros(1, 3, 2, 16, 16)
    frames[0, 1, 0, 0, 0] = 1
    loader = DataLoader(TensorDataset(frames, torch.tensor([0]), torch.arange(1)))

    independent = MiniQKFormer(
        in_channels=2,
        num_classes=2,
        embed_dim=16,
        num_heads=4,
        lif_cross_time=False,
    )
    independent_profile = profile_model(
        independent,
        loader,
        torch.device("cpu"),
        max_samples=1,
    )
    assert independent_profile["state"]["persistent_state_elements"] == 0
    assert independent_profile["state"]["reads_per_sample"] == 0
    assert independent_profile["operations_per_sample"]["lif_comparison"] > 0
    assert independent_profile["operations_per_sample"]["lif_reset_gate_potential"] == 0
    assert independent_profile["execution"]["causal_sequence_equations"] is False

    gated = MiniQKFormer(
        in_channels=2,
        num_classes=2,
        embed_dim=16,
        num_heads=4,
        lif_cross_time=False,
        readout="diagonal_gated",
        readout_time="last_event",
    )
    gated_profile = profile_model(gated, loader, torch.device("cpu"), max_samples=1)
    assert gated_profile["state"]["persistent_state_elements"] == 16
    assert gated_profile["inference_non_linearities"]["sigmoid_per_sample"] == 3 * 16
    assert gated_profile["inference_non_linearities"]["tanh_per_sample"] == 3 * 16
    assert gated_profile["execution"]["causal_sequence_equations"] is True
    assert gated_profile["layers"]["gated_readout"]["persistent_state_elements"] == 16
    gate = gated_profile["layers"]["gated_readout"]["observed_update_gate"]
    assert 0.0 <= gate["minimum_channel_mean"] <= gate["maximum_channel_mean"] <= 1.0
    assert len(gate["mean_by_channel"]) == 16
    assert gated_profile["layers"]["gated_readout"][
        "active_state_update_fraction"
    ] == pytest.approx(2 / 3)
