import torch

from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.spiking import MultiStepLIF


def test_mini_qkformer_output_shape_and_backward():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
    )
    frames = torch.rand(2, 4, 2, 32, 32)
    logits = model(frames)
    assert logits.shape == (2, 6)
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_temporal_trace_matches_standard_and_explicit_prefix_forward():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
    ).eval()
    frames = torch.rand(1, 4, 2, 16, 16)

    with torch.no_grad():
        trace = model.forward_with_trace(frames)

    assert torch.allclose(trace["logits"], model(frames), atol=1e-6)
    for timestep in range(frames.shape[1]):
        explicit = model(frames[:, : timestep + 1])
        assert torch.allclose(trace["cumulative_logits"][:, timestep], explicit, atol=1e-6)


def test_mini_qkformer_propagates_one_surrogate_to_every_lif():
    reference = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=32,
        num_heads=4,
    )
    model = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=32,
        num_heads=4,
        surrogate_name="sigmoid",
        surrogate_alpha=4.0,
    )
    model.load_state_dict(reference.state_dict())

    lif_modules = [module for module in model.modules() if isinstance(module, MultiStepLIF)]
    assert lif_modules
    assert all(module.surrogate_name == "sigmoid" for module in lif_modules)
    assert all(module.surrogate_alpha == 4.0 for module in lif_modules)
