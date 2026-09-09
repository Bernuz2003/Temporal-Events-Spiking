from __future__ import annotations

from typing import Any

from torch import nn

from etsr.models.mini_qkformer import MiniQKFormer


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    name = config["name"]
    if name != "mini_qkformer":
        raise ValueError(f"Unsupported model: {name}")
    return MiniQKFormer(
        in_channels=int(config.get("in_channels", 2)),
        num_classes=num_classes,
        embed_dim=int(config.get("embed_dim", 128)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 2.0)),
        lif_tau=float(config.get("lif_tau", 2.0)),
        lif_threshold=float(config.get("lif_threshold", 1.0)),
        surrogate_alpha=float(config.get("surrogate_alpha", 4.0)),
        lif_cross_time=config.get("lif_cross_time", True),
        readout=str(config.get("readout", "mean")),
        readout_time=str(config.get("readout_time", "fixed_window")),
        frontend=str(config.get("frontend", "baseline")),
        temporal_fir=config.get("temporal_fir", False),
        temporal_fir_kernel_size=int(config.get("temporal_fir_kernel_size", 3)),
        temporal_fir_dilations=tuple(config.get("temporal_fir_dilations", (1, 2))),
        temporal_channel_mixer=config.get("temporal_channel_mixer", False),
        temporal_channel_mixer_delays=tuple(config.get("temporal_channel_mixer_delays", (1, 2, 4))),
        learnable_lif_tau=config.get("learnable_lif_tau", False),
        gated_initial_memory_steps=config.get("gated_initial_memory_steps"),
        multigranular=config.get("multigranular", False),
        multigranular_fine_channels=int(config.get("multigranular_fine_channels", 16)),
        multigranular_fine_mid_channels=config.get("multigranular_fine_mid_channels"),
        multigranular_temporal_groups=config.get("multigranular_temporal_groups"),
        multigranular_fusion=str(config.get("multigranular_fusion", "add")),
        multigranular_micro_steps=int(config.get("multigranular_micro_steps", 8)),
    )
