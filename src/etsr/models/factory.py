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
    )
