from __future__ import annotations

import torch
from torch import nn

from etsr.models.layers import (
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    SpikingBlock,
    SpikingSelfAttention,
    TokenQKAttention,
)


class MiniQKFormer(nn.Module):
    """Compact QKFormer-like baseline for controlled temporal audits.

    This implementation is intentionally self-contained and CPU-compatible. It preserves the main
    research structure—hierarchical SPEDS-like embedding, Q-K token gating, spike self-attention—
    without claiming bit-exact reproduction of the official repository.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        lif_tau: float = 2.0,
        lif_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by four")
        self.num_classes = num_classes
        half = embed_dim // 2

        self.patch_embed1 = InitialPatchEmbedding(
            in_channels, embed_dim, lif_tau, lif_threshold
        )
        self.stage1 = SpikingBlock(
            attention=TokenQKAttention(half, num_heads, lif_tau, lif_threshold),
            dim=half,
            mlp_ratio=mlp_ratio,
            tau=lif_tau,
            threshold=lif_threshold,
        )
        self.patch_embed2 = PatchEmbeddingStage(
            half, embed_dim, lif_tau, lif_threshold
        )
        self.stage2 = SpikingBlock(
            attention=SpikingSelfAttention(embed_dim, num_heads, lif_tau, lif_threshold),
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            tau=lif_tau,
            threshold=lif_threshold,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.op_kind = "mac"

        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("Expected input [B, T, C, H, W].")
        x = frames.permute(1, 0, 2, 3, 4).contiguous()
        x = self.patch_embed1(x)
        x = self.stage1(x)
        x = self.patch_embed2(x)
        x = self.stage2(x)
        pooled = x.mean(dim=(0, 3, 4))
        return self.head(pooled)
