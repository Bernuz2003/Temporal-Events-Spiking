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

    def _encode(
        self, frames: torch.Tensor, record_trace: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if frames.ndim != 5:
            raise ValueError("Expected input [B, T, C, H, W].")
        x = frames.permute(1, 0, 2, 3, 4).contiguous()
        trace = {"input": x} if record_trace else {}
        x = self.patch_embed1(x)
        if record_trace:
            trace["patch_embed1"] = x
        x = self.stage1(x)
        if record_trace:
            trace["stage1"] = x
        x = self.patch_embed2(x)
        if record_trace:
            trace["patch_embed2"] = x
        x = self.stage2(x)
        if record_trace:
            trace["stage2"] = x
        return x, trace

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x, _trace = self._encode(frames, record_trace=False)
        pooled = x.mean(dim=(0, 3, 4))
        return self.head(pooled)

    def forward_with_trace(self, frames: torch.Tensor) -> dict[str, object]:
        """Return compact time-resolved features without changing the standard forward path."""
        x, raw_trace = self._encode(frames, record_trace=True)
        compact_trace = {
            name: value.mean(dim=(3, 4)).permute(1, 0, 2).contiguous()
            for name, value in raw_trace.items()
        }
        final_features = compact_trace["stage2"]
        per_timestep_logits = self.head(final_features)
        denominator = torch.arange(
            1,
            per_timestep_logits.shape[1] + 1,
            device=per_timestep_logits.device,
            dtype=per_timestep_logits.dtype,
        ).view(1, -1, 1)
        cumulative_logits = per_timestep_logits.cumsum(dim=1) / denominator
        return {
            "logits": cumulative_logits[:, -1],
            "cumulative_logits": cumulative_logits,
            "per_timestep_logits": per_timestep_logits,
            "evidence_without_bias": torch.nn.functional.linear(
                final_features, self.head.weight, bias=None
            ),
            "head_bias": self.head.bias,
            "features": compact_trace,
        }
