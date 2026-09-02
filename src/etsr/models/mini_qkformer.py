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
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF


class MiniQKFormer(nn.Module):
    """Compact QKFormer-like DVS-Lip baseline.

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
        surrogate_alpha: float = 4.0,
        lif_cross_time: bool = True,
        readout: str = "mean",
        readout_time: str = "fixed_window",
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by four")
        if surrogate_alpha <= 0.0:
            raise ValueError("surrogate_alpha must be positive")
        if type(lif_cross_time) is not bool:
            raise ValueError("lif_cross_time must be boolean")
        if readout not in {"mean", "last", "diagonal_gated"}:
            raise ValueError(f"Unsupported readout: {readout}")
        if readout_time not in {"fixed_window", "last_event"}:
            raise ValueError(f"Unsupported readout time: {readout_time}")
        self.num_classes = num_classes
        self.readout_name = readout
        self.readout_time = readout_time
        half = embed_dim // 2

        self.patch_embed1 = InitialPatchEmbedding(in_channels, embed_dim, lif_tau, lif_threshold)
        self.stage1 = SpikingBlock(
            attention=TokenQKAttention(half, num_heads, lif_tau, lif_threshold),
            dim=half,
            mlp_ratio=mlp_ratio,
            tau=lif_tau,
            threshold=lif_threshold,
        )
        self.patch_embed2 = PatchEmbeddingStage(half, embed_dim, lif_tau, lif_threshold)
        self.stage2 = SpikingBlock(
            attention=SpikingSelfAttention(embed_dim, num_heads, lif_tau, lif_threshold),
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            tau=lif_tau,
            threshold=lif_threshold,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self.gated_readout = (
            DiagonalGatedReadout(embed_dim) if readout == "diagonal_gated" else None
        )

        for module in self.modules():
            if isinstance(module, MultiStepLIF):
                module.surrogate_alpha = float(surrogate_alpha)
                module.cross_time = lif_cross_time
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
            module, (nn.Conv1d, nn.Conv2d, nn.Linear)
        ):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("Expected input [B, T, C, H, W].")
        x = frames.permute(1, 0, 2, 3, 4).contiguous()
        x = self.patch_embed1(x)
        x = self.stage1(x)
        x = self.patch_embed2(x)
        return self.stage2(x)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._encode(frames)
        valid_steps = self._last_event_steps(frames) if self.readout_time == "last_event" else None
        pooled = self._readout(x, valid_steps)
        return self.head(pooled)

    @staticmethod
    def _last_event_steps(frames: torch.Tensor) -> torch.Tensor:
        """Return the one-based final occupied bin for each encoded sample."""

        occupied = frames.flatten(2).ne(0).any(dim=2)
        positions = torch.arange(
            1,
            occupied.shape[1] + 1,
            device=frames.device,
            dtype=torch.long,
        )
        # Valid dataset samples always contain events; clamping defines silent synthetic inputs.
        return (occupied * positions.unsqueeze(0)).amax(dim=1).clamp_min(1)

    def _readout(
        self,
        encoded: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spatial = encoded.mean(dim=(3, 4))
        if self.readout_name == "mean":
            if valid_steps is None:
                return spatial.mean(dim=0)
            time = torch.arange(spatial.shape[0], device=spatial.device).unsqueeze(1)
            mask = (time < valid_steps.unsqueeze(0)).unsqueeze(2)
            return (spatial * mask).sum(dim=0) / valid_steps.unsqueeze(1)
        if self.readout_name == "last":
            if valid_steps is None:
                return spatial[-1]
            batch = torch.arange(spatial.shape[1], device=spatial.device)
            return spatial[valid_steps - 1, batch]
        assert self.gated_readout is not None
        return self.gated_readout(spatial, valid_steps)
