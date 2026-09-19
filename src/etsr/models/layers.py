from __future__ import annotations

import math

import torch
from torch import nn

from etsr.models.spiking import MultiStepLIF
from etsr.models.temporal import CausalTemporalChannelMixer, CausalTemporalFIR


def _time_distributed(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    time_steps, batch_size = x.shape[:2]
    output = module(x.flatten(0, 1))
    return output.reshape(time_steps, batch_size, *output.shape[1:])


def _lif(channels: int, tau: float, threshold: float, learnable_tau: bool = False) -> MultiStepLIF:
    return MultiStepLIF(
        tau=tau,
        threshold=threshold,
        channels=channels if learnable_tau else None,
        learnable_tau=learnable_tau,
    )


class ConvBNLIF2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        tau: float = 2.0,
        threshold: float = 1.0,
        temporal_fir_kernel_size: int | None = None,
        temporal_fir_dilation: int = 1,
        temporal_channel_mixer_delays: tuple[int, ...] | None = None,
        temporal_channel_mixer_learnable_delays: bool = False,
        temporal_channel_mixer_dynamic_routing: bool = False,
        temporal_channel_mixer_predictive_auxiliary: bool = False,
        temporal_channel_mixer_surprise_routing: bool = False,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.temporal_fir = (
            CausalTemporalFIR(
                out_channels,
                kernel_size=temporal_fir_kernel_size,
                dilation=temporal_fir_dilation,
            )
            if temporal_fir_kernel_size is not None
            else None
        )
        self.temporal_channel_mixer = (
            CausalTemporalChannelMixer(
                out_channels,
                temporal_channel_mixer_delays,
                learnable_delays=temporal_channel_mixer_learnable_delays,
                dynamic_routing=temporal_channel_mixer_dynamic_routing,
                predictive_auxiliary=temporal_channel_mixer_predictive_auxiliary,
                surprise_routing=temporal_channel_mixer_surprise_routing,
            )
            if temporal_channel_mixer_delays is not None
            else None
        )
        if self.temporal_fir is not None and self.temporal_channel_mixer is not None:
            raise ValueError("Select only one explicit temporal core per layer")
        self.lif = _lif(out_channels, tau, threshold, learnable_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _time_distributed(self.conv, x)
        x = _time_distributed(self.bn, x)
        if self.temporal_fir is not None:
            x = self.temporal_fir(x)
        if self.temporal_channel_mixer is not None:
            x = self.temporal_channel_mixer(x)
        return self.lif(x)


class ConvBNMaxPoolLIF2d(nn.Module):
    """Time-distributed convolution and pooling with one LIF after spatial reduction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tau: float,
        threshold: float,
        temporal_fir_kernel_size: int | None = None,
        temporal_fir_dilation: int = 1,
        temporal_channel_mixer_delays: tuple[int, ...] | None = None,
        temporal_channel_mixer_learnable_delays: bool = False,
        temporal_channel_mixer_dynamic_routing: bool = False,
        temporal_channel_mixer_predictive_auxiliary: bool = False,
        temporal_channel_mixer_surprise_routing: bool = False,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.temporal_fir = (
            CausalTemporalFIR(
                out_channels,
                kernel_size=temporal_fir_kernel_size,
                dilation=temporal_fir_dilation,
            )
            if temporal_fir_kernel_size is not None
            else None
        )
        self.temporal_channel_mixer = (
            CausalTemporalChannelMixer(
                out_channels,
                temporal_channel_mixer_delays,
                learnable_delays=temporal_channel_mixer_learnable_delays,
                dynamic_routing=temporal_channel_mixer_dynamic_routing,
                predictive_auxiliary=temporal_channel_mixer_predictive_auxiliary,
                surprise_routing=temporal_channel_mixer_surprise_routing,
            )
            if temporal_channel_mixer_delays is not None
            else None
        )
        if self.temporal_fir is not None and self.temporal_channel_mixer is not None:
            raise ValueError("Select only one explicit temporal core per layer")
        self.lif = _lif(out_channels, tau, threshold, learnable_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _time_distributed(self.conv, x)
        x = _time_distributed(self.bn, x)
        x = _time_distributed(self.pool, x)
        if self.temporal_fir is not None:
            x = self.temporal_fir(x)
        if self.temporal_channel_mixer is not None:
            x = self.temporal_channel_mixer(x)
        return self.lif(x)


class InitialPatchEmbedding(nn.Module):
    """Compact SPEDS-like front-end whose spiking branches are added directly."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        tau: float,
        threshold: float,
        temporal_fir_kernel_size: int | None = None,
        temporal_fir_dilation: int = 1,
        temporal_channel_mixer_delays: tuple[int, ...] | None = None,
        temporal_channel_mixer_learnable_delays: bool = False,
        temporal_channel_mixer_dynamic_routing: bool = False,
        temporal_channel_mixer_predictive_auxiliary: bool = False,
        temporal_channel_mixer_surprise_routing: bool = False,
        learnable_tau: bool = False,
    ):
        super().__init__()
        quarter = embed_dim // 4
        half = embed_dim // 2
        self.main1 = ConvBNLIF2d(
            in_channels,
            quarter,
            3,
            stride=1,
            padding=1,
            tau=tau,
            threshold=threshold,
            learnable_tau=learnable_tau,
        )
        self.main2 = ConvBNLIF2d(
            quarter,
            half,
            3,
            stride=2,
            padding=1,
            tau=tau,
            threshold=threshold,
            learnable_tau=learnable_tau,
        )
        self.main3 = ConvBNLIF2d(
            half,
            half,
            3,
            stride=2,
            padding=1,
            tau=tau,
            threshold=threshold,
            learnable_tau=learnable_tau,
        )
        self.main4 = ConvBNLIF2d(
            half,
            half,
            3,
            stride=2,
            padding=1,
            tau=tau,
            threshold=threshold,
            temporal_fir_kernel_size=temporal_fir_kernel_size,
            temporal_fir_dilation=temporal_fir_dilation,
            temporal_channel_mixer_delays=temporal_channel_mixer_delays,
            temporal_channel_mixer_learnable_delays=temporal_channel_mixer_learnable_delays,
            temporal_channel_mixer_dynamic_routing=temporal_channel_mixer_dynamic_routing,
            temporal_channel_mixer_predictive_auxiliary=temporal_channel_mixer_predictive_auxiliary,
            temporal_channel_mixer_surprise_routing=temporal_channel_mixer_surprise_routing,
            learnable_tau=learnable_tau,
        )
        self.shortcut = ConvBNLIF2d(
            quarter, half, 1, stride=8, tau=tau, threshold=threshold, learnable_tau=learnable_tau
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main1(x)
        shortcut_source = x
        x = self.main2(x)
        x = self.main3(x)
        x = self.main4(x)
        shortcut = self.shortcut(shortcut_source)
        return x + shortcut


class PyramidalPatchEmbedding(nn.Module):
    """Low-state Conv-BN-MaxPool front-end with a direct spiking shortcut."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        tau: float,
        threshold: float,
        temporal_fir_kernel_size: int | None = None,
        temporal_fir_dilation: int = 1,
        temporal_channel_mixer_delays: tuple[int, ...] | None = None,
        temporal_channel_mixer_learnable_delays: bool = False,
        temporal_channel_mixer_dynamic_routing: bool = False,
        temporal_channel_mixer_predictive_auxiliary: bool = False,
        temporal_channel_mixer_surprise_routing: bool = False,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % 16 != 0:
            raise ValueError("pyramidal front-end requires embed_dim divisible by 16")
        first = embed_dim // 16
        second = embed_dim // 8
        third = embed_dim // 4
        output = embed_dim // 2
        self.main1 = ConvBNLIF2d(
            in_channels,
            first,
            3,
            padding=1,
            tau=tau,
            threshold=threshold,
            learnable_tau=learnable_tau,
        )
        self.main2 = ConvBNMaxPoolLIF2d(
            first, second, tau=tau, threshold=threshold, learnable_tau=learnable_tau
        )
        self.main3 = ConvBNMaxPoolLIF2d(
            second, third, tau=tau, threshold=threshold, learnable_tau=learnable_tau
        )
        self.main4 = ConvBNMaxPoolLIF2d(
            third,
            output,
            tau=tau,
            threshold=threshold,
            temporal_fir_kernel_size=temporal_fir_kernel_size,
            temporal_fir_dilation=temporal_fir_dilation,
            temporal_channel_mixer_delays=temporal_channel_mixer_delays,
            temporal_channel_mixer_learnable_delays=temporal_channel_mixer_learnable_delays,
            temporal_channel_mixer_dynamic_routing=temporal_channel_mixer_dynamic_routing,
            temporal_channel_mixer_predictive_auxiliary=temporal_channel_mixer_predictive_auxiliary,
            temporal_channel_mixer_surprise_routing=temporal_channel_mixer_surprise_routing,
            learnable_tau=learnable_tau,
        )
        self.shortcut = ConvBNLIF2d(
            second, output, 1, stride=4, tau=tau, threshold=threshold, learnable_tau=learnable_tau
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main1(x)
        x = self.main2(x)
        shortcut_source = x
        x = self.main3(x)
        x = self.main4(x)
        return x + self.shortcut(shortcut_source)


class FineTemporalBranch(nn.Module):
    """Configurable high-rate branch compressed to the coarse clock before token processing.

    ``mid_channels`` optionally enables learned residual spatial reduction; ``temporal_groups``
    selects full or depthwise temporal mixing. Both settings share the same forward path.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        micro_steps_per_macro: int,
        tau: float,
        threshold: float,
        mid_channels: int | None = None,
        temporal_groups: int | None = None,
    ) -> None:
        super().__init__()
        if micro_steps_per_macro <= 1:
            raise ValueError("micro_steps_per_macro must be greater than one")
        self.micro_steps_per_macro = micro_steps_per_macro
        self.mid_channels = mid_channels
        self.spatial = ConvBNLIF2d(
            in_channels,
            hidden_channels,
            3,
            padding=1,
            tau=tau,
            threshold=threshold,
        )
        self.spatial_down = (
            ConvBNLIF2d(
                hidden_channels,
                mid_channels,
                3,
                stride=2,
                padding=1,
                tau=tau,
                threshold=threshold,
            )
            if mid_channels is not None
            else None
        )
        self.spatial_shortcut = (
            ConvBNLIF2d(
                hidden_channels,
                mid_channels,
                1,
                stride=2,
                tau=tau,
                threshold=threshold,
            )
            if mid_channels is not None
            else None
        )
        project_input = mid_channels if mid_channels is not None else hidden_channels
        self.project = ConvBNLIF2d(
            project_input,
            out_channels,
            1,
            tau=tau,
            threshold=threshold,
        )
        groups = out_channels if temporal_groups is None else temporal_groups
        if out_channels % groups:
            raise ValueError("temporal_groups must divide out_channels")
        self.temporal_reduce = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=micro_steps_per_macro,
            stride=micro_steps_per_macro,
            groups=groups,
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm1d(out_channels)
        self.output_lif = _lif(out_channels, tau, threshold)

    def initialize_temporal_reducer(self) -> None:
        with torch.no_grad():
            self.temporal_reduce.weight.zero_()
            diagonal = torch.arange(self.temporal_reduce.out_channels)
            channels_per_group = (
                self.temporal_reduce.out_channels // self.temporal_reduce.groups
            )
            self.temporal_reduce.weight[
                diagonal, diagonal.remainder(channels_per_group)
            ] = 1.0 / self.micro_steps_per_macro

    def forward_pre_lif(self, fine_frames: torch.Tensor) -> torch.Tensor:
        if fine_frames.ndim != 5:
            raise ValueError("Fine branch expects [B, T_micro, C, H, W].")
        if fine_frames.shape[1] % self.micro_steps_per_macro:
            raise ValueError("Fine time axis must contain complete macro-windows.")
        x = fine_frames.permute(1, 0, 2, 3, 4).contiguous()
        x = self.spatial(x)
        if self.spatial_down is not None:
            assert self.spatial_shortcut is not None
            x = self.spatial_down(x) + self.spatial_shortcut(x)
        x = self.project(x)
        micro_steps, batch, channels, height, width = x.shape
        x = x.permute(1, 3, 4, 2, 0).reshape(batch * height * width, channels, micro_steps)
        x = self.temporal_bn(self.temporal_reduce(x))
        macro_steps = x.shape[-1]
        x = x.reshape(batch, height, width, channels, macro_steps).permute(4, 0, 3, 1, 2)
        return x.contiguous()

    def forward(self, fine_frames: torch.Tensor) -> torch.Tensor:
        return self.output_lif(self.forward_pre_lif(fine_frames))


class PatchEmbeddingStage(nn.Module):
    """Downsample two spiking branches and add them without another threshold."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tau: float,
        threshold: float,
        temporal_fir_kernel_size: int | None = None,
        temporal_fir_dilation: int = 1,
        temporal_channel_mixer_delays: tuple[int, ...] | None = None,
        temporal_channel_mixer_learnable_delays: bool = False,
        temporal_channel_mixer_dynamic_routing: bool = False,
        temporal_channel_mixer_predictive_auxiliary: bool = False,
        temporal_channel_mixer_surprise_routing: bool = False,
        learnable_tau: bool = False,
    ):
        super().__init__()
        self.proj = ConvBNLIF2d(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            tau=tau,
            threshold=threshold,
            learnable_tau=learnable_tau,
        )
        self.down = ConvBNLIF2d(
            out_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            tau=tau,
            threshold=threshold,
            temporal_fir_kernel_size=temporal_fir_kernel_size,
            temporal_fir_dilation=temporal_fir_dilation,
            temporal_channel_mixer_delays=temporal_channel_mixer_delays,
            temporal_channel_mixer_learnable_delays=temporal_channel_mixer_learnable_delays,
            temporal_channel_mixer_dynamic_routing=temporal_channel_mixer_dynamic_routing,
            temporal_channel_mixer_predictive_auxiliary=temporal_channel_mixer_predictive_auxiliary,
            temporal_channel_mixer_surprise_routing=temporal_channel_mixer_surprise_routing,
            learnable_tau=learnable_tau,
        )
        self.shortcut = ConvBNLIF2d(
            in_channels,
            out_channels,
            1,
            stride=2,
            tau=tau,
            threshold=threshold,
            learnable_tau=learnable_tau,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.proj(x)) + self.shortcut(x)


class TokenQKAttention(nn.Module):
    """Q-K token gating inspired by QKFormer, without an N x N attention map."""

    def __init__(
        self, dim: int, num_heads: int, tau: float, threshold: float, learnable_tau: bool = False
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.proj_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.q_lif = _lif(dim, tau, threshold, learnable_tau)
        self.k_lif = _lif(dim, tau, threshold, learnable_tau)
        self.attn_lif = _lif(num_heads, tau, 0.5, learnable_tau)
        self.proj_lif = _lif(dim, tau, threshold, learnable_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, batch_size, channels, height, width = x.shape
        tokens = x.flatten(3)
        q = _time_distributed(self.q_bn, _time_distributed(self.q_conv, tokens))
        k = _time_distributed(self.k_bn, _time_distributed(self.k_conv, tokens))
        q = self.q_lif(q).reshape(
            time_steps, batch_size, self.num_heads, channels // self.num_heads, -1
        )
        k = self.k_lif(k).reshape_as(q)

        # Sum over head channels -> binary token importance vector, then mask K.
        token_current = q.sum(dim=3, keepdim=True)
        attention = self.attn_lif(token_current)
        output = (attention * k).flatten(2, 3)
        output = _time_distributed(self.proj_bn, _time_distributed(self.proj_conv, output))
        output = output.reshape(time_steps, batch_size, channels, height, width)
        output = self.proj_lif(output)

        return output


class SpikingDepthwiseLocalMixer(nn.Module):
    """Local spatial token mixer for the early high-frequency probe.

    The depthwise convolution acts independently on each spike channel at every timestep. Batch
    normalization and the following LIF use the same ordering as the other convolutional blocks.
    The enclosing :class:`SpikingBlock` owns the residual path.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        tau: float,
        threshold: float,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("depthwise local mixer kernel_size must be an odd integer >= 3")
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(dim)
        self.lif = _lif(dim, tau, threshold, learnable_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _time_distributed(self.conv, x)
        x = _time_distributed(self.bn, x)
        return self.lif(x)


class SpikingSelfAttention(nn.Module):
    """Spike-form self-attention using K^T V followed by Q(K^T V).

    The projection is reshaped to time-major form before its LIF, avoiding cross-sample membrane
    mixing in flattened tensors.
    """

    def __init__(
        self, dim: int, num_heads: int, tau: float, threshold: float, learnable_tau: bool = False
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(dim // num_heads)
        self.q_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.proj_conv = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.q_lif = _lif(dim, tau, threshold, learnable_tau)
        self.k_lif = _lif(dim, tau, threshold, learnable_tau)
        self.v_lif = _lif(dim, tau, threshold, learnable_tau)
        self.attn_lif = _lif(num_heads, tau, 0.5, learnable_tau)
        self.proj_lif = _lif(dim, tau, threshold, learnable_tau)

    def _project(self, x: torch.Tensor, conv, bn, lif) -> torch.Tensor:
        projected = _time_distributed(bn, _time_distributed(conv, x))
        return lif(projected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, batch_size, channels, height, width = x.shape
        token_count = height * width
        flattened = x.flatten(3)
        head_dim = channels // self.num_heads

        q = self._project(flattened, self.q_conv, self.q_bn, self.q_lif)
        k = self._project(flattened, self.k_conv, self.k_bn, self.k_lif)
        v = self._project(flattened, self.v_conv, self.v_bn, self.v_lif)

        def to_heads(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.transpose(-1, -2)
                .reshape(time_steps, batch_size, token_count, self.num_heads, head_dim)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

        qh, kh, vh = map(to_heads, (q, k, v))
        kv = kh.transpose(-2, -1) @ vh
        output = (qh @ kv) * self.scale
        output = self.attn_lif(output)
        output = output.transpose(3, 4).reshape(time_steps, batch_size, channels, token_count)
        output = _time_distributed(self.proj_bn, _time_distributed(self.proj_conv, output))
        output = output.reshape(time_steps, batch_size, channels, height, width)
        output = self.proj_lif(output)

        return output


class SpikingMLP(nn.Module):
    def __init__(
        self, dim: int, ratio: float, tau: float, threshold: float, learnable_tau: bool = False
    ):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = ConvBNLIF2d(
            dim, hidden, 1, tau=tau, threshold=threshold, learnable_tau=learnable_tau
        )
        self.fc2 = ConvBNLIF2d(
            hidden, dim, 1, tau=tau, threshold=threshold, learnable_tau=learnable_tau
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class SpikingBlock(nn.Module):
    """QKFormer block with identity-preserving residual additions outside the LIF nodes."""

    def __init__(
        self,
        attention: nn.Module,
        dim: int,
        mlp_ratio: float,
        tau: float,
        threshold: float,
        learnable_tau: bool = False,
    ):
        super().__init__()
        self.attention = attention
        self.mlp = SpikingMLP(dim, mlp_ratio, tau, threshold, learnable_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.mlp(x)
