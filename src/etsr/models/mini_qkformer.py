from __future__ import annotations

import torch
from torch import nn

from etsr.models.layers import (
    ConvBNLIF2d,
    FineTemporalBranch,
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    PyramidalPatchEmbedding,
    SpikingBlock,
    SpikingDepthwiseLocalMixer,
    SpikingSelfAttention,
    TokenQKAttention,
)
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF
from etsr.models.temporal import CausalTemporalChannelMixer


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
        frontend: str = "baseline",
        temporal_fir: bool = False,
        temporal_fir_kernel_size: int = 3,
        temporal_fir_dilations: tuple[int, int] = (1, 2),
        temporal_channel_mixer: bool = False,
        temporal_channel_mixer_delays: tuple[int, ...] = (1, 2, 4),
        temporal_channel_mixer_learnable_delays: bool = False,
        temporal_channel_mixer_dynamic_routing: bool = False,
        temporal_channel_mixer_router_pooling: str = "global",
        temporal_channel_mixer_router_hidden_divisor: int | None = None,
        temporal_channel_mixer_predictive_auxiliary: bool = False,
        temporal_channel_mixer_predictor_channel_groups: int | None = None,
        temporal_channel_mixer_predictor_spatial_kernel_size: int = 1,
        temporal_channel_mixer_surprise_routing: bool = False,
        temporal_channel_mixer_routing_stages: tuple[int, ...] = (1, 2),
        temporal_channel_mixer_predictive_stages: tuple[int, ...] = (1, 2),
        temporal_channel_mixer_routing_parameterization: str = "independent",
        learnable_lif_tau: bool = False,
        gated_initial_memory_steps: float | None = None,
        multigranular: bool = False,
        multigranular_fine_channels: int = 16,
        multigranular_fine_mid_channels: int | None = None,
        multigranular_temporal_groups: int | None = None,
        multigranular_fusion: str = "add",
        multigranular_micro_steps: int = 8,
        stage1_mixer: str = "token_qk",
        stage1_depthwise_kernel_size: int = 3,
        predictive_head: bool = False,
        predictive_head_spatial_kernel_size: int = 1,
        predictive_head_hidden_channels: int | None = None,
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
        if frontend not in {"baseline", "pyramidal"}:
            raise ValueError(f"Unsupported front-end: {frontend}")
        if frontend == "pyramidal" and embed_dim % 16 != 0:
            raise ValueError("pyramidal front-end requires embed_dim divisible by 16")
        if type(temporal_fir) is not bool:
            raise ValueError("temporal_fir must be boolean")
        if type(temporal_channel_mixer) is not bool:
            raise ValueError("temporal_channel_mixer must be boolean")
        if type(temporal_channel_mixer_learnable_delays) is not bool:
            raise ValueError("temporal_channel_mixer_learnable_delays must be boolean")
        if temporal_channel_mixer_learnable_delays and not temporal_channel_mixer:
            raise ValueError("learnable delays require the temporal channel mixer")
        for name, enabled in (
            ("temporal_channel_mixer_dynamic_routing", temporal_channel_mixer_dynamic_routing),
            ("temporal_channel_mixer_predictive_auxiliary", temporal_channel_mixer_predictive_auxiliary),
            ("temporal_channel_mixer_surprise_routing", temporal_channel_mixer_surprise_routing),
            ("predictive_head", predictive_head),
        ):
            if type(enabled) is not bool:
                raise ValueError(f"{name} must be boolean")
        if temporal_channel_mixer_surprise_routing and not temporal_channel_mixer_predictive_auxiliary:
            raise ValueError("surprise routing requires the predictive auxiliary")
        if temporal_channel_mixer_router_pooling not in {"global", "local"}:
            raise ValueError("temporal mixer router pooling must be global or local")
        if (
            temporal_channel_mixer_router_hidden_divisor is not None
            and temporal_channel_mixer_router_hidden_divisor <= 0
        ):
            raise ValueError("temporal mixer router hidden divisor must be positive or null")
        if not (
            temporal_channel_mixer_dynamic_routing
            or temporal_channel_mixer_surprise_routing
        ) and (
            temporal_channel_mixer_router_pooling != "global"
            or temporal_channel_mixer_router_hidden_divisor is not None
        ):
            raise ValueError("temporal router geometry requires dynamic routing")
        if (
            not temporal_channel_mixer_routing_stages
            or any(stage not in {1, 2} for stage in temporal_channel_mixer_routing_stages)
            or tuple(sorted(set(temporal_channel_mixer_routing_stages)))
            != temporal_channel_mixer_routing_stages
        ):
            raise ValueError("temporal routing stages must be an ordered subset of (1, 2)")
        if (
            not temporal_channel_mixer_predictive_stages
            or any(stage not in {1, 2} for stage in temporal_channel_mixer_predictive_stages)
            or tuple(sorted(set(temporal_channel_mixer_predictive_stages)))
            != temporal_channel_mixer_predictive_stages
        ):
            raise ValueError("temporal predictive stages must be an ordered subset of (1, 2)")
        if temporal_channel_mixer_surprise_routing and not set(
            temporal_channel_mixer_routing_stages
        ) <= set(temporal_channel_mixer_predictive_stages):
            raise ValueError("surprise routing needs a causal predictor in every routed stage")
        if temporal_channel_mixer_routing_parameterization not in {
            "independent",
            "amplitude_allocation",
        }:
            raise ValueError("unsupported temporal routing parameterization")
        if (
            temporal_channel_mixer_predictor_channel_groups is not None
            and temporal_channel_mixer_predictor_channel_groups <= 0
        ):
            raise ValueError("temporal predictor channel groups must be positive or null")
        if (
            temporal_channel_mixer_predictor_spatial_kernel_size <= 0
            or temporal_channel_mixer_predictor_spatial_kernel_size % 2 == 0
        ):
            raise ValueError("temporal predictor spatial kernel must be a positive odd integer")
        if not temporal_channel_mixer_predictive_auxiliary and (
            temporal_channel_mixer_predictor_channel_groups is not None
            or temporal_channel_mixer_predictor_spatial_kernel_size != 1
        ):
            raise ValueError("temporal predictor geometry requires predictive auxiliary training")
        if (
            temporal_channel_mixer_dynamic_routing
            or temporal_channel_mixer_predictive_auxiliary
        ) and not temporal_channel_mixer:
            raise ValueError("conditional temporal options require the temporal channel mixer")
        if type(learnable_lif_tau) is not bool:
            raise ValueError("learnable_lif_tau must be boolean")
        if temporal_fir and temporal_channel_mixer:
            raise ValueError("temporal FIR and channel mixer are mutually exclusive")
        if temporal_fir_kernel_size < 2:
            raise ValueError("temporal_fir_kernel_size must be at least two")
        if len(temporal_fir_dilations) != 2 or any(
            type(dilation) is not int or dilation <= 0 for dilation in temporal_fir_dilations
        ):
            raise ValueError("temporal_fir_dilations must contain two positive integers")
        if temporal_fir and not lif_cross_time:
            raise ValueError("no-cross-time control cannot include temporal FIR memory")
        if temporal_channel_mixer and not lif_cross_time:
            raise ValueError("no-cross-time control cannot include a temporal channel mixer")
        if (
            not temporal_channel_mixer_delays
            or any(type(delay) is not int or delay <= 0 for delay in temporal_channel_mixer_delays)
            or tuple(sorted(set(temporal_channel_mixer_delays))) != temporal_channel_mixer_delays
        ):
            raise ValueError("temporal channel mixer delays must be increasing positive integers")
        if gated_initial_memory_steps is not None and gated_initial_memory_steps <= 1.0:
            raise ValueError("gated_initial_memory_steps must be greater than one")
        if type(multigranular) is not bool:
            raise ValueError("multigranular must be boolean")
        if multigranular and frontend != "pyramidal":
            raise ValueError("multi-granular branches require the pyramidal front-end")
        if multigranular_fine_channels <= 0:
            raise ValueError("multigranular_fine_channels must be positive")
        if (
            multigranular_fine_mid_channels is not None
            and multigranular_fine_mid_channels <= 0
        ):
            raise ValueError("multigranular_fine_mid_channels must be positive or null")
        if multigranular_temporal_groups is not None and multigranular_temporal_groups <= 0:
            raise ValueError("multigranular_temporal_groups must be positive or null")
        if multigranular_fusion not in {"add", "concat_residual"}:
            raise ValueError("multigranular_fusion must be add or concat_residual")
        if not multigranular and multigranular_fusion != "add":
            raise ValueError("A learned multi-granular fusion requires multigranular=true")
        if multigranular_micro_steps <= 1:
            raise ValueError("multigranular_micro_steps must be greater than one")
        if stage1_mixer not in {"token_qk", "depthwise_conv"}:
            raise ValueError("stage1_mixer must be token_qk or depthwise_conv")
        if stage1_depthwise_kernel_size < 3 or stage1_depthwise_kernel_size % 2 == 0:
            raise ValueError("stage1_depthwise_kernel_size must be an odd integer >= 3")
        if predictive_head_spatial_kernel_size <= 0 or predictive_head_spatial_kernel_size % 2 == 0:
            raise ValueError("predictive head spatial kernel must be a positive odd integer")
        if predictive_head_hidden_channels is not None and predictive_head_hidden_channels <= 0:
            raise ValueError("predictive head hidden channels must be positive or null")
        if not predictive_head and (
            predictive_head_spatial_kernel_size != 1
            or predictive_head_hidden_channels is not None
        ):
            raise ValueError("predictive-head geometry requires predictive_head=true")
        self.num_classes = num_classes
        self.readout_name = readout
        self.readout_time = readout_time
        self.frontend_name = frontend
        self.temporal_fir_enabled = temporal_fir
        self.temporal_channel_mixer_enabled = temporal_channel_mixer
        self.temporal_channel_mixer_learnable_delays = temporal_channel_mixer_learnable_delays
        self.temporal_channel_mixer_dynamic_routing = temporal_channel_mixer_dynamic_routing
        self.temporal_channel_mixer_predictive_auxiliary = (
            temporal_channel_mixer_predictive_auxiliary
        )
        self.temporal_channel_mixer_surprise_routing = temporal_channel_mixer_surprise_routing
        self.temporal_channel_mixer_routing_stages = temporal_channel_mixer_routing_stages
        self.temporal_channel_mixer_predictive_stages = temporal_channel_mixer_predictive_stages
        self.learnable_lif_tau_enabled = learnable_lif_tau
        self.multigranular_enabled = multigranular
        self.multigranular_fusion_name = multigranular_fusion
        self.stage1_mixer_name = stage1_mixer
        half = embed_dim // 2

        first_fir = temporal_fir_kernel_size if temporal_fir else None
        channel_mixer_delays = temporal_channel_mixer_delays if temporal_channel_mixer else None
        frontend_class = (
            PyramidalPatchEmbedding if frontend == "pyramidal" else InitialPatchEmbedding
        )
        self.patch_embed1 = frontend_class(
            in_channels,
            embed_dim,
            lif_tau,
            lif_threshold,
            temporal_fir_kernel_size=first_fir,
            temporal_fir_dilation=temporal_fir_dilations[0],
            temporal_channel_mixer_delays=channel_mixer_delays,
            temporal_channel_mixer_learnable_delays=temporal_channel_mixer_learnable_delays,
            temporal_channel_mixer_dynamic_routing=(
                temporal_channel_mixer_dynamic_routing
                and 1 in temporal_channel_mixer_routing_stages
            ),
            temporal_channel_mixer_router_pooling=temporal_channel_mixer_router_pooling,
            temporal_channel_mixer_router_hidden_divisor=temporal_channel_mixer_router_hidden_divisor,
            temporal_channel_mixer_predictive_auxiliary=(
                temporal_channel_mixer_predictive_auxiliary
                and 1 in temporal_channel_mixer_predictive_stages
            ),
            temporal_channel_mixer_predictor_channel_groups=temporal_channel_mixer_predictor_channel_groups,
            temporal_channel_mixer_predictor_spatial_kernel_size=temporal_channel_mixer_predictor_spatial_kernel_size,
            temporal_channel_mixer_surprise_routing=(
                temporal_channel_mixer_surprise_routing
                and 1 in temporal_channel_mixer_routing_stages
            ),
            temporal_channel_mixer_routing_parameterization=(
                temporal_channel_mixer_routing_parameterization
            ),
            learnable_tau=learnable_lif_tau,
        )
        self.fine_temporal_branch = (
            FineTemporalBranch(
                in_channels=2,
                hidden_channels=multigranular_fine_channels,
                out_channels=half,
                micro_steps_per_macro=multigranular_micro_steps,
                tau=lif_tau,
                threshold=lif_threshold,
                mid_channels=multigranular_fine_mid_channels,
                temporal_groups=multigranular_temporal_groups,
            )
            if self.multigranular_enabled
            else None
        )
        self.multigranular_fusion = (
            ConvBNLIF2d(
                embed_dim,
                half,
                1,
                tau=lif_tau,
                threshold=lif_threshold,
            )
            if multigranular and multigranular_fusion == "concat_residual"
            else None
        )
        stage1_attention = (
            TokenQKAttention(half, num_heads, lif_tau, lif_threshold, learnable_lif_tau)
            if stage1_mixer == "token_qk"
            else SpikingDepthwiseLocalMixer(
                half,
                stage1_depthwise_kernel_size,
                lif_tau,
                lif_threshold,
                learnable_lif_tau,
            )
        )
        self.stage1 = SpikingBlock(
            attention=stage1_attention,
            dim=half,
            mlp_ratio=mlp_ratio,
            tau=lif_tau,
            threshold=lif_threshold,
            learnable_tau=learnable_lif_tau,
        )
        self.patch_embed2 = PatchEmbeddingStage(
            half,
            embed_dim,
            lif_tau,
            lif_threshold,
            temporal_fir_kernel_size=first_fir,
            temporal_fir_dilation=temporal_fir_dilations[1],
            temporal_channel_mixer_delays=channel_mixer_delays,
            temporal_channel_mixer_learnable_delays=temporal_channel_mixer_learnable_delays,
            temporal_channel_mixer_dynamic_routing=(
                temporal_channel_mixer_dynamic_routing
                and 2 in temporal_channel_mixer_routing_stages
            ),
            temporal_channel_mixer_router_pooling=temporal_channel_mixer_router_pooling,
            temporal_channel_mixer_router_hidden_divisor=temporal_channel_mixer_router_hidden_divisor,
            temporal_channel_mixer_predictive_auxiliary=(
                temporal_channel_mixer_predictive_auxiliary
                and 2 in temporal_channel_mixer_predictive_stages
            ),
            temporal_channel_mixer_predictor_channel_groups=temporal_channel_mixer_predictor_channel_groups,
            temporal_channel_mixer_predictor_spatial_kernel_size=temporal_channel_mixer_predictor_spatial_kernel_size,
            temporal_channel_mixer_surprise_routing=(
                temporal_channel_mixer_surprise_routing
                and 2 in temporal_channel_mixer_routing_stages
            ),
            temporal_channel_mixer_routing_parameterization=(
                temporal_channel_mixer_routing_parameterization
            ),
            learnable_tau=learnable_lif_tau,
        )
        self.stage2 = SpikingBlock(
            attention=SpikingSelfAttention(
                embed_dim, num_heads, lif_tau, lif_threshold, learnable_lif_tau
            ),
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            tau=lif_tau,
            threshold=lif_threshold,
            learnable_tau=learnable_lif_tau,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        if not predictive_head:
            self.predictive_head = None
        elif predictive_head_hidden_channels is None and predictive_head_spatial_kernel_size == 1:
            self.predictive_head = nn.Conv2d(half, half, 1, bias=True)
        else:
            hidden = predictive_head_hidden_channels or half
            kernel = predictive_head_spatial_kernel_size
            self.predictive_head = nn.Sequential(
                nn.Conv2d(half, half, kernel, padding=kernel // 2, groups=half, bias=False),
                nn.Conv2d(half, hidden, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, half, 1, bias=True),
            )
        self.gated_readout = (
            DiagonalGatedReadout(embed_dim, gated_initial_memory_steps)
            if readout == "diagonal_gated"
            else None
        )

        for module in self.modules():
            if isinstance(module, MultiStepLIF):
                module.surrogate_alpha = float(surrogate_alpha)
                module.cross_time = lif_cross_time
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, CausalTemporalChannelMixer):
                module._initialize_conditional_modules()
                # V_delta is a stop metric for every stage whenever a predictive objective exists,
                # including stages that carry no predictor but receive its backpropagated gradient.
                module.record_temporal_variation = temporal_channel_mixer_predictive_auxiliary
        if self.fine_temporal_branch is not None:
            self.fine_temporal_branch.initialize_temporal_reducer()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
            module, (nn.Conv1d, nn.Conv2d, nn.Linear)
        ):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def extract_stage1(self, frames: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(frames, dict):
            if set(frames) != {"coarse", "fine"}:
                raise ValueError("A coarse/fine input must contain exactly coarse and fine.")
            coarse_frames = frames["coarse"]
            fine_frames = frames["fine"]
        else:
            if self.multigranular_enabled:
                raise ValueError("A multi-granular branch requires coarse and fine input streams.")
            coarse_frames = frames
            fine_frames = None
        if coarse_frames.ndim != 5:
            raise ValueError("Expected input [B, T, C, H, W].")
        x = coarse_frames.permute(1, 0, 2, 3, 4).contiguous()
        x = self.patch_embed1(x)
        if fine_frames is not None and self.multigranular_enabled:
            assert self.fine_temporal_branch is not None
            fine = self.fine_temporal_branch(fine_frames)
            if fine.shape != x.shape:
                raise ValueError(
                    f"Fine branch output {tuple(fine.shape)} does not match coarse {tuple(x.shape)}."
                )
            if self.multigranular_fusion_name == "concat_residual":
                assert self.multigranular_fusion is not None
                x = x + self.multigranular_fusion(torch.cat((x, fine), dim=2))
            else:
                x = x + fine
        x = self.stage1(x)
        return x

    def _encode(self, frames: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.extract_stage1(frames)
        return self.encode_from_stage1(x)

    def encode_from_stage1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed2(x)
        return self.stage2(x)

    def logits_from_stage1(
        self,
        stage1: torch.Tensor,
        frames: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded = self.encode_from_stage1(stage1)
        valid_steps = self._last_event_steps(frames) if self.readout_time == "last_event" else None
        return self.head(self._readout(encoded, valid_steps))

    def extract_fine_pre_lif(self, frames: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fine_temporal_branch is None:
            raise ValueError("Fine targets require a multi-granular teacher.")
        if set(frames) != {"coarse", "fine"}:
            raise ValueError("Fine targets require coarse and fine input streams.")
        return self.fine_temporal_branch.forward_pre_lif(frames["fine"])

    def predict_stage1_target(self, context: torch.Tensor) -> torch.Tensor:
        if self.predictive_head is None:
            raise ValueError("This model has no training-only predictive head.")
        time_steps, batch = context.shape[:2]
        projected = self.predictive_head(context.flatten(0, 1))
        return projected.reshape(time_steps, batch, *projected.shape[1:])

    def temporal_auxiliary_loss(self) -> torch.Tensor | None:
        losses = [
            module.auxiliary_loss()
            for module in self.modules()
            if isinstance(module, CausalTemporalChannelMixer)
            and module.predictive_auxiliary
        ]
        valid = [loss for loss in losses if loss is not None]
        return torch.stack(valid).mean() if valid else None

    def temporal_auxiliary_statistics(
        self,
        valid_steps: torch.Tensor,
        *,
        region_weights: dict[str, float] | None = None,
        stage_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, float], dict[str, torch.Tensor]]:
        """Return an explicitly stage/region-weighted causal predictor objective."""

        region_weights = region_weights or {"active": 1.0, "tail": 0.25}
        stage_weights = stage_weights or {"stage1": 1.0, "stage2": 1.0}
        staged = self._temporal_prediction_region_values_by_stage(valid_steps)
        metrics: dict[str, float] = {}
        components: dict[str, torch.Tensor] = {}
        weighted_terms: list[torch.Tensor] = []
        weights: list[float] = []
        for stage, values in staged.items():
            metrics.update(
                {
                    f"{stage}_{name}": value
                    for name, value in self._temporal_prediction_metrics(values).items()
                }
            )
            for region in ("active", "tail"):
                key = f"prediction_{region}"
                weight = stage_weights[stage] * region_weights[region]
                if key in values and weight > 0:
                    weighted_terms.append(values[key] * weight)
                    weights.append(weight)
        if not weighted_terms:
            return None, metrics, components
        total_weight = sum(weights)
        for region in ("active", "tail"):
            terms = []
            for stage, values in staged.items():
                key = f"prediction_{region}"
                weight = stage_weights[stage] * region_weights[region]
                if key in values and weight > 0:
                    terms.append(values[key] * weight)
            if terms:
                # Components are their exact contribution to the total auxiliary objective.
                # They therefore sum to the returned loss and can be differentiated separately.
                components[region] = torch.stack(terms).sum() / total_weight
        combined = self._temporal_prediction_region_values(valid_steps)
        metrics.update(self._temporal_prediction_metrics(combined))
        metrics.update(self._temporal_variation_metrics(valid_steps))
        return torch.stack(weighted_terms).sum() / total_weight, metrics, components

    def temporal_prediction_statistics(self, valid_steps: torch.Tensor) -> dict[str, float]:
        """Measure predictor skill against causal persistence and delay-mean references."""

        metrics = self._temporal_prediction_metrics(
            self._temporal_prediction_region_values(valid_steps)
        )
        for stage, values in self._temporal_prediction_region_values_by_stage(
            valid_steps
        ).items():
            metrics.update(
                {
                    f"{stage}_{name}": value
                    for name, value in self._temporal_prediction_metrics(values).items()
                }
            )
        metrics.update(self._temporal_variation_metrics(valid_steps))
        return metrics

    def _temporal_prediction_region_values(
        self, valid_steps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        staged = self._temporal_prediction_region_values_by_stage(valid_steps)
        collected: dict[str, list[torch.Tensor]] = {}
        for values in staged.values():
            for name, value in values.items():
                collected.setdefault(name, []).append(value)
        return {
            name: torch.stack(region_values).mean()
            for name, region_values in collected.items()
        }

    def _temporal_prediction_region_values_by_stage(
        self, valid_steps: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        staged: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in self.named_modules():
            if not isinstance(module, CausalTemporalChannelMixer) or not module.predictive_auxiliary:
                continue
            diagnostic = module.prediction_diagnostics()
            if diagnostic is None:
                continue
            stage = "stage1" if name.startswith("patch_embed1") else "stage2"
            collected: dict[str, list[torch.Tensor]] = {}
            reference = diagnostic["prediction_error"]
            positions = torch.arange(reference.shape[0], device=reference.device).unsqueeze(1)
            active_mask = positions < valid_steps.unsqueeze(0)
            tail_mask = ~active_mask
            for region, mask in (("active", active_mask), ("tail", tail_mask)):
                counts = mask.sum(0)
                present = counts > 0
                if not bool(present.any().item()):
                    continue
                for name, error in diagnostic.items():
                    per_sample = (error * mask).sum(0) / counts.clamp_min(1)
                    collected.setdefault(f"{name.removesuffix('_error')}_{region}", []).append(
                        per_sample[present].mean()
                    )
            staged[stage] = {
                key: torch.stack(region_values).mean()
                for key, region_values in collected.items()
            }
        return staged

    def _temporal_variation_metrics(self, valid_steps: torch.Tensor) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, module in self.named_modules():
            if not isinstance(module, CausalTemporalChannelMixer):
                continue
            variation = module.last_temporal_variation
            if variation is None or variation.shape[0] == 0:
                continue
            stage = "stage1" if name.startswith("patch_embed1") else "stage2"
            positions = torch.arange(1, variation.shape[0] + 1, device=variation.device).unsqueeze(1)
            mask = positions < valid_steps.unsqueeze(0)
            counts = mask.sum(0)
            present = counts > 0
            if bool(present.any().item()):
                per_sample = (variation * mask).sum(0) / counts.clamp_min(1)
                metrics[f"temporal_variation_{stage}_active"] = float(
                    per_sample[present].mean().detach()
                )
        return metrics

    @staticmethod
    def _temporal_prediction_metrics(
        values: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        display_names = {
            "prediction": "temporal_prediction",
            "persistence": "temporal_persistence",
            "delay_mean": "temporal_delay_mean",
            "target_variance": "temporal_target",
        }
        for region in ("active", "tail"):
            for source, prefix in display_names.items():
                key = f"{source}_{region}"
                if key not in values:
                    continue
                suffix = "variance" if source == "target_variance" else "loss"
                metrics[f"{prefix}_{region}_{suffix}"] = float(values[key].detach())
            prediction_key = f"prediction_{region}"
            if prediction_key not in values:
                continue
            for reference in ("persistence", "delay_mean"):
                reference_key = f"{reference}_{region}"
                if reference_key in values:
                    skill = 1.0 - values[prediction_key].detach() / values[
                        reference_key
                    ].detach().clamp_min(1e-12)
                    metrics[
                        f"temporal_prediction_{region}_skill_vs_{reference}"
                    ] = float(skill)
        return metrics

    def forward(self, frames: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._encode(frames)
        valid_steps = self._last_event_steps(frames) if self.readout_time == "last_event" else None
        pooled = self._readout(x, valid_steps)
        return self.head(pooled)

    @staticmethod
    def _last_event_steps(frames: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the one-based final occupied bin for each encoded sample."""

        if isinstance(frames, dict):
            frames = frames["coarse"]
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
