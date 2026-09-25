"""Small hardware-oriented profiler for the active spiking models."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from etsr.data.events import (
    encoded_batch_size,
    move_encoded_input,
    select_encoded_batch,
    slice_encoded_batch,
)
from etsr.models.layers import (
    FineTemporalBranch,
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    PyramidalPatchEmbedding,
    SpikingBlock,
    SpikingSelfAttention,
    TokenQKAttention,
)
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF
from etsr.models.temporal import CausalTemporalChannelMixer, CausalTemporalFIR
from etsr.profiling.energy import horowitz_reference


class _HardwareProfiler:
    """Collect analytical operation counts and activity observed during ordinary forwards."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.batch_size = 1
        self.totals: dict[str, float] = defaultdict(float)
        self.layers: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.state_shapes: dict[str, tuple[int, int]] = {}
        self.gate_channel_sums: dict[str, torch.Tensor] = {}
        self.gate_step_counts: dict[str, int] = defaultdict(int)
        self.handles: list[Any] = []
        for name, module in model.named_modules():
            if isinstance(module, FineTemporalBranch):
                hook = self._fine_temporal_branch_hook(name)
            elif isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
                module, (nn.Conv1d, nn.Conv2d, nn.Linear)
            ):
                hook = self._dense_hook(name)
            elif isinstance(module, MultiStepLIF):
                hook = self._lif_hook(name)
            elif isinstance(module, CausalTemporalFIR):
                hook = self._temporal_fir_hook(name)
            elif isinstance(module, CausalTemporalChannelMixer):
                hook = self._temporal_channel_mixer_hook(name)
            elif isinstance(module, nn.MaxPool2d):
                hook = self._maxpool_hook(name)
            elif isinstance(module, DiagonalGatedReadout):
                hook = self._gated_readout_hook(name)
            elif isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
                module, (TokenQKAttention, SpikingSelfAttention)
            ):
                hook = self._attention_hook(name)
            elif isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
                module,
                (
                    InitialPatchEmbedding,
                    PyramidalPatchEmbedding,
                    PatchEmbeddingStage,
                    SpikingBlock,
                ),
            ):
                hook = self._residual_hook(name)
            else:
                continue
            self.handles.append(module.register_forward_hook(hook))

    def _fine_temporal_branch_hook(self, name: str):
        def record(
            module: FineTemporalBranch,
            _inputs: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            ratio = module.micro_steps_per_macro
            elements = output.numel()
            additions = elements
            if module.mid_channels is not None:
                additions += (
                    elements
                    * ratio
                    * module.mid_channels
                    / module.temporal_reduce.out_channels
                )
            history_reads = elements * ratio
            history_writes = elements * ratio
            layer = self.layers[name]
            layer["elementwise_add"] += additions
            layer["state_reads"] += history_reads
            layer["state_writes"] += history_writes
            self.totals["elementwise_add"] += additions
            self.totals["state_reads"] += history_reads
            self.totals["recurrent_state_updates"] += history_writes
            state_elements = (ratio - 1) * output[0].numel() // output.shape[1]
            self.state_shapes[name] = (
                state_elements,
                output.element_size() * 8,
            )

        return record

    def _maxpool_hook(self, name: str):
        def record(
            module: nn.MaxPool2d, _inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            kernel_size = module.kernel_size
            if isinstance(kernel_size, int):
                kernel_elements = kernel_size**2
            else:
                kernel_elements = kernel_size[0] * kernel_size[1]
            comparisons = output.numel() * (kernel_elements - 1)
            self.layers[name]["maxpool_comparison"] += comparisons
            self.totals["maxpool_comparison"] += comparisons
            self._activation(output)

        return record

    def _temporal_fir_hook(self, name: str):
        def record(
            module: CausalTemporalFIR,
            _inputs: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            elements = output.numel()
            multiplies = elements * module.kernel_size
            additions = elements * (module.kernel_size - 1)
            history_reads = elements * (module.kernel_size - 1)
            layer = self.layers[name]
            layer["temporal_fir_multiply"] += multiplies
            layer["temporal_fir_add"] += additions
            layer["state_reads"] += history_reads
            layer["state_writes"] += elements
            self.totals["temporal_fir_multiply"] += multiplies
            self.totals["temporal_fir_add"] += additions
            self.totals["elementwise_multiply"] += multiplies
            self.totals["elementwise_add"] += additions
            self.totals["state_reads"] += history_reads
            self.totals["recurrent_state_updates"] += elements
            state_elements = module.max_delay * output[0].numel() // output.shape[1]
            self.state_shapes[name] = (state_elements, output.element_size() * 8)
            self._activation(output)

        return record

    def _dense_hook(self, name: str):
        def record(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
            values = inputs[0].detach()
            if isinstance(module, nn.Conv2d):
                kernel_ops = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.in_channels
                    // module.groups
                )
            elif isinstance(module, nn.Conv1d):
                kernel_ops = module.kernel_size[0] * module.in_channels // module.groups
            else:
                kernel_ops = module.in_features
            operations = float(output.numel() * kernel_ops)
            nonzero_elements = torch.count_nonzero(values).item()
            nonzero_rate = float(nonzero_elements / max(1, values.numel()))
            binary_input = bool(torch.all((values == 0) | (values == 1)).item())
            layer = self.layers[name]
            layer["calls"] += 1
            layer["input_elements"] += values.numel()
            layer["input_nonzero"] += nonzero_elements
            layer["binary_calls"] += int(binary_input)
            kind = "binary_ac" if binary_input else "multivalued_mac"
            layer[f"{kind}_potential"] += operations
            self.totals[f"{kind}_potential"] += operations
            if binary_input:
                layer["binary_ac_activity_estimate"] += operations * nonzero_rate
                self.totals["binary_ac_activity_estimate"] += operations * nonzero_rate
            self._activation(output)

        return record

    def _temporal_channel_mixer_hook(self, name: str):
        def record(
            module: CausalTemporalChannelMixer,
            _inputs: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            macs = output.numel() * module.channels * len(module.delays)
            history_reads = output.numel() * len(module.delays)
            layer = self.layers[name]
            layer["temporal_channel_mixer_mac"] += macs
            layer["state_reads"] += history_reads
            layer["state_writes"] += output.numel()
            if module.learnable_delays:
                layer["delay_address_select"] += history_reads
                self.totals["delay_address_select"] += history_reads
            else:
                layer["delays"] = module.delays
            self.totals["temporal_channel_mixer_mac"] += macs
            self.totals["multivalued_mac_potential"] += macs
            self.totals["state_reads"] += history_reads
            self.totals["recurrent_state_updates"] += output.numel()
            has_conditional_gating = (
                module.content_router is not None or module.surprise_router is not None
            )
            if has_conditional_gating:
                spatial_positions = math.prod(output.shape[3:]) if output.ndim > 3 else 1
                routed_positions = spatial_positions if module.router_pooling == "local" else 1
                router_outputs = len(module.delays) + int(
                    module.routing_parameterization == "amplitude_allocation"
                )
                router_macs = 0
                if module.content_router is not None:
                    router_macs = (
                        output.shape[0]
                        * output.shape[1]
                        * routed_positions
                        * module.channels
                        * router_outputs
                    )
                pooling_adds = (
                    output.numel()
                    if module.content_router is not None and module.router_pooling == "global"
                    else 0
                )
                gate_multiplies = output.numel() * len(module.delays)
                layer["content_router_mac"] += router_macs
                layer["global_pool_add"] += pooling_adds
                layer["tap_gate_multiply"] += gate_multiplies
                # Router Linear/Conv MACs are already included by their ordinary hooks.
                self.totals["elementwise_add"] += pooling_adds
                self.totals["elementwise_multiply"] += gate_multiplies
                routed_observations = output.shape[0] * output.shape[1] * routed_positions
                if module.routing_parameterization == "independent":
                    self.totals["sigmoid"] += routed_observations * len(module.delays)
                else:
                    self.totals["sigmoid"] += routed_observations
                    self.totals["softmax"] += routed_observations * len(module.delays)
            if module.predictive_auxiliary:
                predictor_multiplies = (
                    output.numel() * len(module.delays)
                    if module.predictor_logits is not None
                    else 0
                )
                predictor_adds = output.numel() * max(0, len(module.delays) - 1)
                layer["causal_predictor_multiply"] += predictor_multiplies
                layer["causal_predictor_add"] += predictor_adds
                self.totals["elementwise_multiply"] += predictor_multiplies
                self.totals["elementwise_add"] += predictor_adds
            if module.surprise_routing:
                observations = output.shape[0] * output.shape[1] * module.channels
                # The vector-error projection is an nn.Linear and is counted by its own hook.
                surprise_gate_multiplies = 0
                surprise_elements = output.numel()
                surprise_reduction_adds = 2 * (surprise_elements - observations)
                surprise_mean_scales = 2 * observations
                surprise_logit_adds = (
                    output.shape[0] * output.shape[1] * len(module.delays)
                    if module.content_router is not None
                    else 0
                )
                layer["surprise_gate_multiply"] += surprise_gate_multiplies
                layer["surprise_error_subtract"] += surprise_elements
                layer["surprise_absolute_value"] += surprise_elements
                layer["surprise_square"] += surprise_elements
                layer["surprise_reduction_add"] += surprise_reduction_adds
                layer["surprise_mean_scale_multiply"] += surprise_mean_scales
                layer["surprise_sqrt"] += observations
                layer["surprise_divide"] += observations
                layer["surprise_logit_add"] += surprise_logit_adds
                self.totals["elementwise_multiply"] += (
                    surprise_gate_multiplies + surprise_mean_scales
                )
                self.totals["elementwise_add"] += surprise_logit_adds
                self.totals["surprise_error_subtract"] += surprise_elements
                self.totals["surprise_absolute_value"] += surprise_elements
                self.totals["surprise_square"] += surprise_elements
                self.totals["surprise_reduction_add"] += surprise_reduction_adds
                self.totals["surprise_mean_scale_multiply"] += surprise_mean_scales
                self.totals["surprise_sqrt"] += observations
                self.totals["surprise_divide"] += observations
                self.totals["surprise_logit_add"] += surprise_logit_adds
            routing = module.last_routing_statistics
            if routing is not None:
                means = routing["gate_mean_by_delay"].cpu()
                stds = routing["gate_std_by_delay"].cpu()
                for index, delay in enumerate(module.delays):
                    layer[f"observed_gate_mean_delay_{delay}"] += float(means[index])
                    layer[f"observed_gate_std_delay_{delay}"] += float(stds[index])
                contributions = routing.get("effective_contribution_mean_abs_by_delay")
                if contributions is not None:
                    for index, delay in enumerate(module.delays):
                        layer[f"observed_effective_contribution_mean_abs_delay_{delay}"] += float(
                            contributions[index].cpu()
                        )
                surprise = float(routing["surprise_mean"].cpu())
                if not torch.isnan(torch.tensor(surprise)):
                    layer["observed_normalized_surprise_mean"] += surprise
            state_elements = module.max_delay * output[0].numel() // output.shape[1]
            self.state_shapes[name] = (state_elements, output.element_size() * 8)
            self._activation(output)

        return record

    def _lif_hook(self, name: str):
        def record(
            module: MultiStepLIF,
            _inputs: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            spikes = output.detach()
            layer = self.layers[name]
            layer["spikes"] += spikes.sum().item()
            layer["spike_elements"] += spikes.numel()
            layer["cross_time_state"] = int(module.cross_time)
            self.totals["spikes"] += spikes.sum().item()
            self.totals["spike_elements"] += spikes.numel()
            self.totals["lif_evaluations"] += spikes.numel()
            if module.cross_time:
                self.totals["lif_recurrent_evaluations"] += spikes.numel()
                self.totals["lif_reset_spikes"] += spikes.sum().item()
                self.totals["recurrent_state_updates"] += spikes.numel()
                self.totals["state_reads"] += (spikes.shape[0] - 1) * spikes[0].numel()
                self.state_shapes[name] = (
                    int(spikes[0].numel() // spikes.shape[1]),
                    spikes.element_size() * 8,
                )
            self._activation(output)

        return record

    def _gated_readout_hook(self, name: str):
        def record(
            module: DiagonalGatedReadout,
            inputs: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            sequence = inputs[0]
            time_steps, batch_size, channels = sequence.shape
            valid_steps = inputs[1] if len(inputs) > 1 else None
            active_steps = (
                int(valid_steps.sum().item())
                if valid_steps is not None
                else time_steps * batch_size
            )
            elements = time_steps * batch_size * channels
            # Gate and candidate each use one input and one recurrent channelwise weight.
            affine_macs = 4 * elements
            layer = self.layers[name]
            layer["multivalued_mac_potential"] += affine_macs
            layer["state_mix_multiply"] += elements
            layer["state_mix_add"] += 2 * elements
            layer["sigmoid"] += elements
            layer["tanh"] += elements
            self.totals["multivalued_mac_potential"] += affine_macs
            self.totals["elementwise_multiply"] += elements
            self.totals["elementwise_add"] += 2 * elements
            self.totals["sigmoid"] += elements
            self.totals["tanh"] += elements
            self.totals["recurrent_state_updates"] += elements
            self.totals["state_reads"] += (time_steps - 1) * batch_size * channels
            self.state_shapes[name] = (channels, output.element_size() * 8)
            layer["active_state_updates"] += active_steps * channels

            state = module.reset_state(sequence[0])
            gate_sum = torch.zeros(channels, device=sequence.device)
            for index, current in enumerate(sequence.unbind(0)):
                next_state, gate = module.transition(current, state)
                if valid_steps is None:
                    active = torch.ones(batch_size, dtype=torch.bool, device=sequence.device)
                else:
                    active = index < valid_steps
                gate_sum += gate[active].sum(dim=0)
                state = torch.where(active.unsqueeze(1), next_state, state)
            previous = self.gate_channel_sums.get(name)
            detached_sum = gate_sum.detach().cpu()
            self.gate_channel_sums[name] = (
                detached_sum if previous is None else previous + detached_sum
            )
            self.gate_step_counts[name] += active_steps
            self._activation(output)

        return record

    def _attention_hook(self, name: str):
        def record(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
            time_steps, batch_size, channels, height, width = inputs[0].shape
            tokens = height * width
            head_dim = channels // module.num_heads
            if isinstance(module, TokenQKAttention):
                reduction_adds = (
                    time_steps * batch_size * module.num_heads * tokens * (head_dim - 1)
                )
                attention_sops = time_steps * batch_size * channels * tokens
                self.layers[name]["token_reduction_add"] += reduction_adds
                self.totals["elementwise_add"] += reduction_adds
            else:
                attention_sops = (
                    2 * time_steps * batch_size * module.num_heads * tokens * head_dim**2
                )
                scale_ops = time_steps * batch_size * channels * tokens
                self.layers[name]["scale_multiply"] += scale_ops
                self.totals["attention_scale_multiply"] += scale_ops
            self.layers[name]["attention_sop_potential"] += attention_sops
            self.totals["attention_sop_potential"] += attention_sops
            self._activation(output)

        return record

    def _residual_hook(self, name: str):
        def record(module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
            additions = output.numel() * (2 if isinstance(module, SpikingBlock) else 1)
            self.layers[name]["residual_add"] += additions
            self.totals["elementwise_add"] += additions

        return record

    def _activation(self, output: torch.Tensor) -> None:
        self.totals["max_activation_elements"] = max(
            self.totals["max_activation_elements"], output.numel() // self.batch_size
        )
        self.totals["max_activation_bits"] = max(
            self.totals["max_activation_bits"], output.element_size() * 8
        )

    def summary(self, samples: int) -> dict[str, Any]:
        per_sample = {
            name: self.totals[name] / samples
            for name in (
                "multivalued_mac_potential",
                "binary_ac_potential",
                "binary_ac_activity_estimate",
                "attention_sop_potential",
                "attention_scale_multiply",
                "temporal_fir_multiply",
                "temporal_fir_add",
                "temporal_channel_mixer_mac",
                "delay_address_select",
                "maxpool_comparison",
                "elementwise_add",
                "elementwise_multiply",
                "surprise_error_subtract",
                "surprise_absolute_value",
                "surprise_square",
                "surprise_reduction_add",
                "surprise_mean_scale_multiply",
                "surprise_sqrt",
                "surprise_divide",
                "surprise_logit_add",
            )
        }
        per_sample["sop_potential"] = (
            per_sample["binary_ac_potential"] + per_sample["attention_sop_potential"]
        )
        lif_evaluations = self.totals["lif_evaluations"] / samples
        state_updates = self.totals["recurrent_state_updates"] / samples
        layers: dict[str, dict[str, float | int]] = {}
        for name, values in self.layers.items():
            layer = {
                key: value / samples
                for key, value in values.items()
                if key
                not in {
                    "calls",
                    "input_elements",
                    "input_nonzero",
                    "binary_calls",
                    "spikes",
                    "spike_elements",
                }
                and key not in {"cross_time_state", "persistent_state_elements", "delays"}
            }
            if values.get("input_elements"):
                layer["input_nonzero_rate"] = values["input_nonzero"] / values["input_elements"]
                layer["binary_input_call_fraction"] = values["binary_calls"] / values["calls"]
            if values.get("spike_elements"):
                layer["firing_rate"] = values["spikes"] / values["spike_elements"]
                layer["cross_time_state"] = bool(values["cross_time_state"])
            if name in self.state_shapes:
                layer["persistent_state_elements"] = self.state_shapes[name][0]
            elif values.get("spike_elements"):
                layer["persistent_state_elements"] = 0
            if name in self.gate_channel_sums:
                channel_means = self.gate_channel_sums[name] / self.gate_step_counts[name]
                layer["active_state_update_fraction"] = (
                    values["active_state_updates"] / values["sigmoid"]
                )
                layer["observed_update_gate"] = {
                    "mean": float(channel_means.mean().item()),
                    "minimum_channel_mean": float(channel_means.min().item()),
                    "maximum_channel_mean": float(channel_means.max().item()),
                    "mean_by_channel": channel_means.tolist(),
                }
            if "delays" in values:
                layer["delays"] = list(values["delays"])
            layers[name] = layer

        deployable_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("predictive_head.")
        ]
        training_only_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name.startswith("predictive_head.")
        ]
        parameter_dtypes: dict[str, int] = defaultdict(int)
        for parameter in deployable_parameters:
            parameter_dtypes[str(parameter.dtype)] += parameter.numel()
        state_elements = sum(elements for elements, _bits in self.state_shapes.values())
        state_bits = sum(elements * bits for elements, bits in self.state_shapes.values())
        plif_layers = {}
        learned_delay_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, CausalTemporalChannelMixer) and module.learnable_delays:
                learned_delay_layers[name] = module.learned_delay_summary()
            if isinstance(module, MultiStepLIF) and module.learnable_tau:
                effective_tau = module.effective_tau().detach().cpu()
                plif_layers[name] = {
                    "parameter_elements": effective_tau.numel(),
                    "initial_tau": module.tau,
                    "effective_tau_mean": float(effective_tau.mean().item()),
                    "effective_tau_min": float(effective_tau.min().item()),
                    "effective_tau_max": float(effective_tau.max().item()),
                    "granularity": "feature_channel_or_attention_head",
                }
        return {
            "schema_version": 4,
            "samples_profiled": samples,
            "parameters": {
                "trainable_elements": sum(
                    parameter.numel()
                    for parameter in deployable_parameters
                    if parameter.requires_grad
                ),
                "runtime_parameter_bits": sum(
                    parameter.numel() * parameter.element_size() * 8
                    for parameter in deployable_parameters
                ),
                "training_only_elements_excluded": sum(
                    parameter.numel() for parameter in training_only_parameters
                ),
                "dtypes": dict(parameter_dtypes),
                "quantization_status": "unquantized_runtime",
            },
            "operations_per_sample": {
                **per_sample,
                "lif_comparison": lif_evaluations,
                "lif_decay": lif_evaluations,
                "lif_reset_gate_potential": self.totals["lif_recurrent_evaluations"] / samples,
                "lif_reset_gate_activity": self.totals["lif_reset_spikes"] / samples,
                "elementwise_multiply": self.totals["elementwise_multiply"] / samples,
            },
            "activity": {
                "global_firing_rate": self.totals["spikes"]
                / max(1.0, self.totals["spike_elements"]),
                "zero_skip_assumption": "observed input density with ideal fanout skipping",
            },
            "neuron_dynamics": {
                "type": "per_channel_plif" if plif_layers else "fixed_lif",
                "plif_parameter_elements": sum(
                    values["parameter_elements"] for values in plif_layers.values()
                ),
                "layers": plif_layers,
                "deployment_note": (
                    "sigmoid-derived inverse tau can be precomputed after training"
                    if plif_layers
                    else None
                ),
            },
            "learned_temporal_delays": learned_delay_layers,
            "energy_reference": horowitz_reference(per_sample),
            "state": {
                "persistent_state_elements": state_elements,
                "runtime_persistent_state_bits": state_bits,
                "reads_per_sample": self.totals["state_reads"] / samples,
                "writes_per_sample": state_updates + state_elements,
                "reset_writes_per_sample": state_elements,
                "maximum_layer_state_elements": max(
                    (elements for elements, _bits in self.state_shapes.values()), default=0
                ),
                "hardware_precision_bits": None,
            },
            "buffers": {
                "maximum_observed_activation_elements_per_sample": int(
                    self.totals["max_activation_elements"]
                ),
                "runtime_activation_precision_bits": int(self.totals["max_activation_bits"]),
                "scheduled_hardware_buffer_bits": None,
            },
            "execution": {
                "causal_sequence_equations": bool(self.state_shapes),
                "streaming_state_api": False,
                "feedback_path": (
                    "local LIF membrane, temporal delay buffer and/or diagonal readout state"
                    if self.state_shapes
                    else None
                ),
                "feedback_critical_path": None,
                "bram_expectation": None,
                "dsp_expectation": None,
            },
            "layers": layers,
            "inference_non_linearities": {
                "threshold_comparisons_per_sample": lif_evaluations,
                "sigmoid_per_sample": self.totals["sigmoid"] / samples,
                "tanh_per_sample": self.totals["tanh"] / samples,
                "exp_lut_per_sample": self.totals["exponential"] / samples,
                "softmax_elements_per_sample": self.totals["softmax"] / samples,
                "note": (
                    "surrogate sigmoid is training-only; gated-readout and conditional TCAP "
                    "sigmoid/exp/softmax/tanh are inference operations"
                ),
            },
            "limitations": [
                "BatchNorm is assumed fused into preceding affine layers at inference.",
                "MAC versus AC classification is observational, not a quantized datapath guarantee.",
                "Binary AC uses observed density; attention SOP is a dense potential count.",
                "A single activity-weighted SOP total is omitted because attention sparsity is not observed.",
                "Memory movement, routing, control, clocking and softmax are excluded.",
                "Energy reference covers only specified arithmetic, not total hardware energy.",
                "FIR traffic assumes a ring-buffer hardware schedule, not PyTorch copy traffic.",
                "Channel-mixer traffic also assumes a ring-buffer schedule; MACs are dense potential.",
                "Learned-delay address selection is counted, but its routing energy is excluded from Horowitz.",
                "Max-pool comparisons include padded positions (potential upper bound).",
                "Causality holds in eval mode; training BatchNorm aggregates time and batch.",
                "Scheduled buffers, BRAM/DSP mapping and feedback timing remain unresolved.",
            ],
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def profile_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> dict[str, Any]:
    """Profile a deterministic prefix of a loader without retaining outputs."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    model.eval()
    profiler = _HardwareProfiler(model)
    samples = 0
    sample_indices = []
    sample_targets = []
    try:
        for frames, targets, indices in loader:
            remaining = max_samples - samples
            if remaining <= 0:
                break
            frames = move_encoded_input(slice_encoded_batch(frames, remaining), device)
            # Observational binary classification must not depend on loader batching.
            profiler.batch_size = 1
            for sample_index in range(encoded_batch_size(frames)):
                model(select_encoded_batch(frames, sample_index))
            sample_indices.extend(int(index) for index in indices[:remaining])
            sample_targets.extend(int(target) for target in targets[:remaining])
            samples += encoded_batch_size(frames)
    finally:
        profiler.close()
    if samples == 0:
        raise ValueError("Cannot profile an empty loader.")
    result = profiler.summary(samples)
    result["sampling"] = {
        "indices": sample_indices,
        "targets": sample_targets,
        "classes_observed": len(set(sample_targets)),
        "forward_batch_size": 1,
        "indices_targets_sha256": hashlib.sha256(
            json.dumps([sample_indices, sample_targets]).encode()
        ).hexdigest(),
    }
    return result
