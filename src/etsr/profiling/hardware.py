"""Small hardware-oriented profiler for the active spiking models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from etsr.models.layers import (
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    SpikingBlock,
    SpikingSelfAttention,
    TokenQKAttention,
)
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF


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
            if isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
                module, (nn.Conv1d, nn.Conv2d, nn.Linear)
            ):
                hook = self._dense_hook(name)
            elif isinstance(module, MultiStepLIF):
                hook = self._lif_hook(name)
            elif isinstance(module, DiagonalGatedReadout):
                hook = self._gated_readout_hook(name)
            elif isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
                module, (TokenQKAttention, SpikingSelfAttention)
            ):
                hook = self._attention_hook(name)
            elif isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
                module, (InitialPatchEmbedding, PatchEmbeddingStage, SpikingBlock)
            ):
                hook = self._residual_hook(name)
            else:
                continue
            self.handles.append(module.register_forward_hook(hook))

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
                int(valid_steps.sum().item()) if valid_steps is not None else time_steps * batch_size
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
                    2
                    * time_steps
                    * batch_size
                    * module.num_heads
                    * tokens
                    * head_dim**2
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
                "elementwise_add",
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
                not in {"calls", "input_elements", "input_nonzero", "binary_calls", "spikes", "spike_elements"}
                and key not in {"cross_time_state", "persistent_state_elements"}
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
            layers[name] = layer

        parameter_dtypes: dict[str, int] = defaultdict(int)
        for parameter in self.model.parameters():
            parameter_dtypes[str(parameter.dtype)] += parameter.numel()
        state_elements = sum(elements for elements, _bits in self.state_shapes.values())
        state_bits = sum(elements * bits for elements, bits in self.state_shapes.values())
        return {
            "schema_version": 2,
            "samples_profiled": samples,
            "parameters": {
                "trainable_elements": sum(
                    parameter.numel()
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
                "runtime_parameter_bits": sum(
                    parameter.numel() * parameter.element_size() * 8
                    for parameter in self.model.parameters()
                ),
                "dtypes": dict(parameter_dtypes),
                "quantization_status": "unquantized_runtime",
            },
            "operations_per_sample": {
                **per_sample,
                "lif_comparison": lif_evaluations,
                "lif_decay": lif_evaluations,
                "lif_reset_gate_potential": self.totals["lif_recurrent_evaluations"]
                / samples,
                "lif_reset_gate_activity": self.totals["lif_reset_spikes"] / samples,
                "elementwise_multiply": self.totals["elementwise_multiply"] / samples,
            },
            "activity": {
                "global_firing_rate": self.totals["spikes"]
                / max(1.0, self.totals["spike_elements"]),
                "zero_skip_assumption": "observed input density with ideal fanout skipping",
            },
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
                    "local LIF membrane and/or diagonal readout state from t-1 to t"
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
                "exp_lut_per_sample": 0,
                "note": "surrogate sigmoid is training-only; gated-readout sigmoid/tanh are inference operations",
            },
            "limitations": [
                "BatchNorm is assumed fused into preceding affine layers at inference.",
                "MAC versus AC classification is observational, not a quantized datapath guarantee.",
                "Binary AC uses observed density; attention SOP is a dense potential count.",
                "A single activity-weighted SOP total is omitted because attention sparsity is not observed.",
                "Memory movement, routing, control, clocking and softmax are excluded.",
                "Energy is omitted until numeric format and hardware target are fixed.",
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
    try:
        for frames, _targets, _indices in loader:
            remaining = max_samples - samples
            if remaining <= 0:
                break
            frames = frames[:remaining].to(device, non_blocking=True)
            profiler.batch_size = int(frames.shape[0])
            model(frames)
            samples += int(frames.shape[0])
    finally:
        profiler.close()
    if samples == 0:
        raise ValueError("Cannot profile an empty loader.")
    return profiler.summary(samples)
