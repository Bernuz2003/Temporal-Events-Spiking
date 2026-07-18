from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from etsr.models.layers import SpikingSelfAttention, TokenQKAttention


class OperationProfiler:
    """High-level MAC/AC estimator.

    Convolutional and linear modules are tagged with `op_kind`. Spike-driven layers use the observed
    non-zero ratio of their input. Attention mixing costs are supplied by the custom modules.
    """

    def __init__(self, model: nn.Module) -> None:
        self.totals = defaultdict(float)
        self.by_layer: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.batch_size = 1
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(self._dense_hook(name)))
            elif isinstance(module, (TokenQKAttention, SpikingSelfAttention)):
                self.handles.append(module.register_forward_hook(self._attention_hook(name)))

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, int(batch_size))

    def _dense_hook(self, name: str):
        def record(module: nn.Module, inputs, output: torch.Tensor):
            x = inputs[0]
            if isinstance(module, nn.Conv2d):
                kernel_ops = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.in_channels
                    / module.groups
                )
            elif isinstance(module, nn.Conv1d):
                kernel_ops = module.kernel_size[0] * module.in_channels / module.groups
            else:
                kernel_ops = module.in_features

            dense_ops = float(output.numel() * kernel_ops)
            op_kind = getattr(module, "op_kind", "mac")
            if op_kind == "ac":
                nonzero = float((x.detach() != 0).float().mean().item())
                ac_ops = dense_ops * nonzero
                self.totals["ac_ops"] += ac_ops
                self.by_layer[name]["ac_ops"] += ac_ops
                self.by_layer[name]["input_nonzero_rate_sum"] += nonzero
                self.by_layer[name]["calls"] += 1
            else:
                self.totals["mac_ops"] += dense_ops
                self.by_layer[name]["mac_ops"] += dense_ops

        return record

    def _attention_hook(self, name: str):
        def record(module, _inputs, _output):
            value = float(module.last_mixing_ac_per_sample) * self.batch_size
            self.totals["ac_ops"] += value
            self.by_layer[name]["ac_ops"] += value

        return record

    def summary(self, samples: int) -> dict:
        divisor = max(1, samples)
        layers = {}
        for name, values in self.by_layer.items():
            layer = {key: value / divisor for key, value in values.items() if key != "calls"}
            if values.get("calls", 0):
                layer["mean_input_nonzero_rate"] = (
                    values["input_nonzero_rate_sum"] / values["calls"]
                )
                layer.pop("input_nonzero_rate_sum", None)
            layers[name] = layer
        return {
            "mac_ops_per_sample": self.totals["mac_ops"] / divisor,
            "ac_ops_per_sample": self.totals["ac_ops"] / divisor,
            "layers": layers,
            "method": "activity-weighted high-level estimate",
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
