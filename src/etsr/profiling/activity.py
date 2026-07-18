from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from etsr.models.spiking import MultiStepLIF


class FiringRateProfiler:
    """Collect element-weighted and layer-wise firing rates from LIF outputs."""

    def __init__(self, model: nn.Module) -> None:
        self.spike_sums: dict[str, float] = defaultdict(float)
        self.element_counts: dict[str, int] = defaultdict(int)
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, MultiStepLIF):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def record(_module, _inputs, output: torch.Tensor):
            detached = output.detach()
            self.spike_sums[name] += float(detached.sum().item())
            self.element_counts[name] += int(detached.numel())

        return record

    def summary(self) -> dict:
        layer_rates = {
            name: self.spike_sums[name] / max(1, self.element_counts[name])
            for name in self.spike_sums
        }
        total_spikes = sum(self.spike_sums.values())
        total_elements = sum(self.element_counts.values())
        return {
            "global_firing_rate": total_spikes / max(1, total_elements),
            "layer_firing_rates": layer_rates,
            "aggregation": "element-weighted across profiled LIF outputs",
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
