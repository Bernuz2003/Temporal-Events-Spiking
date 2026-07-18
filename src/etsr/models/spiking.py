from __future__ import annotations

import torch
from torch import nn


class _FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor, slope: float):
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.slope = slope
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        surrogate = 1.0 / (1.0 + slope * x.abs()).pow(2)
        return grad_output * surrogate, None


def spike_function(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return _FastSigmoidSpike.apply(x, slope)


class MultiStepLIF(nn.Module):
    """Minimal multi-step LIF with soft reset.

    The temporal dimension must be first. State is local to a forward pass, matching the sample-level
    reset used in direct-training visual SNNs.
    """

    def __init__(
        self,
        tau: float = 2.0,
        threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1 for a leaky neuron.")
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.detach_reset = detach_reset
        self.surrogate_slope = float(surrogate_slope)
        self.last_firing_rate = 0.0

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim < 2:
            raise ValueError("MultiStepLIF expects time-major input [T, ...].")
        membrane = torch.zeros_like(current[0])
        spikes = []
        for current_t in current.unbind(0):
            membrane = membrane + (current_t - membrane) / self.tau
            spike = spike_function(membrane - self.threshold, self.surrogate_slope)
            reset_spike = spike.detach() if self.detach_reset else spike
            membrane = membrane - reset_spike * self.threshold
            spikes.append(spike)
        output = torch.stack(spikes, dim=0)
        self.last_firing_rate = float(output.detach().mean().item())
        return output
