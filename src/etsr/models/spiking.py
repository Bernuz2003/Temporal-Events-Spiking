from __future__ import annotations

import math

import torch
from torch import nn


class _SigmoidSpike(torch.autograd.Function):
    """Hard spike in the forward pass with a logistic derivative in the backward pass."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor, alpha: float):
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.alpha = alpha
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sigmoid = torch.sigmoid(alpha * x)
        surrogate = alpha * sigmoid * (1.0 - sigmoid)
        return grad_output * surrogate, None


def spike_function(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Apply a hard spike with the logistic surrogate used during backward only."""

    return _SigmoidSpike.apply(x, alpha)


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
        surrogate_alpha: float = 4.0,
        cross_time: bool = True,
        channels: int | None = None,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1 for a leaky neuron.")
        if surrogate_alpha <= 0.0:
            raise ValueError("surrogate_alpha must be positive.")
        if type(cross_time) is not bool:
            raise ValueError("cross_time must be boolean.")
        if type(learnable_tau) is not bool:
            raise ValueError("learnable_tau must be boolean.")
        if learnable_tau and (channels is None or channels <= 0):
            raise ValueError("learnable_tau requires a positive channel count.")
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.detach_reset = detach_reset
        self.surrogate_alpha = float(surrogate_alpha)
        self.cross_time = cross_time
        self.channels = int(channels) if channels is not None else None
        self.learnable_tau = learnable_tau
        if learnable_tau:
            inverse_tau = 1.0 / self.tau
            initial_logit = math.log(inverse_tau / (1.0 - inverse_tau))
            self.inverse_tau_logit = nn.Parameter(torch.full((self.channels,), initial_logit))
        else:
            self.register_parameter("inverse_tau_logit", None)

    def inverse_tau(self, reference: torch.Tensor | None = None) -> torch.Tensor | float:
        if self.inverse_tau_logit is None:
            return 1.0 / self.tau
        value = torch.sigmoid(self.inverse_tau_logit)
        if reference is not None:
            if reference.ndim < 2 or reference.shape[1] != self.channels:
                raise ValueError("PLIF input must have its configured feature channels on axis 1.")
            value = value.to(dtype=reference.dtype).reshape(
                1, self.channels, *((1,) * (reference.ndim - 2))
            )
        return value

    def effective_tau(self) -> torch.Tensor:
        if self.inverse_tau_logit is None:
            return torch.tensor(self.tau)
        return torch.sigmoid(self.inverse_tau_logit).reciprocal()

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim < 2:
            raise ValueError("MultiStepLIF expects time-major input [T, ...].")
        inverse_tau = self.inverse_tau(current[0])
        if not self.cross_time:
            return spike_function(current * inverse_tau - self.threshold, self.surrogate_alpha)
        membrane = torch.zeros_like(current[0])
        spikes = []
        for current_t in current.unbind(0):
            membrane = membrane + (current_t - membrane) * inverse_tau
            spike = spike_function(membrane - self.threshold, self.surrogate_alpha)
            reset_spike = spike.detach() if self.detach_reset else spike
            membrane = membrane - reset_spike * self.threshold
            spikes.append(spike)
        return torch.stack(spikes, dim=0)
