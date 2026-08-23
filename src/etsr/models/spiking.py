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


SUPPORTED_SURROGATES = ("fast_sigmoid", "sigmoid")


def spike_function(
    x: torch.Tensor,
    slope: float = 25.0,
    *,
    surrogate_name: str = "fast_sigmoid",
) -> torch.Tensor:
    """Apply a hard spike with a selectable backward-only surrogate.

    ``slope`` retains the original public argument name for compatibility. For the logistic
    surrogate it is the conventional alpha parameter.
    """

    if surrogate_name == "fast_sigmoid":
        return _FastSigmoidSpike.apply(x, slope)
    if surrogate_name == "sigmoid":
        return _SigmoidSpike.apply(x, slope)
    raise ValueError(
        f"Unsupported surrogate: {surrogate_name}. Expected one of {SUPPORTED_SURROGATES}."
    )


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
        surrogate_name: str = "fast_sigmoid",
        surrogate_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1 for a leaky neuron.")
        if surrogate_name not in SUPPORTED_SURROGATES:
            raise ValueError(
                f"Unsupported surrogate: {surrogate_name}. Expected one of {SUPPORTED_SURROGATES}."
            )
        alpha = surrogate_slope if surrogate_alpha is None else surrogate_alpha
        if alpha <= 0.0:
            raise ValueError("surrogate_alpha must be positive.")
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.detach_reset = detach_reset
        self.surrogate_name = surrogate_name
        self.surrogate_alpha = float(alpha)
        # Retained for callers that inspect the original attribute.
        self.surrogate_slope = self.surrogate_alpha
        self.last_firing_rate = 0.0

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim < 2:
            raise ValueError("MultiStepLIF expects time-major input [T, ...].")
        membrane = torch.zeros_like(current[0])
        spikes = []
        for current_t in current.unbind(0):
            membrane = membrane + (current_t - membrane) / self.tau
            spike = spike_function(
                membrane - self.threshold,
                self.surrogate_alpha,
                surrogate_name=self.surrogate_name,
            )
            reset_spike = spike.detach() if self.detach_reset else spike
            membrane = membrane - reset_spike * self.threshold
            spikes.append(spike)
        output = torch.stack(spikes, dim=0)
        self.last_firing_rate = float(output.detach().mean().item())
        return output
