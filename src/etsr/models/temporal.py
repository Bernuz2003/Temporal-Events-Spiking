from __future__ import annotations

import torch
from torch import nn


class CausalTemporalFIR(nn.Module):
    """Channel-wise causal finite-impulse-response memory for time-major tensors.

    Tap zero multiplies the current input; later taps multiply inputs delayed by
    ``tap * dilation`` steps. The caller owns no persistent state during an ordinary full-sequence
    forward, while ``transition`` exposes the exact state needed by a streaming implementation.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least two")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size))
        with torch.no_grad():
            self.weight[:, 0] = 1.0

    @property
    def max_delay(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def reset_state(self, reference: torch.Tensor) -> torch.Tensor:
        self._validate_step(reference)
        return reference.new_zeros((self.max_delay, *reference.shape))

    @staticmethod
    def detach_state(state: torch.Tensor) -> torch.Tensor:
        return state.detach()

    def _validate_step(self, current: torch.Tensor) -> None:
        if current.ndim < 2 or current.shape[1] != self.channels:
            raise ValueError("FIR step input must have shape [B, C, ...].")

    def transition(
        self,
        current: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the filtered current step and updated causal delay buffer."""

        self._validate_step(current)
        expected = (self.max_delay, *current.shape)
        if state.shape != expected:
            raise ValueError(f"FIR state must have shape {expected}.")
        if state.dtype != current.dtype or state.device != current.device:
            raise ValueError("FIR state must match current dtype and device.")
        weight_shape = (1, self.channels, *((1,) * (current.ndim - 2)))
        weights = self.weight.to(dtype=current.dtype)
        output = current * weights[:, 0].reshape(weight_shape)
        for tap in range(1, self.kernel_size):
            delayed = state[-tap * self.dilation]
            output = output + delayed * weights[:, tap].reshape(weight_shape)
        next_state = torch.cat((state[1:], current.unsqueeze(0)), dim=0)
        return output, next_state

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim < 3 or sequence.shape[0] == 0 or sequence.shape[2] != self.channels:
            raise ValueError("Expected non-empty time-major input [T, B, C, ...].")
        if state is None:
            state = self.reset_state(sequence[0])
        if state.shape != (self.max_delay, *sequence.shape[1:]):
            raise ValueError("FIR state does not match the sequence shape.")
        if state.dtype != sequence.dtype or state.device != sequence.device:
            raise ValueError("FIR state must match sequence dtype and device.")
        # One concatenation for the sequence, rather than one buffer copy per timestep.
        history = torch.cat((state, sequence), dim=0)
        weights = self.weight.to(dtype=sequence.dtype)
        weight_shape = (1, 1, self.channels, *((1,) * (sequence.ndim - 3)))
        output = sequence * weights[:, 0].reshape(weight_shape)
        for tap in range(1, self.kernel_size):
            start = self.max_delay - tap * self.dilation
            delayed = history[start : start + sequence.shape[0]]
            output = output + delayed * weights[:, tap].reshape(weight_shape)
        return output, history[-self.max_delay :].clone()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(sequence)[0]


class CausalTemporalChannelMixer(nn.Module):
    """Causal multi-delay MIMO FIR acting independently at each spatial/token position.

    Only delayed features are projected across channels. The current-step path is a fixed identity,
    so the zero initialization is exactly the unmodified backbone and cannot add a static channel
    transform. The state API defines the corresponding streaming implementation.
    """

    def __init__(self, channels: int, delays: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not delays or any(type(delay) is not int or delay <= 0 for delay in delays):
            raise ValueError("delays must contain positive integers")
        if tuple(sorted(set(delays))) != delays:
            raise ValueError("delays must be strictly increasing and unique")
        self.channels = int(channels)
        self.delays = tuple(delays)
        self.weight = nn.Parameter(torch.zeros(len(delays), channels, channels))

    @property
    def max_delay(self) -> int:
        return self.delays[-1]

    def _validate_step(self, current: torch.Tensor) -> None:
        if current.ndim < 2 or current.shape[1] != self.channels:
            raise ValueError("Temporal mixer step input must have shape [B, C, ...].")

    def reset_state(self, reference: torch.Tensor) -> torch.Tensor:
        self._validate_step(reference)
        return reference.new_zeros((self.max_delay, *reference.shape))

    @staticmethod
    def detach_state(state: torch.Tensor) -> torch.Tensor:
        return state.detach()

    def transition(
        self, current: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_step(current)
        expected = (self.max_delay, *current.shape)
        if state.shape != expected:
            raise ValueError(f"Temporal mixer state must have shape {expected}.")
        if state.dtype != current.dtype or state.device != current.device:
            raise ValueError("Temporal mixer state must match current dtype and device.")
        weights = self.weight.to(dtype=current.dtype)
        output = current
        for index, delay in enumerate(self.delays):
            output = output + torch.einsum("oc,bc...->bo...", weights[index], state[-delay])
        next_state = torch.cat((state[1:], current.unsqueeze(0)), dim=0)
        return output, next_state

    def forward_sequence(
        self, sequence: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim < 3 or sequence.shape[0] == 0 or sequence.shape[2] != self.channels:
            raise ValueError("Expected non-empty time-major input [T, B, C, ...].")
        if state is None:
            state = self.reset_state(sequence[0])
        expected = (self.max_delay, *sequence.shape[1:])
        if state.shape != expected:
            raise ValueError(f"Temporal mixer state must have shape {expected}.")
        if state.dtype != sequence.dtype or state.device != sequence.device:
            raise ValueError("Temporal mixer state must match sequence dtype and device.")
        history = torch.cat((state, sequence), dim=0)
        weights = self.weight.to(dtype=sequence.dtype)
        output = sequence
        for index, delay in enumerate(self.delays):
            start = self.max_delay - delay
            delayed = history[start : start + sequence.shape[0]]
            output = output + torch.einsum("oc,tbc...->tbo...", weights[index], delayed)
        return output, history[-self.max_delay :].clone()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(sequence)[0]
