from __future__ import annotations

import torch
from torch import nn


class DiagonalGatedReadout(nn.Module):
    """Single-gate causal readout with caller-owned channelwise recurrent state."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.gate_input = nn.Parameter(torch.ones(channels))
        self.gate_state = nn.Parameter(torch.zeros(channels))
        self.gate_bias = nn.Parameter(torch.zeros(channels))
        self.candidate_input = nn.Parameter(torch.ones(channels))
        self.candidate_state = nn.Parameter(torch.zeros(channels))
        self.candidate_bias = nn.Parameter(torch.zeros(channels))

    def reset_state(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 2 or reference.shape[-1] != self.channels:
            raise ValueError("Readout state reference must have shape [B, C].")
        return torch.zeros_like(reference)

    @staticmethod
    def detach_state(state: torch.Tensor) -> torch.Tensor:
        return state.detach()

    def transition(
        self,
        current: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next state and update gate for one causal step."""

        if current.ndim != 2 or current.shape[-1] != self.channels:
            raise ValueError("Readout step input must have shape [B, C].")
        if state.shape != current.shape:
            raise ValueError("Readout state must match the step input shape.")
        gate = torch.sigmoid(current * self.gate_input + state * self.gate_state + self.gate_bias)
        candidate = torch.tanh(
            current * self.candidate_input + state * self.candidate_state + self.candidate_bias
        )
        return state + gate * (candidate - state), gate

    def step(self, current: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.transition(current, state)[0]

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        state: torch.Tensor | None = None,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.channels or sequence.shape[0] == 0:
            raise ValueError("Expected non-empty time-major input [T, B, C].")
        if state is None:
            state = self.reset_state(sequence[0])
        if valid_steps is not None:
            if valid_steps.shape != (sequence.shape[1],):
                raise ValueError("valid_steps must contain one length per batch sample.")
            if valid_steps.dtype not in (torch.int32, torch.int64):
                raise ValueError("valid_steps must use an integer dtype.")
            if bool(((valid_steps < 1) | (valid_steps > sequence.shape[0])).any().item()):
                raise ValueError("valid_steps must fit the readout time axis.")
        for index, current in enumerate(sequence.unbind(0)):
            next_state = self.step(current, state)
            if valid_steps is None:
                state = next_state
            else:
                active = (index < valid_steps).unsqueeze(1)
                state = torch.where(active, next_state, state)
        return state

    def forward(
        self,
        sequence: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_sequence(sequence, valid_steps=valid_steps)
