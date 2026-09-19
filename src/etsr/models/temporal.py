from __future__ import annotations

import math
from collections.abc import Callable, Iterator

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

    def __init__(
        self,
        channels: int,
        delays: tuple[int, ...] = (1, 2, 4),
        learnable_delays: bool = False,
        dynamic_routing: bool = False,
        predictive_auxiliary: bool = False,
        surprise_routing: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not delays or any(type(delay) is not int or delay <= 0 for delay in delays):
            raise ValueError("delays must contain positive integers")
        if tuple(sorted(set(delays))) != delays:
            raise ValueError("delays must be strictly increasing and unique")
        if type(learnable_delays) is not bool:
            raise ValueError("learnable_delays must be boolean")
        for name, enabled in (
            ("dynamic_routing", dynamic_routing),
            ("predictive_auxiliary", predictive_auxiliary),
            ("surprise_routing", surprise_routing),
        ):
            if type(enabled) is not bool:
                raise ValueError(f"{name} must be boolean")
        if surprise_routing and not predictive_auxiliary:
            raise ValueError("surprise routing requires the predictive auxiliary")
        if learnable_delays and (dynamic_routing or predictive_auxiliary):
            raise ValueError("conditional routing is defined only for fixed TCAP delays")
        self.channels = int(channels)
        self.delays = tuple(delays)
        self.learnable_delays = learnable_delays
        self.dynamic_routing = dynamic_routing
        self.predictive_auxiliary = predictive_auxiliary
        self.surprise_routing = surprise_routing
        self.weight = nn.Parameter(torch.zeros(len(delays), channels, channels))
        self.content_router = (
            nn.Linear(channels, len(delays), bias=True) if dynamic_routing else None
        )
        self.predictor_logits = (
            nn.Parameter(torch.zeros(len(delays), channels)) if predictive_auxiliary else None
        )
        self.surprise_router = (
            nn.Parameter(torch.zeros(len(delays))) if surprise_routing else None
        )
        self.last_auxiliary_loss: torch.Tensor | None = None
        self.last_auxiliary_error: torch.Tensor | None = None
        self.last_routing_statistics: dict[str, torch.Tensor] | None = None
        if self.content_router is not None:
            nn.init.zeros_(self.content_router.weight)
            nn.init.zeros_(self.content_router.bias)
        if learnable_delays:
            centers = torch.tensor(delays, dtype=torch.float32)[:, None].repeat(1, channels)
            self.delay_centers = nn.Parameter(centers)
        else:
            self.register_parameter("delay_centers", None)
        # Training uses a triangular distribution; evaluation always uses integer delays.
        self.delay_temperature = max(0.501, self.max_delay / 2)

    @property
    def max_delay(self) -> int:
        return self.delays[-1]

    def set_delay_progress(self, epoch: int, total_epochs: int) -> None:
        """Normalized squared raised cosine from Dmax/2 to approximately 0.5."""
        if not self.learnable_delays:
            return
        if total_epochs <= 0 or not 1 <= epoch <= total_epochs:
            raise ValueError("epoch must be within the configured training horizon")
        progress = (epoch - 1) / max(1, total_epochs - 1)
        # MD-Mixer supplementary Eq. 11 uses (1+cos)^2/2, which exceeds its stated
        # tau_max at epoch zero. Dividing by four preserves the published squared
        # shape while satisfying the stated endpoints tau_max and tau_min.
        fraction = ((1 + math.cos(math.pi * progress)) / 2) ** 2
        minimum = 0.501  # Keeps at least one integer delay in the triangular support.
        self.delay_temperature = minimum + (max(minimum, self.max_delay / 2) - minimum) * fraction

    @torch.no_grad()
    def project_delay_centers_(self) -> None:
        if self.delay_centers is not None:
            self.delay_centers.clamp_(1, self.max_delay)

    @torch.no_grad()
    def discrete_delays(self) -> torch.Tensor:
        if self.delay_centers is None:
            return (
                torch.tensor(self.delays, dtype=torch.long).unsqueeze(1).expand(-1, self.channels)
            )
        return torch.floor(self.delay_centers.clamp(1, self.max_delay) + 0.5).long()

    def delay_distribution(self, *, hard: bool | None = None) -> torch.Tensor:
        """Return [branch, delay, input channel] coefficients in float32."""
        if self.delay_centers is None:
            raise ValueError("Fixed-delay TCAP has no trainable delay distribution")
        if hard is None:
            hard = not self.training
        if hard:
            return (
                torch.nn.functional.one_hot(self.discrete_delays() - 1, num_classes=self.max_delay)
                .permute(0, 2, 1)
                .float()
            )
        grid = torch.arange(
            1, self.max_delay + 1, device=self.delay_centers.device, dtype=torch.float32
        )
        centers = self.delay_centers.float().clamp(1, self.max_delay)
        unnormalized = torch.relu(
            1 - (grid[None, :, None] - centers[:, None, :]).abs() / self.delay_temperature
        )
        return unnormalized / unnormalized.sum(dim=1, keepdim=True)

    @torch.no_grad()
    def learned_delay_summary(self) -> dict:
        if self.delay_centers is None:
            raise ValueError("Fixed-delay TCAP has no learned delay summary")
        centers = self.delay_centers.detach().cpu()
        hard = self.discrete_delays().cpu()
        rounding_distance = (centers - hard).abs()
        distinct_by_channel = [
            len(set(hard[:, channel].tolist())) for channel in range(self.channels)
        ]
        return {
            "initial_delays": list(self.delays),
            "maximum_delay": self.max_delay,
            "centers_mean_by_branch": centers.mean(dim=1).tolist(),
            "discrete_histogram_by_branch": [
                torch.bincount(row, minlength=self.max_delay + 1)[1:].tolist() for row in hard
            ],
            "mean_rounding_distance_bins": rounding_distance.mean().item(),
            "maximum_rounding_distance_bins": rounding_distance.max().item(),
            "mean_distinct_delays_per_channel": sum(distinct_by_channel) / self.channels,
            "evaluation_mode": "hard_discrete",
        }

    def _learned_history_inputs(
        self,
        current: torch.Tensor,
        read_delay: Callable[[int], torch.Tensor],
        channel_axis: int,
    ) -> Iterator[torch.Tensor]:
        if not self.training:
            # Evaluation performs actual integer channel-wise selection, not soft interpolation.
            selected_delays = self.discrete_delays()
            for branch in range(len(self.delays)):
                mixed = torch.empty_like(current)
                for delay in range(1, self.max_delay + 1):
                    channels = selected_delays[branch] == delay
                    if channel_axis == 1:
                        mixed[:, channels] = read_delay(delay)[:, channels]
                    else:
                        mixed[:, :, channels] = read_delay(delay)[:, :, channels]
                yield mixed
            return

        coefficients = self.delay_distribution(hard=False).to(dtype=current.dtype)
        shape = [1] * current.ndim
        shape[channel_axis] = self.channels
        for branch in range(len(self.delays)):
            mixed = None
            for delay in range(1, self.max_delay + 1):
                contribution = read_delay(delay) * coefficients[branch, delay - 1].reshape(shape)
                mixed = contribution if mixed is None else mixed + contribution
            yield mixed

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
        if self.learnable_delays:
            output = current
            for branch, mixed in enumerate(
                self._learned_history_inputs(current, lambda delay: state[-delay], 1)
            ):
                output = output + torch.einsum("oc,bc...->bo...", weights[branch], mixed)
        else:
            output = current
            sequence = current.unsqueeze(0)
            history = torch.cat((state, sequence), dim=0)
            gates, prediction = self._conditional_gates(sequence, history)
            for index, delay in enumerate(self.delays):
                contribution = torch.einsum("oc,bc...->bo...", weights[index], state[-delay])
                if gates is not None:
                    gate_shape = (current.shape[0], *((1,) * (current.ndim - 1)))
                    contribution = contribution * gates[0, :, index].reshape(gate_shape)
                output = output + contribution
            if prediction is not None:
                pointwise = torch.nn.functional.smooth_l1_loss(
                    prediction[0], current.detach(), reduction="none"
                )
                self.last_auxiliary_error = pointwise.flatten(1).mean(1).unsqueeze(0)
                self.last_auxiliary_loss = self.last_auxiliary_error.mean()
            else:
                self.last_auxiliary_loss = None
                self.last_auxiliary_error = None
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
        if self.learnable_delays:
            output = sequence
            for branch, mixed in enumerate(
                self._learned_history_inputs(
                    sequence,
                    lambda delay: history[
                        self.max_delay - delay : self.max_delay - delay + sequence.shape[0]
                    ],
                    2,
                )
            ):
                output = output + torch.einsum("oc,tbc...->tbo...", weights[branch], mixed)
        else:
            output = sequence
            gates, prediction = self._conditional_gates(sequence, history)
            for index, delay in enumerate(self.delays):
                start = self.max_delay - delay
                delayed = history[start : start + sequence.shape[0]]
                contribution = torch.einsum("oc,tbc...->tbo...", weights[index], delayed)
                if gates is not None:
                    gate_shape = (sequence.shape[0], sequence.shape[1], *((1,) * (sequence.ndim - 2)))
                    contribution = contribution * gates[..., index].reshape(gate_shape)
                output = output + contribution
            if prediction is not None:
                # The current feature is detached only on the target side. This prevents the
                # predictor from manufacturing an easy target while still training its history.
                pointwise = torch.nn.functional.smooth_l1_loss(
                    prediction,
                    sequence.detach(),
                    reduction="none",
                )
                self.last_auxiliary_error = pointwise.flatten(2).mean(2)
                self.last_auxiliary_loss = self.last_auxiliary_error.mean()
            else:
                self.last_auxiliary_loss = None
                self.last_auxiliary_error = None
        return output, history[-self.max_delay :].clone()

    def _conditional_gates(
        self,
        sequence: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return unit-centred tap gates and the optional causal prediction.

        The router uses only the current feature and/or a detached, normalized prediction error.
        A zero-initialized router therefore preserves the exact fixed-TCAP function.
        """

        prediction = None
        surprise = None
        if self.predictor_logits is not None and (self.training or self.surprise_routing):
            coefficients = self.predictor_logits.softmax(dim=0).to(sequence.dtype)
            prediction = torch.zeros_like(sequence)
            shape = (1, 1, self.channels, *((1,) * (sequence.ndim - 3)))
            for index, delay in enumerate(self.delays):
                start = self.max_delay - delay
                delayed = history[start : start + sequence.shape[0]]
                prediction = prediction + delayed * coefficients[index].reshape(shape)
            reduce_dims = tuple(range(2, sequence.ndim))
            scale = sequence.detach().square().mean(dim=reduce_dims).sqrt().clamp_min(1e-4)
            error = (sequence.detach() - prediction.detach()).abs().mean(dim=reduce_dims)
            surprise = error / scale

        gate_logits = None
        if self.content_router is not None:
            spatial_dims = tuple(range(3, sequence.ndim))
            pooled = sequence.mean(dim=spatial_dims) if spatial_dims else sequence
            gate_logits = self.content_router(pooled)
        if self.surprise_router is not None:
            assert surprise is not None
            surprise_logits = surprise.unsqueeze(-1) * self.surprise_router.to(sequence.dtype)
            gate_logits = surprise_logits if gate_logits is None else gate_logits + surprise_logits
        gates = None if gate_logits is None else 2.0 * torch.sigmoid(gate_logits)
        self.last_routing_statistics = None
        if gates is not None:
            self.last_routing_statistics = {
                "gate_mean_by_delay": gates.detach().mean(dim=(0, 1)),
                "gate_std_by_delay": gates.detach().std(dim=(0, 1), unbiased=False),
                "surprise_mean": (
                    surprise.detach().mean() if surprise is not None else gates.new_tensor(float("nan"))
                ),
            }
        return gates, prediction

    def auxiliary_loss(self) -> torch.Tensor | None:
        return self.last_auxiliary_loss

    def auxiliary_error(self) -> torch.Tensor | None:
        return self.last_auxiliary_error

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(sequence)[0]
