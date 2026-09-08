import pytest
import torch

from etsr.models.spiking import MultiStepLIF, spike_function


def test_multistep_lif_resets_state_between_forward_calls():
    lif = MultiStepLIF(tau=2.0, threshold=1.0)
    signal = torch.tensor([[0.0], [2.0], [0.0], [2.0]])
    distractor = torch.full_like(signal, 8.0)

    first = lif(signal)
    lif(distractor)
    repeated = lif(signal)

    assert torch.equal(first, repeated)


def test_multistep_lif_keeps_batch_samples_independent():
    lif = MultiStepLIF(tau=2.0, threshold=1.0)
    first_signal = torch.tensor([[0.0], [2.0], [0.0], [2.0]])
    second_signal = torch.tensor([[4.0], [0.0], [4.0], [0.0]])

    expected_first = lif(first_signal)
    expected_second = lif(second_signal)
    combined = lif(torch.cat([first_signal, second_signal], dim=1))

    assert torch.equal(combined[:, :1], expected_first)
    assert torch.equal(combined[:, 1:], expected_second)


def test_sigmoid_surrogate_changes_only_the_backward_function():
    x = torch.tensor([-1.0, -0.2, 0.0, 0.2, 1.0], requires_grad=True)

    spikes = spike_function(x)
    spikes.sum().backward()

    expected_spikes = (x.detach() >= 0).to(x.dtype)
    sigmoid = torch.sigmoid(4.0 * x.detach())
    expected_gradient = 4.0 * sigmoid * (1.0 - sigmoid)
    assert torch.equal(spikes.detach(), expected_spikes)
    assert torch.allclose(x.grad, expected_gradient)


def test_no_cross_time_lif_matches_independent_single_step_calls():
    signal = torch.tensor([[0.0], [2.0], [4.0], [0.0]])
    independent = MultiStepLIF(tau=2.0, threshold=1.0, cross_time=False)
    reference = MultiStepLIF(tau=2.0, threshold=1.0)

    expected = torch.cat([reference(step.unsqueeze(0)) for step in signal], dim=0)

    assert torch.equal(independent(signal), expected)


def test_fixed_tau_lif_preserves_the_historical_tau_two_recurrence_exactly():
    lif = MultiStepLIF(tau=2.0, threshold=1.0)
    signal = torch.rand(7, 3, 4, 2, 2) * 4
    membrane = torch.zeros_like(signal[0])
    expected = []
    for current in signal:
        membrane = membrane + (current - membrane) / 2.0
        spike = spike_function(membrane - 1.0)
        membrane = membrane - spike.detach()
        expected.append(spike)

    assert torch.equal(lif(signal), torch.stack(expected))


def test_per_channel_plif_matches_tau_two_lif_at_initialization_and_learns():
    fixed = MultiStepLIF(tau=2.0, threshold=1.0)
    plif = MultiStepLIF(tau=2.0, threshold=1.0, channels=3, learnable_tau=True)
    signal = torch.rand(5, 2, 3, 2, 2, requires_grad=True) * 4
    assert torch.equal(fixed(signal), plif(signal))
    assert torch.equal(plif.effective_tau(), torch.full((3,), 2.0))
    loss = plif(signal).sum()
    loss.backward()
    assert plif.inverse_tau_logit.grad is not None
    assert torch.isfinite(plif.inverse_tau_logit.grad).all()


def test_plif_rejects_a_wrong_feature_axis_and_tau_stays_above_one():
    plif = MultiStepLIF(tau=2.0, channels=3, learnable_tau=True)
    with pytest.raises(ValueError, match="feature channels"):
        plif(torch.rand(4, 2, 4))
    with torch.no_grad():
        plif.inverse_tau_logit.copy_(torch.tensor([-20.0, 0.0, 20.0]))
    assert torch.all(plif.effective_tau() >= 1.0)
