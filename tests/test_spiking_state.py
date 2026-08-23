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

    spikes = spike_function(x, 4.0, surrogate_name="sigmoid")
    spikes.sum().backward()

    expected_spikes = (x.detach() >= 0).to(x.dtype)
    sigmoid = torch.sigmoid(4.0 * x.detach())
    expected_gradient = 4.0 * sigmoid * (1.0 - sigmoid)
    assert torch.equal(spikes.detach(), expected_spikes)
    assert torch.allclose(x.grad, expected_gradient)


def test_original_fast_sigmoid_remains_the_default():
    x = torch.tensor([-0.2, 0.0, 0.2], requires_grad=True)

    spike_function(x, 25.0).sum().backward()

    expected_gradient = 1.0 / (1.0 + 25.0 * x.detach().abs()).pow(2)
    assert torch.allclose(x.grad, expected_gradient)
