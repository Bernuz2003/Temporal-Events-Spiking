import torch

from etsr.models.spiking import MultiStepLIF


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
