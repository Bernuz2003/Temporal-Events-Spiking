from __future__ import annotations

import torch


def segment_aligned_time_indices(
    recipient_lengths: torch.Tensor, donor_lengths: torch.Tensor
) -> torch.Tensor:
    """Map each recipient timestep to the same normalized position in a donor segment."""
    if recipient_lengths.ndim != 2 or donor_lengths.shape != recipient_lengths.shape:
        raise ValueError("Segment lengths must be aligned matrices [B, S].")
    if torch.any(recipient_lengths <= 0) or torch.any(donor_lengths <= 0):
        raise ValueError("Segment lengths must be positive.")
    recipient_totals = recipient_lengths.sum(dim=1)
    donor_totals = donor_lengths.sum(dim=1)
    if not torch.all(recipient_totals == recipient_totals[0]):
        raise ValueError("Recipient samples must share a padded temporal length.")
    if not torch.all(donor_totals == donor_totals[0]):
        raise ValueError("Donor samples must share a padded temporal length.")
    if int(recipient_totals[0]) != int(donor_totals[0]):
        raise ValueError("Recipient and donor total lengths must match.")

    batch_size = recipient_lengths.shape[0]
    total = int(recipient_totals[0])
    indices = torch.empty((batch_size, total), dtype=torch.long, device=recipient_lengths.device)
    for batch_index in range(batch_size):
        recipient_offset = 0
        donor_offset = 0
        for recipient_length_tensor, donor_length_tensor in zip(
            recipient_lengths[batch_index], donor_lengths[batch_index], strict=True
        ):
            recipient_length = int(recipient_length_tensor)
            donor_length = int(donor_length_tensor)
            positions = torch.arange(
                recipient_length, device=recipient_lengths.device, dtype=torch.float32
            )
            relative = torch.floor((positions + 0.5) * donor_length / recipient_length).long()
            relative.clamp_(max=donor_length - 1)
            indices[batch_index, recipient_offset : recipient_offset + recipient_length] = (
                donor_offset + relative
            )
            recipient_offset += recipient_length
            donor_offset += donor_length
    return indices


@torch.no_grad()
def patch_with_counterfactual_activations(
    model: torch.nn.Module,
    module_name: str,
    frames: torch.Tensor,
    counterfactual_frames: torch.Tensor,
    time_mask: torch.Tensor,
    donor_time_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patch a canonical module output with matched counterfactual activations."""
    if frames.shape != counterfactual_frames.shape:
        raise ValueError("Original and counterfactual frame batches must have identical shapes.")
    if time_mask.shape != frames.shape[:2] or time_mask.dtype != torch.bool:
        raise ValueError("time_mask must be boolean [B, T].")
    if donor_time_indices is not None:
        if donor_time_indices.shape != time_mask.shape or donor_time_indices.dtype != torch.long:
            raise ValueError("donor_time_indices must be int64 [B, T].")
        if torch.any((donor_time_indices < 0) | (donor_time_indices >= frames.shape[1])):
            raise ValueError("donor_time_indices contains an out-of-range timestep.")
    module = model.get_submodule(module_name)
    captured: dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inputs, output):
        captured["activation"] = output.detach()

    capture_handle = module.register_forward_hook(capture_hook)
    try:
        model(counterfactual_frames)
    finally:
        capture_handle.remove()
    if "activation" not in captured:
        raise RuntimeError(f"Failed to capture counterfactual activation for {module_name}")

    donor = captured["activation"]
    if donor_time_indices is not None:
        time_indices = donor_time_indices.transpose(0, 1)
        batch_indices = torch.arange(donor.shape[1], device=donor.device).view(1, -1)
        batch_indices = batch_indices.expand_as(time_indices)
        donor = donor[time_indices, batch_indices]

    def patch_hook(_module, _inputs, output):
        if output.shape != donor.shape:
            raise RuntimeError("Counterfactual activation shape mismatch.")
        mask = time_mask.transpose(0, 1)
        while mask.ndim < output.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, donor, output)

    patch_handle = module.register_forward_hook(patch_hook)
    try:
        logits = model(frames)
    finally:
        patch_handle.remove()
    return logits


def pairwise_margin(
    logits: torch.Tensor, targets: torch.Tensor, counterfactual_targets: torch.Tensor
) -> torch.Tensor:
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    counterfactual = logits.gather(1, counterfactual_targets[:, None]).squeeze(1)
    return correct - counterfactual
