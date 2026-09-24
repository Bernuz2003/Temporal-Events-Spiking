"""Gradient-authority diagnostics shared by training, preflight, calibration and audits."""

from __future__ import annotations

import torch
from torch import nn

# Parameters that are not part of the shared backbone: training-only predictors and the routers
# added by conditional TCAP. Their gradients are reported separately and never enter the
# shared-authority ratio.
NON_BACKBONE_PARAMETER_TOKENS = (
    "predictive_head",
    "content_router",
    "predictor_logits",
    "predictor_spatial",
    "predictor_projections",
    "surprise_router",
)
SHARED_GRADIENT_BLOCKS = ("stage1_shared", "tcap1_weights", "stage2_shared", "tcap2_weights")
GRADIENT_BLOCKS = (*SHARED_GRADIENT_BLOCKS, "head")


def gradient_block(name: str) -> str | None:
    """Assign one parameter name to the backbone block used by authority diagnostics."""

    if name.startswith("head."):
        return "head"
    if ".temporal_channel_mixer.weight" in name:
        return "tcap1_weights" if name.startswith("patch_embed1") else "tcap2_weights"
    if any(token in name for token in NON_BACKBONE_PARAMETER_TOKENS):
        return None
    if name.startswith(("patch_embed1.", "stage1.")):
        return "stage1_shared"
    if name.startswith(("patch_embed2.", "stage2.")):
        return "stage2_shared"
    return None


def objective_gradient_diagnostics(
    model: nn.Module,
    classification_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    auxiliary_weight: float,
) -> dict[str, float]:
    """Compare CE and weighted auxiliary gradients on identical parameter blocks.

    Per-block ratios use the same parameter support in numerator and denominator. The
    ``gradient_shared_*`` aggregate pools the four backbone blocks that can receive both
    gradients; the classification head is excluded because no auxiliary objective reaches it.
    ``gradient_shared_unit_ratio`` is independent of the current weight and is the quantity used
    to calibrate auxiliary authority.
    """

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = tuple(parameter for _name, parameter in named_parameters)
    classification_gradients = torch.autograd.grad(
        classification_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    records = dict(
        zip(
            (name for name, _parameter in named_parameters),
            zip(classification_gradients, auxiliary_gradients, strict=True),
            strict=True,
        )
    )
    weight = abs(float(auxiliary_weight))
    zero = classification_loss.new_zeros((), dtype=torch.float64)
    output: dict[str, float] = {}
    global_dot = zero.clone()
    global_both_ce = zero.clone()
    global_both_aux = zero.clone()
    shared_ce = zero.clone()
    shared_aux = zero.clone()
    shared_dot = zero.clone()
    for block_name in GRADIENT_BLOCKS:
        ce_squared = zero.clone()
        aux_squared = zero.clone()
        dot = zero.clone()
        for name, (ce_gradient, aux_gradient) in records.items():
            if gradient_block(name) != block_name:
                continue
            if ce_gradient is not None:
                ce_squared += ce_gradient.detach().double().square().sum()
            if aux_gradient is not None:
                aux_squared += aux_gradient.detach().double().square().sum()
            if ce_gradient is not None and aux_gradient is not None:
                ce = ce_gradient.detach().double()
                aux = aux_gradient.detach().double()
                dot += (ce * aux).sum()
                global_both_ce += ce.square().sum()
                global_both_aux += aux.square().sum()
        ce_norm = ce_squared.sqrt()
        aux_norm = aux_squared.sqrt()
        weighted_aux_norm = aux_norm * weight
        denominator = ce_norm * aux_norm
        cosine = dot / denominator if bool((denominator > 0).item()) else dot.new_tensor(float("nan"))
        ratio = (
            weighted_aux_norm / ce_norm
            if bool((ce_norm > 0).item())
            else ce_norm.new_tensor(float("nan"))
        )
        prefix = f"gradient_{block_name}"
        output.update(
            {
                f"{prefix}_classification_norm": float(ce_norm.cpu()),
                f"{prefix}_weighted_auxiliary_norm": float(weighted_aux_norm.cpu()),
                f"{prefix}_ratio": float(ratio.cpu()),
                f"{prefix}_cosine": float(cosine.cpu()),
            }
        )
        global_dot += dot
        if block_name in SHARED_GRADIENT_BLOCKS:
            shared_ce += ce_squared
            shared_aux += aux_squared
            shared_dot += dot
    shared_ce_norm = shared_ce.sqrt()
    shared_aux_norm = shared_aux.sqrt()
    shared_denominator = shared_ce_norm * shared_aux_norm
    unit_ratio = (
        shared_aux_norm / shared_ce_norm
        if bool((shared_ce_norm > 0).item())
        else shared_ce_norm.new_tensor(float("nan"))
    )
    output.update(
        {
            "gradient_shared_classification_norm": float(shared_ce_norm.cpu()),
            "gradient_shared_auxiliary_norm": float(shared_aux_norm.cpu()),
            "gradient_shared_weighted_auxiliary_norm": float((shared_aux_norm * weight).cpu()),
            "gradient_shared_unit_ratio": float(unit_ratio.cpu()),
            "gradient_shared_ratio": float((unit_ratio * weight).cpu()),
            "gradient_shared_cosine": float(
                (shared_dot / shared_denominator).cpu()
                if bool((shared_denominator > 0).item())
                else float("nan")
            ),
        }
    )
    global_denominator = global_both_ce.sqrt() * global_both_aux.sqrt()
    output["classification_auxiliary_gradient_cosine"] = float(
        (global_dot / global_denominator).cpu()
        if bool((global_denominator > 0).item())
        else float("nan")
    )
    return output
