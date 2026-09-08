"""Fixed diagnostic acceptance rule; not a generalization or model-selection test."""

import math
from typing import Any


def overfit_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_fields = ("train_loss", "validation_loss", "gradient_norm_mean")
    healthy = bool(rows) and all(
        all(math.isfinite(float(row[field])) for field in finite_fields)
        and float(row["gradient_nonfinite_fraction"]) == 0.0
        and float(row["amp_overflow_fraction"]) == 0.0
        for row in rows
    )
    passed = healthy and len(rows) >= 5 and all(
        float(row["validation_accuracy"]) >= 0.95
        and float(row["validation_loss"]) < 1.5
        for row in rows[-5:]
    )
    return {
        "passed": passed,
        "epochs_observed": len(rows),
        "required_consecutive_epochs": 5,
        "minimum_accuracy": 0.95,
        "maximum_loss_exclusive": 1.5,
        "all_epochs_finite": healthy,
        "last_epoch": rows[-1] if rows else None,
        "selection_basis": "last five epochs, not the best checkpoint",
        "generalization_evidence": False,
    }
