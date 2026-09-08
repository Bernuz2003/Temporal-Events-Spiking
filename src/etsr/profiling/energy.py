"""Arithmetic-only Horowitz FP32 reference proxy, never measured device energy."""

from typing import Any


def horowitz_reference(operations: dict[str, float]) -> dict[str, Any]:
    # Horowitz, ISSCC 2014, Fig. 1.1.9: 45 nm FP32 add 0.9 pJ, multiply 3.7 pJ.
    # MAC = one multiply + one add. AC assumes ideal spike-triggered accumulation.
    mac, ac, multiply = 4.6, 0.9, 3.7
    dense_affine = mac * operations["multivalued_mac_potential"]
    attention = ac * operations["attention_sop_potential"]
    extra = ac * operations["elementwise_add"] + multiply * (
        operations["elementwise_multiply"] + operations["attention_scale_multiply"]
    )
    return {
        "model": "Horowitz 2014, 45nm FP32 arithmetic reference",
        "source": "https://doi.org/10.1109/ISSCC.2014.6757323",
        "coefficients_pj": {"mac": mac, "ac": ac, "multiply": multiply},
        "covered_arithmetic_dense_potential_uj_per_sample": (
            dense_affine + ac * operations["binary_ac_potential"] + attention + extra
        ) / 1e6,
        "covered_arithmetic_activity_proxy_uj_per_sample": (
            dense_affine + ac * operations["binary_ac_activity_estimate"] + attention + extra
        ) / 1e6,
        "extra_arithmetic_uj_per_sample": extra / 1e6,
        "attention_assumption": "dense potential SOP charged as AC; activity not observed",
        "total_hardware_energy_uj_per_sample": None,
        "excluded": [
            "LIF integration, reset and threshold arithmetic",
            "pooling comparisons, sigmoid, tanh and readout reductions",
            "memory access, routing, control and leakage",
        ],
        "interpretation": (
            "Partial arithmetic proxy with ideal binary zero skipping; neither measured GPU "
            "energy nor a complete FPGA/ASIC estimate. FIR is included once in elementwise totals."
        ),
    }
