from __future__ import annotations


def estimate_horowitz_energy(
    mac_ops_per_sample: float,
    ac_ops_per_sample: float,
    mac_energy_pj: float = 4.6,
    ac_energy_pj: float = 0.9,
) -> dict[str, float | str | list[str]]:
    mac_pj = mac_ops_per_sample * mac_energy_pj
    ac_pj = ac_ops_per_sample * ac_energy_pj
    total_pj = mac_pj + ac_pj
    return {
        "mac_energy_pj": mac_pj,
        "ac_energy_pj": ac_pj,
        "total_energy_pj": total_pj,
        "total_energy_uj": total_pj / 1e6,
        "total_energy_mj": total_pj / 1e9,
        "mac_cost_pj_per_op": mac_energy_pj,
        "ac_cost_pj_per_op": ac_energy_pj,
        "label": "theoretical Horowitz-style arithmetic energy estimate",
        "limitations": [
            "excludes memory access and data movement",
            "excludes routing, control and clocking",
            "assumes zero-valued spike operations can be skipped",
            "depends on technology node and numeric precision",
            "is not measured FPGA energy",
            "BatchNorm is assumed fused at inference",
        ],
    }
