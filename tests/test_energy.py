from etsr.profiling.energy import estimate_horowitz_energy


def test_energy_estimate_units():
    result = estimate_horowitz_energy(100.0, 200.0, mac_energy_pj=4.6, ac_energy_pj=0.9)
    assert result["total_energy_pj"] == 640.0
    assert result["total_energy_uj"] == 0.00064
    assert "excludes memory access and data movement" in result["limitations"]
