import numpy as np

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from enufft import EMSConfig, elastic_mode_selection, ems_retain_count, mode_values, select_sparse_conjugate_modes


def test_scalar_ems_reports_pair_and_signed_counts():
    energies = np.array([10.0, 1.0, 0.5, 0.25, 0.1])
    config = EMSConfig(k_min=1, k_max=5, alpha_min=0.5, alpha_max=0.5)

    result = elastic_mode_selection(energies, config)

    assert result.k_star == 1
    assert result.retain_count("pairs") == 1
    assert result.retain_count("signed") == 2
    assert ems_retain_count(energies, config, count="signed") == 2


def test_sparse_conjugate_selection_keeps_both_signed_modes():
    modes = mode_values(3)
    lookup = {int(value): index for index, value in enumerate(modes)}
    spectrum = np.zeros((modes.size, modes.size), dtype=complex)
    spectrum[lookup[1], lookup[2]] = 3.0 + 1.0j
    spectrum[lookup[-1], lookup[-2]] = 3.0 - 1.0j
    spectrum[lookup[2], lookup[0]] = 0.4
    spectrum[lookup[-2], lookup[0]] = 0.4

    selection = select_sparse_conjugate_modes(
        spectrum,
        modes,
        modes,
        EMSConfig(k_min=1, k_max=1, alpha_min=0.0, alpha_max=1.0),
    )

    assert selection.mode_pair_count == 1
    assert selection.signed_mode_count == 2
    assert tuple(selection.selected_modes[0]) == (1, 2)
    assert selection.spectrum[lookup[1], lookup[2]] != 0.0
    assert selection.spectrum[lookup[-1], lookup[-2]] != 0.0
    assert selection.spectrum[lookup[2], lookup[0]] == 0.0
