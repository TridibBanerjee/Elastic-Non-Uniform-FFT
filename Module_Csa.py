# Module_Csa.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the reusable CSA Fourier ranking and sparse refitting pipeline.

import numpy as np


# Build the dense signed Fourier mode list used by the full-analysis CSA fit.
def build_dense_mode_list(m_values, n_values):
    dense_modes = []
    for m_mode in m_values:
        for n_mode in n_values:
            dense_modes.append((int(m_mode), int(n_mode)))
    return np.asarray(dense_modes, dtype=int)


# Assemble the Fourier normal equations for one mode list and one sample cloud.
def build_normal_equations(x_values, y_values, h_values, mode_list, domain_length_x, domain_length_y, chunk_size=None):
    mode_count = len(mode_list)
    normal_matrix = np.zeros((mode_count, mode_count), dtype=complex)
    normal_rhs = np.zeros(mode_count, dtype=complex)
    k_m = 2.0 * np.pi * mode_list[:, 0] / domain_length_x
    l_n = 2.0 * np.pi * mode_list[:, 1] / domain_length_y
    if chunk_size is None:
        chunk_size = len(x_values)
    for start in range(0, len(x_values), int(chunk_size)):
        stop = min(start + int(chunk_size), len(x_values))
        phase = x_values[start:stop, None] * k_m[None, :] + y_values[start:stop, None] * l_n[None, :]
        basis = np.exp(1j * phase)
        basis_h = np.conj(basis.T)
        normal_matrix += basis_h @ basis
        normal_rhs += basis_h @ h_values[start:stop]
    return normal_matrix, normal_rhs


# Solve one regularized complex normal system.
def solve_tikhonov_from_normal(normal_matrix, normal_rhs, regularization):
    regularized_matrix = normal_matrix + max(float(regularization), 0.0) * np.eye(normal_matrix.shape[0], dtype=complex)
    try:
        return np.linalg.solve(regularized_matrix, normal_rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(regularized_matrix, normal_rhs, rcond=None)[0]


# Fit one Tikhonov-regularized Fourier model on a prescribed mode list.
def fit_fourier_modes(x_values, y_values, h_values, mode_list, domain_length_x, domain_length_y, regularization, chunk_size=None):
    normal_matrix, normal_rhs = build_normal_equations(x_values, y_values, h_values, mode_list, domain_length_x, domain_length_y, chunk_size)
    return solve_tikhonov_from_normal(normal_matrix, normal_rhs, regularization)


# Compute the two-stage CSA spectrum from full-analysis ranking and sparse-analysis refitting.
def compute_csa_spectrum(x_fa, y_fa, h_fa, x_sa, y_sa, h_sa, m_values, n_values, domain_length_x, domain_length_y, lambda_fa, lambda_sa, sparse_modes, chunk_size=None):
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    dense_modes = build_dense_mode_list(m_values, n_values)
    if sparse_modes is None:
        sparse_modes = 2 * max(int(np.max(np.abs(m_values))), int(np.max(np.abs(n_values))))
    sparse_modes = int(np.clip(int(sparse_modes), 1, len(dense_modes)))
    spectrum = np.zeros((m_values.size, n_values.size), dtype=complex)
    if len(x_fa) < 3 or len(x_sa) < 3:
        return {"spectrum": spectrum, "selected_modes": np.zeros((0, 2), dtype=int), "energy_sorted": np.array([], dtype=float)}
    coeff_fa = fit_fourier_modes(x_fa, y_fa, h_fa, dense_modes, domain_length_x, domain_length_y, lambda_fa, chunk_size)
    energy = np.abs(coeff_fa) ** 2
    order = np.argsort(-energy)
    selected_modes = dense_modes[order[:sparse_modes]]
    coeff_sa = fit_fourier_modes(x_sa, y_sa, h_sa, selected_modes, domain_length_x, domain_length_y, lambda_sa, chunk_size)
    m_lookup = {int(value): index for index, value in enumerate(m_values)}
    n_lookup = {int(value): index for index, value in enumerate(n_values)}
    for (m_mode, n_mode), value in zip(selected_modes, coeff_sa):
        spectrum[m_lookup[int(m_mode)], n_lookup[int(n_mode)]] = value
    return {"spectrum": spectrum, "selected_modes": selected_modes, "energy_sorted": energy[order]}


# Reconstruct sample values from a CSA-style forward-synthesis Fourier spectrum.
def reconstruct_at_points(spectrum, x_values, y_values, m_values, n_values, domain_length_x, domain_length_y):
    h_recon = np.zeros(len(x_values), dtype=complex)
    for m_index, m_mode in enumerate(m_values):
        k_m = 2.0 * np.pi * m_mode / domain_length_x
        for n_index, n_mode in enumerate(n_values):
            l_n = 2.0 * np.pi * n_mode / domain_length_y
            h_recon += spectrum[m_index, n_index] * np.exp(1j * (k_m * x_values + l_n * y_values))
    return np.real(h_recon)


# Sort non-DC signed Fourier amplitudes in descending order.
def compute_sorted_spectral_amplitudes(spectrum, m_values, n_values):
    amplitude = np.abs(np.asarray(spectrum)).copy()
    m_zero = int(np.where(np.asarray(m_values) == 0)[0][0])
    n_zero = int(np.where(np.asarray(n_values) == 0)[0][0])
    amplitude[m_zero, n_zero] = 0.0
    return np.sort(amplitude.ravel())[::-1]


# Find the strongest non-DC signed Fourier mode.
def find_dominant_mode_pair(spectrum, m_values, n_values):
    amplitude = np.abs(np.asarray(spectrum)).copy()
    m_zero = int(np.where(np.asarray(m_values) == 0)[0][0])
    n_zero = int(np.where(np.asarray(n_values) == 0)[0][0])
    amplitude[m_zero, n_zero] = 0.0
    mode_index = np.unravel_index(np.argmax(amplitude), amplitude.shape)
    return int(m_values[mode_index[0]]), int(n_values[mode_index[1]]), float(amplitude[mode_index])


# Count canonical conjugate pairs in a selected signed mode list.
def count_unique_mode_pairs(selected_modes, include_dc=True):
    if selected_modes is None or len(selected_modes) == 0:
        return 0
    pair_set = set()
    for m_value, n_value in np.asarray(selected_modes, dtype=int):
        pair_m, pair_n = int(m_value), int(n_value)
        if pair_m < 0 or (pair_m == 0 and pair_n < 0):
            pair_m, pair_n = -pair_m, -pair_n
        if include_dc or pair_m != 0 or pair_n != 0:
            pair_set.add((pair_m, pair_n))
    return int(len(pair_set))


# Count retained signed non-DC Fourier coefficients above a tolerance.
def count_signed_nonzero_modes(spectrum, m_values, n_values, tolerance=1e-15):
    amplitude = np.abs(np.asarray(spectrum))
    m_zero = int(np.where(np.asarray(m_values) == 0)[0][0])
    n_zero = int(np.where(np.asarray(n_values) == 0)[0][0])
    mask = amplitude > tolerance
    mask[m_zero, n_zero] = False
    return int(np.count_nonzero(mask))
