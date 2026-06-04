# Module_Nufft.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the shared NUFFT pipeline and direct DFT pipeline used by ENUFFT cases.

import numpy as np
from scipy.special import i0


# Choose an oversampled auxiliary grid that can resolve the retained Fourier modes.
def build_auxiliary_grid(mode_limit_x, mode_limit_y, domain_length_x, domain_length_y, oversample):
    nx_aux = max(int(np.ceil(oversample * 2.0 * mode_limit_x)), 2 * mode_limit_x, 2)
    ny_aux = max(int(np.ceil(oversample * 2.0 * mode_limit_y)), 2 * mode_limit_y, 2)
    nx_aux += nx_aux % 2
    ny_aux += ny_aux % 2
    dx_aux = domain_length_x / nx_aux
    dy_aux = domain_length_y / ny_aux
    x_aux = np.linspace(0.0, domain_length_x, nx_aux, endpoint=False)
    y_aux = np.linspace(0.0, domain_length_y, ny_aux, endpoint=False)
    return nx_aux, ny_aux, dx_aux, dy_aux, x_aux, y_aux


# Convert optional quadrature weights into the same normalization as a sample mean.
def scale_sample_values(h_values, sample_weights):
    sample_values = np.asarray(h_values, dtype=float)
    if sample_weights is None:
        return sample_values
    sample_weights = np.asarray(sample_weights, dtype=float)
    return sample_values * sample_weights / np.sum(sample_weights) * sample_values.size


# Return the KB shape parameter gamma for the chosen kernel rule.
def kaiser_bessel_alpha(kernel_type, kernel_half_width, kernel_beta):
    if kernel_type == "optimized":
        return np.pi * np.sqrt((2.0 * kernel_half_width) ** 2 * (kernel_beta / np.pi) ** 2 - 0.8)
    if kernel_type == "baseline":
        return kernel_beta
    raise ValueError(f"Unknown kernel type: {kernel_type}")


# Evaluate the compact Kaiser-Bessel kernel phi(zeta) on grid-space offsets zeta.
def kaiser_bessel_kernel(grid_distance, kernel_type, kernel_half_width, kernel_beta):
    grid_distance = np.asarray(grid_distance, dtype=float)
    alpha = kaiser_bessel_alpha(kernel_type, kernel_half_width, kernel_beta)
    absolute_distance = np.abs(grid_distance)
    kernel_values = np.zeros_like(absolute_distance, dtype=float)
    support_mask = absolute_distance <= kernel_half_width
    inside_sqrt = 1.0 - (grid_distance[support_mask] / kernel_half_width) ** 2
    kernel_values[support_mask] = i0(alpha * np.sqrt(inside_sqrt)) / i0(kernel_beta)
    return kernel_values


# Evaluate the 1D Fourier transform Phi(kappa) used in the deconvolution step.
def kb_fourier_transform(wavenumber, grid_spacing, kernel_type, kernel_half_width, kernel_beta):
    alpha = kaiser_bessel_alpha(kernel_type, kernel_half_width, kernel_beta)
    scaled_wavenumber = np.asarray(wavenumber, dtype=float) * kernel_half_width * grid_spacing
    inside = alpha ** 2 - scaled_wavenumber ** 2
    ratio = np.empty_like(scaled_wavenumber, dtype=float)
    positive_mask = inside >= 0.0
    positive_root = np.sqrt(inside[positive_mask] + 1e-15)
    ratio[positive_mask] = np.sinh(positive_root) / positive_root
    negative_root = np.sqrt(-inside[~positive_mask] + 1e-15)
    ratio[~positive_mask] = np.sin(negative_root) / negative_root
    return (2.0 * kernel_half_width / i0(kernel_beta)) * ratio


# Map each displacement to its shortest periodic representative.
def wrap_periodic_distances(distance, period_length):
    wrapped_distance = np.where(distance > period_length / 2.0, distance - period_length, distance)
    wrapped_distance = np.where(wrapped_distance < -period_length / 2.0, wrapped_distance + period_length, wrapped_distance)
    return wrapped_distance


# Spread the scattered samples onto the auxiliary grid with the tensor-product KB stencil.
def spread_samples_to_grid(x_values, y_values, h_values, kernel_type, kernel_half_width, kernel_beta, dx_aux, dy_aux, nx_aux, ny_aux, x_aux, y_aux, domain_length_x, domain_length_y):
    offsets = np.arange(-kernel_half_width, kernel_half_width + 1)
    di_offsets, dj_offsets = np.meshgrid(offsets, offsets, indexing="xy")
    di_offsets = di_offsets.ravel()
    dj_offsets = dj_offsets.ravel()
    i_center = (x_values / dx_aux).astype(int)
    j_center = (y_values / dy_aux).astype(int)
    i_indices = (i_center[:, None] + di_offsets[None, :]) % nx_aux
    j_indices = (j_center[:, None] + dj_offsets[None, :]) % ny_aux
    x_distance = x_values[:, None] - x_aux[i_indices]
    y_distance = y_values[:, None] - y_aux[j_indices]
    x_distance = wrap_periodic_distances(x_distance, domain_length_x)
    y_distance = wrap_periodic_distances(y_distance, domain_length_y)
    x_weights = kaiser_bessel_kernel(x_distance / dx_aux, kernel_type, kernel_half_width, kernel_beta)
    y_weights = kaiser_bessel_kernel(y_distance / dy_aux, kernel_type, kernel_half_width, kernel_beta)
    weights = h_values[:, None] * x_weights * y_weights
    auxiliary_grid = np.zeros((ny_aux, nx_aux), dtype=float)
    np.add.at(auxiliary_grid, (j_indices.ravel(), i_indices.ravel()), weights.ravel())
    auxiliary_grid *= (nx_aux * ny_aux) / h_values.size
    return auxiliary_grid


# Extract the retained signed mode block with first index m and second index n.
def extract_target_mode_block(fourier_grid, m_values, n_values, nx_aux, ny_aux):
    coefficients = np.zeros((m_values.size, n_values.size), dtype=complex)
    for m_index, m_mode in enumerate(m_values):
        for n_index, n_mode in enumerate(n_values):
            wrapped_x_index = m_mode % nx_aux
            wrapped_y_index = n_mode % ny_aux
            coefficients[m_index, n_index] = fourier_grid[wrapped_y_index, wrapped_x_index]
    return coefficients


# Apply the full NUFFT chain for one kernel and return the retained coefficient block.
def compute_nufft_coefficients(x_values, y_values, h_values, m_values, n_values, domain_length_x, domain_length_y, oversample, kernel_half_width, kernel_beta, kernel_type, sample_weights=None):
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    mode_limit_x = int(np.max(np.abs(m_values)))
    mode_limit_y = int(np.max(np.abs(n_values)))
    nx_aux, ny_aux, dx_aux, dy_aux, x_aux, y_aux = build_auxiliary_grid(mode_limit_x, mode_limit_y, domain_length_x, domain_length_y, oversample)
    sample_values = scale_sample_values(h_values, sample_weights)
    auxiliary_grid = spread_samples_to_grid(x_values, y_values, sample_values, kernel_type, kernel_half_width, kernel_beta, dx_aux, dy_aux, nx_aux, ny_aux, x_aux, y_aux, domain_length_x, domain_length_y)
    fft_coefficients = np.fft.fft2(auxiliary_grid) / (nx_aux * ny_aux)
    kx_fft = 2.0 * np.pi * np.fft.fftfreq(nx_aux, dx_aux)
    ky_fft = 2.0 * np.pi * np.fft.fftfreq(ny_aux, dy_aux)
    kx_grid, ky_grid = np.meshgrid(kx_fft, ky_fft, indexing="xy")
    phi_x = kb_fourier_transform(kx_grid, dx_aux, kernel_type, kernel_half_width, kernel_beta)
    phi_y = kb_fourier_transform(ky_grid, dy_aux, kernel_type, kernel_half_width, kernel_beta)
    phi_2d = phi_x * phi_y
    deconvolved_fft = fft_coefficients / np.where(np.abs(phi_2d) > 1e-10, phi_2d, 1.0)
    return extract_target_mode_block(deconvolved_fft, m_values, n_values, nx_aux, ny_aux)


# Reuse the same scattered field and mode set for several kernel choices.
def compute_nufft_for_kernels(x_values, y_values, h_values, m_values, n_values, domain_length_x, domain_length_y, oversample, kernel_half_width, kernel_beta, kernel_types, sample_weights=None):
    nufft_coefficients = {}
    for kernel_type in kernel_types:
        nufft_coefficients[kernel_type] = compute_nufft_coefficients(x_values, y_values, h_values, m_values, n_values, domain_length_x, domain_length_y, oversample, kernel_half_width, kernel_beta, kernel_type, sample_weights)
    return nufft_coefficients


# Evaluate the scattered-point Fourier sum directly on the retained mode set.
def compute_direct_dft_coefficients(x_values, y_values, h_values, m_values, n_values, domain_length_x, domain_length_y, sample_weights=None):
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    sample_values = scale_sample_values(h_values, sample_weights)
    dft_coefficients = np.zeros((m_values.size, n_values.size), dtype=complex)
    for m_index, m_mode in enumerate(m_values):
        k_m = 2.0 * np.pi * m_mode / domain_length_x
        for n_index, n_mode in enumerate(n_values):
            l_n = 2.0 * np.pi * n_mode / domain_length_y
            phase = np.exp(-1j * (k_m * x_values + l_n * y_values))
            dft_coefficients[m_index, n_index] = np.mean(sample_values * phase)
    return dft_coefficients
