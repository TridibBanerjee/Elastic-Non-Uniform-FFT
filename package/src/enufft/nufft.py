"""NUFFT and direct DFT coefficient routines.

The coefficient convention is the same as the upstream ENUFFT scripts:

    h_hat[m, n] = mean_q h_q exp[-i(k_m x_q + l_n y_q)]

Optional sample weights are normalized into this same sample-mean convention.
"""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import i0


@dataclass(frozen=True)
class NUFFTConfig:
    """Kaiser-Bessel NUFFT settings."""

    oversample: float = 1.5
    kernel_half_width: int = 4
    kernel_beta: float = 2.34
    kernel_type: str = "optimized"

    def validate(self) -> None:
        if self.oversample <= 0.0:
            raise ValueError("oversample must be positive")
        if int(self.kernel_half_width) < 1:
            raise ValueError("kernel_half_width must be at least 1")
        if self.kernel_type not in {"optimized", "baseline"}:
            raise ValueError("kernel_type must be 'optimized' or 'baseline'")


def _as_float_1d(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validate_samples(x_values, y_values, h_values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = _as_float_1d(x_values, "x_values")
    y_values = _as_float_1d(y_values, "y_values")
    h_values = _as_float_1d(h_values, "h_values")
    if x_values.size != y_values.size or x_values.size != h_values.size:
        raise ValueError("x_values, y_values, and h_values must have the same length")
    if x_values.size == 0:
        raise ValueError("at least one sample is required")
    return x_values, y_values, h_values


def build_auxiliary_grid(
    mode_limit_x: int,
    mode_limit_y: int,
    domain_length_x: float,
    domain_length_y: float,
    oversample: float,
) -> tuple[int, int, float, float, np.ndarray, np.ndarray]:
    """Choose an even oversampled auxiliary grid for the retained modes."""

    if domain_length_x <= 0.0 or domain_length_y <= 0.0:
        raise ValueError("domain lengths must be positive")
    nx_aux = max(int(np.ceil(oversample * 2.0 * mode_limit_x)), 2 * mode_limit_x, 2)
    ny_aux = max(int(np.ceil(oversample * 2.0 * mode_limit_y)), 2 * mode_limit_y, 2)
    nx_aux += nx_aux % 2
    ny_aux += ny_aux % 2
    dx_aux = domain_length_x / nx_aux
    dy_aux = domain_length_y / ny_aux
    x_aux = np.linspace(0.0, domain_length_x, nx_aux, endpoint=False)
    y_aux = np.linspace(0.0, domain_length_y, ny_aux, endpoint=False)
    return nx_aux, ny_aux, dx_aux, dy_aux, x_aux, y_aux


def scale_sample_values(h_values, sample_weights=None) -> np.ndarray:
    """Normalize optional quadrature weights into sample-mean units."""

    sample_values = np.asarray(h_values, dtype=float)
    if sample_weights is None:
        return sample_values
    sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.ndim != 1 or sample_weights.size != sample_values.size:
        raise ValueError("sample_weights must be one-dimensional and match h_values")
    total_weight = float(np.sum(sample_weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("sample_weights must have a positive finite sum")
    return sample_values * sample_weights / total_weight * sample_values.size


def kaiser_bessel_alpha(kernel_type: str, kernel_half_width: int, kernel_beta: float) -> float:
    """Return the Kaiser-Bessel shape parameter for the selected kernel rule."""

    if kernel_type == "optimized":
        return float(np.pi * np.sqrt((2.0 * kernel_half_width) ** 2 * (kernel_beta / np.pi) ** 2 - 0.8))
    if kernel_type == "baseline":
        return float(kernel_beta)
    raise ValueError(f"Unknown kernel type: {kernel_type}")


def kaiser_bessel_kernel(grid_distance, kernel_type: str, kernel_half_width: int, kernel_beta: float) -> np.ndarray:
    """Evaluate the compact Kaiser-Bessel spreading kernel."""

    grid_distance = np.asarray(grid_distance, dtype=float)
    alpha = kaiser_bessel_alpha(kernel_type, kernel_half_width, kernel_beta)
    absolute_distance = np.abs(grid_distance)
    kernel_values = np.zeros_like(absolute_distance, dtype=float)
    support_mask = absolute_distance <= kernel_half_width
    inside_sqrt = 1.0 - (grid_distance[support_mask] / kernel_half_width) ** 2
    kernel_values[support_mask] = i0(alpha * np.sqrt(inside_sqrt)) / i0(kernel_beta)
    return kernel_values


def kb_fourier_transform(wavenumber, grid_spacing: float, kernel_type: str, kernel_half_width: int, kernel_beta: float) -> np.ndarray:
    """Evaluate the 1-D Fourier transform used for NUFFT deconvolution."""

    alpha = kaiser_bessel_alpha(kernel_type, kernel_half_width, kernel_beta)
    scaled_wavenumber = np.asarray(wavenumber, dtype=float) * kernel_half_width * grid_spacing
    inside = alpha**2 - scaled_wavenumber**2
    ratio = np.empty_like(scaled_wavenumber, dtype=float)
    positive_mask = inside >= 0.0
    positive_root = np.sqrt(inside[positive_mask] + 1e-15)
    ratio[positive_mask] = np.sinh(positive_root) / positive_root
    negative_root = np.sqrt(-inside[~positive_mask] + 1e-15)
    ratio[~positive_mask] = np.sin(negative_root) / negative_root
    return (2.0 * kernel_half_width / i0(kernel_beta)) * ratio


def wrap_periodic_distances(distance, period_length: float) -> np.ndarray:
    """Map displacements to their shortest periodic representatives."""

    wrapped_distance = np.where(distance > period_length / 2.0, distance - period_length, distance)
    wrapped_distance = np.where(wrapped_distance < -period_length / 2.0, wrapped_distance + period_length, wrapped_distance)
    return wrapped_distance


def spread_samples_to_grid(
    x_values,
    y_values,
    h_values,
    kernel_type: str,
    kernel_half_width: int,
    kernel_beta: float,
    dx_aux: float,
    dy_aux: float,
    nx_aux: int,
    ny_aux: int,
    x_aux,
    y_aux,
    domain_length_x: float,
    domain_length_y: float,
) -> np.ndarray:
    """Spread scattered samples onto the auxiliary grid with a KB stencil."""

    offsets = np.arange(-kernel_half_width, kernel_half_width + 1)
    di_offsets, dj_offsets = np.meshgrid(offsets, offsets, indexing="xy")
    di_offsets = di_offsets.ravel()
    dj_offsets = dj_offsets.ravel()
    i_center = np.floor(x_values / dx_aux).astype(int)
    j_center = np.floor(y_values / dy_aux).astype(int)
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


def extract_target_mode_block(fourier_grid, m_values, n_values, nx_aux: int, ny_aux: int) -> np.ndarray:
    """Extract the signed retained mode block with axes m then n."""

    coefficients = np.zeros((m_values.size, n_values.size), dtype=complex)
    for m_index, m_mode in enumerate(m_values):
        for n_index, n_mode in enumerate(n_values):
            coefficients[m_index, n_index] = fourier_grid[int(n_mode) % ny_aux, int(m_mode) % nx_aux]
    return coefficients


def compute_nufft_coefficients(
    x_values,
    y_values,
    h_values,
    m_values,
    n_values,
    domain_length_x: float,
    domain_length_y: float,
    oversample: float = 1.5,
    kernel_half_width: int = 4,
    kernel_beta: float = 2.34,
    kernel_type: str = "optimized",
    sample_weights=None,
) -> np.ndarray:
    """Apply the Kaiser-Bessel NUFFT chain and return retained coefficients."""

    x_values, y_values, h_values = _validate_samples(x_values, y_values, h_values)
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    if m_values.ndim != 1 or n_values.ndim != 1:
        raise ValueError("m_values and n_values must be one-dimensional")
    if m_values.size == 0 or n_values.size == 0:
        raise ValueError("at least one m and one n mode are required")
    config = NUFFTConfig(oversample, int(kernel_half_width), float(kernel_beta), kernel_type)
    config.validate()
    mode_limit_x = int(np.max(np.abs(m_values)))
    mode_limit_y = int(np.max(np.abs(n_values)))
    nx_aux, ny_aux, dx_aux, dy_aux, x_aux, y_aux = build_auxiliary_grid(
        mode_limit_x, mode_limit_y, domain_length_x, domain_length_y, config.oversample
    )
    sample_values = scale_sample_values(h_values, sample_weights)
    auxiliary_grid = spread_samples_to_grid(
        x_values,
        y_values,
        sample_values,
        config.kernel_type,
        config.kernel_half_width,
        config.kernel_beta,
        dx_aux,
        dy_aux,
        nx_aux,
        ny_aux,
        x_aux,
        y_aux,
        domain_length_x,
        domain_length_y,
    )
    fft_coefficients = np.fft.fft2(auxiliary_grid) / (nx_aux * ny_aux)
    kx_fft = 2.0 * np.pi * np.fft.fftfreq(nx_aux, dx_aux)
    ky_fft = 2.0 * np.pi * np.fft.fftfreq(ny_aux, dy_aux)
    kx_grid, ky_grid = np.meshgrid(kx_fft, ky_fft, indexing="xy")
    phi_x = kb_fourier_transform(kx_grid, dx_aux, config.kernel_type, config.kernel_half_width, config.kernel_beta)
    phi_y = kb_fourier_transform(ky_grid, dy_aux, config.kernel_type, config.kernel_half_width, config.kernel_beta)
    phi_2d = phi_x * phi_y
    deconvolved_fft = fft_coefficients / np.where(np.abs(phi_2d) > 1e-10, phi_2d, 1.0)
    return extract_target_mode_block(deconvolved_fft, m_values, n_values, nx_aux, ny_aux)


def compute_nufft_for_kernels(
    x_values,
    y_values,
    h_values,
    m_values,
    n_values,
    domain_length_x: float,
    domain_length_y: float,
    oversample: float,
    kernel_half_width: int,
    kernel_beta: float,
    kernel_types=("optimized", "baseline"),
    sample_weights=None,
) -> dict[str, np.ndarray]:
    """Reuse one field and mode set for several kernel choices."""

    return {
        kernel_type: compute_nufft_coefficients(
            x_values,
            y_values,
            h_values,
            m_values,
            n_values,
            domain_length_x,
            domain_length_y,
            oversample,
            kernel_half_width,
            kernel_beta,
            kernel_type,
            sample_weights,
        )
        for kernel_type in kernel_types
    }


def compute_direct_dft_coefficients(
    x_values,
    y_values,
    h_values,
    m_values,
    n_values,
    domain_length_x: float,
    domain_length_y: float,
    sample_weights=None,
) -> np.ndarray:
    """Evaluate the scattered-point Fourier sum directly."""

    x_values, y_values, h_values = _validate_samples(x_values, y_values, h_values)
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
