import numpy as np

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from enufft import compute_direct_dft_coefficients, compute_nufft_coefficients, mode_values


def test_direct_dft_recovers_grid_cosine_coefficients():
    nx = 32
    ny = 32
    lx = 1.0
    ly = 1.0
    x_grid, y_grid = np.meshgrid(np.linspace(0.0, lx, nx, endpoint=False), np.linspace(0.0, ly, ny, endpoint=False))
    x = x_grid.ravel()
    y = y_grid.ravel()
    m0 = 2
    n0 = -1
    phase = 0.37
    amplitude = 3.0
    values = amplitude * np.cos(2.0 * np.pi * (m0 * x / lx + n0 * y / ly) + phase)
    modes = mode_values(4)

    coeffs = compute_direct_dft_coefficients(x, y, values, modes, modes, lx, ly)
    lookup = {int(value): index for index, value in enumerate(modes)}

    assert np.isclose(coeffs[lookup[m0], lookup[n0]], 0.5 * amplitude * np.exp(1j * phase), atol=1e-12)
    assert np.isclose(coeffs[lookup[-m0], lookup[-n0]], 0.5 * amplitude * np.exp(-1j * phase), atol=1e-12)
    coeffs[lookup[m0], lookup[n0]] = 0.0
    coeffs[lookup[-m0], lookup[-n0]] = 0.0
    assert np.max(np.abs(coeffs)) < 1e-12


def test_nufft_tracks_direct_dft_on_scattered_samples():
    rng = np.random.default_rng(17)
    lx = 1.0
    ly = 1.0
    x = rng.uniform(0.0, lx, 900)
    y = rng.uniform(0.0, ly, 900)
    values = (
        np.cos(2.0 * np.pi * (2 * x / lx + 1 * y / ly) + 0.2)
        + 0.35 * np.sin(2.0 * np.pi * (-1 * x / lx + 3 * y / ly) - 0.4)
    )
    modes = mode_values(4)

    direct = compute_direct_dft_coefficients(x, y, values, modes, modes, lx, ly)
    approx = compute_nufft_coefficients(x, y, values, modes, modes, lx, ly, oversample=2.0)

    assert np.median(np.abs(approx - direct)) < 5e-2
    assert np.max(np.abs(approx - direct)) < 2.5e-1
