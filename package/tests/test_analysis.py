import numpy as np

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from enufft import EMSConfig, WindowConfig, enufft_on_polygon


def test_enufft_on_polygon_prescribed_square_recovers_dominant_pair():
    side = 1.0
    grid = np.linspace(0.0, side, 48, endpoint=False)
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="xy")
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    polygon = np.array([[0.18, 0.10], [0.82, 0.05], [0.94, 0.70], [0.50, 0.92], [0.12, 0.68]])
    m0 = 2
    n0 = 1
    phase = 0.25
    values = np.cos(2.0 * np.pi * (m0 * points[:, 0] / side + n0 * points[:, 1] / side) + phase)

    result = enufft_on_polygon(
        points,
        values,
        polygon,
        mode_limit=4,
        window_config=WindowConfig(support="square", alignment="centroid", side_length=side, center=(0.5, 0.5)),
        ems_config=EMSConfig(k_min=1, k_max=1, alpha_min=0.0, alpha_max=1.0),
        direct=True,
    )

    assert tuple(result.selected_modes[0]) == (m0, n0)
    assert result.mode_pair_count == 1
    assert result.signed_mode_count == 2
    recon = result.reconstruct()
    assert np.sqrt(np.mean((recon - result.raw.values) ** 2)) < 1e-12
