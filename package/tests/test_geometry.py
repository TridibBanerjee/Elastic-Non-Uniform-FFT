import numpy as np

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from enufft import WindowConfig, build_analysis_window, points_in_polygon, polygon_area


def test_points_in_arbitrary_polygon_and_area():
    polygon = np.array([[0.0, 0.0], [2.0, 0.2], [1.6, 1.5], [0.7, 2.0], [-0.2, 1.0]])
    points = np.array([[0.5, 0.5], [1.5, 1.0], [2.5, 2.5], [0.0, 0.0]])

    mask = points_in_polygon(points, polygon)

    assert polygon_area(polygon) > 2.5
    assert mask.tolist() == [True, True, False, True]


def test_build_square_window_from_polygon_with_polygon_support():
    polygon = np.array([[0.1, 0.1], [1.2, 0.0], [1.4, 0.8], [0.6, 1.3], [0.0, 0.7]])
    x_grid, y_grid = np.meshgrid(np.linspace(-0.2, 1.6, 28), np.linspace(-0.2, 1.5, 27))
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    mask, window = build_analysis_window(
        points,
        polygon,
        WindowConfig(support="polygon", alignment="edge_aligned", expansion=1.2),
    )

    assert window.support == "polygon"
    assert window.Lx == window.Ly
    assert np.sum(mask) > 50
    assert window.area <= window.Lx * window.Ly
