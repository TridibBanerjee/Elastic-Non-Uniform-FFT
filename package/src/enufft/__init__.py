"""Elastic non-uniform FFT and elastic mode selection."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

__version__ = "1.0.0"

from .analysis import (
    ENUFFTResult,
    PolygonNUFFTResult,
    dominant_mode_pair,
    enufft_on_polygon,
    mode_values,
    nufft_on_polygon,
    reconstruct_at_points,
    spectrum_amplitudes,
)
from .ems import (
    EMSConfig,
    EMSResult,
    SparseModeSelection,
    count_signed_nonzero_modes,
    count_unique_mode_pairs,
    elastic_mode_selection,
    ems_retain_count,
    select_sparse_conjugate_modes,
)
from .geometry import (
    AnalysisWindow,
    WindowConfig,
    build_analysis_window,
    points_in_polygon,
    points_to_window_coordinates,
    polygon_area,
)
from .nufft import (
    NUFFTConfig,
    build_auxiliary_grid,
    compute_direct_dft_coefficients,
    compute_nufft_coefficients,
    compute_nufft_for_kernels,
)
from .synthetic import generate_dem_points, irregular_orography

__all__ = [
    "__version__",
    "AnalysisWindow",
    "EMSConfig",
    "EMSResult",
    "ENUFFTResult",
    "NUFFTConfig",
    "PolygonNUFFTResult",
    "SparseModeSelection",
    "WindowConfig",
    "build_analysis_window",
    "build_auxiliary_grid",
    "compute_direct_dft_coefficients",
    "compute_nufft_coefficients",
    "compute_nufft_for_kernels",
    "count_signed_nonzero_modes",
    "count_unique_mode_pairs",
    "dominant_mode_pair",
    "elastic_mode_selection",
    "ems_retain_count",
    "enufft_on_polygon",
    "generate_dem_points",
    "irregular_orography",
    "mode_values",
    "nufft_on_polygon",
    "points_in_polygon",
    "points_to_window_coordinates",
    "polygon_area",
    "reconstruct_at_points",
    "select_sparse_conjugate_modes",
    "spectrum_amplitudes",
]
