"""High-level polygon NUFFT and ENUFFT workflows."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .ems import EMSConfig, SparseModeSelection, select_sparse_conjugate_modes
from .geometry import (
    AnalysisWindow,
    WindowConfig,
    build_analysis_window,
    compute_local_voronoi_weights,
    points_to_window_coordinates,
)
from .nufft import NUFFTConfig, compute_direct_dft_coefficients, compute_nufft_coefficients


WeightKind = Literal["uniform", "voronoi", None]


@dataclass(frozen=True)
class PolygonNUFFTResult:
    """NUFFT result for a polygon-prescribed square analysis window."""

    spectrum: np.ndarray
    m_values: np.ndarray
    n_values: np.ndarray
    window: AnalysisWindow
    mask: np.ndarray
    local_points: np.ndarray
    values: np.ndarray
    weights: np.ndarray | None
    config: NUFFTConfig

    @property
    def sample_count(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class ENUFFTResult:
    """ENUFFT result: raw NUFFT spectrum plus EMS sparse selection."""

    raw: PolygonNUFFTResult
    selection: SparseModeSelection

    @property
    def raw_spectrum(self) -> np.ndarray:
        return self.raw.spectrum

    @property
    def spectrum(self) -> np.ndarray:
        return self.selection.spectrum

    @property
    def selected_modes(self) -> np.ndarray:
        return self.selection.selected_modes

    @property
    def mode_pair_count(self) -> int:
        return self.selection.mode_pair_count

    @property
    def signed_mode_count(self) -> int:
        return self.selection.signed_mode_count

    @property
    def power_retained(self) -> float:
        return self.selection.power_retained

    @property
    def window(self) -> AnalysisWindow:
        return self.raw.window

    @property
    def m_values(self) -> np.ndarray:
        return self.raw.m_values

    @property
    def n_values(self) -> np.ndarray:
        return self.raw.n_values

    def reconstruct(self, points=None, sparse: bool = True) -> np.ndarray:
        """Reconstruct the selected or raw spectrum at local or original points."""

        if points is None:
            local_points = self.raw.local_points
        else:
            local_points = np.asarray(points, dtype=float)
            if local_points.ndim != 2 or local_points.shape[1] != 2:
                raise ValueError("points must have shape (N, 2)")
        spectrum = self.spectrum if sparse else self.raw_spectrum
        return reconstruct_at_points(
            spectrum,
            local_points[:, 0],
            local_points[:, 1],
            self.m_values,
            self.n_values,
            self.window.Lx,
            self.window.Ly,
        )


def mode_values(mode_limit: int) -> np.ndarray:
    """Return signed integer modes from ``-mode_limit`` to ``mode_limit``."""

    mode_limit = int(mode_limit)
    if mode_limit < 0:
        raise ValueError("mode_limit must be nonnegative")
    return np.arange(-mode_limit, mode_limit + 1, dtype=int)


def _mode_arrays(mode_limit: int, m_values=None, n_values=None) -> tuple[np.ndarray, np.ndarray]:
    if m_values is None:
        m_values = mode_values(mode_limit)
    else:
        m_values = np.asarray(m_values, dtype=int)
    if n_values is None:
        n_values = mode_values(mode_limit)
    else:
        n_values = np.asarray(n_values, dtype=int)
    return m_values, n_values


def _finite_mask(points, values) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    return np.isfinite(points).all(axis=1) & np.isfinite(values)


def nufft_on_polygon(
    points,
    values,
    polygon,
    mode_limit: int,
    window_config: WindowConfig | None = None,
    nufft_config: NUFFTConfig | None = None,
    weight_type: WeightKind = "uniform",
    m_values=None,
    n_values=None,
    direct: bool = False,
) -> PolygonNUFFTResult:
    """Compute a coefficient block from samples selected by a polygon window."""

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if values.ndim != 1 or values.size != points.shape[0]:
        raise ValueError("values must be one-dimensional and match points")
    nufft_config = NUFFTConfig() if nufft_config is None else nufft_config
    nufft_config.validate()
    m_values, n_values = _mode_arrays(mode_limit, m_values, n_values)

    finite = _finite_mask(points, values)
    mask, window = build_analysis_window(points, polygon, window_config)
    active = mask & finite
    if int(np.sum(active)) < 3:
        raise ValueError("analysis window contains fewer than three finite samples")
    local_x, local_y = points_to_window_coordinates(points[active], window)
    local_points = np.column_stack([local_x, local_y])
    local_values = values[active]
    if weight_type in {None, "uniform"}:
        weights = None
    elif weight_type == "voronoi":
        weights = compute_local_voronoi_weights(
            local_x,
            local_y,
            window.Lx,
            window.Ly,
            window=window,
        )
    else:
        raise ValueError("weight_type must be 'uniform', 'voronoi', or None")

    if direct:
        spectrum = compute_direct_dft_coefficients(
            local_x,
            local_y,
            local_values,
            m_values,
            n_values,
            window.Lx,
            window.Ly,
            weights,
        )
    else:
        spectrum = compute_nufft_coefficients(
            local_x,
            local_y,
            local_values,
            m_values,
            n_values,
            window.Lx,
            window.Ly,
            nufft_config.oversample,
            nufft_config.kernel_half_width,
            nufft_config.kernel_beta,
            nufft_config.kernel_type,
            weights,
        )
    return PolygonNUFFTResult(
        spectrum=spectrum,
        m_values=m_values,
        n_values=n_values,
        window=window,
        mask=active,
        local_points=local_points,
        values=local_values,
        weights=weights,
        config=nufft_config,
    )


def enufft_on_polygon(
    points,
    values,
    polygon,
    mode_limit: int,
    window_config: WindowConfig | None = None,
    nufft_config: NUFFTConfig | None = None,
    ems_config: EMSConfig | None = None,
    weight_type: WeightKind = "uniform",
    m_values=None,
    n_values=None,
    direct: bool = False,
) -> ENUFFTResult:
    """Compute NUFFT coefficients on a polygon-prescribed square and apply EMS."""

    raw = nufft_on_polygon(
        points,
        values,
        polygon,
        mode_limit,
        window_config=window_config,
        nufft_config=nufft_config,
        weight_type=weight_type,
        m_values=m_values,
        n_values=n_values,
        direct=direct,
    )
    selection = select_sparse_conjugate_modes(
        raw.spectrum,
        raw.m_values,
        raw.n_values,
        config=EMSConfig(k_max=mode_limit) if ems_config is None else ems_config,
    )
    return ENUFFTResult(raw=raw, selection=selection)


def reconstruct_at_points(
    spectrum,
    x_values,
    y_values,
    m_values,
    n_values,
    domain_length_x: float,
    domain_length_y: float,
) -> np.ndarray:
    """Evaluate an inverse Fourier series at arbitrary local coordinates."""

    spectrum = np.asarray(spectrum)
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    if x_values.shape != y_values.shape:
        raise ValueError("x_values and y_values must have the same shape")
    if spectrum.shape != (m_values.size, n_values.size):
        raise ValueError("spectrum shape must be (len(m_values), len(n_values))")

    field = np.zeros(x_values.size, dtype=complex)
    for m_index, m_mode in enumerate(m_values):
        phase_x = 2.0 * np.pi * int(m_mode) * x_values.ravel() / domain_length_x
        for n_index, n_mode in enumerate(n_values):
            coeff = spectrum[m_index, n_index]
            if np.abs(coeff) <= 1e-15:
                continue
            phase = phase_x + 2.0 * np.pi * int(n_mode) * y_values.ravel() / domain_length_y
            field += coeff * np.exp(1j * phase)
    field = field.reshape(x_values.shape)
    return np.real_if_close(field, tol=1000).real


def spectrum_amplitudes(spectrum) -> np.ndarray:
    """Return coefficient magnitudes sorted from largest to smallest."""

    return np.sort(np.abs(np.asarray(spectrum)).ravel())[::-1]


def dominant_mode_pair(spectrum, m_values, n_values) -> tuple[int, int, float]:
    """Return the strongest non-DC canonical mode pair and amplitude."""

    spectrum = np.asarray(spectrum)
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    best_pair = (0, 0)
    best_amp = 0.0
    m_lookup = {int(value): index for index, value in enumerate(m_values)}
    n_lookup = {int(value): index for index, value in enumerate(n_values)}
    for m_mode in m_values:
        for n_mode in n_values:
            m_mode = int(m_mode)
            n_mode = int(n_mode)
            if m_mode == 0 and n_mode == 0:
                continue
            if m_mode < 0 or (m_mode == 0 and n_mode < 0):
                continue
            amp = float(
                np.sqrt(
                    (
                        np.abs(spectrum[m_lookup[m_mode], n_lookup[n_mode]]) ** 2
                        + np.abs(spectrum[m_lookup[-m_mode], n_lookup[-n_mode]]) ** 2
                    )
                    / 2.0
                )
            )
            if amp > best_amp:
                best_pair = (m_mode, n_mode)
                best_amp = amp
    return best_pair[0], best_pair[1], best_amp
