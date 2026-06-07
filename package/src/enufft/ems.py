"""Elastic mode selection for spectra and Fourier coefficient blocks."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


CountKind = Literal["pairs", "pair", "mode_pairs", "signed", "signed_modes"]


@dataclass(frozen=True)
class EMSConfig:
    """Parameters for elastic mode selection."""

    k_min: int = 1
    k_max: int | None = 12
    alpha_min: float = 0.0
    alpha_max: float = 0.7
    delta: float = 0.02
    w1: float = 0.5
    w2: float = 0.5

    def resolved(self, available_count: int) -> "EMSConfig":
        if available_count < 0:
            raise ValueError("available_count must be nonnegative")
        if self.k_max is None:
            k_max = max(available_count, 0)
        else:
            k_max = int(self.k_max)
        if available_count > 0:
            k_max = int(np.clip(k_max, 1, available_count))
            k_min = int(np.clip(int(self.k_min), 1, k_max))
        else:
            k_max = 0
            k_min = 0
        if self.alpha_min < 0.0 or self.alpha_max < 0.0:
            raise ValueError("alpha_min and alpha_max must be nonnegative")
        if self.alpha_min > self.alpha_max:
            raise ValueError("alpha_min must be <= alpha_max")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")
        if self.w1 < 0.0 or self.w2 < 0.0:
            raise ValueError("EMS weights must be nonnegative")
        if self.w1 + self.w2 <= 0.0:
            raise ValueError("at least one EMS weight must be positive")
        total = self.w1 + self.w2
        return EMSConfig(
            k_min=k_min,
            k_max=k_max,
            alpha_min=float(self.alpha_min),
            alpha_max=float(self.alpha_max),
            delta=float(self.delta),
            w1=float(self.w1) / total,
            w2=float(self.w2) / total,
        )


@dataclass(frozen=True)
class EMSResult:
    """Scalar EMS diagnostics for one nonnegative spectrum."""

    e_sorted: np.ndarray
    j_star: int
    j_window: int
    sum_e: float
    n_eff: float
    n_eff_clip: float
    n_eff_norm: float
    gaps: np.ndarray
    s_delta: float
    c_measure: float
    alpha_c: float
    cumulative_energy: np.ndarray
    k_star: int
    alpha_c_final: float
    config: EMSConfig

    @property
    def mode_pair_count(self) -> int:
        return int(self.k_star)

    @property
    def signed_mode_count(self) -> int:
        return int(2 * self.k_star)

    def retain_count(self, count: CountKind = "pairs") -> int:
        """Return the selected count as mode pairs or signed non-DC modes."""

        if count in {"pairs", "pair", "mode_pairs"}:
            return self.mode_pair_count
        if count in {"signed", "signed_modes"}:
            return self.signed_mode_count
        raise ValueError("count must be 'pairs' or 'signed'")

    def asdict(self) -> dict:
        data = asdict(self)
        data["config"] = asdict(self.config)
        return data


@dataclass(frozen=True)
class SparseModeSelection:
    """Pair-aware EMS result for a signed Fourier coefficient block."""

    spectrum: np.ndarray
    selected_modes: np.ndarray
    result: EMSResult
    m_values: np.ndarray
    n_values: np.ndarray
    energy_sorted: np.ndarray
    raw_mode_pairs: np.ndarray
    raw_pair_energy: np.ndarray
    power_retained: float

    @property
    def k_star(self) -> int:
        return self.result.k_star

    @property
    def k_max(self) -> int:
        return int(self.result.config.k_max or 0)

    @property
    def mode_pair_count(self) -> int:
        return int(len(self.selected_modes))

    @property
    def signed_mode_count(self) -> int:
        return count_signed_nonzero_modes(self.spectrum, self.m_values, self.n_values)

    def retain_count(self, count: CountKind = "pairs") -> int:
        if count in {"pairs", "pair", "mode_pairs"}:
            return self.mode_pair_count
        if count in {"signed", "signed_modes"}:
            return self.signed_mode_count
        raise ValueError("count must be 'pairs' or 'signed'")

    def asdict(self) -> dict:
        return {
            "spectrum": self.spectrum,
            "selected_modes": self.selected_modes,
            "k_star": self.k_star,
            "k_max": self.k_max,
            "n_eff": self.result.n_eff,
            "s_delta": self.result.s_delta,
            "c_measure": self.result.c_measure,
            "alpha_c": self.result.alpha_c,
            "power_retained": self.power_retained,
            "energy_sorted": self.energy_sorted,
        }


def elastic_mode_selection(e_values, config: EMSConfig | None = None, **overrides) -> EMSResult:
    """Run EMS on any nonnegative one-dimensional spectrum."""

    e_values = np.asarray(e_values, dtype=float).ravel()
    if np.any(e_values < 0.0):
        raise ValueError("EMS energy values must be nonnegative")
    if config is None:
        config = EMSConfig(**overrides) if overrides else EMSConfig()
    elif overrides:
        config = EMSConfig(**{**asdict(config), **overrides})
    available = int(e_values.size)
    resolved = config.resolved(available)
    if available == 0 or float(np.sum(e_values)) <= 1e-15:
        empty = np.sort(e_values)[::-1]
        return EMSResult(
            e_sorted=empty,
            j_star=available,
            j_window=0,
            sum_e=float(np.sum(e_values)),
            n_eff=0.0,
            n_eff_clip=0.0,
            n_eff_norm=0.0,
            gaps=np.array([], dtype=float),
            s_delta=0.0,
            c_measure=0.0,
            alpha_c=resolved.alpha_min,
            cumulative_energy=np.zeros_like(empty, dtype=float),
            k_star=0,
            alpha_c_final=0.0,
            config=resolved,
        )

    e_sorted = np.sort(e_values)[::-1]
    sum_e = float(np.sum(e_sorted))
    n_eff = float(sum_e**2 / max(float(np.sum(e_sorted**2)), 1e-15))
    k_max = int(resolved.k_max or 0)
    n_eff_clip = float(min(n_eff, k_max))
    j_window = min(k_max, available)
    gaps = e_sorted[: j_window - 1] / np.maximum(e_sorted[1:j_window], 1e-15)
    if j_window > 1:
        s_delta = float(np.mean(np.exp(-(gaps - 1.0) / resolved.delta)))
    else:
        s_delta = 1.0
    n_eff_norm = float(n_eff_clip / k_max) if k_max else 0.0
    c_measure = float(resolved.w1 * n_eff_norm + resolved.w2 * s_delta)
    alpha_c = float(np.clip(resolved.alpha_min + (resolved.alpha_max - resolved.alpha_min) * c_measure, 0.0, 1.0))
    cumulative_energy = np.cumsum(e_sorted) / sum_e
    k_star = k_max
    for k_value in range(int(resolved.k_min), k_max + 1):
        if cumulative_energy[k_value - 1] >= alpha_c:
            k_star = k_value
            break
    alpha_c_final = float(cumulative_energy[k_star - 1]) if k_star else 0.0
    return EMSResult(
        e_sorted=e_sorted,
        j_star=available,
        j_window=j_window,
        sum_e=sum_e,
        n_eff=n_eff,
        n_eff_clip=n_eff_clip,
        n_eff_norm=n_eff_norm,
        gaps=gaps,
        s_delta=s_delta,
        c_measure=c_measure,
        alpha_c=alpha_c,
        cumulative_energy=cumulative_energy,
        k_star=int(k_star),
        alpha_c_final=alpha_c_final,
        config=resolved,
    )


def ems_retain_count(e_values, config: EMSConfig | None = None, count: CountKind = "pairs", **overrides) -> int:
    """Return only the EMS retained count for a supplied energy spectrum."""

    return elastic_mode_selection(e_values, config=config, **overrides).retain_count(count)


def _mode_lookup(values: np.ndarray, axis_name: str) -> dict[int, int]:
    lookup = {int(value): index for index, value in enumerate(values)}
    missing = [int(-value) for value in values if int(-value) not in lookup]
    if missing:
        raise ValueError(f"{axis_name} must contain conjugate negative modes, missing {missing[:4]}")
    return lookup


def conjugate_pair_energies(spectrum, m_values, n_values) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical non-DC mode pairs and their Parseval pair energies."""

    spectrum = np.asarray(spectrum)
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    if spectrum.shape != (m_values.size, n_values.size):
        raise ValueError("spectrum shape must be (len(m_values), len(n_values))")
    m_lookup = _mode_lookup(m_values, "m_values")
    n_lookup = _mode_lookup(n_values, "n_values")
    mode_pairs: list[tuple[int, int]] = []
    energy: list[float] = []
    for m_mode in m_values:
        for n_mode in n_values:
            m_mode = int(m_mode)
            n_mode = int(n_mode)
            if m_mode == 0 and n_mode == 0:
                continue
            if (m_mode > 0) or (m_mode == 0 and n_mode > 0):
                mode_pairs.append((m_mode, n_mode))
                pair_energy = (
                    np.abs(spectrum[m_lookup[m_mode], n_lookup[n_mode]]) ** 2
                    + np.abs(spectrum[m_lookup[-m_mode], n_lookup[-n_mode]]) ** 2
                )
                energy.append(float(pair_energy))
    return np.asarray(mode_pairs, dtype=int), np.asarray(energy, dtype=float)


def select_sparse_conjugate_modes(
    spectrum,
    m_values,
    n_values,
    config: EMSConfig | None = None,
    **overrides,
) -> SparseModeSelection:
    """Select a sparse conjugate-symmetric Fourier spectrum from pair energies."""

    spectrum = np.asarray(spectrum)
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    if config is None:
        config = EMSConfig(**overrides) if overrides else EMSConfig(k_max=None)
    elif overrides:
        config = EMSConfig(**{**asdict(config), **overrides})
    mode_pairs, energy = conjugate_pair_energies(spectrum, m_values, n_values)
    if energy.size == 0 or float(np.sum(energy)) <= 1e-15:
        resolved = config.resolved(0)
        empty_result = elastic_mode_selection(np.array([], dtype=float), resolved)
        return SparseModeSelection(
            spectrum=np.zeros_like(spectrum),
            selected_modes=np.zeros((0, 2), dtype=int),
            result=empty_result,
            m_values=m_values,
            n_values=n_values,
            energy_sorted=energy,
            raw_mode_pairs=mode_pairs,
            raw_pair_energy=energy,
            power_retained=0.0,
        )
    order = np.argsort(-energy)
    sorted_pairs = mode_pairs[order]
    sorted_energy = energy[order]
    resolved = config.resolved(len(sorted_energy))
    result = elastic_mode_selection(sorted_energy, resolved)
    k_star = result.k_star
    selected_modes = sorted_pairs[:k_star]
    sparse_spectrum = np.zeros_like(spectrum)
    m_lookup = {int(value): index for index, value in enumerate(m_values)}
    n_lookup = {int(value): index for index, value in enumerate(n_values)}
    for m_mode, n_mode in selected_modes:
        m_mode = int(m_mode)
        n_mode = int(n_mode)
        sparse_spectrum[m_lookup[m_mode], n_lookup[n_mode]] = spectrum[m_lookup[m_mode], n_lookup[n_mode]]
        sparse_spectrum[m_lookup[-m_mode], n_lookup[-n_mode]] = spectrum[m_lookup[-m_mode], n_lookup[-n_mode]]
    power_retained = float(result.cumulative_energy[k_star - 1]) if k_star else 0.0
    return SparseModeSelection(
        spectrum=sparse_spectrum,
        selected_modes=selected_modes,
        result=result,
        m_values=m_values,
        n_values=n_values,
        energy_sorted=sorted_energy,
        raw_mode_pairs=sorted_pairs,
        raw_pair_energy=sorted_energy,
        power_retained=power_retained,
    )


def count_signed_nonzero_modes(spectrum, m_values, n_values, threshold: float = 1e-15, include_dc: bool = False) -> int:
    """Count retained signed modes in a coefficient block."""

    spectrum = np.asarray(spectrum)
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    keep = np.abs(spectrum) > threshold
    if not include_dc:
        m_zero = np.where(m_values == 0)[0]
        n_zero = np.where(n_values == 0)[0]
        if m_zero.size and n_zero.size:
            keep[int(m_zero[0]), int(n_zero[0])] = False
    return int(np.sum(keep))


def count_unique_mode_pairs(selected_modes, include_dc: bool = False) -> int:
    """Count unique conjugate mode pairs from signed or canonical mode rows."""

    selected_modes = np.asarray(selected_modes, dtype=int)
    if selected_modes.size == 0:
        return 0
    selected_modes = selected_modes.reshape(-1, 2)
    pairs = set()
    for m_mode, n_mode in selected_modes:
        m_mode = int(m_mode)
        n_mode = int(n_mode)
        if m_mode == 0 and n_mode == 0 and not include_dc:
            continue
        if m_mode < 0 or (m_mode == 0 and n_mode < 0):
            m_mode, n_mode = -m_mode, -n_mode
        pairs.add((m_mode, n_mode))
    return len(pairs)
