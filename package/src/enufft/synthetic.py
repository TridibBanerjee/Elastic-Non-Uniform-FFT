"""Synthetic fields and spectra used by examples and proof plots."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

import numpy as np


def generate_dem_points(sample_count, domain_length_x, domain_length_y, distribution="uniform", random_seed=None):
    """Draw scattered points from uniform or ridge-biased distributions."""

    rng = np.random.default_rng(random_seed)
    if distribution == "uniform":
        return rng.uniform(0.0, domain_length_x, sample_count), rng.uniform(0.0, domain_length_y, sample_count)
    if distribution != "ridge":
        raise ValueError("distribution must be 'uniform' or 'ridge'")
    accepted_points = []
    max_trials = max(200000, 70 * sample_count)
    total_trials = 0
    while len(accepted_points) < sample_count and total_trials < max_trials:
        batch_size = min(8000, sample_count - len(accepted_points) + 3000)
        candidates = rng.uniform(0.0, 1.0, size=(batch_size, 2)) * np.array([domain_length_x, domain_length_y])
        ridge_1 = 0.30 * domain_length_x + 0.14 * domain_length_x * np.sin(2.0 * np.pi * candidates[:, 1] / domain_length_y + 0.7)
        ridge_2 = 0.72 * domain_length_x + 0.10 * domain_length_x * np.cos(2.0 * np.pi * candidates[:, 1] / domain_length_y - 0.4)
        density = (
            0.20
            + 0.90 * np.exp(-((candidates[:, 0] - ridge_1) / (0.10 * domain_length_x)) ** 2)
            + 0.75 * np.exp(-((candidates[:, 0] - ridge_2) / (0.12 * domain_length_x)) ** 2)
        )
        density /= np.max(density) + 1e-12
        accepted_points.extend(candidates[rng.random(batch_size) < density].tolist())
        total_trials += batch_size
    if len(accepted_points) < sample_count:
        raise RuntimeError(f"Could not generate {sample_count} points after {total_trials} trials")
    accepted = np.asarray(accepted_points[:sample_count], dtype=float)
    return accepted[:, 0], accepted[:, 1]


def irregular_orography(x_values, y_values, kind="multi_peak"):
    """Evaluate one smooth synthetic terrain field."""

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if kind == "multi_peak":
        heights = (
            980.0 * np.exp(-((x_values - 4.9) ** 2 / 3.8 + (y_values - 5.2) ** 2 / 1.3))
            + 430.0 * np.exp(-((x_values - 2.4) ** 2 / 0.9 + (y_values - 7.2) ** 2 / 1.8))
            + 310.0 * np.exp(-((x_values - 7.7) ** 2 / 1.2 + (y_values - 2.7) ** 2 / 0.8))
            + 170.0 * np.sin(2.8 * x_values + 0.7 * np.sin(1.2 * y_values)) * np.cos(1.4 * y_values)
        )
    elif kind == "ridge":
        ridge_envelope = np.exp(-((y_values - (4.9 + 0.9 * np.sin(0.75 * x_values))) ** 2) / 0.55)
        heights = (
            760.0 * ridge_envelope
            + 260.0 * np.sin(2.4 * x_values + 0.8 * y_values)
            + 180.0 * np.cos(3.3 * y_values - 0.5 * x_values)
            - 250.0 * np.exp(-((x_values - 6.4) ** 2 / 1.5 + (y_values - 6.1) ** 2 / 0.6))
        )
    elif kind == "basin":
        heights = (
            620.0 * np.exp(-((x_values - 2.8) ** 2 / 1.0 + (y_values - 3.1) ** 2 / 1.7))
            + 520.0 * np.exp(-((x_values - 7.2) ** 2 / 1.3 + (y_values - 7.4) ** 2 / 1.0))
            - 370.0 * np.exp(-((x_values - 5.1) ** 2 / 4.8 + (y_values - 5.1) ** 2 / 2.6))
            + 145.0 * np.sin(3.4 * x_values) * np.sin(2.1 * y_values + 0.6 * np.cos(x_values))
        )
    else:
        raise ValueError("kind must be 'multi_peak', 'ridge', or 'basin'")
    roughness = 45.0 * np.sin(7.2 * x_values + 1.8 * np.cos(y_values)) + 35.0 * np.cos(6.5 * y_values - 0.7 * np.sin(1.5 * x_values))
    heights = heights + roughness
    return heights - np.min(heights)


def analytic_mode_field(points, domain_length, m_mode, n_mode, amplitude=1.0, phase=0.0):
    """Evaluate a real monochromatic field on points in a square domain."""

    points = np.asarray(points, dtype=float)
    theta = 2.0 * np.pi * (m_mode * points[:, 0] / domain_length + n_mode * points[:, 1] / domain_length) + phase
    return amplitude * np.cos(theta)


def demo_ems_spectra(j_max=20):
    """Return the six analytical spectra used in the EMS proof figure."""

    titles = ["Uniform", "Exponential", "Peak", "Step", "Geometric", "Cosine"]
    labels = [
        r"$E_0$",
        r"$e^{J^\star-i}$",
        r"$1,\epsilon,\epsilon,\ldots$",
        r"$E_1,\ldots,E_2,\ldots$",
        r"$r^{i-1}$",
        r"$\cos\,\frac{\pi i}{2J^\star}$",
    ]
    spectra = []
    for title, label in zip(titles, labels):
        e_values = []
        for i in range(j_max):
            if title == "Uniform":
                e_value = 0.5
            elif title == "Exponential":
                e_value = np.exp(j_max - i) / np.exp(j_max)
            elif title == "Peak":
                e_value = 1.0 if i < 1 else 0.05
            elif title == "Step":
                e_value = 0.7 if i < 5 else 0.35
            elif title == "Geometric":
                e_value = 0.9 * 0.85**i
            elif title == "Cosine":
                e_value = np.cos(np.pi * i / (j_max * 2.0)) * 0.9
            e_values.append(e_value)
        spectra.append({"title": title, "label": label, "e_values": np.asarray(e_values, dtype=float)})
    return spectra
