# Module_Orography.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the shared DEM-point generation and synthetic orography functions.

import numpy as np


# Evaluate the sinusoidal coastline x=x_coast(y) used in the DEM sampling density.
def coast_x_global(y_value, domain_length_x, domain_length_y):
    return 0.55 * domain_length_x + 0.05 * domain_length_x * np.sin(2.0 * np.pi * y_value / domain_length_y)


# Evaluate the ridge-biased DEM density used by the monochromatic test.
def mono_ridge_density(x_values, y_values, domain_length_x, domain_length_y):
    ridge_1 = 0.30 * domain_length_x + 0.14 * domain_length_x * np.sin(2.0 * np.pi * y_values / domain_length_y + 0.7)
    ridge_2 = 0.72 * domain_length_x + 0.10 * domain_length_x * np.cos(2.0 * np.pi * y_values / domain_length_y - 0.4)
    ridge_dist_1 = np.abs(x_values - ridge_1) / (0.10 * domain_length_x)
    ridge_dist_2 = np.abs(x_values - ridge_2) / (0.12 * domain_length_x)
    waviness = 0.55 + 0.25 * np.sin(2.0 * np.pi * x_values / domain_length_x + 0.3) + 0.20 * np.cos(4.0 * np.pi * y_values / domain_length_y - 0.6) + 0.15 * np.sin(2.0 * np.pi * (x_values / domain_length_x + y_values / domain_length_y))
    waviness = np.clip(waviness, 0.05, None)
    return 0.20 + 0.90 * np.exp(-ridge_dist_1 ** 2) + 0.75 * np.exp(-ridge_dist_2 ** 2) + waviness


# Draw scattered DEM points from the prescribed sampling density.
def generate_dem_points(sample_count, domain_length_x, domain_length_y, distribution, random_seed=None):
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    if distribution == "uniform":
        x_values = np.random.uniform(0.0, domain_length_x, sample_count) if rng is None else rng.uniform(0.0, domain_length_x, sample_count)
        y_values = np.random.uniform(0.0, domain_length_y, sample_count) if rng is None else rng.uniform(0.0, domain_length_y, sample_count)
        return x_values, y_values
    if distribution == "mono_ridge":
        rng = np.random.default_rng() if rng is None else rng
        accepted_points = []
        total_trials = 0
        max_trials = max(200000, 70 * sample_count)
        while len(accepted_points) < sample_count and total_trials < max_trials:
            batch_size = min(8000, sample_count - len(accepted_points) + 3000)
            if domain_length_x == domain_length_y:
                candidates = rng.uniform(0.0, domain_length_x, size=(batch_size, 2))
            else:
                candidates = rng.uniform(0.0, 1.0, size=(batch_size, 2)) * np.array([domain_length_x, domain_length_y])
            probability = mono_ridge_density(candidates[:, 0], candidates[:, 1], domain_length_x, domain_length_y)
            probability /= np.max(probability) + 1e-12
            accepted_points.extend(candidates[rng.random(batch_size) < probability].tolist())
            total_trials += batch_size
        if len(accepted_points) < sample_count:
            raise RuntimeError(f"Could not generate {sample_count} DEM points after {total_trials} trials")
        accepted_points = np.asarray(accepted_points[:sample_count], dtype=float)
        return accepted_points[:, 0], accepted_points[:, 1]
    if distribution == "mild":
        sea_density, coast_boost = 0.7, 0.4
    elif distribution == "moderate":
        sea_density, coast_boost = 0.35, 0.9
    elif distribution == "strong":
        sea_density, coast_boost = 0.12, 1.4
    else:
        raise ValueError(f"Unknown DEM distribution: {distribution}")
    accepted_points = []
    total_trials = 0
    max_trials = max(200000, 50 * sample_count)
    while len(accepted_points) < sample_count and total_trials < max_trials:
        batch_size = min(5000, sample_count - len(accepted_points) + 2000)
        x_candidates = np.random.uniform(0.0, domain_length_x, batch_size) if rng is None else rng.uniform(0.0, domain_length_x, batch_size)
        y_candidates = np.random.uniform(0.0, domain_length_y, batch_size) if rng is None else rng.uniform(0.0, domain_length_y, batch_size)
        coast_x = coast_x_global(y_candidates, domain_length_x, domain_length_y)
        land_mask = x_candidates >= coast_x
        acceptance_probability = np.where(land_mask, 1.0, sea_density)
        coast_distance = np.abs(x_candidates - coast_x) / (0.08 * domain_length_x)
        acceptance_probability *= 1.0 + coast_boost * np.exp(-(coast_distance ** 2))
        acceptance_probability /= np.max(acceptance_probability) + 1e-12
        random_values = np.random.rand(batch_size) if rng is None else rng.random(batch_size)
        accept_mask = random_values < acceptance_probability
        for x_value, y_value in zip(x_candidates[accept_mask], y_candidates[accept_mask]):
            accepted_points.append((x_value, y_value))
            if len(accepted_points) >= sample_count:
                break
        total_trials += batch_size
    if len(accepted_points) < sample_count:
        raise RuntimeError(f"Could not generate {sample_count} DEM points after {total_trials} trials")
    accepted_points = np.asarray(accepted_points[:sample_count], dtype=float)
    return accepted_points[:, 0], accepted_points[:, 1]


# Evaluate the chosen synthetic orography h(x,y) on arbitrary coordinates.
def irregular_orography(x_values, y_values, kind):
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
        raise ValueError(f"Unknown terrain kind: {kind}")
    roughness = 45.0 * np.sin(7.2 * x_values + 1.8 * np.cos(y_values)) + 35.0 * np.cos(6.5 * y_values - 0.7 * np.sin(1.5 * x_values))
    heights = heights + roughness
    return heights - np.min(heights)
