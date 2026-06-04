# Code_Mono.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script computes the monochromatic ENUFFT and CSA sweep and writes details, sparse modes, and summaries to CSV.

import itertools
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from Module_Csa import compute_csa_spectrum, count_signed_nonzero_modes, count_unique_mode_pairs
from Module_Csv import write_dict_rows_csv
from Module_Ems import select_sparse_conjugate_modes
from Module_Helpers import points_in_triangle_mask, polygon_box_intersection_area, triangle_area_2d
from Module_Nufft import compute_nufft_coefficients
from Module_Orography import generate_dem_points


csv_dir = Path("./csv")
details_csv = csv_dir / "Banerjee_2026_Enufft_Mono_Details.csv"
modes_csv = csv_dir / "Banerjee_2026_Enufft_Mono_Modes.csv"
summary_csv = csv_dir / "Banerjee_2026_Enufft_Mono_Summary.csv"
domain_length_km = 30.0
amplitude_m = 1000.0
kernel_half_width = 4
kernel_beta = 2.34
mode_min = 1
random_phase = True
signal_master_seed = 12345
dem_seed = 42
dem_distribution = "mono_ridge"
ems_k_min = 1
ems_alpha_min = 0.0
ems_alpha_max = 0.7
ems_delta = 0.02
ems_w1 = 0.5
ems_w2 = 0.5
csa_lambda_fa = 1e-1
csa_lambda_sa = 1e-6
circle_radius_ratio = 0.5
clip_triangle_to_domain = True
physical_grid = {
    "mode_limit": [4, 6, 8, 10],
    "sample_count": [500, 1000, 2000],
    "expansion_ratio": [1.0, 1.5, 2.0],
    "triangle_orientation": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
    "center_offset": [-0.211, -0.077, 0.0],
    "uniformity": [0.0, 0.5, 1.0],
}
algorithm_grid = {
    "mask_condition": ["triangle", "circle", "square"],
    "weight_type": ["uniform", "voronoi"],
    "oversample": [1.25, 1.5, 2.0],
}


# Merge the fixed defaults with one sweep update.
def build_case(updates=None):
    case = {
        "domain_length_km": domain_length_km,
        "mode_limit": 8,
        "oversample": 1.5,
        "kernel_half_width": kernel_half_width,
        "kernel_beta": kernel_beta,
        "sample_count": 3500,
        "amplitude_m": amplitude_m,
        "expansion_ratio": 2.0,
        "triangle_orientation": 0.0,
        "center_offset": 0.0,
        "uniformity": 0.75,
        "mask_condition": "circle",
        "weight_type": "uniform",
        "mode_min": mode_min,
        "mode_max": None,
        "random_phase": random_phase,
        "signal_master_seed": signal_master_seed,
        "dem_seed": dem_seed,
        "dem_distribution": dem_distribution,
        "ems_k_min": ems_k_min,
        "ems_k_max": None,
        "ems_alpha_min": ems_alpha_min,
        "ems_alpha_max": ems_alpha_max,
        "ems_delta": ems_delta,
        "ems_w1": ems_w1,
        "ems_w2": ems_w2,
        "csa_lambda_fa": csa_lambda_fa,
        "csa_lambda_sa": csa_lambda_sa,
        "csa_sparse_modes": None,
        "clip_triangle_to_domain": clip_triangle_to_domain,
    }
    if updates is not None:
        case.update(updates)
    return case


# Build one target triangle relative to the square domain.
def build_main_triangle(case):
    length = case["domain_length_km"]
    center = np.array([0.5 * length, 0.5 * length + case["center_offset"] * length])
    longest_edge = length / max(case["expansion_ratio"], 1e-12)
    uniformity_value = float(np.clip(case["uniformity"], 0.0, 1.0))
    height_equilateral = np.sqrt(3.0) * longest_edge / 2.0
    height = height_equilateral * (0.45 + 0.55 * uniformity_value)
    skew = (1.0 - uniformity_value) * 0.42 * longest_edge
    local = np.array([[skew, 2.0 * height / 3.0], [-0.5 * longest_edge, -height / 3.0], [0.5 * longest_edge, -height / 3.0]], dtype=float)
    angle = np.deg2rad(case["triangle_orientation"])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + center


# Assign triangle and angular outer-region labels.
def classify_regions(points, triangle_vertices, center):
    inside = points_in_triangle_mask(points, triangle_vertices)
    vertex_angles = np.mod(np.arctan2(triangle_vertices[:, 1] - center[1], triangle_vertices[:, 0] - center[0]), 2.0 * np.pi)
    cuts = np.sort(vertex_angles)
    theta = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2.0 * np.pi)
    theta_shifted = np.mod(theta - cuts[0], 2.0 * np.pi)
    cuts_shifted = np.mod(cuts - cuts[0], 2.0 * np.pi)
    region = np.ones(points.shape[0], dtype=int)
    region[theta_shifted >= cuts_shifted[1]] = 2
    region[theta_shifted >= cuts_shifted[2]] = 3
    region[inside] = 0
    return region.astype(int)


# Assign one fixed monochromatic mode pair to each region.
def assign_region_modes(case):
    rng = np.random.default_rng(case["signal_master_seed"])
    region_m = np.zeros(4, dtype=int)
    region_n = np.zeros(4, dtype=int)
    region_phase = np.zeros(4, dtype=float)
    max_mode = case["mode_limit"] - 1 if case["mode_max"] is None else int(case["mode_max"])
    min_mode = max(1, int(case["mode_min"]))
    max_mode = max(min_mode, min(max_mode, case["mode_limit"] - 1 if case["mode_limit"] > 1 else 1))
    for region_index in range(4):
        m_abs = int(rng.integers(min_mode, max_mode + 1))
        n_abs = int(rng.integers(min_mode, max_mode + 1))
        region_m[region_index] = int(rng.choice([-1, 1]) * m_abs)
        region_n[region_index] = int(rng.choice([-1, 1]) * n_abs)
        region_phase[region_index] = rng.uniform(0.0, 2.0 * np.pi) if case["random_phase"] else 0.0
    return region_m, region_n, region_phase


# Evaluate the four-region monochromatic height field on the DEM points.
def build_piecewise_dem(case, points, region_id, region_m, region_n, region_phase):
    local_region = region_id.astype(int)
    phase = 2.0 * np.pi * region_m[local_region] * points[:, 0] / case["domain_length_km"] + 2.0 * np.pi * region_n[local_region] * points[:, 1] / case["domain_length_km"] + region_phase[local_region]
    return case["amplitude_m"] * np.cos(phase)


# Build the fixed mesh and field data for one physical setup.
def build_mesh_data(case):
    triangle_vertices = build_main_triangle(case)
    center = np.array([0.5 * case["domain_length_km"], 0.5 * case["domain_length_km"] + case["center_offset"] * case["domain_length_km"]])
    x_values, y_values = generate_dem_points(case["sample_count"], case["domain_length_km"], case["domain_length_km"], case["dem_distribution"], random_seed=case["dem_seed"])
    dem_points = np.column_stack([x_values, y_values])
    region_id = classify_regions(dem_points, triangle_vertices, center)
    region_m, region_n, region_phase = assign_region_modes(case)
    h_values = build_piecewise_dem(case, dem_points, region_id, region_m, region_n, region_phase)
    return {"dem_points": dem_points, "h_dem": h_values, "region_id": region_id, "region_m": region_m, "region_n": region_n, "region_phase": region_phase, "triangle_vertices": triangle_vertices, "center": center}


# Return the active ENUFFT support mask and its geometric metadata.
def get_analysis_mask(case, mesh_data):
    points = mesh_data["dem_points"]
    triangle_vertices = mesh_data["triangle_vertices"]
    length = case["domain_length_km"]
    clip_triangle = bool(case.get("clip_triangle_to_domain", True))
    domain_mask = (points[:, 0] >= 0.0) & (points[:, 0] <= length) & (points[:, 1] >= 0.0) & (points[:, 1] <= length)
    if case["mask_condition"] in {"triangle", "triangle_only"}:
        triangle_mask = points_in_triangle_mask(points, triangle_vertices)
        if clip_triangle:
            triangle_mask = triangle_mask & domain_mask
            triangle_area_km2 = polygon_box_intersection_area(triangle_vertices, 0.0, length, 0.0, length)
        else:
            triangle_area_km2 = triangle_area_2d(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2])
        return triangle_mask, {"type": "triangle", "area_km2": triangle_area_km2, "vertices": triangle_vertices, "clip_triangle_to_domain": clip_triangle}
    if case["mask_condition"] == "square":
        return domain_mask, {"type": "square", "area_km2": length * length}
    if case["mask_condition"] == "circle":
        center = np.array([0.5 * length, 0.5 * length], dtype=float)
        radius = circle_radius_ratio * length
        distance = np.linalg.norm(points - center, axis=1)
        return (distance <= radius) & domain_mask, {"type": "circle", "area_km2": np.pi * radius ** 2, "center": center, "radius": radius}
    raise ValueError(f"Unknown mask condition {case['mask_condition']}")


# Approximate sample areas by nearest-neighbour ownership on a support grid.
def compute_voronoi_weights(x_values, y_values, length_km, mask_info, grid_res=96):
    point_count = len(x_values)
    support_area_km2 = float(mask_info.get("area_km2", length_km * length_km)) if mask_info is not None else length_km * length_km
    mask_type = mask_info.get("type") if mask_info is not None else None
    use_legacy_triangle_fallback = mask_type == "triangle" and not mask_info.get("clip_triangle_to_domain", True)
    if point_count < 4:
        if use_legacy_triangle_fallback:
            return np.ones(point_count, dtype=float) * (length_km * length_km / max(point_count, 1))
        return np.ones(point_count, dtype=float) * (max(support_area_km2, 1e-12 * length_km * length_km) / max(point_count, 1))
    grid_x_km = np.linspace(0.0, length_km, grid_res, endpoint=False) + length_km / (2 * grid_res)
    grid_y_km = np.linspace(0.0, length_km, grid_res, endpoint=False) + length_km / (2 * grid_res)
    gx, gy = np.meshgrid(grid_x_km, grid_y_km, indexing="xy")
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    if mask_type == "triangle":
        grid_points = grid_points[points_in_triangle_mask(grid_points, np.asarray(mask_info["vertices"]))]
    elif mask_type == "circle":
        center = np.asarray(mask_info["center"])
        radius = float(mask_info["radius"])
        grid_points = grid_points[np.linalg.norm(grid_points - center, axis=1) <= radius]
    if len(grid_points) == 0:
        if use_legacy_triangle_fallback:
            grid_points = np.column_stack([gx.ravel(), gy.ravel()])
            support_area_km2 = length_km * length_km
        else:
            return np.ones(point_count, dtype=float) * (max(support_area_km2, 1e-12 * length_km * length_km) / point_count)
    _, owner = cKDTree(np.column_stack([x_values, y_values])).query(grid_points, k=1)
    counts = np.bincount(owner, minlength=point_count).astype(float)
    weights = counts * (support_area_km2 / len(grid_points))
    weights = np.maximum(weights, 1e-12 * support_area_km2 / point_count)
    return weights * (support_area_km2 / np.sum(weights))


# Compare the dominant recovered mode with the hidden triangle signal.
def compute_peak_metrics(spectrum, case, mesh_data, truth_region_ids):
    m_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    n_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    signal_pairs = []
    for region_id in np.asarray(truth_region_ids, dtype=int):
        m_true = int(mesh_data["region_m"][region_id])
        n_true = int(mesh_data["region_n"][region_id])
        if m_true < 0 or (m_true == 0 and n_true < 0):
            pair = (-m_true, -n_true)
        else:
            pair = (m_true, n_true)
        if pair not in signal_pairs:
            signal_pairs.append(pair)
    signal_pairs = sorted(signal_pairs)
    pair_amplitudes = {}
    m_lookup = {int(value): index for index, value in enumerate(m_values)}
    n_lookup = {int(value): index for index, value in enumerate(n_values)}
    for m_mode in m_values:
        for n_mode in n_values:
            if m_mode == 0 and n_mode == 0:
                continue
            pair_m, pair_n = int(m_mode), int(n_mode)
            if pair_m < 0 or (pair_m == 0 and pair_n < 0):
                pair_m, pair_n = -pair_m, -pair_n
            pair = (pair_m, pair_n)
            if pair in pair_amplitudes:
                continue
            h_pos = spectrum[m_lookup[pair_m], n_lookup[pair_n]]
            h_neg = spectrum[m_lookup[-pair_m], n_lookup[-pair_n]]
            pair_amplitudes[pair] = float(np.sqrt((np.abs(h_pos) ** 2 + np.abs(h_neg) ** 2) / 2.0))
    if not signal_pairs or not pair_amplitudes:
        return {"max_mode_direction_deviation_deg": 0.0, "max_peak_amplitude_deviation": 0.0, "max_peak_amplitude_deviation_pct": 0.0, "max_spurious_amplitude": 0.0}
    peak_pair, peak_amplitude = max(pair_amplitudes.items(), key=lambda item: item[1])
    signal_amplitudes = {pair: pair_amplitudes.get(pair, 0.0) for pair in signal_pairs}
    expected_signal_pair = signal_pairs[0]
    spurious_amplitudes = [amp for pair, amp in pair_amplitudes.items() if pair not in signal_amplitudes]
    physical_amplitude = float(0.5 * case["amplitude_m"])
    amplitude_deviation = float(abs(peak_amplitude - physical_amplitude))
    amplitude_deviation_pct = float(100.0 * amplitude_deviation / max(physical_amplitude, 1e-12))
    dominant_angle = float(np.degrees(np.arctan2(float(expected_signal_pair[1]), float(expected_signal_pair[0]))) % 180.0)
    peak_angle = float(np.degrees(np.arctan2(float(peak_pair[1]), float(peak_pair[0]))) % 180.0)
    direction_error = abs(dominant_angle - peak_angle) % 180.0
    direction_error = float(min(direction_error, 180.0 - direction_error))
    return {"max_mode_direction_deviation_deg": direction_error, "max_peak_amplitude_deviation": amplitude_deviation, "max_peak_amplitude_deviation_pct": amplitude_deviation_pct, "max_spurious_amplitude": float(max(spurious_amplitudes)) if spurious_amplitudes else 0.0}


# Build the CSA result once for one physical setup.
def compute_csa_reference(case, mesh_data):
    m_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    n_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    triangle_case = dict(case)
    triangle_case["mask_condition"] = "triangle"
    triangle_mask, _ = get_analysis_mask(triangle_case, mesh_data)
    full_mask = np.ones(len(mesh_data["dem_points"]), dtype=bool)
    csa = compute_csa_spectrum(mesh_data["dem_points"][full_mask, 0], mesh_data["dem_points"][full_mask, 1], mesh_data["h_dem"][full_mask], mesh_data["dem_points"][triangle_mask, 0], mesh_data["dem_points"][triangle_mask, 1], mesh_data["h_dem"][triangle_mask], m_values, n_values, case["domain_length_km"], case["domain_length_km"], case["csa_lambda_fa"], case["csa_lambda_sa"], case["csa_sparse_modes"] if case["csa_sparse_modes"] is not None else 2 * case["mode_limit"])
    csa_diag = compute_peak_metrics(csa["spectrum"], case, mesh_data, np.array([0], dtype=int))
    return {"fa_mask": full_mask, "sa_mask": triangle_mask, "triangle_mask": triangle_mask, "spectrum": csa["spectrum"], "selected_modes": csa["selected_modes"], "n_signed_modes_used": count_signed_nonzero_modes(csa["spectrum"], m_values, n_values), "n_pairs_used": count_unique_mode_pairs(csa["selected_modes"]), "diag": csa_diag}


# Store one sparse spectrum in row form.
def build_modes_rows(case, spectrum, method):
    rows = []
    m_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    n_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    amplitude_values = np.abs(spectrum)
    keep = amplitude_values > 1e-15
    keep[case["mode_limit"], case["mode_limit"]] = True
    for m_index, m_mode in enumerate(m_values):
        for n_index, n_mode in enumerate(n_values):
            if not keep[m_index, n_index]:
                continue
            value = spectrum[m_index, n_index]
            rows.append({"method": method, "mode_limit": case["mode_limit"], "sample_count": case["sample_count"], "mask": case["mask_condition"], "weight": case["weight_type"], "expansion_ratio": case["expansion_ratio"], "oversample": case["oversample"], "triangle_orientation": case["triangle_orientation"], "center_offset": case["center_offset"], "uniformity": case["uniformity"], "m_mode": int(m_mode), "n_mode": int(n_mode), "coeff_real": float(np.real(value)), "coeff_imag": float(np.imag(value)), "coeff_abs": float(np.abs(value)), "is_dc": bool(m_mode == 0 and n_mode == 0)})
    rows.sort(key=lambda row: -row["coeff_abs"])
    for rank, row in enumerate(rows, start=1):
        row["rank_abs"] = rank
    return rows


# Run one ENUFFT mask choice against one fixed physical setup.
def run_single_configuration(case, mesh_data, csa_reference, include_csa_modes):
    mask, mask_info = get_analysis_mask(case, mesh_data)
    if int(np.sum(mask)) < 3:
        raise RuntimeError("Analysis mask contains fewer than three DEM samples")
    x_values = mesh_data["dem_points"][mask, 0]
    y_values = mesh_data["dem_points"][mask, 1]
    h_values = mesh_data["h_dem"][mask]
    weights = compute_voronoi_weights(x_values, y_values, case["domain_length_km"], mask_info) if case["weight_type"] == "voronoi" else None
    m_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    n_values = np.arange(-case["mode_limit"], case["mode_limit"] + 1)
    raw_spectrum = compute_nufft_coefficients(x_values, y_values, h_values, m_values, n_values, case["domain_length_km"], case["domain_length_km"], case["oversample"], case["kernel_half_width"], case["kernel_beta"], "optimized", weights)
    enufft = select_sparse_conjugate_modes(raw_spectrum, m_values, n_values, case["ems_k_min"], case["ems_k_max"] if case["ems_k_max"] is not None else case["mode_limit"], case["ems_alpha_min"], case["ems_alpha_max"], case["ems_delta"], case["ems_w1"], case["ems_w2"])
    enufft_diag = compute_peak_metrics(enufft["spectrum"], case, mesh_data, np.array([0], dtype=int))
    csa_diag = csa_reference["diag"]
    triangle_case = dict(case)
    triangle_case["mask_condition"] = "triangle"
    _, triangle_info = get_analysis_mask(triangle_case, mesh_data)
    triangle_area_km2 = float(triangle_info["area_km2"])
    mask_area_km2 = float(mask_info["area_km2"])
    full_region_counts = np.bincount(mesh_data["region_id"], minlength=4)
    row = {"mode_limit": case["mode_limit"], "mask": case["mask_condition"], "weight": case["weight_type"], "expansion_ratio": case["expansion_ratio"], "oversample": case["oversample"], "triangle_id": 0, "triangle_orientation": case["triangle_orientation"], "center_offset": case["center_offset"], "uniformity": case["uniformity"], "sample_count": case["sample_count"], "n_mask_points": int(np.sum(mask)), "n_dem_effective_mask": int(np.sum(mask)), "n_csa_fa_points": int(np.sum(csa_reference["fa_mask"])), "n_csa_sa_points": int(np.sum(csa_reference["sa_mask"])), "n_triangle_points": int(np.sum(csa_reference["triangle_mask"])), "n_points_in_triangle": int(np.sum(csa_reference["triangle_mask"])), "n_region_triangle": int(full_region_counts[0]), "n_region_outer_1": int(full_region_counts[1]), "n_region_outer_2": int(full_region_counts[2]), "n_region_outer_3": int(full_region_counts[3]), "triangle_area": triangle_area_km2, "mask_area": mask_area_km2, "k_star_enufft": enufft["k_star"], "k_max_enufft": enufft["k_max"], "n_eff_enufft": enufft["n_eff"], "s_delta_enufft": enufft["s_delta"], "alpha_enufft": enufft["alpha_c"], "max_mode_direction_deviation_deg_enufft": enufft_diag["max_mode_direction_deviation_deg"], "max_peak_amplitude_deviation_enufft": enufft_diag["max_peak_amplitude_deviation"], "max_peak_amplitude_deviation_pct_enufft": enufft_diag["max_peak_amplitude_deviation_pct"], "max_spurious_amplitude_enufft": enufft_diag["max_spurious_amplitude"], "max_mode_direction_deviation_deg_csa": csa_diag["max_mode_direction_deviation_deg"], "max_peak_amplitude_deviation_csa": csa_diag["max_peak_amplitude_deviation"], "max_peak_amplitude_deviation_pct_csa": csa_diag["max_peak_amplitude_deviation_pct"], "max_spurious_amplitude_csa": csa_diag["max_spurious_amplitude"], "enufft_signed_modes_used": count_signed_nonzero_modes(enufft["spectrum"], m_values, n_values), "csa_signed_modes_used": int(csa_reference["n_signed_modes_used"]), "csa_pairs_used": int(csa_reference["n_pairs_used"]), "csa_signed_modes_selected": int(len(csa_reference["selected_modes"]))}
    for region_id, region_name in enumerate(["triangle", "outer_1", "outer_2", "outer_3"]):
        row[f"mode_{region_name}_m"] = int(mesh_data["region_m"][region_id])
        row[f"mode_{region_name}_n"] = int(mesh_data["region_n"][region_id])
        row[f"mode_{region_name}_phase"] = float(mesh_data["region_phase"][region_id])
    modes_rows = build_modes_rows(case, enufft["spectrum"], "enufft")
    if include_csa_modes:
        csa_case = dict(case)
        csa_case["mask_condition"] = "csa"
        modes_rows.extend(build_modes_rows(csa_case, csa_reference["spectrum"], "csa"))
    return {"summary_row": dict(row), "detail_row": dict(row), "modes_rows": modes_rows}


# Build all physical and algorithmic sweep combinations.
def build_parameter_combinations():
    physical_keys = list(physical_grid.keys())
    algorithm_keys = list(algorithm_grid.keys())
    physical_combinations = list(itertools.product(*[physical_grid[key] for key in physical_keys]))
    algorithm_combinations = list(itertools.product(*[algorithm_grid[key] for key in algorithm_keys]))
    return physical_keys, physical_combinations, algorithm_keys, algorithm_combinations


# Run the full monochromatic sweep.
def run_parameter_sweep():
    physical_keys, physical_combinations, algorithm_keys, algorithm_combinations = build_parameter_combinations()
    summary_rows = []
    detail_rows = []
    modes_rows = []
    for physical_index, physical_values in enumerate(physical_combinations, start=1):
        physical_case = build_case(dict(zip(physical_keys, physical_values)))
        mesh_data = build_mesh_data(physical_case)
        csa_reference = compute_csa_reference(physical_case, mesh_data)
        skipped_masks = 0
        for algorithm_index, algorithm_values in enumerate(algorithm_combinations):
            local_case = build_case({**dict(zip(physical_keys, physical_values)), **dict(zip(algorithm_keys, algorithm_values))})
            try:
                result = run_single_configuration(local_case, mesh_data, csa_reference, algorithm_index == 0)
            except RuntimeError as error:
                if "Analysis mask contains fewer than three DEM samples" in str(error):
                    skipped_masks += 1
                    continue
                raise
            summary_rows.append(result["summary_row"])
            detail_rows.append(result["detail_row"])
            modes_rows.extend(result["modes_rows"])
        if physical_index == 1 or physical_index % 100 == 0 or physical_index == len(physical_combinations):
            print(f"{physical_index:4d}/{len(physical_combinations)} mode_limit={physical_case['mode_limit']} sample_count={physical_case['sample_count']} skipped={skipped_masks}")
    return summary_rows, detail_rows, modes_rows


# Write all Mono tables and print a compact summary.
def main():
    print("Mono ENUFFT and CSA sweep")
    summary_rows, detail_rows, modes_rows = run_parameter_sweep()
    if not summary_rows:
        raise RuntimeError("No valid configurations were produced")
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(detail_rows, details_csv)
    write_dict_rows_csv(modes_rows, modes_csv)
    write_dict_rows_csv(summary_rows, summary_csv)
    enufft_direction = np.median([row["max_mode_direction_deviation_deg_enufft"] for row in summary_rows])
    csa_direction = np.median([row["max_mode_direction_deviation_deg_csa"] for row in summary_rows])
    print(f"valid configurations={len(summary_rows)}")
    print(f"median direction error ENUFFT={enufft_direction:.6g} deg")
    print(f"median direction error CSA={csa_direction:.6g} deg")
    print(f"\nWrote {details_csv}")
    print(f"Wrote {modes_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
