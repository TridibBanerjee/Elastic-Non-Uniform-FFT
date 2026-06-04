# Module_Csv.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the shared CSV-writing functions for ENUFFT cases.

import csv
import numpy as np

from Module_Helpers import format_float


# Write a list of dictionaries with first-seen column order.
def write_dict_rows_csv(rows, output_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Write the scattered terrain heights h_q for each terrain case in one wide table.
def write_terrain_csv(case_data, terrain_csv):
    x_dem = case_data["x_dem"]
    y_dem = case_data["y_dem"]
    height_columns = {
        "multi_peak": case_data["case_results"][0]["h_dem"],
        "ridge": case_data["case_results"][1]["h_dem"],
        "basin": case_data["case_results"][2]["h_dem"],
    }
    with terrain_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "x_km", "y_km", "height_multi_peak_m", "height_ridge_m", "height_basin_m"])
        for sample_index in range(x_dem.size):
            writer.writerow([
                sample_index,
                format_float(x_dem[sample_index]),
                format_float(y_dem[sample_index]),
                format_float(height_columns["multi_peak"][sample_index]),
                format_float(height_columns["ridge"][sample_index]),
                format_float(height_columns["basin"][sample_index]),
            ])


# Write the complex Fourier coefficients and errors for every resolved mode pair.
def write_modes_csv(case_data, modes_csv, m_values, n_values):
    with modes_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "terrain_title", "terrain_kind", "m_mode", "n_mode", "dft_real", "dft_imag",
            "optimized_real", "optimized_imag", "baseline_real", "baseline_imag",
            "optimized_error_real", "optimized_error_imag", "optimized_error_abs",
            "baseline_error_real", "baseline_error_imag", "baseline_error_abs",
        ])
        for result in case_data["case_results"]:
            for m_index, m_mode in enumerate(m_values):
                for n_index, n_mode in enumerate(n_values):
                    dft_value = result["h_dft"][m_index, n_index]
                    optimized_value = result["h_nufft_optimized"][m_index, n_index]
                    baseline_value = result["h_nufft_baseline"][m_index, n_index]
                    optimized_error = result["error_optimized"][m_index, n_index]
                    baseline_error = result["error_baseline"][m_index, n_index]
                    writer.writerow([
                        result["title"],
                        result["kind"],
                        m_mode,
                        n_mode,
                        format_float(dft_value.real),
                        format_float(dft_value.imag),
                        format_float(optimized_value.real),
                        format_float(optimized_value.imag),
                        format_float(baseline_value.real),
                        format_float(baseline_value.imag),
                        format_float(optimized_error.real),
                        format_float(optimized_error.imag),
                        format_float(np.abs(optimized_error)),
                        format_float(baseline_error.real),
                        format_float(baseline_error.imag),
                        format_float(np.abs(baseline_error)),
                    ])


# Write the pooled error statistics shown in the terminal summary.
def write_summary_csv(case_data, summary_csv):
    optimized_errors = case_data["all_errors_optimized"]
    baseline_errors = case_data["all_errors_baseline"]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kernel_type", "median_abs_error", "mean_abs_error"])
        writer.writerow(["optimized", format_float(np.median(optimized_errors)), format_float(np.mean(optimized_errors))])
        writer.writerow(["baseline", format_float(np.median(baseline_errors)), format_float(np.mean(baseline_errors))])


# Write the sorted EMS spectra E(j) for every theory example.
def write_ems_spectra_csv(case_results, spectra_csv):
    with spectra_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spectrum_index", "spectrum_title", "spectrum_label", "j_mode", "energy"])
        for spectrum_index, result in enumerate(case_results):
            for j_mode, energy in enumerate(result["ems"]["e_sorted"], start=1):
                writer.writerow([spectrum_index, result["title"], result["label"], j_mode, format_float(energy)])


# Write the EMS cumulative fractions and gap ratios for every retained j index.
def write_ems_modes_csv(case_results, modes_csv):
    with modes_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spectrum_index", "spectrum_title", "spectrum_label", "j_mode", "cumulative_fraction", "gap_ratio", "in_window", "retained_mode"])
        for spectrum_index, result in enumerate(case_results):
            ems = result["ems"]
            for j_mode, cumulative_fraction in enumerate(ems["f_k"], start=1):
                if j_mode <= ems["g"].size:
                    gap_ratio = ems["g"][j_mode - 1]
                else:
                    gap_ratio = np.nan
                writer.writerow([
                    spectrum_index,
                    result["title"],
                    result["label"],
                    j_mode,
                    format_float(cumulative_fraction),
                    format_float(gap_ratio),
                    1 if j_mode <= ems["j_window"] else 0,
                    1 if j_mode <= ems["k_star"] else 0,
                ])


# Write the EMS scalar diagnostics and control parameters for each theory spectrum.
def write_ems_summary_csv(case_results, summary_csv, delta, w1, w2, alpha_min, alpha_max, k_min, k_max):
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "spectrum_index", "spectrum_title", "spectrum_label", "delta", "w1", "w2", "alpha_min", "alpha_max", "k_min", "k_max",
            "j_star", "j_window", "sum_energy", "n_eff", "n_eff_clip", "n_eff_norm", "s_delta", "c_measure", "alpha_c", "k_star", "alpha_c_final",
        ])
        for spectrum_index, result in enumerate(case_results):
            ems = result["ems"]
            writer.writerow([
                spectrum_index,
                result["title"],
                result["label"],
                format_float(delta),
                format_float(w1),
                format_float(w2),
                format_float(alpha_min),
                format_float(alpha_max),
                k_min,
                k_max,
                ems["j_star"],
                ems["j_window"],
                format_float(ems["sum_e"]),
                format_float(ems["n_eff"]),
                format_float(ems["n_eff_clip"]),
                format_float(ems["n_eff_norm"]),
                format_float(ems["s_delta"]),
                format_float(ems["c"]),
                format_float(ems["alpha_c"]),
                ems["k_star"],
                format_float(ems["alpha_c_final"]),
            ])
