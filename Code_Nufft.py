# Code_Nufft.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script computes the NUFFT-versus-DFT kernel comparison and writes the scattered terrain samples, Fourier coefficients, and error summaries to CSV.

import numpy as np
from pathlib import Path

from Module_Csv import write_modes_csv, write_summary_csv, write_terrain_csv
from Module_Nufft import build_auxiliary_grid, compute_direct_dft_coefficients, compute_nufft_for_kernels
from Module_Orography import generate_dem_points, irregular_orography


csv_dir = Path("./csv")
terrain_csv = csv_dir / "Banerjee_2026_Enufft_Nufft_Terrain.csv"
modes_csv = csv_dir / "Banerjee_2026_Enufft_Nufft_Modes.csv"
summary_csv = csv_dir / "Banerjee_2026_Enufft_Nufft_Summary.csv"
random_seed = 42
lx = 10.0
ly = 10.0
mode_limit = 20
q_dem = 2000
dem_distribution = "moderate"
sigma = 1.5
kernel_half_width = 4
kernel_beta = 2.34
terrain_specs = [("Multi-peak", "multi_peak"), ("Meandering ridge", "ridge"), ("Basin and peaks", "basin")]
m_values = np.arange(-mode_limit, mode_limit + 1)
n_values = np.arange(-mode_limit, mode_limit + 1)
kernel_types = ("optimized", "baseline")


# Compute the fixed DEM realization, the terrain heights, and the two kernel comparisons.
def run_case_studies():
    np.random.seed(random_seed)
    x_dem, y_dem = generate_dem_points(q_dem, lx, ly, dem_distribution)
    case_results = []
    for title, kind in terrain_specs:
        h_dem = irregular_orography(x_dem, y_dem, kind)
        dft_coefficients = compute_direct_dft_coefficients(x_dem, y_dem, h_dem, m_values, n_values, lx, ly)
        nufft_coefficients = compute_nufft_for_kernels(
            x_dem, y_dem, h_dem, m_values, n_values, lx, ly, sigma,
            kernel_half_width, kernel_beta, kernel_types,
        )
        optimized_error = nufft_coefficients["optimized"] - dft_coefficients
        baseline_error = nufft_coefficients["baseline"] - dft_coefficients
        case_results.append({
            "title": title,
            "kind": kind,
            "h_dem": h_dem,
            "h_dft": dft_coefficients,
            "h_nufft_optimized": nufft_coefficients["optimized"],
            "h_nufft_baseline": nufft_coefficients["baseline"],
            "error_optimized": optimized_error,
            "error_baseline": baseline_error,
        })
    all_errors_optimized = np.concatenate([np.abs(result["error_optimized"]).ravel() for result in case_results])
    all_errors_baseline = np.concatenate([np.abs(result["error_baseline"]).ravel() for result in case_results])
    return {
        "x_dem": x_dem,
        "y_dem": y_dem,
        "case_results": case_results,
        "all_errors_optimized": all_errors_optimized,
        "all_errors_baseline": all_errors_baseline,
    }


# Compute the full comparison and write every plot-ready table to disk.
def main():
    nx_aux, ny_aux, _, _, _, _ = build_auxiliary_grid(mode_limit, mode_limit, lx, ly, sigma)
    case_data = run_case_studies()
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_terrain_csv(case_data, terrain_csv)
    write_modes_csv(case_data, modes_csv, m_values, n_values)
    write_summary_csv(case_data, summary_csv)
    optimized_median = np.median(case_data["all_errors_optimized"])
    baseline_median = np.median(case_data["all_errors_baseline"])
    optimized_mean = np.mean(case_data["all_errors_optimized"])
    baseline_mean = np.mean(case_data["all_errors_baseline"])
    print("NUFFT kernel comparison")
    print(f"Q={q_dem}, density={dem_distribution}, auxiliary grid={nx_aux}x{ny_aux}, half-width={kernel_half_width}")
    print("\nError summary")
    print(f"optimized median |error| = {optimized_median:.3e}")
    print(f"baseline  median |error| = {baseline_median:.3e}")
    print(f"optimized mean   |error| = {optimized_mean:.3e}")
    print(f"baseline  mean   |error| = {baseline_mean:.3e}")
    print(f"\nWrote {terrain_csv}")
    print(f"Wrote {modes_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
