# Code_Ems.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script computes the EMS theory spectra and writes the sorted energies, cumulative diagnostics, and scalar summaries to CSV.

from pathlib import Path

import numpy as np

from Module_Csv import write_ems_modes_csv, write_ems_spectra_csv, write_ems_summary_csv
from Module_Ems import alpha_max, alpha_min, delta, elastic_mode_selection, k_max, k_min, w1, w2


csv_dir = Path("./csv")
spectra_csv = csv_dir / "Banerjee_2026_Enufft_Ems_Spectra.csv"
modes_csv = csv_dir / "Banerjee_2026_Enufft_Ems_Modes.csv"
summary_csv = csv_dir / "Banerjee_2026_Enufft_Ems_Summary.csv"
j_max = 20
spectrum_titles = ["Uniform", "Exponential", "Peak", "Step", "Geometric", "Cosine"]
spectrum_math_labels = [
    r"$E_0$",
    r"$e^{J^{\star}-i}$",
    r"$1, \varepsilon, \varepsilon, \ldots$",
    r"$E_1, \ldots, E_2, \ldots$",
    r"$r^{i-1}$",
    r"$\cos\,\frac{\pi i}{2J^{\star}}$",
]


# Build the six analytical spectra used only by this EMS theory case.
def build_demo_spectra():
    spectra = []
    for title, label in zip(spectrum_titles, spectrum_math_labels):
        e_values = []
        for i in range(j_max):
            if title == "Uniform":
                e_value = 0.5
            elif title == "Exponential":
                e_value = np.exp(j_max - i) / np.exp(j_max)
            elif title == "Peak":
                e_value = 1 if i < 1 else 0.05
            elif title == "Step":
                e_value = 0.7 if i < 5 else 0.35
            elif title == "Geometric":
                e_value = 0.9 * 0.85 ** i
            elif title == "Cosine":
                e_value = np.cos(np.pi * i / (j_max * 2.0)) * 0.9
            e_values.append(e_value)
        spectra.append({"title": title, "label": label, "e_values": np.asarray(e_values, dtype=float)})
    return spectra


# Compute the six EMS theory examples with the shared selection pipeline.
def run_case():
    case_results = []
    for spectrum in build_demo_spectra():
        ems = elastic_mode_selection(spectrum["e_values"], k_min, k_max, alpha_min, alpha_max, delta, w1, w2)
        case_results.append({"title": spectrum["title"], "label": spectrum["label"], "e_values": spectrum["e_values"], "ems": ems})
    return case_results


# Write the EMS tables and print the retained-mode summary.
def main():
    case_results = run_case()
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_ems_spectra_csv(case_results, spectra_csv)
    write_ems_modes_csv(case_results, modes_csv)
    write_ems_summary_csv(case_results, summary_csv, delta, w1, w2, alpha_min, alpha_max, k_min, k_max)
    print("EMS theory comparison")
    print(f"j_max={j_max}, k_range={k_min}:{k_max}, delta={delta:g}, weights=({w1:g}, {w2:g}), alpha_range={alpha_min:g}:{alpha_max:g}")
    for result in case_results:
        ems = result["ems"]
        print(f"{result['title']:>11}: K*={ems['k_star']}, alpha_C={ems['alpha_c']:.6f}, alpha_final={ems['alpha_c_final']:.6f}, N_eff={ems['n_eff']:.6f}, S_delta={ems['s_delta']:.6f}")
    print(f"\nWrote {spectra_csv}")
    print(f"Wrote {modes_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
