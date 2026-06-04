# Plot_Ems.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script reads the EMS theory CSV tables and rebuilds the six-panel spectrum figure.

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from Module_Plot_Template import mm, apply_james_style, figure_output_path, save_png_and_pdf


csv_dir = Path("./csv")
spectra_csv = csv_dir / "Banerjee_2026_Enufft_Ems_Spectra.csv"
summary_csv = csv_dir / "Banerjee_2026_Enufft_Ems_Summary.csv"
spectrum_color = "#000000"
fill_color = "#959791"
kstar_color = "#c0392b"
neff_color = "#2980b9"
sdelta_color = "#27ae60"


# Read the sorted EMS input spectra and scalar diagnostics from the saved CSV tables.
def read_case_data():
    energies = {}
    with spectra_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            spectrum_index = int(row["spectrum_index"])
            if spectrum_index not in energies:
                energies[spectrum_index] = []
            energies[spectrum_index].append(float(row["energy"]))
    case_results = []
    with summary_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            spectrum_index = int(row["spectrum_index"])
            case_results.append({
                "spectrum_index": spectrum_index,
                "title": row["spectrum_title"],
                "label": row["spectrum_label"],
                "e_sorted": np.asarray(energies[spectrum_index], dtype=float),
                "delta": float(row["delta"]),
                "w1": float(row["w1"]),
                "w2": float(row["w2"]),
                "alpha_min": float(row["alpha_min"]),
                "alpha_max": float(row["alpha_max"]),
                "k_min": int(row["k_min"]),
                "k_max": int(row["k_max"]),
                "j_star": int(row["j_star"]),
                "j_window": int(row["j_window"]),
                "sum_e": float(row["sum_energy"]),
                "n_eff": float(row["n_eff"]),
                "n_eff_clip": float(row["n_eff_clip"]),
                "n_eff_norm": float(row["n_eff_norm"]),
                "s_delta": float(row["s_delta"]),
                "c": float(row["c_measure"]),
                "alpha_c": float(row["alpha_c"]),
                "k_star": int(row["k_star"]),
                "alpha_c_final": float(row["alpha_c_final"]),
            })
    case_results.sort(key=lambda result: result["spectrum_index"])
    return case_results


# Draw one EMS spectrum panel from the saved theory data.
def add_spectrum_panel(axis, result):
    e_sorted = result["e_sorted"]
    x_values = np.arange(1, len(e_sorted) + 1)
    k_star = result["k_star"]
    alpha_c = result["alpha_c"]
    alpha_c_final = result["alpha_c_final"]
    n_eff_line = int(np.ceil(result["n_eff_norm"] * result["k_max"]))
    s_delta_line = int(np.ceil(result["s_delta"] * result["k_max"]))
    x_smooth = np.linspace(x_values[0], x_values[-1], 400)
    e_smooth = PchipInterpolator(x_values, e_sorted)(x_smooth)
    fill_end = min(result["k_max"], x_values[-1])
    fill_mask = x_smooth <= fill_end
    axis.set_facecolor("none")
    axis.fill_between(x_smooth[fill_mask], e_smooth[fill_mask], color=fill_color, alpha=0.18, linewidth=0)
    if k_star < fill_end:
        hatch_mask = (x_smooth >= k_star) & (x_smooth <= fill_end)
        axis.fill_between(x_smooth[hatch_mask], e_smooth[hatch_mask], facecolor="none", edgecolor="#666666", hatch="////", linewidth=0.0, zorder=2.5)
    axis.plot(x_smooth, e_smooth, "-", color=spectrum_color, linewidth=1.55, zorder=3)
    axis.axvline(k_star, color=kstar_color, linestyle="--", linewidth=1.2, alpha=0.95, zorder=2)
    axis.annotate(rf"$K^{{\star}}={k_star}$", xy=(k_star, 0.60), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", color=kstar_color, fontsize=8, fontweight="bold", ha="left", va="top", bbox=dict(boxstyle="square,pad=0.14", facecolor="white", edgecolor="white", linewidth=0.0, alpha=0.96))
    axis.axvline(n_eff_line, color=neff_color, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    axis.annotate(rf"$K_{{\mathrm{{N}}}}={n_eff_line}$", xy=(n_eff_line, 0.45), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", color=neff_color, fontsize=8, ha="left", va="top", bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="white", linewidth=0.0, alpha=0.90))
    axis.axvline(s_delta_line, color=sdelta_color, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    axis.annotate(rf"$K_{{S}}={s_delta_line}$", xy=(s_delta_line, 0.25), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", color=sdelta_color, fontsize=8, ha="left", va="top", bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="white", linewidth=0.0, alpha=0.90))
    axis.text(0.98, 0.98, rf"$\alpha_C={alpha_c:.2f}$" "\n" rf"$\alpha_C^{{\mathrm{{final}}}}={alpha_c_final:.2f}$", transform=axis.transAxes, ha="right", va="top", fontsize=7.5, color="#222222", bbox=dict(boxstyle="square,pad=0.14", facecolor="white", edgecolor="white", linewidth=0.0, alpha=0.92))
    axis.set_title(result["label"], fontsize=9, color="#222222", pad=4)
    axis.set_xlim(1, len(e_sorted))
    axis.set_xticks([1, 4, 8, 12, 16, 20])
    axis.set_ylim(0, 1)
    axis.set_yticks([0, 0.5, 1.0])
    axis.grid(True, axis="y", alpha=0.24, linestyle=":", linewidth=0.6)


# Read the EMS spectra and summary tables, rebuild the figure, and export the matching plot files.
def main():
    print("Loading EMS CSV outputs")
    case_results = read_case_data()
    apply_james_style()
    print("Building figure from CSV data")
    figure, axes = plt.subplots(2, 3, figsize=(170 * mm, 100 * mm), sharex=True, sharey=True)
    figure.patch.set_alpha(0.0)
    for panel_index, result in enumerate(case_results):
        axis = axes[panel_index // 3, panel_index % 3]
        add_spectrum_panel(axis, result)
        if panel_index // 3 == 1:
            axis.set_xlabel(r"Mode pair index $j$")
        if panel_index % 3 == 0:
            axis.set_ylabel(r"$E(j)$")
    figure.align_ylabels(axes[:, 0])
    plt.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    output_path = figure_output_path("Banerjee_2026_Enufft_Ems")
    save_png_and_pdf(figure, output_path)
    print(f"Figure saved to {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
