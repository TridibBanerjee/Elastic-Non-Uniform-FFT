# Plot_Nufft.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script reads the NUFFT comparison CSV tables and builds the figure for this case.

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.tri import Triangulation

from Module_Plot_Template import mm, apply_james_style, figure_output_path, save_png_and_pdf


csv_dir = Path("./csv")
terrain_csv = csv_dir / "Banerjee_2026_Enufft_Nufft_Terrain.csv"
modes_csv = csv_dir / "Banerjee_2026_Enufft_Nufft_Modes.csv"
lx = 10.0
ly = 10.0
terrain_columns = [
    ("Multi-peak", "multi_peak", "height_multi_peak_m"),
    ("Meandering ridge", "ridge", "height_ridge_m"),
    ("Basin and peaks", "basin", "height_basin_m"),
]


# Load the shared scattered coordinates and terrain heights from CSV.
def read_terrain_data():
    x_values = []
    y_values = []
    terrain_heights = {kind: [] for _, kind, _ in terrain_columns}
    with terrain_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_values.append(float(row["x_km"]))
            y_values.append(float(row["y_km"]))
            for _, kind, column_name in terrain_columns:
                terrain_heights[kind].append(float(row[column_name]))
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    terrain_cases = []
    for title, kind, _ in terrain_columns:
        terrain_cases.append({"title": title, "kind": kind, "h_dem": np.asarray(terrain_heights[kind], dtype=float)})
    return x_values, y_values, terrain_cases


# Load the absolute Fourier-coefficient errors used in the histogram panel.
def read_error_arrays():
    optimized_errors = []
    baseline_errors = []
    with modes_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            optimized_errors.append(float(row["optimized_error_abs"]))
            baseline_errors.append(float(row["baseline_error_abs"]))
    return np.asarray(optimized_errors, dtype=float), np.asarray(baseline_errors, dtype=float)


# Plot the pooled PDFs of log10 absolute coefficient errors for the two kernels.
def build_histogram_panel(axis, optimized_errors, baseline_errors, style_values):
    axis.set_facecolor("none")
    epsilon = 1e-14
    log_optimized = np.log10(optimized_errors + epsilon)
    log_baseline = np.log10(baseline_errors + epsilon)
    histogram_min = np.floor(min(log_optimized.min(), log_baseline.min()) * 2.0) / 2.0
    histogram_max = np.ceil(max(log_optimized.max(), log_baseline.max()) * 2.0) / 2.0
    bins = np.linspace(histogram_min, histogram_max, 42)
    _, _, baseline_patches = axis.hist(log_baseline, bins=bins, density=True, color=style_values["baseline_hist_color"], edgecolor="#222222", linewidth=0.8)
    for patch in baseline_patches:
        patch.set_facecolor((*to_rgb(style_values["baseline_hist_color"]), 0.55))
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)
        patch.set_hatch("///")
    _, _, optimized_patches = axis.hist(log_optimized, bins=bins, density=True, color=style_values["optimized_hist_color"], edgecolor="#222222", linewidth=0.8)
    for patch in optimized_patches:
        patch.set_facecolor((*to_rgb(style_values["optimized_hist_color"]), 0.82))
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)
    baseline_median = np.median(log_baseline)
    optimized_median = np.median(log_optimized)
    axis.axvline(baseline_median, color=style_values["baseline_median_color"], linestyle="--", linewidth=1.2, alpha=1.0)
    axis.axvline(optimized_median, color=style_values["optimized_median_color"], linestyle="--", linewidth=1.2, alpha=1.0)
    axis.annotate(rf"median $={baseline_median:.2f}$", xy=(baseline_median, 0.18), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=style_values["baseline_median_color"], bbox=dict(boxstyle="square,pad=0.16", facecolor="white", edgecolor="#222222", linewidth=0.7, alpha=1.0))
    axis.annotate(rf"median $={optimized_median:.2f}$", xy=(optimized_median, 0.18), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=style_values["optimized_median_color"], bbox=dict(boxstyle="square,pad=0.16", facecolor="white", edgecolor="#222222", linewidth=0.7, alpha=1.0))
    legend_handles = [
        Patch(facecolor=style_values["baseline_hist_color"], edgecolor="#222222", hatch="///", label=r"$\gamma=\beta$"),
        Patch(facecolor=style_values["optimized_hist_color"], edgecolor="#222222", label=r"$\gamma=\pi\sqrt{(2a)^2(\beta/\pi)^2-0.8}$"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=1.4, borderpad=0.2)
    axis.set_title(r"Coefficient Errors", fontsize=9, color="#222222", pad=4)
    axis.set_xlabel(r"$e=\log_{10}\,|\widehat{h}_{\mathrm{NUFFT}}-\widehat{h}_{\mathrm{DFT}}|$")
    axis.set_ylabel(r"PDF $p(e)$")
    axis.grid(True, axis="y", alpha=0.30, linestyle=":", linewidth=0.6)


# Plot the three triangulated terrain surfaces on a shared height scale.
def build_terrain_panels(figure, grid_spec, x_values, y_values, terrain_cases, style_values):
    triangulation = Triangulation(x_values, y_values)
    z_min = min(float(case["h_dem"].min()) for case in terrain_cases)
    z_max = max(float(case["h_dem"].max()) for case in terrain_cases)
    norm = Normalize(vmin=z_min, vmax=z_max)
    terrain_axes = []
    for panel_index, terrain_case in enumerate(terrain_cases):
        axis = figure.add_subplot(grid_spec[1, panel_index], projection="3d")
        terrain_axes.append(axis)
        surface = axis.plot_trisurf(triangulation, terrain_case["h_dem"], cmap=style_values["terrain_cmap"], norm=norm, linewidth=0.0, antialiased=True, shade=True, alpha=0.95)
        surface.set_edgecolor("none")
        axis.tricontour(triangulation, terrain_case["h_dem"], zdir="z", offset=z_min - 110.0, levels=8, cmap="Greys", linewidths=0.45, alpha=0.35)
        axis.set_title(terrain_case["title"], fontsize=9, color="#222222", pad=2)
        axis.set_facecolor("none")
        axis.set_xlabel(r"$x$ (km)", labelpad=-9)
        axis.set_ylabel(r"$y$ (km)", labelpad=-9)
        axis.set_xlim(0.0, lx)
        axis.set_ylim(0.0, ly)
        axis.set_zlim(z_min - 110.0, z_max)
        axis.set_xticks([0, 5, 10])
        axis.set_yticks([0, 5, 10])
        axis.set_zticks([])
        axis.view_init(elev=30, azim=-58)
        axis.set_box_aspect((1.0, 1.0, 0.46))
        axis.grid(False)
        axis.xaxis.pane.set_facecolor((1, 1, 1, 0))
        axis.yaxis.pane.set_facecolor((1, 1, 1, 0))
        axis.zaxis.pane.set_facecolor((1, 1, 1, 0))
        axis.xaxis.pane.set_edgecolor("#dddddd")
        axis.yaxis.pane.set_edgecolor("#dddddd")
        axis.zaxis.pane.set_edgecolor("#dddddd")
        axis.xaxis.set_label_coords(0.08, -0.03)
        axis.yaxis.set_label_coords(-0.03, 0.08)
        axis.tick_params(axis="both", which="major", pad=-3)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=style_values["terrain_cmap"])
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=terrain_axes, shrink=0.72, pad=0.040, fraction=0.024)
    colorbar.set_label("Height (km)")
    colorbar.ax.yaxis.set_major_locator(FixedLocator([0.0, 500.0, 1000.0, 1500.0]))
    colorbar.ax.yaxis.set_major_formatter(FixedFormatter(["0", "0.5", "1.0", "1.5"]))
    colorbar.ax.tick_params(length=2.5, width=0.7)
    colorbar.outline.set_linewidth(0.6)


# Read the saved tables, rebuild the figure, and export the matching plot files.
def main():
    print("Loading NUFFT CSV outputs")
    x_values, y_values, terrain_cases = read_terrain_data()
    optimized_errors, baseline_errors = read_error_arrays()
    style_values = apply_james_style()
    print("Building figure from CSV data")
    figure = plt.figure(figsize=(170 * mm, 128 * mm))
    figure.patch.set_alpha(0.0)
    grid_spec = figure.add_gridspec(2, 3, height_ratios=[0.78, 1.20], hspace=0.18, wspace=0.06)
    histogram_axis = figure.add_subplot(grid_spec[0, :])
    build_histogram_panel(histogram_axis, optimized_errors, baseline_errors, style_values)
    build_terrain_panels(figure, grid_spec, x_values, y_values, terrain_cases, style_values)
    output_path = figure_output_path("Banerjee_2026_Enufft_Nufft")
    save_png_and_pdf(figure, output_path)
    print(f"Figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
