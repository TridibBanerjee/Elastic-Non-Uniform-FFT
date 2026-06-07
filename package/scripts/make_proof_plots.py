#!/usr/bin/env python3
"""Generate local proof plots for the ENUFFT package."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import PchipInterpolator

from enufft import (
    EMSConfig,
    WindowConfig,
    build_analysis_window,
    compute_direct_dft_coefficients,
    compute_nufft_for_kernels,
    elastic_mode_selection,
    enufft_on_polygon,
    mode_values,
    reconstruct_at_points,
)
from enufft.plotting import apply_enufft_style, mm, save_png_and_pdf
from enufft.synthetic import demo_ems_spectra, generate_dem_points, irregular_orography


FIGURE_DIR = ROOT / "proof" / "figures"


def _style_3d_axis(axis, lx: float, ly: float, z_min: float, z_max: float) -> None:
    axis.set_facecolor("none")
    axis.set_xlim(0.0, lx)
    axis.set_ylim(0.0, ly)
    axis.set_zlim(z_min, z_max)
    axis.set_xticks([0, 5, 10])
    axis.set_yticks([0, 5, 10])
    axis.set_zticks([])
    axis.set_xlabel(r"$x$ (km)", labelpad=-8)
    axis.set_ylabel(r"$y$ (km)", labelpad=-8)
    axis.view_init(elev=30, azim=-58)
    axis.set_box_aspect((1.0, 1.0, 0.46))
    axis.grid(False)
    axis.xaxis.pane.set_facecolor((1, 1, 1, 0))
    axis.yaxis.pane.set_facecolor((1, 1, 1, 0))
    axis.zaxis.pane.set_facecolor((1, 1, 1, 0))
    axis.xaxis.pane.set_edgecolor("#dddddd")
    axis.yaxis.pane.set_edgecolor("#dddddd")
    axis.zaxis.pane.set_edgecolor("#dddddd")
    axis.tick_params(axis="both", which="major", pad=-3)


def build_nufft_accuracy_figure(output_dir: Path) -> dict[str, float]:
    style = apply_enufft_style()
    rng_seed = 42
    lx = 10.0
    ly = 10.0
    mode_limit = 12
    x_dem, y_dem = generate_dem_points(1200, lx, ly, distribution="ridge", random_seed=rng_seed)
    m_values = mode_values(mode_limit)
    n_values = mode_values(mode_limit)
    terrain_specs = [("Multi-peak", "multi_peak"), ("Meandering ridge", "ridge"), ("Basin and peaks", "basin")]

    terrain_cases = []
    optimized_errors = []
    baseline_errors = []
    for title, kind in terrain_specs:
        heights = irregular_orography(x_dem, y_dem, kind)
        direct = compute_direct_dft_coefficients(x_dem, y_dem, heights, m_values, n_values, lx, ly)
        kernels = compute_nufft_for_kernels(
            x_dem,
            y_dem,
            heights,
            m_values,
            n_values,
            lx,
            ly,
            1.5,
            4,
            2.34,
            ("optimized", "baseline"),
        )
        optimized_errors.append(np.abs(kernels["optimized"] - direct).ravel())
        baseline_errors.append(np.abs(kernels["baseline"] - direct).ravel())
        terrain_cases.append((title, heights))

    optimized_errors = np.concatenate(optimized_errors)
    baseline_errors = np.concatenate(baseline_errors)
    figure = plt.figure(figsize=(176 * mm, 132 * mm))
    figure.patch.set_alpha(0.0)
    grid = figure.add_gridspec(2, 3, height_ratios=[0.76, 1.24], hspace=0.12, wspace=0.05)
    axis = figure.add_subplot(grid[0, :])
    eps = 1e-14
    log_optimized = np.log10(optimized_errors + eps)
    log_baseline = np.log10(baseline_errors + eps)
    bins = np.linspace(np.floor(min(log_optimized.min(), log_baseline.min()) * 2) / 2, np.ceil(max(log_optimized.max(), log_baseline.max()) * 2) / 2, 40)
    _, _, baseline_patches = axis.hist(log_baseline, bins=bins, density=True, color=style["baseline_hist_color"], edgecolor="#222222", linewidth=0.65)
    for patch in baseline_patches:
        patch.set_facecolor((*to_rgb(style["baseline_hist_color"]), 0.52))
        patch.set_hatch("///")
    axis.hist(log_optimized, bins=bins, density=True, color=style["optimized_hist_color"], edgecolor="#222222", linewidth=0.65, alpha=0.86)
    baseline_median = float(np.median(log_baseline))
    optimized_median = float(np.median(log_optimized))
    axis.axvline(baseline_median, color=style["baseline_median_color"], linestyle="--", linewidth=1.2)
    axis.axvline(optimized_median, color=style["optimized_median_color"], linestyle="--", linewidth=1.2)
    axis.text(
        0.03,
        0.88,
        rf"optimized median $={optimized_median:.2f}$",
        transform=axis.transAxes,
        color=style["optimized_median_color"],
        fontsize=8,
        bbox=dict(boxstyle="square,pad=0.16", facecolor="white", edgecolor="white", alpha=0.92),
    )
    axis.text(
        0.03,
        0.74,
        rf"baseline median $={baseline_median:.2f}$",
        transform=axis.transAxes,
        color=style["baseline_median_color"],
        fontsize=8,
        bbox=dict(boxstyle="square,pad=0.16", facecolor="white", edgecolor="white", alpha=0.92),
    )
    axis.legend(
        handles=[
            Patch(facecolor=style["baseline_hist_color"], edgecolor="#222222", hatch="///", label=r"$\gamma=\beta$"),
            Patch(facecolor=style["optimized_hist_color"], edgecolor="#222222", label="optimized KB"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 1.24),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=1.5,
    )
    axis.set_title("Coefficient errors", pad=6)
    axis.set_xlabel(r"$\log_{10}|\widehat h_{\mathrm{NUFFT}}-\widehat h_{\mathrm{DFT}}|$")
    axis.set_ylabel("PDF")
    axis.grid(True, axis="y", linestyle=":", alpha=0.30, linewidth=0.6)

    triangulation = Triangulation(x_dem, y_dem)
    z_min = min(float(np.min(h)) for _, h in terrain_cases) / 1000.0
    z_max = max(float(np.max(h)) for _, h in terrain_cases) / 1000.0
    norm = Normalize(vmin=z_min, vmax=z_max)
    terrain_axes = []
    for index, (title, heights) in enumerate(terrain_cases):
        terrain_axis = figure.add_subplot(grid[1, index], projection="3d")
        terrain_axes.append(terrain_axis)
        heights_km = heights / 1000.0
        surface = terrain_axis.plot_trisurf(
            triangulation,
            heights_km,
            cmap=style["terrain_cmap"],
            norm=norm,
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=0.96,
        )
        surface.set_edgecolor("none")
        terrain_axis.tricontour(
            triangulation,
            heights_km,
            zdir="z",
            offset=z_min - 0.12,
            levels=8,
            cmap="Greys",
            linewidths=0.45,
            alpha=0.35,
        )
        terrain_axis.set_title(title, pad=0)
        _style_3d_axis(terrain_axis, lx, ly, z_min - 0.12, z_max)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=style["terrain_cmap"])
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=terrain_axes, shrink=0.73, pad=0.035, fraction=0.025)
    colorbar.set_label("height (km)")
    output_path = output_dir / "enufft_proof_nufft.png"
    save_png_and_pdf(figure, output_path)
    plt.close(figure)
    return {
        "nufft_optimized_median_abs_error": float(np.median(optimized_errors)),
        "nufft_baseline_median_abs_error": float(np.median(baseline_errors)),
    }


def build_ems_figure(output_dir: Path) -> dict[str, float]:
    apply_enufft_style()
    config = EMSConfig(k_min=1, k_max=12, alpha_min=0.0, alpha_max=0.7, delta=0.02, w1=0.5, w2=0.5)
    spectra = []
    for spectrum in demo_ems_spectra(20):
        result = elastic_mode_selection(spectrum["e_values"], config)
        spectra.append({**spectrum, "ems": result})

    figure, axes = plt.subplots(2, 3, figsize=(170 * mm, 100 * mm), sharex=True, sharey=True)
    figure.patch.set_alpha(0.0)
    for panel_index, spectrum in enumerate(spectra):
        axis = axes[panel_index // 3, panel_index % 3]
        result = spectrum["ems"]
        e_sorted = result.e_sorted
        x_values = np.arange(1, len(e_sorted) + 1)
        x_smooth = np.linspace(1, len(e_sorted), 400)
        y_smooth = PchipInterpolator(x_values, e_sorted)(x_smooth)
        fill_end = min(result.config.k_max or len(e_sorted), len(e_sorted))
        axis.fill_between(x_smooth[x_smooth <= fill_end], y_smooth[x_smooth <= fill_end], color="#959791", alpha=0.18, linewidth=0)
        if result.k_star < fill_end:
            hatch_mask = (x_smooth >= result.k_star) & (x_smooth <= fill_end)
            axis.fill_between(x_smooth[hatch_mask], y_smooth[hatch_mask], facecolor="none", edgecolor="#666666", hatch="////", linewidth=0.0)
        axis.plot(x_smooth, y_smooth, color="#000000", linewidth=1.55)
        axis.axvline(result.k_star, color="#c0392b", linestyle="--", linewidth=1.2)
        axis.annotate(rf"$K^\star={result.k_star}$", xy=(result.k_star, 0.60), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", color="#c0392b", fontweight="bold", fontsize=8)
        axis.text(0.98, 0.98, rf"$\alpha_C={result.alpha_c:.2f}$" "\n" rf"$\vartheta={result.alpha_c_final:.2f}$", transform=axis.transAxes, ha="right", va="top", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.14", facecolor="white", edgecolor="white", alpha=0.92))
        axis.set_title(spectrum["label"])
        axis.set_xlim(1, len(e_sorted))
        axis.set_ylim(0, 1)
        axis.grid(True, axis="y", alpha=0.24, linestyle=":", linewidth=0.6)
        if panel_index // 3 == 1:
            axis.set_xlabel(r"Mode pair index $j$")
        if panel_index % 3 == 0:
            axis.set_ylabel(r"$E(j)$")
    figure.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    output_path = output_dir / "enufft_proof_ems.png"
    save_png_and_pdf(figure, output_path)
    plt.close(figure)
    return {"ems_uniform_k_star": float(spectra[0]["ems"].k_star), "ems_peak_k_star": float(spectra[2]["ems"].k_star)}


def build_polygon_enufft_figure(output_dir: Path) -> dict[str, float]:
    style = apply_enufft_style()
    polygon = np.array([[0.10, 0.16], [1.35, 0.02], [1.72, 0.82], [1.05, 1.55], [0.16, 1.18]])
    window_config = WindowConfig(support="square", alignment="centroid", expansion=1.42)
    rng = np.random.default_rng(8)
    seed_points = rng.uniform([-0.45, -0.45], [2.05, 2.05], size=(3600, 2))
    _, preview_window = build_analysis_window(seed_points, polygon, window_config)
    local_seed = preview_window.to_local(seed_points)
    values = (
        900.0 * np.cos(2.0 * np.pi * (3 * local_seed[:, 0] / preview_window.Lx + 2 * local_seed[:, 1] / preview_window.Ly) + 0.35)
        + 220.0 * np.cos(2.0 * np.pi * (-1 * local_seed[:, 0] / preview_window.Lx + 4 * local_seed[:, 1] / preview_window.Ly) - 0.4)
    )
    result = enufft_on_polygon(
        seed_points,
        values,
        polygon,
        mode_limit=6,
        window_config=window_config,
        ems_config=EMSConfig(k_min=1, k_max=6, alpha_min=0.0, alpha_max=0.78),
        weight_type="voronoi",
    )

    figure = plt.figure(figsize=(180 * mm, 170 * mm))
    figure.patch.set_alpha(0.0)
    grid = figure.add_gridspec(2, 2)

    def _attach_cax(axis):
        return make_axes_locatable(axis).append_axes("right", size="4.5%", pad=0.08)

    domain_axis = figure.add_subplot(grid[0, 0])
    local_points = result.raw.local_points
    value_norm = Normalize(vmin=float(np.percentile(result.raw.values, 1)), vmax=float(np.percentile(result.raw.values, 99)))
    scatter = domain_axis.scatter(
        local_points[:, 0],
        local_points[:, 1],
        c=result.raw.values,
        s=5,
        cmap=style["terrain_cmap"],
        norm=value_norm,
        linewidths=0,
        alpha=0.82,
    )
    domain_axis.add_patch(Rectangle((0.0, 0.0), result.window.Lx, result.window.Ly, fill=False, edgecolor="#222222", linewidth=1.05))
    domain_axis.add_patch(Polygon(result.window.polygon_local, fill=False, closed=True, edgecolor="#0072b2", linewidth=1.4))
    domain_axis.set_title("Polygon samples")
    domain_axis.set_xlabel(r"local $x$")
    domain_axis.set_ylabel(r"local $y$")
    domain_axis.set_xlim(-0.03 * result.window.Lx, 1.03 * result.window.Lx)
    domain_axis.set_ylim(-0.03 * result.window.Ly, 1.03 * result.window.Ly)
    colorbar = figure.colorbar(scatter, cax=_attach_cax(domain_axis))
    colorbar.set_label("height")

    raw_axis = figure.add_subplot(grid[0, 1])
    amplitude = np.abs(result.raw_spectrum)
    image = raw_axis.imshow(
        amplitude.T,
        origin="lower",
        cmap=style["spectrum_cmap"],
        norm=LogNorm(vmin=max(amplitude[amplitude > 0].min(), 1e-8), vmax=amplitude.max()),
        extent=[result.m_values[0] - 0.5, result.m_values[-1] + 0.5, result.n_values[0] - 0.5, result.n_values[-1] + 0.5],
        aspect="auto",
    )
    raw_axis.scatter(result.selected_modes[:, 0], result.selected_modes[:, 1], s=38, marker="o", facecolors="none", edgecolors="#4aa3df", linewidths=1.2)
    raw_axis.scatter(-result.selected_modes[:, 0], -result.selected_modes[:, 1], s=38, marker="o", facecolors="none", edgecolors="#4aa3df", linewidths=1.2)
    raw_axis.set_title("Raw spectrum and EMS picks")
    raw_axis.set_xlabel(r"$m$")
    raw_axis.set_ylabel(r"$n$")
    raw_axis.set_xticks([-6, -3, 0, 3, 6])
    raw_axis.set_yticks([-6, -3, 0, 3, 6])
    spectrum_bar = figure.colorbar(image, cax=_attach_cax(raw_axis))
    spectrum_bar.set_label(r"$|\widehat h|$")

    recon_axis = figure.add_subplot(grid[1, 0])
    grid_n = 120
    gx = np.linspace(0.0, result.window.Lx, grid_n)
    gy = np.linspace(0.0, result.window.Ly, grid_n)
    x_grid, y_grid = np.meshgrid(gx, gy, indexing="xy")
    recon = reconstruct_at_points(result.spectrum, x_grid.ravel(), y_grid.ravel(), result.m_values, result.n_values, result.window.Lx, result.window.Ly).reshape(grid_n, grid_n)
    recon_axis.contourf(x_grid, y_grid, recon, levels=28, cmap=style["terrain_cmap"], norm=value_norm)
    recon_axis.add_patch(Polygon(result.window.polygon_local, fill=False, closed=True, edgecolor="#0072b2", linewidth=1.2))
    recon_axis.set_title("Sparse reconstruction")
    recon_axis.set_xlabel(r"local $x$")
    recon_axis.set_ylabel(r"local $y$")
    recon_sm = plt.cm.ScalarMappable(norm=value_norm, cmap=style["terrain_cmap"])
    recon_sm.set_array([])
    recon_bar = figure.colorbar(recon_sm, cax=_attach_cax(recon_axis))
    recon_bar.set_label("height")

    energy_axis = figure.add_subplot(grid[1, 1])
    energy = result.selection.energy_sorted
    ranks = np.arange(1, energy.size + 1)
    normalized_energy = energy / np.max(energy)
    energy_axis.semilogy(ranks, normalized_energy, color="#222222", linewidth=1.4)
    energy_axis.fill_between(ranks[: result.mode_pair_count], normalized_energy[: result.mode_pair_count], 1e-5, color="#4aa3df", alpha=0.25)
    energy_axis.axvline(result.mode_pair_count, color="#c0392b", linestyle="--", linewidth=1.2)
    energy_axis.text(
        0.06,
        0.82,
        rf"$K^\star={result.mode_pair_count}$" "\n" rf"signed modes $={result.signed_mode_count}$" "\n" rf"power $={result.power_retained:.2f}$",
        transform=energy_axis.transAxes,
        color="#c0392b",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="white", alpha=0.94),
    )
    energy_axis.set_title("Pair-energy decay")
    energy_axis.set_xlabel("mode pair rank")
    energy_axis.set_ylabel("normalized energy")
    energy_axis.set_xlim(1, min(30, energy.size))
    energy_axis.set_ylim(1e-5, 1.35)
    energy_axis.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.28, which="major")
    _attach_cax(energy_axis).set_axis_off()

    figure.subplots_adjust(left=0.08, right=0.965, top=0.95, bottom=0.08, wspace=0.40, hspace=0.30)

    output_path = output_dir / "enufft_proof_polygon.png"
    save_png_and_pdf(figure, output_path)
    plt.close(figure)
    return {
        "polygon_k_star": float(result.mode_pair_count),
        "polygon_signed_modes": float(result.signed_mode_count),
        "polygon_power_retained": float(result.power_retained),
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    summaries.update(build_nufft_accuracy_figure(FIGURE_DIR))
    summaries.update(build_ems_figure(FIGURE_DIR))
    summaries.update(build_polygon_enufft_figure(FIGURE_DIR))
    summary_path = FIGURE_DIR / "proof_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        for key, value in sorted(summaries.items()):
            handle.write(f"{key}: {value:.8g}\n")
    print(f"Wrote proof figures to {FIGURE_DIR}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
