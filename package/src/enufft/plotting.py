"""Shared plotting style for proof figures."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


mm = 1.0 / 25.4


def apply_enufft_style():
    """Apply the paper-ready ENUFFT plotting style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": "#6f6f6f",
            "savefig.dpi": 600,
            "hatch.linewidth": 0.8,
        }
    )
    return {
        "baseline_hist_color": "#f06b4f",
        "optimized_hist_color": "#4aa3df",
        "baseline_median_color": "#c0392b",
        "optimized_median_color": "#1f78b4",
        "terrain_cmap": plt.get_cmap("YlOrRd_r"),
        "spectrum_cmap": plt.get_cmap("magma"),
    }


def save_png_and_pdf(figure, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
