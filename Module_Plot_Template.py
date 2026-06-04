# Module_Plot_Template.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the shared plotting style and figure-export helpers for these ENUFFT plots.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


mm = 1.0 / 25.4


# Apply the shared figure style and return the recurring color choices.
def apply_james_style():
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
    }


# Build the common figure output path inside the shared figures folder.
def figure_output_path(file_stem):
    figure_dir = Path("./figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir / f"{file_stem}.png"


# Save the figure as matching PNG and PDF files.
def save_png_and_pdf(figure, output_path):
    figure.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
