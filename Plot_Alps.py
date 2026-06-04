# Plot_Alps.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script rebuilds Alps preprocessing, diagnostics, and pooled sweep figures from local outputs.

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, LightSource, TwoSlopeNorm, to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from Module_Alps import (
    build_alps_case,
    build_alps_mesh,
    csa_window_strategy,
    find_mesh_triangle_ids,
    is_csa_supported_window,
    load_alps_dem,
    lonlat_to_xy,
    make_alps_sweep_tag,
    normalize_mesh_name,
    preprocessed_dem,
    require_preprocessed_dem,
    alps_mesh_presets,
    relative_rmse_variance_floor,
    window_strategies,
    xy_to_lonlat,
)
from Module_Plot_Template import mm, save_png_and_pdf


csv_dir = Path("./csv")
figure_dir = Path("./figures")
per_triangle_prefix = "Banerjee_2026_Enufft_Alps_PerTriangle"
summary_prefix = "Banerjee_2026_Enufft_Alps_Summary"
spectra_prefix = "Banerjee_2026_Enufft_Alps_Spectra"
sweep_summary_prefix = "Banerjee_2026_Enufft_Alps_SweepSummary"
earth_radius_km = 6371.0
alps_r2b5_target_dx_km = 80.0
icon_grid_n = 2
icon_grid_k = 5
icon_r2b5_effective_dx_km = 5050.0 / (icon_grid_n * 2 ** icon_grid_k)
text_color = "#202020"
muted_color = "#5f6468"
line_color = "#3a3a3a"
natural_earth_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
country_names = {"France", "Italy", "Switzerland", "Austria", "Germany", "Slovenia", "Liechtenstein"}
region_labels = [("FRANCE", 6.0, 44.55), ("ITALY", 10.00, 45.00), ("SWITZERLAND", 7.8, 47.00), ("AUSTRIA", 14.0, 47.45), ("GERMANY", 10.8, 48.00)]
target_hires_nx = 6601
target_stage_nx = 560
fallback_cell_size = 80.0


method_color = {"enufft": "#0072b2", "csa": "#4d4d4d", "truth": "#262626"}
summary_method_colors = {"CSA": "#4d4d4d", "Square": "#0072b2", "Tri.": "#009e73", "Circle": "#d55e00"}
method_hatches = {"enufft": "", "csa": "////"}
summary_method_hatches = {"CSA": "////", "Square": "", "Tri.": "\\\\\\\\", "Circle": "...."}
method_markers = {"enufft": "o", "csa": "s"}
summary_method_markers = {"CSA": "s", "Square": "o", "Tri.": "^", "Circle": "D"}
method_spread_hatches = {"enufft": "", "csa": "////"}
summary_method_spread_hatches = {"CSA": "////", "Square": "", "Tri.": "\\\\\\\\", "Circle": "...."}
combined_legend_handler = {tuple: HandlerTuple(ndivide=1, pad=0.0)}
terminal_true_values = {"--terminal_only", "--terminal-only", "--terminal_only=true", "--terminal-only=true"}
terminal_false_values = {"--terminal_only=false", "--terminal-only=false"}


terrain_cmap = LinearSegmentedColormap.from_list("alps_james_terrain", [(0.00, "#f4f2ee"), (0.15, "#d8e2d0"), (0.34, "#7baa70"), (0.52, "#d2c56d"), (0.70, "#c77b4c"), (0.88, "#9d4b3d"), (1.00, "#f3eee3")])
direction_cmap = LinearSegmentedColormap.from_list("alps_james_direction", [(0.00, "#f7f5f0"), (0.18, "#e7dfce"), (0.36, "#c8b783"), (0.54, "#8fa58f"), (0.72, "#587d83"), (0.88, "#3d5a80"), (1.00, "#202832")])
variance_cmap = LinearSegmentedColormap.from_list("alps_james_variance", [(0.00, "#1f2528"), (0.20, "#344f5c"), (0.42, "#5f8f8a"), (0.66, "#c3aa65"), (0.84, "#e7d8a6"), (1.00, "#fbf6e8")])
viz_terrain_cmap = LinearSegmentedColormap.from_list("alps_publication_relief", [(0.00, "#f2f2ef"), (0.10, "#e5ece2"), (0.22, "#72a56a"), (0.42, "#d6c96d"), (0.62, "#c77945"), (0.82, "#a5533d"), (1.00, "#f5f0e6")])
residual_cmap = LinearSegmentedColormap.from_list("alps_publication_residual", [(0.00, "#23557b"), (0.25, "#78aec4"), (0.49, "#f3f1e9"), (0.51, "#f3ede4"), (0.76, "#d58769"), (1.00, "#933a35")])


# Apply the Alps manuscript plotting style.
def apply_alps_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#222222", "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222", "hatch.linewidth": 0.75, "savefig.dpi": 600})


# Parse a CSV scalar into a Python value.
def parse_scalar(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return int(number)
    return number


# Read one CSV table into parsed dictionaries.
def read_csv_rows(path):
    with path.open(newline="") as handle:
        return [{key: parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


# Extract a run tag from a tagged Alps CSV path.
def tag_from_per_triangle_path(path):
    stem = Path(path).stem
    if not stem.startswith(per_triangle_prefix):
        raise ValueError(f"Unexpected per-triangle filename {Path(path).name}")
    return stem[len(per_triangle_prefix):]


# Build the CSV paths for one run tag.
def paths_for_tag(tag):
    return (csv_dir / f"{per_triangle_prefix}{tag}.csv", csv_dir / f"{summary_prefix}{tag}.csv", csv_dir / f"{spectra_prefix}{tag}.csv")


# Check whether one discovered tag belongs to the current proxy-mesh format.
def is_current_proxy_tag(tag):
    return re.search(r"^_r2b[45]_.*_dx[0-9mp]+(?:_first\d+)?$", str(tag)) is not None


# Discover complete per-configuration output tags.
def discover_tags(pattern=f"{per_triangle_prefix}_*.csv"):
    tags = []
    for path in sorted(csv_dir.glob(pattern)):
        tag = tag_from_per_triangle_path(path)
        if not is_current_proxy_tag(tag):
            continue
        _, summary_path, spectra_path = paths_for_tag(tag)
        if summary_path.exists() and spectra_path.exists():
            tags.append(tag)
    return tags


# Load one run and attach sorted spectral amplitudes.
def load_case(tag):
    per_triangle_path, summary_path, spectra_path = paths_for_tag(tag)
    rows = read_csv_rows(per_triangle_path)
    summary = read_csv_rows(summary_path)[0]
    spectra = read_spectra_csv(spectra_path)
    for row in rows:
        tri_num = int(row["tri_num"])
        row["centroid"] = [float(row["centroid_x"]), float(row["centroid_y"])]
        row["sorted_amplitudes_en"] = spectra.get((tri_num, "enufft"), [])
        row["sorted_amplitudes_csa"] = spectra.get((tri_num, "csa"), [])
    return rows, summary


# Read compact sorted-amplitude spectra from CSV.
def read_spectra_csv(path):
    spectra = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            tri_num = int(float(row["tri_num"]))
            method = str(row["method"])
            rank = int(float(row["rank"]))
            value = float(row["amplitude"])
            key = (tri_num, method)
            values = spectra.setdefault(key, [])
            while len(values) < rank:
                values.append(np.nan)
            values[rank - 1] = value
    return spectra


# Build the output figure path for one Alps figure.
def alps_figure_path(stem):
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir / f"{stem}.png"


# Proxy handle that overlays a hatched patch and a median line.
def combined_line_patch_handle(color, hatch, label, marker=None, lw=1.45, fill_alpha=0.0, hatch_alpha=0.62, hatch_color=None):
    patch = Patch(facecolor=to_rgba(color, fill_alpha), edgecolor=hatch_color if hatch_color is not None else to_rgba(color, hatch_alpha if hatch else 0.0), hatch=hatch, linewidth=0.0)
    marker_kwargs = {}
    if marker:
        marker_kwargs = {"marker": marker, "markersize": 3.4, "markerfacecolor": color, "markeredgecolor": color, "markeredgewidth": 0.75}
    line = Line2D([0.0, 1.0], [0.5, 0.5], color=color, lw=lw, **marker_kwargs)
    return (patch, line), label


# Blend a hatch color toward white for faint PDF hatching.
def spread_hatch_color(color):
    r_value, g_value, b_value, _ = to_rgba(color)
    blend = 0.55
    return (r_value + (1.0 - r_value) * blend, g_value + (1.0 - g_value) * blend, b_value + (1.0 - b_value) * blend, 1.0)


# Draw a percentile spread with optional hatching.
def plot_hatched_spread(axis, x_values, lo_values, hi_values, color, hatch):
    axis.fill_between(x_values, lo_values, hi_values, facecolor=to_rgba(color, 0.09), edgecolor="none", linewidth=0.0, zorder=1)
    if hatch:
        axis.fill_between(x_values, lo_values, hi_values, facecolor=(1.0, 1.0, 1.0, 0.0), edgecolor=spread_hatch_color(color), hatch=hatch, linewidth=0.32, zorder=2)


# Choose a readable marker cadence for spectral decay curves.
def decay_markevery(x_values):
    return max(1, int(np.ceil(len(x_values) / 10.0)))


# Plot a histogram with the same hatched method encoding as the maps.
def plot_hatched_histogram(axis, values, bins, color, hatch, label, alpha=0.68):
    counts, edges = np.histogram(values, bins=bins)
    axis.stairs(counts, edges, color=color, alpha=alpha, linewidth=1.45, label="_nolegend_", zorder=3)
    if hatch:
        widths = np.diff(edges)
        bars = axis.bar(edges[:-1], counts, width=widths, align="edge", facecolor=(1.0, 1.0, 1.0, 0.0), edgecolor=color, linewidth=0.0, hatch=hatch, alpha=alpha, zorder=2)
        for bar in bars:
            bar.set_facecolor((1.0, 1.0, 1.0, 0.0))
            bar.set_edgecolor(color)
            bar.set_alpha(alpha)
            bar.set_linewidth(0.0)
    return combined_line_patch_handle(color, hatch, label, lw=1.45, fill_alpha=0.0, hatch_alpha=alpha)


# Return a display name for a window strategy.
def window_support_name(window):
    if str(window).startswith("square_"):
        return "Square"
    if str(window).startswith("triangle_"):
        return "Tri."
    if str(window).startswith("circle_"):
        return "Circle"
    return str(window).replace("_", " ").title()


# Detect window support from a tagged output filename.
def detect_support_from_filename(name):
    for strategy in window_strategies:
        if f"_{strategy}_" in name:
            return window_support_name(strategy)
    return None


# Detect the full window strategy from a tagged output filename.
def detect_strategy_from_filename(name):
    for strategy in window_strategies:
        if f"_{strategy}_" in name:
            return strategy
    return None


# Build the pooled CSA key from a filename and triangle number.
def csa_pooled_spectra_key_from_filename(name, tri_num):
    strategy = detect_strategy_from_filename(name)
    if strategy is None:
        return None
    reference_window = csa_window_strategy(strategy)
    match = re.search(r"_(?:(?P<mesh>r2b4|r2b5)_)?N(?P<n>\d+)_.*_eta(?P<eta>[0-9mp]+)_os[0-9mp]+_csa(?P<csa>\d+)(?:_dx(?P<dx>[0-9mp]+))?", name)
    if match is None:
        return None
    mesh_name = match.group("mesh")
    dx_text = match.group("dx")
    if dx_text is None and mesh_name in alps_mesh_presets:
        cell_size = float(alps_mesh_presets[mesh_name]["nominal_dx_km"])
    elif dx_text is not None:
        cell_size = float(dx_text.replace("m", "-").replace("p", "."))
    else:
        cell_size = np.nan
    return (int(tri_num), str(reference_window), cell_size, int(match.group("n")), int(match.group("csa")), float(match.group("eta").replace("m", "-").replace("p", ".")))


# Build the pooled CSA key for one scalar row.
def csa_pooled_key(row, csv_name, tri_num):
    filename_key = csa_pooled_spectra_key_from_filename(csv_name, tri_num)
    if filename_key is not None:
        return filename_key
    strategy = detect_strategy_from_filename(csv_name)
    reference_window = row.get("csa_reference_window")
    if not reference_window or str(reference_window).lower() == "nan":
        reference_window = csa_window_strategy(strategy) if strategy is not None else None
    return (int(tri_num), str(reference_window), float(row.get("cell_size_km", np.nan)), int(float(row.get("n_modes", np.nan))), int(float(row.get("csa_signed_limit", np.nan))), float(row.get("expansion", np.nan)))


# Create empty pooled diagnostic storage.
def make_pooled_method_store():
    return {label: {"rel_rmse": [], "spectra": [], "by_tri": {}, "samples": []} for label in ["CSA", "Square", "Tri.", "Circle", "ENUFFT", "Physical"]}


# Append one triangle value to pooled map storage.
def add_tri_value(store, tri_num, key, value, centroid=None):
    if not np.isfinite(value):
        return
    tri_store = store["by_tri"].setdefault(int(tri_num), {})
    tri_store.setdefault(key, []).append(float(value))
    if centroid is not None and np.all(np.isfinite(centroid)):
        store["samples"].append({"tri_num": int(tri_num), "key": key, "value": float(value), "centroid": np.asarray(centroid, dtype=float)})


# Return a row centroid when it is available in a scalar CSV.
def row_centroid(row):
    if "centroid_x" not in row or "centroid_y" not in row:
        return None
    centroid = np.asarray([float(row["centroid_x"]), float(row["centroid_y"])], dtype=float)
    return centroid if np.all(np.isfinite(centroid)) else None


# Return true when relative RMSE has a meaningful reference triangle.
def row_has_relative_rmse_reference(row):
    value = row.get("true_var")
    if value is None:
        return True
    try:
        true_var = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(true_var) and true_var > relative_rmse_variance_floor


# Load pooled diagnostics across all selected tags.
def load_pooled_sweep_diagnostics(tags):
    pooled = make_pooled_method_store()
    csa_scalar_keys = set()
    csa_spectra_keys = set()
    for tag in tags:
        csv_path, _, spectra_path = paths_for_tag(tag)
        strategy = detect_strategy_from_filename(csv_path.name)
        support = detect_support_from_filename(csv_path.name)
        if support is None:
            continue
        file_has_valid_csa = strategy is not None and is_csa_supported_window(strategy)
        for row in read_csv_rows(csv_path):
            tri_num = int(row["tri_num"])
            centroid = row_centroid(row)
            rel_rmse_en = float(row["rel_rmse_en"])
            rel_rmse_csa = float(row["rel_rmse_csa"])
            has_relative_reference = row_has_relative_rmse_reference(row)
            if has_relative_reference and np.isfinite(rel_rmse_en):
                pooled[support]["rel_rmse"].append(rel_rmse_en)
            row_has_valid_csa = bool(row.get("csa_valid", file_has_valid_csa))
            add_tri_value(pooled[support], tri_num, "direction", float(row["dom_dir_en"]), centroid)
            add_tri_value(pooled["ENUFFT"], tri_num, "direction", float(row["dom_dir_en"]), centroid)
            add_tri_value(pooled[support], tri_num, "variance", float(row["var_en"]), centroid)
            add_tri_value(pooled["ENUFFT"], tri_num, "variance", float(row["var_en"]), centroid)
            add_tri_value(pooled[support], tri_num, "K_star", float(row["K_star"]), centroid)
            add_tri_value(pooled["ENUFFT"], tri_num, "K_star", float(row["K_star"]), centroid)
            add_tri_value(pooled["Physical"], tri_num, "variance", float(row["true_var"]), centroid)
            csa_key = csa_pooled_key(row, csv_path.name, tri_num) if row_has_valid_csa else None
            if csa_key is not None and csa_key not in csa_scalar_keys:
                csa_scalar_keys.add(csa_key)
                if has_relative_reference and np.isfinite(rel_rmse_csa):
                    pooled["CSA"]["rel_rmse"].append(rel_rmse_csa)
                add_tri_value(pooled["CSA"], tri_num, "direction", float(row["dom_dir_csa"]), centroid)
                add_tri_value(pooled["CSA"], tri_num, "variance", float(row["var_csa"]), centroid)
        spectra = read_spectra_csv(spectra_path)
        for (tri_num, method), curve in spectra.items():
            if method == "enufft":
                pooled[support]["spectra"].append(np.asarray(curve, dtype=float))
            elif method == "csa":
                csa_key = csa_pooled_spectra_key_from_filename(spectra_path.name, int(tri_num))
                if csa_key is not None and csa_key not in csa_spectra_keys:
                    csa_spectra_keys.add(csa_key)
                    pooled["CSA"]["spectra"].append(np.asarray(curve, dtype=float))
    return pooled


# Return true when terminal-only output was requested.
def terminal_only_requested(argv):
    if len(argv) == 1:
        return False
    if len(argv) == 2 and argv[1] in terminal_true_values:
        return True
    if len(argv) == 2 and argv[1] in terminal_false_values:
        return False
    raise SystemExit("Usage: python3 Plot_Alps.py [--terminal_only=true]")


# Add one finite scalar to a terminal metric list.
def append_metric(store, label, metric, value):
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(scalar):
        store[label][metric].append(scalar)


# Create empty terminal summary storage.
def make_terminal_store():
    labels = ["Physical", "CSA", "Square", "Triangle", "Circle", "ENUFFT pooled"]
    metrics = ("rel_rmse", "mode_count", "variance")
    return {label: {metric: [] for metric in metrics} for label in labels}


# Return the terminal label for one support family.
def terminal_support_label(support):
    return "Triangle" if support == "Tri." else support


# Load terminal summary metrics across selected tags.
def load_terminal_metrics(tags):
    store = make_terminal_store()
    physical_keys = set()
    csa_scalar_keys = set()
    for tag in tags:
        csv_path, _, _ = paths_for_tag(tag)
        strategy = detect_strategy_from_filename(csv_path.name)
        support = detect_support_from_filename(csv_path.name)
        if support is None:
            continue
        label = terminal_support_label(support)
        file_has_valid_csa = strategy is not None and is_csa_supported_window(strategy)
        for row in read_csv_rows(csv_path):
            tri_num = int(row["tri_num"])
            if tri_num not in physical_keys:
                physical_keys.add(tri_num)
                append_metric(store, "Physical", "variance", row.get("true_var"))
            has_relative_reference = row_has_relative_rmse_reference(row)
            if has_relative_reference:
                append_metric(store, label, "rel_rmse", row.get("rel_rmse_en"))
            append_metric(store, label, "mode_count", row.get("K_star"))
            append_metric(store, label, "variance", row.get("var_en"))
            if has_relative_reference:
                append_metric(store, "ENUFFT pooled", "rel_rmse", row.get("rel_rmse_en"))
            append_metric(store, "ENUFFT pooled", "mode_count", row.get("K_star"))
            append_metric(store, "ENUFFT pooled", "variance", row.get("var_en"))
            row_has_valid_csa = bool(row.get("csa_valid", file_has_valid_csa))
            csa_key = csa_pooled_key(row, csv_path.name, tri_num) if row_has_valid_csa else None
            if csa_key is not None and csa_key not in csa_scalar_keys:
                csa_scalar_keys.add(csa_key)
                if has_relative_reference:
                    append_metric(store, "CSA", "rel_rmse", row.get("rel_rmse_csa"))
                append_metric(store, "CSA", "mode_count", row.get("csa_pairs_used"))
                append_metric(store, "CSA", "variance", row.get("var_csa"))
    return store


# Return terminal table row order for one metric.
def terminal_metric_labels(metric):
    if metric == "variance":
        return ["Physical", "CSA", "Square", "Triangle", "Circle", "ENUFFT pooled"]
    return ["CSA", "Square", "Triangle", "Circle", "ENUFFT pooled"]


# Compute p10, median, and p90 for one metric list.
def terminal_stats(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    p10, median, p90 = np.nanpercentile(array, [10.0, 50.0, 90.0])
    return float(np.nanmin(array)), p10, median, p90, float(np.nanmax(array)), int(array.size)


# Print one terminal metric table.
def print_terminal_metric(title, store, metric, value_format):
    print("")
    print(title)
    print(f"{'method':<16} {'min':>12} {'p10':>12} {'median':>12} {'p90':>12} {'max':>12} {'n':>8}")
    print("-" * 90)
    for label in terminal_metric_labels(metric):
        stats = terminal_stats(store[label][metric])
        if stats is None:
            print(f"{label:<16} {'-':>12} {'-':>12} {'-':>12} {'-':>12} {'-':>12} {0:>8}")
            continue
        minimum, p10, median, p90, maximum, count = stats
        print(f"{label:<16} {value_format.format(minimum):>12} {value_format.format(p10):>12} {value_format.format(median):>12} {value_format.format(p90):>12} {value_format.format(maximum):>12} {count:>8}")


# Build tag groups for terminal and plot summaries.
def grouped_complete_tags():
    tags = discover_tags()
    summaries = {tag: read_csv_rows(paths_for_tag(tag)[1])[0] for tag in tags}
    groups = {}
    for tag in tags:
        case = case_for_summary(summaries[tag], tag)
        key = (case.get("mesh_name") or "regular", int(case["n_modes"]), float(case["cell_size_km"]))
        groups.setdefault(key, {"case": case, "tags": []})["tags"].append(tag)
    return groups


# Print pooled Alps statistics to the terminal.
def print_terminal_summary():
    groups = grouped_complete_tags()
    if not groups:
        raise RuntimeError("No complete Alps outputs found for terminal summary")
    for group in groups.values():
        case = group["case"]
        output_tag = make_alps_sweep_tag(case)
        print("")
        print("=" * 72)
        print(f"Alps terminal summary {output_tag}")
        print(f"tags: {len(group['tags'])}")
        store = load_terminal_metrics(group["tags"])
        print_terminal_metric("Relative RMSE (reference sigma > 1 m)", store, "rel_rmse", "{:.3f}")
        print_terminal_metric("K* for ENUFFT, retained CSA pairs for CSA", store, "mode_count", "{:.1f}")
        print_terminal_metric("Variance [m^2]", store, "variance", "{:.3e}")


# Compute median spectral decay curves across triangles.
def pooled_decay_curve(curves):
    if not curves:
        return None
    n_show = min(40, max(len(curve) for curve in curves))
    if n_show <= 0:
        return None
    values = np.full((len(curves), n_show), np.nan, dtype=float)
    for index, curve in enumerate(curves):
        take = min(n_show, len(curve))
        values[index, :take] = curve[:take]
    valid = np.any(np.isfinite(values), axis=0)
    ranks = np.arange(1, n_show + 1)[valid]
    median = np.nanmedian(values[:, valid], axis=0)
    low = np.nanpercentile(values[:, valid], 10.0, axis=0)
    high = np.nanpercentile(values[:, valid], 90.0, axis=0)
    return ranks, median, low, high


# Compute the circular mean of angle values in degrees.
def circular_mean_degrees(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    radians = np.deg2rad(vals)
    angle = np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(((angle + 180.0) % 360.0) - 180.0)


# Compute one per-triangle map value from pooled storage.
def median_map_values(store, key, triangle_count, circular=False, mesh_centroids=None):
    values = np.full(triangle_count, np.nan, dtype=float)
    if mesh_centroids is not None and store.get("samples"):
        mesh_centroids = np.asarray(mesh_centroids, dtype=float)
        tree = cKDTree(mesh_centroids)
        domain_scale = max(float(np.ptp(mesh_centroids[:, 0])), float(np.ptp(mesh_centroids[:, 1])), 1.0)
        tolerance = 1e-8 * domain_scale
        buckets = [[] for _ in range(triangle_count)]
        for sample in store["samples"]:
            if sample["key"] != key:
                continue
            distance, index = tree.query(sample["centroid"], k=1)
            if distance <= tolerance:
                buckets[int(index)].append(float(sample["value"]))
        for index, samples in enumerate(buckets):
            sample_array = np.asarray(samples, dtype=float)
            if sample_array.size:
                values[index] = circular_mean_degrees(sample_array) if circular else float(np.nanmedian(sample_array))
        return values
    for tri_num, tri_store in store["by_tri"].items():
        samples = np.asarray(tri_store.get(key, []), dtype=float)
        if samples.size == 0 or int(tri_num) >= triangle_count:
            continue
        values[int(tri_num)] = circular_mean_degrees(samples) if circular else float(np.nanmedian(samples))
    return values


# Match a colorbar axis height to a reference map axis.
def match_colorbar_height(colorbar_axis, reference_axis):
    ref_box = reference_axis.get_position()
    cax_box = colorbar_axis.get_position()
    colorbar_axis.set_position([cax_box.x0, ref_box.y0, cax_box.width, ref_box.height])


# Compute the two-dimensional scalar cross product.
def cross2(a_value, b_value):
    return float(a_value[0] * b_value[1] - a_value[1] * b_value[0])


# Build hatch segments clipped to one triangle polygon.
def hatch_segments_for_triangle(poly, angle_deg, spacing):
    if not np.isfinite(angle_deg):
        return []
    theta = np.deg2rad(angle_deg)
    u_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=float)
    center = np.mean(poly, axis=0)
    radius = float(np.max(np.linalg.norm(poly - center, axis=1))) + spacing
    offsets = np.arange(-radius, radius + spacing, spacing)
    segments = []
    for offset in offsets:
        point = center + offset * v_vec
        hits = []
        for index in range(3):
            q_value = poly[index]
            r_value = poly[(index + 1) % 3]
            edge = r_value - q_value
            denom = cross2(u_vec, edge)
            if abs(denom) < 1e-12:
                continue
            qp = q_value - point
            t_value = cross2(qp, edge) / denom
            s_value = cross2(qp, u_vec) / denom
            if -1e-10 <= s_value <= 1.0 + 1e-10:
                hits.append(t_value)
        if len(hits) >= 2:
            hits = sorted(hits)
            segments.append([point + hits[0] * u_vec, point + hits[-1] * u_vec])
    return segments


# Draw direction hatching over triangle polygons.
def add_direction_hatching(axis, polygons, values, domain_scale):
    spacing = 0.036 * domain_scale
    for polygon, angle in zip(polygons, values):
        if not np.isfinite(angle):
            continue
        segments = hatch_segments_for_triangle(np.asarray(polygon, dtype=float), float(angle), spacing)
        if not segments:
            continue
        hatch_color = "#252525" if float(angle) <= 0.0 else "white"
        collection = LineCollection(segments, colors=hatch_color, linewidths=0.42, alpha=0.42, zorder=4, capstyle="butt")
        axis.add_collection(collection)


# Label projected map axes with longitude and latitude ticks.
def set_lonlat_axes(axis, dem_data, x_limits, y_limits):
    lon_ticks = [6, 8, 10, 12, 14, 16]
    lat_ticks = [44, 45, 46, 47, 48, 49]
    x_ticks = []
    x_labels = []
    for lon_value in lon_ticks:
        x_value, _ = lonlat_to_xy(lon_value, dem_data["lat_ref"], dem_data["lon_ref"], dem_data["lat_ref"])
        if x_limits[0] <= x_value <= x_limits[1]:
            x_ticks.append(x_value)
            x_labels.append(f"{lon_value}E")
    y_ticks = []
    y_labels = []
    for lat_value in lat_ticks:
        _, y_value = lonlat_to_xy(dem_data["lon_ref"], lat_value, dem_data["lon_ref"], dem_data["lat_ref"])
        if y_limits[0] <= y_value <= y_limits[1]:
            y_ticks.append(y_value)
            y_labels.append(f"{lat_value}N")
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_labels)
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(y_labels)


# Expand per-triangle scalar metrics onto the full mesh.
def full_metric(mesh, rows, key):
    values = np.full(len(mesh["triangles"]), np.nan, dtype=float)
    for row in rows:
        values[int(row["tri_num"])] = float(row[key])
    return values


# Draw the single-run Alps diagnostic figure.
def plot_diagnostics(rows, mesh, dem_data, tag):
    if not rows:
        return
    csa_available = any(bool(row.get("csa_valid", False)) for row in rows)
    vertices = mesh["vertices"]
    triangles = mesh["triangles"]
    polygons = [vertices[triangle] for triangle in triangles]
    centroids = np.array([vertices[triangle].mean(axis=0) for triangle in triangles])
    x_limits = (float(vertices[:, 0].min()), float(vertices[:, 0].max()))
    y_limits = (float(vertices[:, 1].min()), float(vertices[:, 1].max()))
    rel_en = np.array([row["rel_rmse_en"] for row in rows], dtype=float)
    rel_csa = np.array([row["rel_rmse_csa"] for row in rows], dtype=float)
    k_star = np.array([row["K_star"] for row in rows], dtype=float)
    csa_pairs = np.array([row["csa_pairs_used"] for row in rows], dtype=float)
    full_k = full_metric(mesh, rows, "K_star")
    full_dir_en = full_metric(mesh, rows, "dom_dir_en")
    full_dir_csa = full_metric(mesh, rows, "dom_dir_csa")
    full_true_var = full_metric(mesh, rows, "true_var")
    full_var_en = full_metric(mesh, rows, "var_en")
    full_var_csa = full_metric(mesh, rows, "var_csa")
    figure = plt.figure(figsize=(170 * mm, 142 * mm))
    outer = figure.add_gridspec(3, 1, height_ratios=[0.86, 1.0, 1.0], hspace=0.30)
    top_grid = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    dir_grid = outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.12)
    var_grid = outer[2].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.12)
    axis_hist = figure.add_subplot(top_grid[0, 0])
    axis_decay = figure.add_subplot(top_grid[0, 1])
    all_rel_parts = [rel_en[np.isfinite(rel_en)]]
    if csa_available:
        all_rel_parts.append(rel_csa[np.isfinite(rel_csa)])
    all_rel = np.concatenate(all_rel_parts)
    hist_upper = max(1.05, float(np.nanpercentile(all_rel, 95)) * 1.25)
    bins = np.linspace(0.0, hist_upper, 24)
    hist_series = [(rel_en, "enufft", rf"ENUFFT {np.nanmedian(rel_en):.2f}")]
    if csa_available:
        hist_series.append((rel_csa, "csa", rf"CSA {np.nanmedian(rel_csa):.2f}"))
    handles = []
    labels = []
    for values, key, label in hist_series:
        handle, hist_label = plot_hatched_histogram(axis_hist, values, bins, method_color[key], method_hatches[key], label)
        handles.append(handle)
        labels.append(hist_label)
    axis_hist.set_title("Relative RMSE", pad=3)
    axis_hist.set_xlabel(r"RMSE/$\sigma_h$")
    axis_hist.set_ylabel("Count")
    axis_hist.grid(True, axis="y", alpha=0.28, linestyle=":", linewidth=0.6)
    if handles:
        axis_hist.legend(handles, labels, frameon=False, loc="upper right", handler_map=combined_legend_handler, handlelength=2.2, handleheight=1.05)
    sorted_en = [np.asarray(row["sorted_amplitudes_en"], dtype=float) for row in rows if len(row["sorted_amplitudes_en"])]
    sorted_csa = [np.asarray(row["sorted_amplitudes_csa"], dtype=float) for row in rows if bool(row.get("csa_valid", False)) and len(row["sorted_amplitudes_csa"])]
    n_show = min(30, min(len(row) for row in sorted_en))
    if csa_available and sorted_csa:
        n_show = min(n_show, min(len(row) for row in sorted_csa))
    ranks = np.arange(1, n_show + 1)
    decay_series = [(np.array([row[:n_show] for row in sorted_en], dtype=float), "enufft", method_color["enufft"], rf"ENUFFT median $K^{{\star}}$={np.median(k_star):.1f}")]
    if csa_available and sorted_csa:
        finite_pairs = csa_pairs[np.isfinite(csa_pairs)]
        pairs_label = np.median(finite_pairs) if finite_pairs.size else np.nan
        decay_series.append((np.array([row[:n_show] for row in sorted_csa], dtype=float), "csa", method_color["csa"], rf"CSA median pairs={pairs_label:.1f}"))
    decay_handles = []
    decay_labels = []
    for values, key, color, label in decay_series:
        low = np.percentile(values, 10, axis=0)
        high = np.percentile(values, 90, axis=0)
        median = np.median(values, axis=0)
        hatch = method_spread_hatches[key]
        marker = method_markers[key]
        plot_hatched_spread(axis_decay, ranks, low, high, color, hatch)
        axis_decay.plot(ranks, median, color=color, lw=1.5, marker=marker, markersize=3.0, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.7, markevery=decay_markevery(ranks), label="_nolegend_", zorder=3)
        handle, decay_label = combined_line_patch_handle(color, hatch, label, marker=marker, lw=1.5, fill_alpha=0.0, hatch_color=spread_hatch_color(color))
        decay_handles.append(handle)
        decay_labels.append(decay_label)
    axis_decay.set_title("Sorted Amplitudes", pad=3)
    axis_decay.set_xlabel("Signed mode rank")
    axis_decay.set_ylabel(r"$|\hat{h}_{mn}|$ [m]")
    axis_decay.set_yscale("log")
    axis_decay.grid(True, axis="y", alpha=0.28, linestyle=":", linewidth=0.6)
    if decay_handles:
        axis_decay.legend(decay_handles, decay_labels, frameon=False, loc="upper right", handler_map=combined_legend_handler, handlelength=2.2, handleheight=1.05)
    axis_k = figure.add_subplot(dir_grid[0, 0])
    axis_de = figure.add_subplot(dir_grid[0, 1])
    axis_dc = figure.add_subplot(dir_grid[0, 2])
    caxis_dir = figure.add_subplot(dir_grid[0, 3])
    axis_vt = figure.add_subplot(var_grid[0, 0])
    axis_ve = figure.add_subplot(var_grid[0, 1])
    axis_vc = figure.add_subplot(var_grid[0, 2])
    caxis_var = figure.add_subplot(var_grid[0, 3])
    for axis in [axis_k, axis_de, axis_dc, axis_vt, axis_ve, axis_vc]:
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal")
        set_lonlat_axes(axis, dem_data, x_limits, y_limits)
    for axis in [axis_k, axis_de, axis_dc]:
        axis.set_anchor("S")
    for axis in [axis_vt, axis_ve, axis_vc]:
        axis.set_anchor("N")
    axis_k.pcolormesh(dem_data["x_km"], dem_data["y_km"], dem_data["elev"], cmap=terrain_cmap, shading="nearest", vmin=0, vmax=3600, rasterized=True, alpha=0.72)
    axis_k.add_collection(PolyCollection(polygons, facecolors="none", edgecolors="#f8f6ef", linewidths=0.35, alpha=0.82))
    for index, center in enumerate(centroids):
        if np.isfinite(full_k[index]):
            axis_k.text(center[0], center[1], f"{int(full_k[index])}", fontsize=5.3, ha="center", va="center", color="#111111")
    axis_k.set_title(r"ENUFFT $K^{\star}$", pad=2)
    axis_k.set_ylabel("Latitude")
    direction_norm = Normalize(vmin=-180.0, vmax=180.0, clip=True)
    coll_de = PolyCollection(polygons, array=np.ma.masked_invalid(full_dir_en), cmap=direction_cmap, norm=direction_norm, edgecolors="0.55", linewidth=0.25)
    coll_dc = PolyCollection(polygons, array=np.ma.masked_invalid(full_dir_csa), cmap=direction_cmap, norm=direction_norm, edgecolors="0.55", linewidth=0.25)
    axis_de.add_collection(coll_de)
    axis_dc.add_collection(coll_dc)
    domain_scale = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    add_direction_hatching(axis_de, polygons, full_dir_en, domain_scale)
    add_direction_hatching(axis_dc, polygons, full_dir_csa, domain_scale)
    axis_de.set_title("ENUFFT Direction", pad=2)
    axis_dc.set_title("CSA Direction" if csa_available else "CSA Direction skipped", pad=2)
    colorbar_dir = figure.colorbar(coll_dc, cax=caxis_dir)
    colorbar_dir.set_label("Degrees")
    all_var_parts = [np.array([row["true_var"] for row in rows], dtype=float), np.array([row["var_en"] for row in rows], dtype=float)]
    if csa_available:
        all_var_parts.append(np.array([row["var_csa"] for row in rows], dtype=float))
    all_var = np.concatenate(all_var_parts)
    all_var = all_var[np.isfinite(all_var) & (all_var > 0.0)]
    vmin = max(float(np.percentile(all_var, 5)), 1e-2)
    vmax = max(float(np.percentile(all_var, 95)), vmin * 10.0)
    var_norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    coll_vt = PolyCollection(polygons, array=np.ma.masked_invalid(full_true_var), cmap=variance_cmap, norm=var_norm, edgecolors="0.55", linewidth=0.25)
    coll_ve = PolyCollection(polygons, array=np.ma.masked_invalid(full_var_en), cmap=variance_cmap, norm=var_norm, edgecolors="0.55", linewidth=0.25)
    coll_vc = PolyCollection(polygons, array=np.ma.masked_invalid(full_var_csa), cmap=variance_cmap, norm=var_norm, edgecolors="0.55", linewidth=0.25)
    axis_vt.add_collection(coll_vt)
    axis_ve.add_collection(coll_ve)
    axis_vc.add_collection(coll_vc)
    axis_vt.set_title("Physical Variance", pad=2)
    axis_ve.set_title("ENUFFT Variance", pad=2)
    axis_vc.set_title("CSA Variance" if csa_available else "CSA Variance skipped", pad=2)
    axis_vt.set_xlabel("Longitude")
    axis_ve.set_xlabel("Longitude")
    axis_vc.set_xlabel("Longitude")
    axis_vt.set_ylabel("Latitude")
    colorbar_var = figure.colorbar(coll_vc, cax=caxis_var)
    colorbar_var.set_label(r"m$^2$")
    for axis in [axis_de, axis_dc, axis_ve, axis_vc]:
        axis.set_yticklabels([])
    for axis in [axis_k, axis_de, axis_dc]:
        axis.set_xticklabels([])
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.07, top=0.96)
    figure.canvas.draw()
    match_colorbar_height(caxis_dir, axis_dc)
    match_colorbar_height(caxis_var, axis_vc)
    output_path = alps_figure_path(f"Banerjee_2026_Enufft_Alps_Diagnostics{tag}")
    save_png_and_pdf(figure, output_path)
    plt.close(figure)
    return output_path


# Draw the pooled Alps sweep summary figure.
def plot_pooled_sweep_summary(tags, dem_data, mesh, output_tag):
    pooled = load_pooled_sweep_diagnostics(tags)
    labels = ["CSA", "Square", "Tri.", "Circle"]
    vertices = mesh["vertices"]
    triangles = mesh["triangles"]
    polygons = [vertices[triangle] for triangle in triangles]
    centroids = np.array([vertices[triangle].mean(axis=0) for triangle in triangles])
    x_limits = (float(vertices[:, 0].min()), float(vertices[:, 0].max()))
    y_limits = (float(vertices[:, 1].min()), float(vertices[:, 1].max()))
    triangle_count = len(triangles)
    figure = plt.figure(figsize=(170 * mm, 142 * mm))
    outer = figure.add_gridspec(3, 1, height_ratios=[0.86, 1.0, 1.0], hspace=0.30)
    top_grid = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    dir_grid = outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.12)
    var_grid = outer[2].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.12)
    axis_hist = figure.add_subplot(top_grid[0, 0])
    axis_decay = figure.add_subplot(top_grid[0, 1])
    rel_values = []
    for label in labels:
        values = np.asarray(pooled[label]["rel_rmse"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            rel_values.append(values)
    hist_upper = max(1.05, float(np.nanpercentile(np.concatenate(rel_values), 95)) * 1.25) if rel_values else 1.05
    bins = np.linspace(0.0, hist_upper, 24)
    hist_handles = []
    hist_labels = []
    for label in labels:
        values = np.asarray(pooled[label]["rel_rmse"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        handle, hist_label = plot_hatched_histogram(axis_hist, values, bins, summary_method_colors[label], summary_method_hatches[label], f"{label} {np.nanmedian(values):.2f}")
        hist_handles.append(handle)
        hist_labels.append(hist_label)
    axis_hist.set_title("Pooled Relative RMSE", pad=3)
    axis_hist.set_xlabel(r"RMSE/$\sigma_h$")
    axis_hist.set_ylabel("Count")
    axis_hist.grid(True, axis="y", alpha=0.28, linestyle=":", linewidth=0.6)
    if hist_handles:
        axis_hist.legend(hist_handles, hist_labels, frameon=False, ncol=2, loc="upper right", handler_map=combined_legend_handler, handlelength=2.2, handleheight=1.05)
    decay_handles = []
    decay_labels = []
    for label in labels:
        decay = pooled_decay_curve(pooled[label]["spectra"])
        if decay is None:
            continue
        ranks, median, low, high = decay
        color = summary_method_colors[label]
        hatch = summary_method_spread_hatches[label]
        marker = summary_method_markers[label]
        plot_hatched_spread(axis_decay, ranks, low, high, color, hatch)
        axis_decay.plot(ranks, median, color=color, lw=1.55, marker=marker, markersize=3.0, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.7, markevery=decay_markevery(ranks), label="_nolegend_", zorder=3)
        handle, decay_label = combined_line_patch_handle(color, hatch, label, marker=marker, lw=1.55, fill_alpha=0.0, hatch_color=spread_hatch_color(color))
        decay_handles.append(handle)
        decay_labels.append(decay_label)
    axis_decay.set_title("Pooled Sorted Amplitudes", pad=3)
    axis_decay.set_xlabel("Signed mode rank")
    axis_decay.set_ylabel(r"$|\hat{h}_{mn}|$ [m]")
    axis_decay.set_yscale("log")
    axis_decay.grid(True, axis="y", alpha=0.28, linestyle=":", linewidth=0.6)
    if decay_handles:
        axis_decay.legend(decay_handles, decay_labels, frameon=False, ncol=2, loc="upper right", handler_map=combined_legend_handler, handlelength=2.2, handleheight=1.05)
    axis_k = figure.add_subplot(dir_grid[0, 0])
    axis_dir_en = figure.add_subplot(dir_grid[0, 1])
    axis_dir_csa = figure.add_subplot(dir_grid[0, 2])
    caxis_dir = figure.add_subplot(dir_grid[0, 3])
    axis_phys = figure.add_subplot(var_grid[0, 0])
    axis_var_en = figure.add_subplot(var_grid[0, 1])
    axis_var_csa = figure.add_subplot(var_grid[0, 2])
    caxis_var = figure.add_subplot(var_grid[0, 3])
    for axis in [axis_k, axis_dir_en, axis_dir_csa, axis_phys, axis_var_en, axis_var_csa]:
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal")
        set_lonlat_axes(axis, dem_data, x_limits, y_limits)
    for axis in [axis_k, axis_dir_en, axis_dir_csa]:
        axis.set_anchor("S")
    for axis in [axis_phys, axis_var_en, axis_var_csa]:
        axis.set_anchor("N")
    pooled_k = median_map_values(pooled["ENUFFT"], "K_star", triangle_count, mesh_centroids=centroids)
    pooled_dir_en = median_map_values(pooled["ENUFFT"], "direction", triangle_count, circular=True, mesh_centroids=centroids)
    pooled_dir_csa = median_map_values(pooled["CSA"], "direction", triangle_count, circular=True, mesh_centroids=centroids)
    pooled_var_en = median_map_values(pooled["ENUFFT"], "variance", triangle_count, mesh_centroids=centroids)
    pooled_var_csa = median_map_values(pooled["CSA"], "variance", triangle_count, mesh_centroids=centroids)
    physical_var = median_map_values(pooled["Physical"], "variance", triangle_count, mesh_centroids=centroids)
    axis_k.pcolormesh(dem_data["x_km"], dem_data["y_km"], dem_data["elev"], cmap=terrain_cmap, shading="nearest", vmin=0, vmax=3600, rasterized=True, alpha=0.72)
    axis_k.add_collection(PolyCollection(polygons, facecolors="none", edgecolors="#f8f6ef", linewidths=0.35, alpha=0.82))
    for index, center in enumerate(centroids):
        if np.isfinite(pooled_k[index]):
            axis_k.text(center[0], center[1], f"{int(round(pooled_k[index]))}", fontsize=5.2, ha="center", va="center", color="#111111")
    axis_k.set_title(r"Pooled ENUFFT $K^{\star}$", pad=2)
    axis_k.set_ylabel("Latitude")
    direction_norm = Normalize(vmin=-180.0, vmax=180.0, clip=True)
    direction_coll_en = PolyCollection(polygons, array=np.ma.masked_invalid(pooled_dir_en), cmap=direction_cmap, norm=direction_norm, edgecolors="0.52", linewidth=0.25)
    direction_coll_csa = PolyCollection(polygons, array=np.ma.masked_invalid(pooled_dir_csa), cmap=direction_cmap, norm=direction_norm, edgecolors="0.52", linewidth=0.25)
    axis_dir_en.add_collection(direction_coll_en)
    axis_dir_csa.add_collection(direction_coll_csa)
    domain_scale = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    add_direction_hatching(axis_dir_en, polygons, pooled_dir_en, domain_scale)
    add_direction_hatching(axis_dir_csa, polygons, pooled_dir_csa, domain_scale)
    axis_dir_en.set_title("Pooled ENUFFT Direction", pad=2)
    axis_dir_csa.set_title("Pooled CSA Direction", pad=2)
    all_var = np.concatenate([physical_var[np.isfinite(physical_var) & (physical_var > 0.0)], pooled_var_en[np.isfinite(pooled_var_en) & (pooled_var_en > 0.0)], pooled_var_csa[np.isfinite(pooled_var_csa) & (pooled_var_csa > 0.0)]])
    if all_var.size == 0:
        all_var = np.array([1.0])
    vmin = max(float(np.percentile(all_var, 5)), 1e-2)
    vmax = max(float(np.percentile(all_var, 95)), vmin * 10.0)
    var_norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    phys_coll = PolyCollection(polygons, array=np.ma.masked_invalid(physical_var), cmap=variance_cmap, norm=var_norm, edgecolors="0.55", linewidth=0.25)
    var_coll_en = PolyCollection(polygons, array=np.ma.masked_invalid(pooled_var_en), cmap=variance_cmap, norm=var_norm, edgecolors="0.55", linewidth=0.25)
    var_coll_csa = PolyCollection(polygons, array=np.ma.masked_invalid(pooled_var_csa), cmap=variance_cmap, norm=var_norm, edgecolors="0.55", linewidth=0.25)
    axis_phys.add_collection(phys_coll)
    axis_var_en.add_collection(var_coll_en)
    axis_var_csa.add_collection(var_coll_csa)
    axis_phys.set_title("Physical Variance", pad=2)
    axis_var_en.set_title("Pooled ENUFFT Variance", pad=2)
    axis_var_csa.set_title("Pooled CSA Variance", pad=2)
    axis_phys.set_xlabel("Longitude")
    axis_var_en.set_xlabel("Longitude")
    axis_var_csa.set_xlabel("Longitude")
    axis_phys.set_ylabel("Latitude")
    for axis in [axis_dir_en, axis_dir_csa, axis_var_en, axis_var_csa]:
        axis.set_yticklabels([])
    for axis in [axis_k, axis_dir_en, axis_dir_csa]:
        axis.set_xticklabels([])
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.955)
    figure.canvas.draw()
    match_colorbar_height(caxis_dir, axis_dir_csa)
    match_colorbar_height(caxis_var, axis_var_csa)
    colorbar_dir = figure.colorbar(direction_coll_csa, cax=caxis_dir)
    colorbar_dir.set_label("Degrees")
    colorbar_var = figure.colorbar(var_coll_csa, cax=caxis_var)
    colorbar_var.set_label(r"m$^2$")
    output_path = alps_figure_path(f"Banerjee_2026_Enufft_Alps_SweepSummary{output_tag}")
    save_png_and_pdf(figure, output_path)
    plt.close(figure)
    return output_path


# Replace missing grid values with a neutral finite terrain level.
def finite_field(z_values):
    z_values = np.asarray(z_values, dtype=float).copy()
    bad = ~np.isfinite(z_values)
    if np.any(bad):
        z_values[bad] = np.nanmedian(z_values)
    return z_values


# Orient one grid so y increases for Matplotlib.
def orient_y_ascending(x_values, y_values, z_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    z_values = finite_field(z_values)
    if y_values[0] > y_values[-1]:
        y_values = y_values[::-1]
        z_values = z_values[::-1, :]
    return x_values, y_values, z_values


# Thin a raster to the requested x rendering density.
def thin_grid(x_values, y_values, z_values, target_nx):
    step = max(1, int(np.ceil(len(x_values) / target_nx)))
    return x_values[::step], y_values[::step], z_values[::step, ::step], step


# Build a deterministic mesh object for the preprocessing-composite panel.
def regular_icon_like_triangles_for_viz(x_values, y_values, target_dx_km):
    mesh = build_alps_mesh({"x_km": x_values, "y_km": y_values}, "r2b5", target_dx_km)
    return SimpleNamespace(points=mesh["vertices"], simplices=mesh["triangles"], nsimplex=len(mesh["triangles"]), mesh=mesh)


# Remove the local mean within each triangular cell.
def deplane_on_triangles(x_values, y_values, z_values, triangulation):
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    simplex_id = find_mesh_triangle_ids(points, triangulation.mesh) if hasattr(triangulation, "mesh") else triangulation.find_simplex(points)
    values = z_values.ravel()
    residual = np.full(values.shape, np.nan, dtype=float)
    valid = (simplex_id >= 0) & np.isfinite(values)
    counts = np.bincount(simplex_id[valid], minlength=triangulation.nsimplex)
    sums = np.bincount(simplex_id[valid], weights=values[valid], minlength=triangulation.nsimplex)
    means = np.full(triangulation.nsimplex, np.nan, dtype=float)
    has_points = counts > 0
    means[has_points] = sums[has_points] / counts[has_points]
    residual[valid] = values[valid] - means[simplex_id[valid]]
    return residual.reshape(z_values.shape), means


# Combine elevation colors with hillshade for raised surfaces.
def surface_colors(z_values, cmap, norm, residual=False):
    light = LightSource(azdeg=314, altdeg=42)
    base = cmap(norm(z_values))[..., :3]
    shade = light.hillshade(z_values, vert_exag=0.75 if residual else 0.48)
    rgb = base * (0.72 + 0.35 * shade[..., None])
    rgb = np.clip(rgb + 0.02, 0.0, 1.0)
    alpha = np.ones((*z_values.shape, 1))
    return np.concatenate([rgb, alpha], axis=-1)


# Fade relief color at raster edges.
def edge_fade_mask(shape, margin_fraction=0.08):
    ny_value, nx_value = shape
    yy = np.minimum(np.arange(ny_value), np.arange(ny_value)[::-1])[:, None]
    xx = np.minimum(np.arange(nx_value), np.arange(nx_value)[::-1])[None, :]
    margin = max(1.0, min(nx_value, ny_value) * margin_fraction)
    distance = np.minimum(xx, yy) / margin
    return np.clip(distance, 0.0, 1.0) ** 0.75


# Build the shaded hero-map RGB image.
def map_colors(z_values, cmap, norm):
    light = LightSource(azdeg=315, altdeg=46)
    shade = light.hillshade(z_values, vert_exag=0.44)
    background = np.ones((*z_values.shape, 3))
    color = cmap(norm(z_values))[..., :3]
    colored_relief = np.clip(color * (0.76 + 0.34 * shade[..., None]) + 0.025, 0.0, 1.0)
    alpha_elev = np.clip(z_values / 2300.0, 0.0, 1.0) ** 0.72
    alpha = np.clip(alpha_elev * edge_fade_mask(z_values.shape), 0.0, 0.92)
    return background * (1.0 - alpha[..., None]) + colored_relief * alpha[..., None]


# Add a shallow Earth-curvature sag to terrain panels.
def earth_curvature(x_values, y_values):
    x_mid = 0.5 * (float(x_values[0]) + float(x_values[-1]))
    y_mid = 0.5 * (float(y_values[0]) + float(y_values[-1]))
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return -((x_grid - x_mid) ** 2 + (y_grid - y_mid) ** 2) / (2.0 * earth_radius_km)


# Sample triangular mesh edges along a plotted terrain surface.
def mesh_segments_3d(triangulation, x_values, y_values, z_values, z_scale):
    interp = RegularGridInterpolator((y_values, x_values), z_values, bounds_error=False, fill_value=np.nan)
    segments = []
    x_mid = 0.5 * (float(x_values[0]) + float(x_values[-1]))
    y_mid = 0.5 * (float(y_values[0]) + float(y_values[-1]))
    for simplex in triangulation.simplices:
        points = triangulation.points[simplex]
        cycle = [points[0], points[1], points[2], points[0]]
        for p0, p1 in zip(cycle[:-1], cycle[1:]):
            sample = np.linspace(0.0, 1.0, 10)
            xs = p0[0] + sample * (p1[0] - p0[0])
            ys = p0[1] + sample * (p1[1] - p0[1])
            zs = interp(np.column_stack([ys, xs]))
            if np.all(np.isfinite(zs)):
                curve = -((xs - x_mid) ** 2 + (ys - y_mid) ** 2) / (2.0 * earth_radius_km)
                segments.append(np.column_stack([xs, ys, zs * z_scale + curve + 0.8]))
    return segments


# Load the DEM archive and prepare preprocessing stages for rendering.
def load_processing_stages(target_hires_nx, target_stage_nx):
    data = np.load(require_preprocessed_dem(), allow_pickle=False)
    lon_ref = float(data["lon_ref"])
    lat_ref = float(data["lat_ref"])
    x_hires, y_hires, z_hires = orient_y_ascending(data["x_km_hires"], data["y_km_hires"], data["elev_hires"])
    x_stage, y_stage, z_block = orient_y_ascending(data["x_km"], data["y_km"], data["elev_block_avg"])
    _, _, z_gaussian = orient_y_ascending(data["x_km"], data["y_km"], data["elev"])
    triangulation = regular_icon_like_triangles_for_viz(x_stage, y_stage, alps_r2b5_target_dx_km)
    z_deplane, tri_means = deplane_on_triangles(x_stage, y_stage, z_gaussian, triangulation)
    x_hires, y_hires, z_hires, hires_step = thin_grid(x_hires, y_hires, z_hires, target_hires_nx)
    x_plot, y_plot, z_block_plot, stage_step = thin_grid(x_stage, y_stage, z_block, target_stage_nx)
    _, _, z_gaussian_plot, _ = thin_grid(x_stage, y_stage, z_gaussian, target_stage_nx)
    _, _, z_deplane_plot, _ = thin_grid(x_stage, y_stage, z_deplane, target_stage_nx)
    lon0, lat0 = xy_to_lonlat(x_stage[0], y_stage[0], lon_ref, lat_ref)
    lon1, lat1 = xy_to_lonlat(x_stage[-1], y_stage[-1], lon_ref, lat_ref)
    print(f"    High-resolution plotting stride: {hires_step}", flush=True)
    print(f"    High-resolution source step: {int(data['hires_subsample'])} arc-sec", flush=True)
    print(f"    Processed-stage plotting stride: {stage_step}", flush=True)
    print(f"    R2B5 effective mesh size: {icon_r2b5_effective_dx_km:.1f} km", flush=True)
    print(f"    Alps proxy target spacing: {alps_r2b5_target_dx_km:.1f} km", flush=True)
    print(f"    R2B5 local triangles: {triangulation.nsimplex}", flush=True)
    return {"lon_ref": lon_ref, "lat_ref": lat_ref, "extent_deg": [float(lon0), float(lon1), float(lat0), float(lat1)], "block_size": int(data["block_size"]), "smooth_km": float(data["smooth_km"]), "hires_subsample": int(data["hires_subsample"]), "hires_plot_arcsec": int(data["hires_subsample"]) * hires_step, "tri": triangulation, "tri_means": tri_means, "hires": {"x": x_hires, "y": y_hires, "z": z_hires}, "block": {"x": x_plot, "y": y_plot, "z": z_block_plot}, "gaussian": {"x": x_plot, "y": y_plot, "z": z_gaussian_plot}, "deplane": {"x": x_plot, "y": y_plot, "z": z_deplane_plot}}


# Normalize Natural Earth geometry to coordinate rings.
def geometry_rings(geometry):
    if geometry is None:
        return []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return coordinates
    if geometry_type == "MultiPolygon":
        rings = []
        for polygon in coordinates:
            rings.extend(polygon)
        return rings
    return []


# Test whether a boundary ring overlaps the plotted geographic extent.
def ring_overlaps_extent(ring, extent):
    lon0, lon1, lat0, lat1 = extent
    points = np.asarray(ring, dtype=float)
    if points.size == 0:
        return False
    return np.nanmax(points[:, 0]) >= lon0 and np.nanmin(points[:, 0]) <= lon1 and np.nanmax(points[:, 1]) >= lat0 and np.nanmin(points[:, 1]) <= lat1


# Fetch or read cached Natural Earth country outlines.
def load_country_boundaries(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ne_10m_admin_0_countries.geojson"
    if not cache_path.exists():
        try:
            with urllib.request.urlopen(natural_earth_url, timeout=12) as response:
                cache_path.write_bytes(response.read())
            print("    Natural Earth country boundaries cached", flush=True)
        except Exception as exc:
            print(f"    Country boundaries unavailable: {exc}", flush=True)
            return []
    try:
        payload = json.loads(cache_path.read_text())
    except Exception as exc:
        print(f"    Country boundary cache unreadable: {exc}", flush=True)
        return []
    boundaries = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        name = props.get("ADMIN") or props.get("NAME") or props.get("NAME_EN")
        if name not in country_names:
            continue
        for ring in geometry_rings(feature.get("geometry")):
            boundaries.append((name, ring))
    return boundaries


# Draw country borders on the hero map.
def draw_country_boundaries(axis, boundaries, extent):
    for _, ring in boundaries:
        if not ring_overlaps_extent(ring, extent):
            continue
        points = np.asarray(ring, dtype=float)
        axis.plot(points[:, 0], points[:, 1], color=line_color, lw=0.42, alpha=0.58, solid_capstyle="round", zorder=4)


# Add latitude and longitude guide lines to the hero map.
def add_hero_graticule(axis, extent):
    lon_ticks = [6, 8, 10, 12, 14]
    lat_ticks = [45, 46, 47, 48]
    lon0, lon1, lat0, lat1 = extent
    for lon_value in lon_ticks:
        axis.axvline(lon_value, color=line_color, lw=0.32, alpha=0.22, zorder=2)
        if lon0 <= lon_value <= lon1:
            axis.text(lon_value, lat0 + 0.08, f"{lon_value}E", color=text_color, fontsize=7.0, ha="center", va="bottom", zorder=7)
    for lat_value in lat_ticks:
        axis.axhline(lat_value, color=line_color, lw=0.32, alpha=0.22, zorder=2)
        if lat0 <= lat_value <= lat1:
            axis.text(lon0 + 0.10, lat_value, f"{lat_value}N", color=text_color, fontsize=7.0, ha="left", va="center", zorder=7)


# Add a compact scale bar to the hero map.
def add_scale_bar(axis, extent):
    lon0, lon1, lat0, _ = extent
    lat = lat0 + 0.26
    length_km = 100.0
    half_length_km = 50.0
    deg_per_km = np.rad2deg(1.0 / (earth_radius_km * np.cos(np.deg2rad(lat))))
    length_deg = length_km * deg_per_km
    half_length_deg = half_length_km * deg_per_km
    x0 = lon1 - length_deg - 0.45
    y0 = lat
    axis.plot([x0, x0 + length_deg], [y0, y0], color=text_color, lw=1.0, zorder=8)
    for x_value in [x0, x0 + half_length_deg, x0 + length_deg]:
        axis.plot([x_value, x_value], [y0 - 0.025, y0 + 0.025], color=text_color, lw=1.0, zorder=8)
    axis.text(x0, y0 + 0.045, "0", color=text_color, fontsize=6.5, ha="center", va="bottom", zorder=8)
    axis.text(x0 + half_length_deg, y0 + 0.045, "50", color=text_color, fontsize=6.5, ha="center", va="bottom", zorder=8)
    axis.text(x0 + length_deg, y0 + 0.045, "100 km", color=text_color, fontsize=6.5, ha="center", va="bottom", zorder=8)


# Strip the 3D axes for terrain-object rendering.
def clean_3d_axis(axis, stage, zlim, zoom=1.20, pad_km=55.0):
    x_values, y_values = stage["x"], stage["y"]
    axis.set_facecolor((1, 1, 1, 0))
    axis.patch.set_alpha(0.0)
    axis.set_axis_off()
    axis.set_proj_type("ortho")
    axis.view_init(elev=34, azim=-126, roll=0)
    axis.set_xlim(float(x_values[0]) - pad_km, float(x_values[-1]) + pad_km)
    axis.set_ylim(float(y_values[0]) - 0.65 * pad_km, float(y_values[-1]) + 0.65 * pad_km)
    axis.set_zlim(zlim[0], zlim[1])
    try:
        axis.set_box_aspect((1.85, 1.0, 0.34), zoom=zoom)
    except TypeError:
        axis.set_box_aspect((1.85, 1.0, 0.34))
    for subaxis in (axis.xaxis, axis.yaxis, axis.zaxis):
        subaxis.pane.fill = False
        subaxis.pane.set_edgecolor((1, 1, 1, 0))
        subaxis._axinfo["grid"]["linewidth"] = 0
        subaxis._axinfo["axisline"]["linewidth"] = 0
        subaxis._axinfo["tick"]["inward_factor"] = 0
        subaxis._axinfo["tick"]["outward_factor"] = 0


# Project subtle contour bands onto the curved base plane.
def add_floor_contours(axis, stage, cmap, norm, z_base, residual=False):
    x_values, y_values, z_values = stage["x"], stage["y"], stage["z"]
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    levels = np.linspace(norm.vmin, norm.vmax, 18) if residual else [250, 500, 1000, 1500, 2000, 3000, 4000]
    axis.contourf(x_grid, y_grid, z_values, levels=levels, zdir="z", offset=z_base, cmap=cmap, norm=norm, alpha=0.16, antialiased=True)
    axis.contour(x_grid, y_grid, z_values, levels=levels, zdir="z", offset=z_base + 0.15, colors=line_color, linewidths=0.18, alpha=0.20)


# Add optional graticule lines to one 3D terrain panel.
def add_latlon_graticule(axis, data, stage, z_base):
    x0, x1 = float(stage["x"][0]), float(stage["x"][-1])
    y0, y1 = float(stage["y"][0]), float(stage["y"][-1])
    for lon_value in [6, 8, 10, 12, 14]:
        x_value, _ = lonlat_to_xy(lon_value, data["extent_deg"][2], data["lon_ref"], data["lat_ref"])
        if x0 <= x_value <= x1:
            axis.plot([x_value, x_value], [y0, y1], [z_base + 0.35, z_base + 0.35], color=line_color, lw=0.32, alpha=0.30)
            axis.text(x_value, y0 - 25.0, z_base + 0.50, f"{lon_value}E", color=muted_color, fontsize=5.2, ha="center", va="top")
    for lat_value in [45, 46, 47, 48]:
        _, y_value = lonlat_to_xy(data["extent_deg"][0], lat_value, data["lon_ref"], data["lat_ref"])
        if y0 <= y_value <= y1:
            axis.plot([x0, x1], [y_value, y_value], [z_base + 0.35, z_base + 0.35], color=line_color, lw=0.32, alpha=0.30)
            axis.text(x0 - 35.0, y_value, z_base + 0.50, f"{lat_value}N", color=muted_color, fontsize=5.2, ha="right", va="center")


# Draw one raised terrain preprocessing stage.
def draw_terrain_on_axis(axis, data, stage, cmap, norm, z_scale, residual=False, mesh=False, graticule=False, zoom=1.20):
    x_values, y_values, z_values = stage["x"], stage["y"], stage["z"]
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    curve = earth_curvature(x_values, y_values)
    z_plot = z_values * z_scale + curve
    if residual:
        z_abs = max(18.0, float(np.nanpercentile(np.abs(z_plot), 99.5)) * 1.18)
        zlim = (float(np.nanmin(curve)) - z_abs * 0.55, z_abs)
        z_base = -z_abs
    else:
        zlim = (float(np.nanmin(curve)) - 6.0, max(84.0, float(np.nanmax(z_plot)) * 1.04))
        z_base = float(np.nanmin(curve)) - 5.0
    clean_3d_axis(axis, stage, zlim, zoom=zoom)
    colors = surface_colors(z_values, cmap, norm, residual=residual)
    axis.plot_surface(x_grid, y_grid, z_plot, rstride=1, cstride=1, facecolors=colors, linewidth=0.0, antialiased=False, shade=False, rasterized=True)
    add_floor_contours(axis, stage, cmap, norm, z_base, residual=residual)
    if graticule:
        add_latlon_graticule(axis, data, stage, z_base)
    if mesh:
        segments = mesh_segments_3d(data["tri"], x_values, y_values, z_values, z_scale)
        collection = Line3DCollection(segments, colors="#f8f4e8", linewidths=0.22, alpha=0.66)
        axis.add_collection3d(collection)
    return axis


# Render a transparent cropped terrain-stage PNG.
def render_terrain_image(data, stage, cmap, norm, z_scale, residual=False, mesh=False, graticule=False, zoom=1.0):
    import io
    figure = plt.figure(figsize=(6.2, 3.7), dpi=300)
    figure.patch.set_alpha(0.0)
    axis = figure.add_axes([0, 0, 1, 1], projection="3d")
    draw_terrain_on_axis(axis, data, stage, cmap, norm, z_scale, residual=residual, mesh=mesh, graticule=graticule, zoom=zoom)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    alpha_box = image.getchannel("A").getbbox()
    if alpha_box is not None:
        pad = 18
        left = max(0, alpha_box[0] - pad)
        upper = max(0, alpha_box[1] - pad)
        right = min(image.size[0], alpha_box[2] + pad)
        lower = min(image.size[1], alpha_box[3] + pad)
        image = image.crop((left, upper, right, lower))
    return np.asarray(image, dtype=float) / 255.0


# Add a compact panel title.
def add_plain_label(figure, rect, title, inside=False):
    x0, y0, _, height = rect
    y_title = y0 + height - 0.020 if inside else y0 + height + 0.018
    vertical_alignment = "top" if inside else "bottom"
    figure.text(x0, y_title, title, color=text_color, fontsize=9.0, fontweight="normal", ha="left", va=vertical_alignment)


# Place a pre-rendered terrain image into the composite.
def add_rendered_panel(figure, rect, image):
    axis = figure.add_axes(rect)
    axis.imshow(image, interpolation="lanczos")
    axis.set_axis_off()
    axis.set_facecolor((1, 1, 1, 0))
    axis.patch.set_alpha(0.0)
    return axis


# Build the full-width geographic hero map.
def add_hero_map(figure, rect, data, terrain_norm, boundaries):
    axis = figure.add_axes(rect)
    axis.set_facecolor((1, 1, 1, 0))
    axis.patch.set_alpha(0.0)
    stage = data["hires"]
    lon0, lat0 = xy_to_lonlat(stage["x"][0], stage["y"][0], data["lon_ref"], data["lat_ref"])
    lon1, lat1 = xy_to_lonlat(stage["x"][-1], stage["y"][-1], data["lon_ref"], data["lat_ref"])
    extent = [float(lon0), float(lon1), float(lat0), float(lat1)]
    axis.imshow(map_colors(stage["z"], viz_terrain_cmap, terrain_norm), origin="lower", extent=extent, interpolation="bilinear", aspect="auto", zorder=1)
    add_hero_graticule(axis, extent)
    draw_country_boundaries(axis, boundaries, extent)
    add_scale_bar(axis, extent)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    for label, lon_value, lat_value in region_labels:
        axis.text(lon_value, lat_value, label, color=text_color, fontsize=6.6, fontweight="normal", ha="center", va="center", alpha=0.78, zorder=5)
    axis.text(0.018, 0.955, f"NASA SRTM Terrain ({data['hires_subsample']} arc-sec source, {data['hires_plot_arcsec']} arc-sec raster)", transform=axis.transAxes, ha="left", va="top", color=text_color, fontsize=9.0, fontweight="normal", zorder=8)
    return axis


# Add the horizontal elevation and residual colorbars.
def add_colorbars(figure, terrain_norm, residual_norm):
    caxis = figure.add_axes([0.070, 0.662, 0.500, 0.008])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=terrain_norm, cmap=viz_terrain_cmap), cax=caxis, orientation="horizontal")
    colorbar.ax.set_title("Elevation [m]", color=text_color, fontsize=7.0, pad=3.0)
    colorbar.outline.set_visible(False)
    colorbar.set_ticks([0, 2000, 4000])
    colorbar.ax.tick_params(labelsize=6.4, colors=text_color, pad=0.5, length=2.0)
    caxis.set_facecolor((1, 1, 1, 0))
    residual_axis = figure.add_axes([0.715, 0.662, 0.210, 0.008])
    residual_bar = figure.colorbar(plt.cm.ScalarMappable(norm=residual_norm, cmap=residual_cmap), cax=residual_axis, orientation="horizontal")
    residual_bar.ax.set_title("R2B5 perturbation [m]", color=text_color, fontsize=7.0, pad=3.0)
    residual_bar.outline.set_visible(False)
    residual_bar.set_ticks([residual_norm.vmin, 0.0, residual_norm.vmax])
    residual_bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.0f}"))
    residual_bar.ax.tick_params(labelsize=6.4, colors=text_color, pad=0.5)
    residual_axis.set_facecolor((1, 1, 1, 0))


# Build the Alps preprocessing-composite figure.
def build_viz_figure(data, output_base):
    terrain_norm = Normalize(vmin=-100.0, vmax=4600.0)
    residual_limit = float(np.nanpercentile(np.abs(data["deplane"]["z"]), 98.5))
    residual_limit = max(500.0, min(residual_limit, 1900.0))
    residual_norm = TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit)
    print("    Rendering cropped 3D stage panels", flush=True)
    block_img = render_terrain_image(data, data["block"], viz_terrain_cmap, terrain_norm, z_scale=0.019, zoom=0.96)
    gaussian_img = render_terrain_image(data, data["gaussian"], viz_terrain_cmap, terrain_norm, z_scale=0.019, zoom=0.96)
    deplane_img = render_terrain_image(data, data["deplane"], residual_cmap, residual_norm, z_scale=0.017, residual=True, mesh=True, graticule=False, zoom=0.98)
    figure = plt.figure(figsize=(170 * mm, 120 * mm))
    figure.patch.set_alpha(0.0)
    boundaries = load_country_boundaries(output_base.parent)
    add_hero_map(figure, [0.012, 0.040, 0.976, 0.590], data, terrain_norm, boundaries)
    block_rect = [0.020, 0.690, 0.312, 0.255]
    smooth_rect = [0.344, 0.690, 0.312, 0.255]
    main_rect = [0.668, 0.690, 0.318, 0.255]
    add_plain_label(figure, block_rect, "Block Averaged (30 arc-sec)")
    add_rendered_panel(figure, block_rect, block_img)
    add_plain_label(figure, smooth_rect, "Gaussian Low-pass (5 km)")
    add_rendered_panel(figure, smooth_rect, gaussian_img)
    add_plain_label(figure, main_rect, "R2B5 Deplaned (80 km cells)")
    add_rendered_panel(figure, main_rect, deplane_img)
    add_colorbars(figure, terrain_norm, residual_norm)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_base.with_suffix(".pdf"), dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output_base.with_suffix(".png")


# Infer an ICON mesh preset from a run tag.
def mesh_name_from_tag(tag):
    match = re.match(r"^_(r2b4|r2b5)_", str(tag))
    return match.group(1) if match else None


# Return true when a parsed CSV field is a usable scalar.
def has_summary_value(value):
    if value is None:
        return False
    if isinstance(value, float) and not np.isfinite(value):
        return False
    return str(value).strip().lower() not in ("", "nan")


# Infer the analysis mesh from the summary CSV or tag.
def mesh_name_for_summary(summary, tag):
    if has_summary_value(summary.get("mesh_name")):
        return normalize_mesh_name(summary["mesh_name"])
    tag_mesh = mesh_name_from_tag(tag)
    if tag_mesh is not None:
        return tag_mesh
    return "regular"


# Build the plotting case dictionary from one summary row.
def case_for_summary(summary, tag):
    mesh_name = mesh_name_for_summary(summary, tag)
    cell_size = float(summary.get("cell_size_km", fallback_cell_size)) if has_summary_value(summary.get("cell_size_km")) else fallback_cell_size
    n_modes = int(summary["n_modes"]) if has_summary_value(summary.get("n_modes")) else None
    return build_alps_case({"mesh_name": mesh_name, "cell_size_km": cell_size, "n_modes": n_modes})


# Load or reuse the mesh needed by one tagged output.
def mesh_for_case(dem_data, case, mesh_cache):
    key = (case.get("mesh_name") or "regular", float(case["cell_size_km"]))
    if key not in mesh_cache:
        mesh_cache[key] = build_alps_mesh(dem_data, case.get("mesh_name"), case["cell_size_km"])
    return mesh_cache[key]


# Run the requested Alps plotting workflow.
def main():
    apply_alps_style()
    if terminal_only_requested(sys.argv):
        print_terminal_summary()
        return
    rendered = []
    print(f"[1] Loading Alpine preprocessing archive: {preprocessed_dem}", flush=True)
    data = load_processing_stages(target_hires_nx, target_stage_nx)
    output_base = figure_dir / "Banerjee_2026_Enufft_Alps_Viz"
    print("[2] Building transparent 3D preprocessing composite", flush=True)
    rendered.append(build_viz_figure(data, output_base))
    tags = discover_tags()
    summaries = {}
    if tags:
        dem_data = load_alps_dem()
        summaries = {tag: read_csv_rows(paths_for_tag(tag)[1])[0] for tag in tags}
        mesh_cache = {}
        groups = {}
        for tag in tags:
            case = case_for_summary(summaries[tag], tag)
            key = (case.get("mesh_name") or "regular", int(case["n_modes"]), float(case["cell_size_km"]))
            groups.setdefault(key, {"case": case, "tags": []})["tags"].append(tag)
        for group in groups.values():
            case = group["case"]
            mesh = mesh_for_case(dem_data, case, mesh_cache)
            output_tag = make_alps_sweep_tag(case)
            if len(group["tags"]) == 1 and "_first" in group["tags"][0]:
                output_tag += group["tags"][0][group["tags"][0].index("_first"):]
            rendered.append(plot_pooled_sweep_summary(group["tags"], dem_data, mesh, output_tag))
            print(f"Rendered sweep summary{output_tag}", flush=True)
    if not rendered:
        raise RuntimeError("No complete Alps outputs found to plot")
    for path in rendered:
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
