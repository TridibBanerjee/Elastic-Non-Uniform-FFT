# Plot_Mono.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script reads the monochromatic ENUFFT and CSA CSV tables and builds the six-panel comparison figure.

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from scipy.spatial import cKDTree

from Code_Mono import build_case, build_mesh_data, classify_regions, get_analysis_mask
from Module_Plot_Template import mm, apply_james_style, figure_output_path, save_png_and_pdf


csv_dir = Path("./csv")
summary_csv = csv_dir / "Banerjee_2026_Enufft_Mono_Summary.csv"
method_colors = {"Square": "#222222", "Tri.": "#7a5a34", "Circle": "#3e6f76", "CSA": "#b35c44"}


# Read the scalar Mono summary table.
def read_summary_rows():
    rows = []
    with summary_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


# Build the representative geometry used in the top row.
def representative_setup():
    case = build_case({"mode_limit": 8, "sample_count": 420, "triangle_orientation": 32.0, "center_offset": 0.08, "uniformity": 0.58, "mask_condition": "circle", "weight_type": "voronoi"})
    mesh_data = build_mesh_data(case)
    return case, mesh_data


# Convert one mode pair into a physical direction angle.
def compute_signal_angle_deg(m_mode, n_mode):
    return float(np.degrees(np.arctan2(float(n_mode), float(m_mode))) % 180.0)


# Sample the four-region partition on a regular plotting grid.
def build_region_grid(case, mesh_data, grid_res=220):
    edges = np.linspace(0.0, case["domain_length_km"], grid_res + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_x, grid_y = np.meshgrid(centers, centers, indexing="xy")
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    region_grid = classify_regions(grid_points, mesh_data["triangle_vertices"], mesh_data["center"]).reshape(grid_res, grid_res)
    return edges, edges, region_grid


# Intersect a ray from the centre to a vertex with the square boundary.
def ray_square_endpoint(origin, target, length):
    direction = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    candidates = []
    if abs(direction[0]) > 1e-12:
        candidates.extend([(0.0 - origin[0]) / direction[0], (length - origin[0]) / direction[0]])
    if abs(direction[1]) > 1e-12:
        candidates.extend([(0.0 - origin[1]) / direction[1], (length - origin[1]) / direction[1]])
    hits = []
    for value in candidates:
        if value <= 0.0:
            continue
        hit = origin + value * direction
        if -1e-12 <= hit[0] <= length + 1e-12 and -1e-12 <= hit[1] <= length + 1e-12:
            hits.append(np.clip(hit, 0.0, length))
    if not hits:
        return np.asarray(target, dtype=float)
    distances = [np.linalg.norm(hit - origin) for hit in hits]
    return hits[int(np.argmax(distances))]


# Estimate label positions from the rasterized region map.
def estimate_region_centroids(case, mesh_data, region_grid):
    grid_res = region_grid.shape[0]
    edges = np.linspace(0.0, case["domain_length_km"], grid_res + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_x, grid_y = np.meshgrid(centers, centers, indexing="xy")
    centroids = {}
    for region_id in range(4):
        mask = region_grid == region_id
        if np.any(mask):
            centroids[region_id] = np.array([np.mean(grid_x[mask]), np.mean(grid_y[mask])], dtype=float)
        else:
            centroids[region_id] = np.asarray(mesh_data["center"], dtype=float)
    return centroids


# Draw clipped stripe segments for one modal region.
def add_region_stripes(axis, case, mesh_data, region_id, angle_deg, spacing, line_color, line_width):
    direction = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    diagonal = np.sqrt(2.0) * case["domain_length_km"]
    offsets = np.arange(-diagonal, diagonal + spacing, spacing)
    t_values = np.linspace(-diagonal, diagonal, 900)
    for offset in offsets:
        points = mesh_data["center"][None, :] + offset * normal[None, :] + t_values[:, None] * direction[None, :]
        inside_domain = (points[:, 0] >= 0.0) & (points[:, 0] <= case["domain_length_km"]) & (points[:, 1] >= 0.0) & (points[:, 1] <= case["domain_length_km"])
        if not np.any(inside_domain):
            continue
        local_points = points[inside_domain]
        region_mask = classify_regions(local_points, mesh_data["triangle_vertices"], mesh_data["center"]) == region_id
        if not np.any(region_mask):
            continue
        x_local = local_points[:, 0]
        y_local = local_points[:, 1]
        hit_index = np.flatnonzero(region_mask)
        for segment in np.split(hit_index, np.where(np.diff(hit_index) > 1)[0] + 1):
            if segment.size < 2:
                continue
            axis.plot(x_local[segment], y_local[segment], color=line_color, linewidth=line_width, alpha=0.65, solid_capstyle="round", zorder=3)


# Style one square-domain axis.
def style_domain_axis(axis, case, show_ylabel):
    tick_positions = [0.0, 0.5 * case["domain_length_km"], case["domain_length_km"]]
    axis.set_xlim(0.0, case["domain_length_km"])
    axis.set_ylim(0.0, case["domain_length_km"])
    axis.set_aspect("equal")
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(["0", "0.5", "1.0"])
    axis.set_yticks(tick_positions)
    axis.set_yticklabels(["0", "0.5", "1.0"] if show_ylabel else [])
    axis.set_xlabel(r"$x/L$")
    if show_ylabel:
        axis.set_ylabel(r"$y/L$")
    axis.grid(True, axis="both", alpha=0.18, linestyle=":", linewidth=0.55)
    axis.set_facecolor("none")
    axis.margins(x=0.0, y=0.0)


# Draw one small panel title on a consistent baseline.
def set_panel_title(axis, title):
    axis.set_title("")
    axis.text(0.5, 1.015, title, transform=axis.transAxes, ha="center", va="bottom", fontsize=8.2, color="#222222", clip_on=False)


# Draw the piecewise modal-domain panel.
def plot_piecewise_domain_map(axis, case, mesh_data):
    region_colors = ["#f3eadc", "#dde8e4", "#ece1d7", "#dbe0ec"]
    stripe_colors = ["#70584b", "#476e67", "#7e6557", "#4c6480"]
    x_edges, y_edges, region_grid = build_region_grid(case, mesh_data, grid_res=240)
    centroids = estimate_region_centroids(case, mesh_data, region_grid)
    axis.pcolormesh(x_edges, y_edges, region_grid, cmap=ListedColormap(region_colors), shading="flat", vmin=-0.5, vmax=3.5, alpha=0.95, rasterized=True, zorder=0)
    axis.add_patch(Rectangle((0.0, 0.0), case["domain_length_km"], case["domain_length_km"], facecolor="none", edgecolor="#333333", linewidth=1.0, zorder=5))
    for vertex in mesh_data["triangle_vertices"]:
        endpoint = ray_square_endpoint(mesh_data["center"], vertex, case["domain_length_km"])
        axis.plot([mesh_data["center"][0], endpoint[0]], [mesh_data["center"][1], endpoint[1]], color="#222222", linewidth=0.75, linestyle=(0, (3, 2)), alpha=0.62, zorder=4)
    axis.add_patch(Polygon(mesh_data["triangle_vertices"], closed=True, facecolor="none", edgecolor="#222222", linewidth=1.2, zorder=5))
    for region_id in range(4):
        angle_deg = compute_signal_angle_deg(mesh_data["region_m"][region_id], mesh_data["region_n"][region_id])
        add_region_stripes(axis, case, mesh_data, region_id, angle_deg, spacing=0.075 * case["domain_length_km"], line_color=stripe_colors[region_id], line_width=0.9)
        axis.text(centroids[region_id][0], centroids[region_id][1], rf"$({int(mesh_data['region_m'][region_id])}, {int(mesh_data['region_n'][region_id])})$", ha="center", va="center", fontsize=6.8, color="#222222", bbox=dict(boxstyle="square,pad=0.14", facecolor="white", edgecolor="white", alpha=0.82), zorder=6)
    axis.scatter(mesh_data["center"][0], mesh_data["center"][1], s=18, color="#222222", zorder=6)
    set_panel_title(axis, "Piecewise modal domain")


# Draw the representative support geometry panel.
def plot_representative_setup(axis, case, mesh_data):
    base_center = np.array([0.5 * case["domain_length_km"], 0.5 * case["domain_length_km"]], dtype=float)
    _, mask_info = get_analysis_mask(case, mesh_data)
    circle_radius = float(mask_info["radius"])
    axis.add_patch(Rectangle((0.0, 0.0), case["domain_length_km"], case["domain_length_km"], facecolor="#f7f5f0", edgecolor="#333333", linewidth=1.0, zorder=1))
    axis.add_patch(Circle(tuple(base_center), circle_radius, facecolor="none", edgecolor="#457b78", linestyle=(0, (5, 2)), linewidth=1.3, zorder=3))
    axis.add_patch(Polygon(mesh_data["triangle_vertices"], closed=True, facecolor="#e4c6a1", edgecolor="#7d5f3d", linewidth=1.15, alpha=0.88, zorder=4))
    axis.scatter(base_center[0], base_center[1], s=16, color="#3d5a80", zorder=5)
    axis.scatter(mesh_data["center"][0], mesh_data["center"][1], s=18, color="#222222", zorder=6)
    axis.annotate("", xy=(mesh_data["center"][0], mesh_data["center"][1]), xytext=(base_center[0], base_center[1]), arrowprops=dict(arrowstyle="-|>", color="#3d5a80", linewidth=1.2, shrinkA=0.0, shrinkB=0.0), zorder=5)
    arc_radius = 0.20 * case["domain_length_km"]
    axis.add_patch(Arc(tuple(mesh_data["center"]), 2.0 * arc_radius, 2.0 * arc_radius, angle=0.0, theta1=0.0, theta2=float(case["triangle_orientation"]), color="#c0392b", linewidth=1.2, zorder=6))
    theta = np.deg2rad(case["triangle_orientation"])
    axis.plot([mesh_data["center"][0], mesh_data["center"][0] + arc_radius], [mesh_data["center"][1], mesh_data["center"][1]], color="#c0392b", linewidth=1.0, alpha=0.75, zorder=6)
    axis.plot([mesh_data["center"][0], mesh_data["center"][0] + arc_radius * np.cos(theta)], [mesh_data["center"][1], mesh_data["center"][1] + arc_radius * np.sin(theta)], color="#c0392b", linewidth=1.0, alpha=0.75, zorder=6)
    axis.text(0.55 * case["domain_length_km"], 0.45 * case["domain_length_km"], rf"$\Delta y={case['center_offset']:.2f}L$", fontsize=6.8, color="#3d5a80", ha="left", va="center", bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="white", alpha=0.96), zorder=7)
    axis.text(0.67 * case["domain_length_km"], 0.70 * case["domain_length_km"], rf"$\theta={case['triangle_orientation']:.0f}^\circ$", fontsize=6.8, color="#c0392b", ha="left", va="center", bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="white", alpha=0.86))
    set_panel_title(axis, "Representative geometry setup")


# Build a nearest-sample area tile map for the DEM panel.
def build_voronoi_tile_map(points, length, grid_res=135):
    edges = np.linspace(0.0, length, grid_res + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_x, grid_y = np.meshgrid(centers, centers, indexing="xy")
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    _, owner = cKDTree(points).query(grid_points, k=1)
    counts = np.bincount(owner, minlength=len(points)).astype(float)
    area_weights = counts * (length * length / len(grid_points))
    area_weights = np.maximum(area_weights, 1e-12)
    tile_values = (area_weights / np.mean(area_weights))[owner].reshape(grid_res, grid_res)
    return edges, edges, tile_values, area_weights


# Draw the DEM cloud and its area-proxy tiling.
def plot_voronoi_panel(axis, case, mesh_data):
    tile_cmap = LinearSegmentedColormap.from_list("mono_voronoi_tiles", ["#f7f2e8", "#d7cdc0", "#7a8f94"])
    x_edges, y_edges, tile_values, area_weights = build_voronoi_tile_map(mesh_data["dem_points"], case["domain_length_km"])
    mesh = axis.pcolormesh(x_edges, y_edges, tile_values, cmap=tile_cmap, shading="flat", alpha=0.96, rasterized=True, zorder=0)
    axis.scatter(mesh_data["dem_points"][:, 0], mesh_data["dem_points"][:, 1], s=4.4, color="#1f1f1f", alpha=0.66, linewidths=0.0, zorder=3)
    axis.add_patch(Rectangle((0.0, 0.0), case["domain_length_km"], case["domain_length_km"], facecolor="none", edgecolor="#333333", linewidth=1.0, zorder=5))
    axis.add_patch(Polygon(mesh_data["triangle_vertices"], closed=True, facecolor="none", edgecolor="white", linewidth=1.1, zorder=5))
    mesh.set_clim(np.percentile(area_weights / np.mean(area_weights), 5), np.percentile(area_weights / np.mean(area_weights), 95))
    axis.text(0.03, 0.96, rf"$Q={len(mesh_data['dem_points'])}$", transform=axis.transAxes, ha="left", va="top", fontsize=6.8, color="#333333", bbox=dict(boxstyle="square,pad=0.10", facecolor="white", edgecolor="white", alpha=0.82))
    set_panel_title(axis, "DEM Voronoi tiling")


# Group a scalar metric by ENUFFT support and CSA.
def collect_metric_by_method(summary_rows, enufft_key, csa_key):
    return {"Square": np.asarray([float(row[enufft_key]) for row in summary_rows if row["mask"] == "square"], dtype=float), "Tri.": np.asarray([float(row[enufft_key]) for row in summary_rows if row["mask"] == "triangle"], dtype=float), "Circle": np.asarray([float(row[enufft_key]) for row in summary_rows if row["mask"] == "circle"], dtype=float), "CSA": np.asarray([float(row[csa_key]) for row in summary_rows], dtype=float)}


# Compute the median and percentile band for one vector.
def compute_summary_band(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.median(finite)), float(np.percentile(finite, 10.0)), float(np.percentile(finite, 90.0))


# Build a readable upper bound for a plotted scale.
def nice_upper_bound(value):
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    for step in [1.0, 1.5, 2.0, 2.5, 5.0, 10.0]:
        if step * magnitude >= value:
            return float(step * magnitude)
    return float(10.0 * magnitude)


# Build the linear radar tick positions.
def build_radial_ticks(r_max, integer_ticks):
    if integer_ticks:
        ticks = np.linspace(0.0, r_max, 5)[1:]
        ticks = np.unique(np.maximum(1.0, np.round(ticks)))
        return [float(tick) for tick in ticks]
    return [float(tick) for tick in np.linspace(0.0, r_max, 5)[1:]]


# Build the logarithmic radar tick positions.
def build_log_radial_ticks(r_floor, r_max):
    if r_max <= r_floor:
        return [r_floor]
    min_exp = int(np.floor(np.log10(r_floor)))
    max_exp = int(np.ceil(np.log10(r_max)))
    ticks = [10.0 ** exponent for exponent in range(min_exp, max_exp + 1)]
    ticks = [tick for tick in ticks if r_floor <= tick <= r_max]
    if not ticks or ticks[0] > r_floor * 1.25:
        ticks = [r_floor] + ticks
    return [float(tick) for tick in ticks]


# Format one linear radar tick label.
def format_tick_label(value, integer_ticks):
    if integer_ticks:
        return f"{int(round(value))}"
    if value >= 1000.0:
        return f"{value / 1000.0:.1f}k"
    if value >= 10.0:
        return f"{value:.0f}"
    if value >= 1.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


# Format one logarithmic radar tick label.
def format_log_tick_label(value):
    if value >= 1000.0:
        return f"{value / 1000.0:.0f}k" if value % 1000.0 == 0.0 else f"{value / 1000.0:.1f}k"
    if value >= 1.0:
        return f"{value:.0f}"
    if value >= 0.1:
        return f"{value:.1f}"
    return f"{value:.2f}"


# Format one radar median annotation.
def format_median_annotation(value, integer_ticks, radial_scale):
    if integer_ticks:
        return f"{int(round(value))}"
    if radial_scale == "log":
        return format_log_tick_label(value)
    if value >= 10.0:
        return f"{value:.0f}"
    return f"{value:.1f}"


# Draw the pooled direction-error boxplot.
def plot_metric_boxplot(axis, summary_rows):
    grouped = collect_metric_by_method(summary_rows, "max_mode_direction_deviation_deg_enufft", "max_mode_direction_deviation_deg_csa")
    labels = ["Square", "Tri.", "Circle", "CSA"]
    data = [grouped[label] for label in labels]
    colors = [method_colors[label] for label in labels]
    boxplot = axis.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, whis=(10, 90), widths=0.58)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for whisker, color in zip(boxplot["whiskers"], np.repeat(colors, 2)):
        whisker.set_color(color)
        whisker.set_linewidth(1.0)
    for cap, color in zip(boxplot["caps"], np.repeat(colors, 2)):
        cap.set_color(color)
        cap.set_linewidth(1.0)
    for median in boxplot["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)
    finite_values = np.concatenate([values[np.isfinite(values)] for values in data if np.any(np.isfinite(values))])
    y_max = nice_upper_bound(1.05 * max(float(np.max(finite_values)), 1.0)) if finite_values.size else 1.0
    axis.set_ylim(0.0, y_max)
    axis.set_ylabel(r"$|\theta_{\mathrm{true}}-\theta_{\mathrm{peak}}|$")
    axis.grid(True, axis="y", alpha=0.30, linestyle=":", linewidth=0.6)
    set_panel_title(axis, r"$|\theta_{\mathrm{true}}-\theta_{\mathrm{peak}}|$")
    for x_position, values, color in zip(range(1, len(labels) + 1), data, colors):
        if values.size == 0:
            continue
        median_value = float(np.median(values))
        axis.text(x_position, min(y_max * 0.97, median_value + 0.045 * y_max), f"{median_value:.2f}", ha="center", va="bottom", fontsize=6.3, color=color, bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="white", alpha=0.84))


# Draw one radar panel from pooled metric summaries.
def plot_metric_radar(axis, summary_rows, enufft_key, csa_key, title, line_color, fixed_r_max=None, integer_ticks=False, radial_scale="linear", tick_values=None):
    grouped = collect_metric_by_method(summary_rows, enufft_key, csa_key)
    labels = list(grouped.keys())
    medians = []
    p10 = []
    p90 = []
    for label in labels:
        median_value, p10_value, p90_value = compute_summary_band(grouped[label])
        medians.append(median_value)
        p10.append(p10_value)
        p90.append(p90_value)
    theta = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    median_closed = np.concatenate([medians, medians[:1]])
    median_points = np.asarray(medians, dtype=float)
    p10_closed = np.concatenate([p10, p10[:1]])
    p90_closed = np.concatenate([p90, p90[:1]])
    axis.set_theta_offset(np.pi / 4.0)
    axis.set_theta_direction(-1)
    axis.set_thetagrids(np.degrees(theta), labels, fontsize=6.4, color="#222222")
    axis.set_rlabel_position(135)
    axis.yaxis.grid(True, alpha=0.28, linestyle=":", linewidth=0.6)
    axis.xaxis.grid(True, alpha=0.18, linestyle="-", linewidth=0.5)
    axis.spines["polar"].set_color("#808080")
    axis.spines["polar"].set_linewidth(0.8)
    axis.set_facecolor("none")
    if radial_scale == "log":
        positive_values = [value for value in p10_closed if value > 0.0] + [value for value in median_closed if value > 0.0]
        floor = min(positive_values) if positive_values else 1e-2
        floor = max(10.0 ** np.floor(np.log10(floor)), 1e-2)
        r_max = fixed_r_max if fixed_r_max is not None else nice_upper_bound(1.15 * max(max(p90_closed), floor))
        median_closed = np.maximum(median_closed, floor)
        median_points = np.maximum(median_points, floor)
        p10_closed = np.maximum(p10_closed, floor)
        p90_closed = np.maximum(p90_closed, floor)
        axis.set_yscale("log")
        axis.set_ylim(floor, r_max)
        axis.set_yticks(build_log_radial_ticks(floor, r_max))
        axis.set_yticklabels([])
        low_ring_value = float(min([value for value in p10_closed if value > 0.0], default=floor))
        high_ring_value = float(max(p90_closed))
    else:
        r_max = fixed_r_max if fixed_r_max is not None else nice_upper_bound(1.05 * max(p90_closed))
        ticks = tick_values if tick_values is not None else build_radial_ticks(r_max, integer_ticks)
        axis.set_ylim(0.0, r_max)
        axis.set_yticks(ticks)
        axis.set_yticklabels([])
        low_ring_value = float(min(p10_closed))
        high_ring_value = float(max(p90_closed))
    band_theta = np.concatenate([theta_closed, theta_closed[::-1]])
    band_radius = np.concatenate([p90_closed, p10_closed[::-1]])
    axis.fill(band_theta, band_radius, color=line_color, alpha=0.16, linewidth=0.0, zorder=1)
    axis.plot(theta_closed, p10_closed, color=line_color, linewidth=0.8, linestyle=":", alpha=0.45, zorder=2)
    axis.plot(theta_closed, p90_closed, color=line_color, linewidth=0.8, linestyle=":", alpha=0.45, zorder=2)
    axis.plot(theta_closed, median_closed, color=line_color, linewidth=1.65, zorder=3)
    axis.scatter(theta, median_points, s=18, color=line_color, edgecolors="white", linewidths=0.5, zorder=4)
    for angle, value in zip(theta, median_points):
        text_radius = value * (1.10 if radial_scale == "log" else 1.06) if value > 0.0 else 0.04 * r_max
        if radial_scale != "log":
            text_radius = min(text_radius, 0.97 * r_max)
        annotation = format_median_annotation(value, integer_ticks, radial_scale)
        axis.text(angle, text_radius, annotation, fontsize=6.2, color=line_color, ha="center", va="center", bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="white", alpha=0.84), zorder=5)
    label_angle = np.deg2rad(118.0)
    low_label = format_log_tick_label(low_ring_value) if radial_scale == "log" else format_tick_label(low_ring_value, integer_ticks)
    high_label = format_log_tick_label(high_ring_value) if radial_scale == "log" else format_tick_label(high_ring_value, integer_ticks)
    axis.text(label_angle, low_ring_value, low_label, fontsize=6.1, color="#666666", ha="left", va="bottom")
    axis.text(label_angle, high_ring_value, high_label, fontsize=6.1, color="#666666", ha="left", va="bottom")
    set_panel_title(axis, title)


# Group normalized retained-mode fractions by method.
def collect_kstar_fraction_by_method(summary_rows):
    labels = ["Square", "Tri.", "Circle", "CSA"]
    grouped = {label: [] for label in labels}
    for row in summary_rows:
        mode_limit = max(int(row["mode_limit"]), 1)
        mask_label = "Tri." if row["mask"] == "triangle" else row["mask"].capitalize()
        grouped[mask_label].append(float(row["k_star_enufft"]) / mode_limit)
        grouped["CSA"].append(float(row["csa_signed_modes_selected"]) / (2.0 * mode_limit))
    return {label: np.asarray(values, dtype=float) for label, values in grouped.items()}


# Draw the retained-mode fraction radar panel.
def plot_kstar_fraction(axis, summary_rows):
    labels = ["Square", "Tri.", "Circle", "CSA"]
    grouped = collect_kstar_fraction_by_method(summary_rows)
    medians = []
    p10 = []
    p90 = []
    for label in labels:
        median_value, p10_value, p90_value = compute_summary_band(grouped[label])
        medians.append(median_value)
        p10.append(p10_value)
        p90.append(p90_value)
    theta = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    median_closed = np.concatenate([medians, medians[:1]])
    p10_closed = np.concatenate([p10, p10[:1]])
    p90_closed = np.concatenate([p90, p90[:1]])
    axis.set_theta_offset(np.pi / 4.0)
    axis.set_theta_direction(-1)
    axis.set_thetagrids(np.degrees(theta), labels, fontsize=6.4, color="#222222")
    axis.set_ylim(0.0, 1.08)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels([])
    axis.yaxis.grid(True, alpha=0.28, linestyle=":", linewidth=0.6)
    axis.xaxis.grid(True, alpha=0.18, linestyle="-", linewidth=0.5)
    axis.spines["polar"].set_color("#808080")
    axis.spines["polar"].set_linewidth(0.8)
    axis.set_facecolor("none")
    band_theta = np.concatenate([theta_closed, theta_closed[::-1]])
    band_radius = np.concatenate([p90_closed, p10_closed[::-1]])
    axis.fill(band_theta, band_radius, color="#8a8a8a", alpha=0.15, linewidth=0.0, zorder=1)
    axis.plot(theta_closed, p10_closed, color="#666666", linewidth=0.75, linestyle=":", alpha=0.55, zorder=2)
    axis.plot(theta_closed, p90_closed, color="#666666", linewidth=0.75, linestyle=":", alpha=0.55, zorder=2)
    axis.plot(theta_closed, median_closed, color="#222222", linewidth=1.55, zorder=3)
    for angle, label, value in zip(theta, labels, medians):
        axis.scatter([angle], [value], s=32, color=method_colors[label], edgecolors="white", linewidths=0.6, zorder=4)
        axis.text(angle, min(1.04, value + 0.05), f"{value:.2f}", fontsize=6.2, color=method_colors[label], ha="center", va="center", bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="white", alpha=0.82), zorder=5)
    low_ring_value = float(min(p10_closed))
    high_ring_value = float(max(p90_closed))
    label_angle = np.deg2rad(118.0)
    axis.text(label_angle, low_ring_value, f"{low_ring_value:.2f}", fontsize=6.1, color="#666666", ha="left", va="bottom")
    axis.text(label_angle, high_ring_value, f"{high_ring_value:.2f}", fontsize=6.1, color="#666666", ha="left", va="bottom")
    set_panel_title(axis, r"$K^{\star}/K_{\mathrm{max}}$")


# Read the CSV outputs, build the figure, and export PNG plus PDF.
def main():
    print("Loading Mono CSV outputs")
    summary_rows = read_summary_rows()
    representative_case, representative_mesh = representative_setup()
    apply_james_style()
    print("Building figure from CSV data")
    figure = plt.figure(figsize=(170.0 * mm, 118.0 * mm))
    figure.patch.set_alpha(0.0)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.98], hspace=0.44, wspace=0.07)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[0, 2])
    axis_d = figure.add_subplot(grid[1, 0])
    axis_e = figure.add_subplot(grid[1, 1], projection="polar")
    axis_f = figure.add_subplot(grid[1, 2], projection="polar")
    plot_piecewise_domain_map(axis_a, representative_case, representative_mesh)
    plot_representative_setup(axis_b, representative_case, representative_mesh)
    plot_voronoi_panel(axis_c, representative_case, representative_mesh)
    style_domain_axis(axis_a, representative_case, True)
    style_domain_axis(axis_b, representative_case, False)
    style_domain_axis(axis_c, representative_case, False)
    plot_metric_boxplot(axis_d, summary_rows)
    plot_metric_radar(axis_e, summary_rows, "max_peak_amplitude_deviation_enufft", "max_peak_amplitude_deviation_csa", r"$|A/2 - A_{\mathrm{peak}}|$", "#355f7c", radial_scale="log")
    plot_kstar_fraction(axis_f, summary_rows)
    figure.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.065)
    output_path = figure_output_path("Banerjee_2026_Enufft_Mono")
    save_png_and_pdf(figure, output_path)
    print(f"Figure saved to {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
