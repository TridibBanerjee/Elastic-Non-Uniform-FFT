# Plot_Mountain.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script reads the mountain-wave EMS CSV tables and rebuilds the figure.

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from Module_Plot_Template import apply_james_style, mm


csv_dir = Path("./csv")
figure_dir = Path("./figures")
terrain_csv = csv_dir / "Banerjee_2026_Enufft_Mountain_Terrain.csv"
modes_csv = csv_dir / "Banerjee_2026_Enufft_Mountain_Modes.csv"
summary_csv = csv_dir / "Banerjee_2026_Enufft_Mountain_Summary.csv"
figure_output = figure_dir / "Banerjee_2026_Enufft_Mountain"
figure_width_mm = 170
figure_height_mm = 128
plot_limit_km = 100.0
color_levels = 10
text_color = "#242424"
hatch_color = "#454545"
topography_colors = ["#ffffff", "#fff8f5", "#fff0eb", "#fbd5ca", "#f6b5a4", "#ed9285", "#e36f62", "#cf5055", "#b43b4a"]
wind_colors = ["#255c99", "#5f95c8", "#a8cce2", "#e7f0f3", "#ffffff", "#fff0eb", "#f6b5a4", "#e36f62", "#b43b4a"]
mode_colors = ["#ffffff", "#d6e6e0", "#9ec4b9", "#5c9589", "#1f5f57"]
loss_colors = ["#ffffff", "#fff8f5", "#fff0eb", "#fbd5ca", "#f6b5a4", "#ed9285", "#e36f62", "#cf5055", "#b43b4a"]


# Apply the shared style and the EMS-specific figure settings.
def apply_mountain_style():
    apply_james_style()
    plt.rcParams.update({
        "legend.fontsize": 7,
        "axes.edgecolor": text_color,
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "hatch.color": hatch_color,
        "hatch.linewidth": 0.30,
        "savefig.dpi": 600,
    })


# Return evenly spaced plotting levels.
def linrange(start, stop, length):
    return np.linspace(start, stop, length)


# Use symmetric color limits around zero.
def symmetric_limits(values):
    peak = float(np.nanmax(np.abs(values)))
    if not np.isfinite(peak) or peak == 0.0:
        peak = 1.0
    return -peak, peak


# Return finite plotting limits with a nonzero range.
def finite_minmax(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


# Build a binned colormap and norm from endpoint colors.
def categorical_gradient(colors, levels, name):
    bin_count = len(levels) - 1
    if len(colors) == bin_count:
        cmap = ListedColormap(colors, name=name)
    else:
        continuous = LinearSegmentedColormap.from_list(name, colors)
        cmap = continuous.resampled(bin_count)
    cmap.set_bad("white")
    return cmap, BoundaryNorm(levels, cmap.N)


# Convert cell centers to plotting cell edges.
def cell_edges(centers):
    centers = np.asarray(centers, dtype=float)
    midpoints = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


# Return the segment where one diagonal line crosses one rectangular cell.
def clip_line_to_box(slope, intercept, x0, x1, y0, y1):
    points = []
    for x_value in (x0, x1):
        y_value = slope * x_value + intercept
        if y0 <= y_value <= y1:
            points.append((x_value, y_value))
    for y_value in (y0, y1):
        x_value = (y_value - intercept) / slope
        if x0 <= x_value <= x1:
            points.append((x_value, y_value))
    unique = []
    for point in points:
        if not any(np.allclose(point, old_point, atol=1.0e-9) for old_point in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    distances = {}
    for i_index, point_i in enumerate(unique):
        for j_index, point_j in enumerate(unique[i_index + 1:], start=i_index + 1):
            distances[(i_index, j_index)] = (point_i[0] - point_j[0]) ** 2 + (point_i[1] - point_j[1]) ** 2
    i_index, j_index = max(distances, key=distances.get)
    return unique[i_index], unique[j_index]


# Build diagonal hatch segments clipped to selected cells.
def hatch_segments_for_mask(x_edges, y_edges, mask, slope, spacing_km):
    x0_all = float(x_edges[0])
    x1_all = float(x_edges[-1])
    y0_all = float(y_edges[0])
    y1_all = float(y_edges[-1])
    b_min = min(y0_all - slope * x0_all, y0_all - slope * x1_all, y1_all - slope * x0_all, y1_all - slope * x1_all)
    b_max = max(y0_all - slope * x0_all, y0_all - slope * x1_all, y1_all - slope * x0_all, y1_all - slope * x1_all)
    offsets = np.arange(b_min - spacing_km, b_max + spacing_km, spacing_km)
    segments = []
    for y_index in range(mask.shape[0]):
        for x_index in range(mask.shape[1]):
            if not mask[y_index, x_index]:
                continue
            x0 = float(x_edges[x_index])
            x1 = float(x_edges[x_index + 1])
            y0 = float(y_edges[y_index])
            y1 = float(y_edges[y_index + 1])
            for intercept in offsets:
                segment = clip_line_to_box(slope, intercept, x0, x1, y0, y1)
                if segment is not None:
                    segments.append(segment)
    return segments


# Add directional hatching to a signed wind field.
def add_signed_hatching(axis, x_km, y_km, values, zero_band):
    if zero_band <= 0.0:
        return
    x_edges = cell_edges(x_km)
    y_edges = cell_edges(y_km)
    positive_mask = np.isfinite(values) & (values > zero_band)
    negative_mask = np.isfinite(values) & (values < -zero_band)
    positive_segments = hatch_segments_for_mask(x_edges, y_edges, positive_mask, slope=1.0, spacing_km=17.0)
    negative_segments = hatch_segments_for_mask(x_edges, y_edges, negative_mask, slope=-1.0, spacing_km=17.0)
    for segments in (positive_segments, negative_segments):
        if segments:
            collection = LineCollection(segments, colors=hatch_color, linewidths=0.42, alpha=0.82, capstyle="butt", zorder=3)
            axis.add_collection(collection)
    legend_handles = [
        Patch(facecolor="white", edgecolor=hatch_color, hatch="////", linewidth=0.45, label=r"$v > 0$"),
        Patch(facecolor="white", edgecolor=hatch_color, hatch="\\\\\\\\", linewidth=0.45, label=r"$v < 0$"),
    ]
    legend = axis.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.965, 0.982), borderaxespad=0.0, handlelength=1.75, handleheight=1.05, handletextpad=0.55, borderpad=0.32, labelspacing=0.34, frameon=True, fancybox=False, framealpha=0.92, edgecolor="#d6d6d6", facecolor="white", fontsize=7.0)
    legend.get_frame().set_linewidth(0.45)


# Format compact colorbar tick labels.
def simple_labels(values, digits=1):
    return [f"{round(float(value), digits):g}" for value in values]


# Apply common map-axis limits and labels.
def configure_map_axis(axis, title):
    axis.set_title(title, color=text_color, pad=4)
    axis.set_xlabel(r"$x$ [km]")
    axis.set_ylabel(r"$y$ [km]")
    axis.set_xlim(-plot_limit_km, plot_limit_km)
    axis.set_ylim(-plot_limit_km, plot_limit_km)
    axis.set_xticks([-100, -50, 0, 50, 100])
    axis.set_yticks([-100, -50, 0, 50, 100])
    axis.grid(False)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor((1, 1, 1, 0))
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)


# Draw one map panel with a compact vertical colorbar.
def add_image_panel(figure, axis, color_axis, x_km, y_km, values, title, cmap, norm, ticks, ticklabels=None):
    image = axis.imshow(values, origin="lower", extent=(x_km.min(), x_km.max(), y_km.min(), y_km.max()), cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    configure_map_axis(axis, title)
    colorbar = figure.colorbar(image, cax=color_axis, ticks=ticks, spacing="proportional")
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=8, colors=text_color, length=2.4, pad=1.5)
    if ticklabels is not None:
        colorbar.ax.set_yticklabels(ticklabels)
    color_axis.set_facecolor((1, 1, 1, 0))


# Read the scalar summary row from the Mountain summary CSV.
def read_summary_row():
    with summary_csv.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    return {key: float(value) if key != "wave_modes" and key != "selected_cells" else int(value) for key, value in row.items()}


# Read the fine-grid topography table and rebuild its arrays.
def read_terrain_data():
    rows = []
    with terrain_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    x_size = max(int(row["x_index"]) for row in rows) + 1
    y_size = max(int(row["y_index"]) for row in rows) + 1
    x_km = np.zeros(x_size, dtype=float)
    y_km = np.zeros(y_size, dtype=float)
    full_topography_km = np.zeros((y_size, x_size), dtype=float)
    for row in rows:
        x_index = int(row["x_index"])
        y_index = int(row["y_index"])
        x_km[x_index] = float(row["x_km"])
        y_km[y_index] = float(row["y_km"])
        full_topography_km[y_index, x_index] = float(row["full_topography_km"])
    return x_km, y_km, full_topography_km


# Read the final surface-field table and rebuild its arrays.
def read_mode_data():
    rows = []
    with modes_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    x_size = max(int(row["column_index"]) for row in rows) + 1
    y_size = max(int(row["row_index"]) for row in rows) + 1
    x_km = np.zeros(x_size, dtype=float)
    y_km = np.zeros(y_size, dtype=float)
    v_surface = np.zeros((y_size, x_size), dtype=float)
    mode_count = np.zeros((y_size, x_size), dtype=float)
    power_fraction = np.zeros((y_size, x_size), dtype=float)
    for row in rows:
        x_index = int(row["column_index"])
        y_index = int(row["row_index"])
        x_km[x_index] = float(row["x_km"])
        y_km[y_index] = float(row["y_km"])
        v_surface[y_index, x_index] = float(row["v_surface_ms"])
        mode_count[y_index, x_index] = float(row["launch_mode_count"])
        power_fraction[y_index, x_index] = float(row["launch_power_fraction"])
    return x_km, y_km, v_surface, mode_count, power_fraction


# Draw the fine-grid mountain-wave orography as a 3D panel.
def add_topography_panel(axis, x_km, y_km, topography_km, setup):
    xx_km, yy_km = np.meshgrid(x_km, y_km)
    topo_max = max(float(np.nanmax(topography_km)), float(setup["h0"]) / 1000.0)
    topo_levels = linrange(0.0, topo_max, color_levels)
    topo_cmap, topo_norm = categorical_gradient(topography_colors, topo_levels, "ems_topography")
    active = topography_km > 1.0e-8
    rows = np.where(np.any(active, axis=1))[0]
    cols = np.where(np.any(active, axis=0))[0]
    row0 = max(0, rows[0] - 1)
    row1 = min(topography_km.shape[0], rows[-1] + 2)
    col0 = max(0, cols[0] - 1)
    col1 = min(topography_km.shape[1], cols[-1] + 2)
    xx_km = xx_km[row0:row1, col0:col1]
    yy_km = yy_km[row0:row1, col0:col1]
    topography_plot = np.ma.masked_less_equal(topography_km[row0:row1, col0:col1], 1.0e-8)
    axis.plot_surface(xx_km, yy_km, topography_plot, cmap=topo_cmap, norm=topo_norm, edgecolor="none", linewidth=0.0, antialiased=True, shade=False, rasterized=True, rcount=topography_plot.shape[0], ccount=topography_plot.shape[1])
    axis.set_title("")
    axis.text2D(0.50, 1.01, "Orography", transform=axis.transAxes, ha="center", va="top", color=text_color, fontsize=9)
    axis.set_xlabel(r"$x$ [km]", labelpad=-1)
    axis.set_ylabel(r"$y$ [km]", labelpad=-1)
    axis.set_zlabel("")
    axis.text2D(0.935, 0.52, r"$z$ [km]", transform=axis.transAxes, rotation=90, ha="center", va="center", color=text_color)
    axis.set_xlim(-50.0, 50.0)
    axis.set_ylim(-50.0, 50.0)
    axis.set_zlim(0, topo_max)
    axis.set_xticks([-50, 0, 50])
    axis.set_yticks([-50, 0, 50])
    axis.set_zticks([0, 0.5, 1.0])
    axis.tick_params(pad=0, labelsize=8, colors=text_color)
    axis.set_proj_type("ortho")
    axis.view_init(elev=25, azim=-40)
    axis.set_box_aspect((1.15, 1.0, 0.52), zoom=1.22)
    axis.grid(False)
    axis.set_facecolor((1, 1, 1, 0))
    for local_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        local_axis.pane.fill = False
        local_axis.pane.set_edgecolor((1, 1, 1, 0))


# Save the mountain-wave EMS figure as PNG and PDF.
def save_figure(figure, output_prefix):
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", transparent=True)


# Read the CSV tables and build the mountain-wave EMS figure.
def main():
    print("Loading mountain-wave EMS CSV outputs")
    setup = read_summary_row()
    x_topo_km, y_topo_km, full_topography_km = read_terrain_data()
    x_km, y_km, v_surface, mode_count, power_fraction = read_mode_data()
    apply_mountain_style()
    power_loss_percent = np.where(power_fraction > 0.0, (1.0 - power_fraction) * 100.0, np.nan)
    wind_min, wind_max = symmetric_limits(v_surface)
    wind_ticks = linrange(wind_min, wind_max, color_levels)
    wind_zero_band = float(np.min(wind_ticks[wind_ticks > 0.0]))
    wind_cmap, wind_norm = categorical_gradient(wind_colors, wind_ticks, "ems_wind")
    mode_ticks = np.arange(0, int(setup["wave_modes"]) + 1)
    mode_boundaries = np.arange(-0.5, int(setup["wave_modes"]) + 1.5, 1.0)
    mode_cmap, mode_norm = categorical_gradient(mode_colors, mode_boundaries, "ems_modes")
    loss_min, loss_max = finite_minmax(power_loss_percent)
    loss_ticks = linrange(min(0.0, loss_min), loss_max, color_levels)
    loss_cmap, loss_norm = categorical_gradient(loss_colors, loss_ticks, "ems_power_loss")
    print("Building figure from CSV data")
    figure = plt.figure(figsize=(figure_width_mm * mm, figure_height_mm * mm))
    figure.patch.set_alpha(0.0)
    topography_axis = figure.add_axes([0.030, 0.510, 0.425, 0.405], projection="3d")
    wind_axis = figure.add_axes([0.505, 0.555, 0.325, 0.335])
    wind_color_axis = figure.add_axes([0.505 + 0.325 + 0.008, 0.555, 0.020, 0.335])
    mode_axis = figure.add_axes([0.085, 0.105, 0.325, 0.335])
    mode_color_axis = figure.add_axes([0.085 + 0.325 + 0.008, 0.105, 0.020, 0.335])
    loss_axis = figure.add_axes([0.505, 0.105, 0.325, 0.335])
    loss_color_axis = figure.add_axes([0.505 + 0.325 + 0.008, 0.105, 0.020, 0.335])
    add_topography_panel(topography_axis, x_topo_km, y_topo_km, full_topography_km, setup)
    add_image_panel(figure, wind_axis, wind_color_axis, x_km, y_km, v_surface, r"Meridional wind $v$ [m s$^{-1}$]", wind_cmap, wind_norm, wind_ticks, simple_labels(wind_ticks, 2))
    add_signed_hatching(wind_axis, x_km, y_km, v_surface, wind_zero_band)
    wind_axis.set_ylabel("")
    add_image_panel(figure, mode_axis, mode_color_axis, x_km, y_km, mode_count, r"Launch-mode count $K^{\star}$", mode_cmap, mode_norm, mode_ticks, [f"{int(value)}" for value in mode_ticks])
    add_image_panel(figure, loss_axis, loss_color_axis, x_km, y_km, np.ma.masked_invalid(power_loss_percent), r"Power loss $|\Delta\mathcal{A}/\mathcal{A}|$ [%]", loss_cmap, loss_norm, loss_ticks, simple_labels(loss_ticks, 2))
    loss_axis.set_ylabel("")
    save_figure(figure, figure_output)
    plt.close(figure)
    print(f"Figure saved to {figure_output}.png")


if __name__ == "__main__":
    main()
