# Code_Mountain.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script extracts the PinCFlow mountain-wave EMS output and writes plot-ready CSV tables.

from pathlib import Path

import h5py
import numpy as np

from Module_Csv import write_dict_rows_csv
from Module_Helpers import format_float


csv_dir = Path("./csv")
input_hdf = Path("./PinCFlow-EMS/Banerjee_2026_Enufft_MountainWaveEms.h5")
terrain_csv = csv_dir / "Banerjee_2026_Enufft_Mountain_Terrain.csv"
modes_csv = csv_dir / "Banerjee_2026_Enufft_Mountain_Modes.csv"
summary_csv = csv_dir / "Banerjee_2026_Enufft_Mountain_Summary.csv"
default_setup = {
    "lx": 200000.0,
    "ly": 200000.0,
    "lz": 10000.0,
    "h0": 1000.0,
    "lambda0": 10000.0,
    "rlambda": 10.0,
    "rh": 2.0,
    "wave_modes": 4,
    "u0": 10.0,
    "ems_d": 0.02,
    "ems_wp": 0.5,
    "ems_ws": 0.5,
    "ems_min_a": 0.9,
    "ems_max_a": 1.0,
}


# Read the final two-dimensional field from one PinCFlow dataset.
def read_last_snapshot(handle, name):
    values = np.asarray(handle[name])
    if values.ndim == 4:
        return values[-1, 0, :, :]
    if values.ndim == 3:
        return values[-1, :, :]
    if values.ndim == 2:
        return values
    raise ValueError(f"Cannot extract final snapshot from {name} with shape {values.shape}")


# Read scalar setup values recorded by the mountain-wave EMS run.
def read_setup_values(handle):
    setup = dict(default_setup)
    if "setup" in handle:
        group = handle["setup"]
        for name in ("lx", "ly", "lz", "h0", "lambda0", "rlambda", "rh", "wave_modes", "u0"):
            if name in group:
                value = np.asarray(group[name]).item()
                setup[name] = int(value) if name == "wave_modes" else float(value)
        if "ems" in group:
            ems = group["ems"]
            for name in ("ems_d", "ems_wp", "ems_ws", "ems_min_a", "ems_max_a"):
                if name in ems:
                    setup[name] = float(np.asarray(ems[name]).item())
    return setup


# Rebuild the analytical full mountain topography on one plotting grid.
def fallback_full_topography(x_m, y_m, setup):
    xx, yy = np.meshgrid(x_m, y_m)
    h0 = float(setup["h0"])
    lambda0 = float(setup["lambda0"])
    rlambda = float(setup["rlambda"])
    rh = float(setup["rh"])
    wave_modes = int(setup["wave_modes"])
    radius = rlambda * lambda0 / 2.0
    radius_grid = np.sqrt(xx ** 2 + yy ** 2)
    envelope = np.where(radius_grid <= radius, 1.0 + np.cos(2.0 * np.pi * radius_grid / (rlambda * lambda0)), 0.0)
    unresolved_amplitude = h0 * envelope / (2.0 * wave_modes * (rh + 1.0))
    full_topography = wave_modes * rh * unresolved_amplitude
    for mode in range(1, wave_modes + 1):
        k_m = 2.0 * np.pi * (mode - 1) / lambda0
        ell_n = 2.0 * np.pi / lambda0
        full_topography += unresolved_amplitude * np.cos(k_m * xx + ell_n * yy)
    return full_topography


# Read or reconstruct the fine-grid topography used in the orography panel.
def read_topography(handle, setup):
    if "setup/topography" in handle:
        group = handle["setup/topography"]
        x_km = np.asarray(group["x_fine"]) / 1000.0
        y_km = np.asarray(group["y_fine"]) / 1000.0
        full_km = np.asarray(group["full_topography"]) / 1000.0
        if full_km.shape == (x_km.size, y_km.size):
            full_km = full_km.T
        return x_km, y_km, full_km
    lx = float(setup["lx"])
    ly = float(setup["ly"])
    wave_modes = int(setup["wave_modes"])
    rlambda = int(round(float(setup["rlambda"])))
    nx = 10 * rlambda * (wave_modes - 1)
    ny = 10 * rlambda
    dx = lx / nx
    dy = ly / ny
    x_m = np.linspace(-lx / 4.0 + dx / 2.0, lx / 4.0 - dx / 2.0, nx)
    y_m = np.linspace(-ly / 4.0 + dy / 2.0, ly / 4.0 - dy / 2.0, ny)
    return x_m / 1000.0, y_m / 1000.0, fallback_full_topography(x_m, y_m, setup) / 1000.0


# Load the mountain-wave EMS fields and setup from the HDF5 run output.
def read_mountain_output(input_path):
    with h5py.File(input_path, "r") as handle:
        setup = read_setup_values(handle)
        topography_x_km, topography_y_km, full_topography_km = read_topography(handle, setup)
        return {
            "time_seconds": float(np.asarray(handle["t"])[-1]),
            "x_km": np.asarray(handle["x"]) / 1000.0,
            "y_km": np.asarray(handle["y"]) / 1000.0,
            "v_surface": read_last_snapshot(handle, "v"),
            "mode_count": read_last_snapshot(handle, "launch_mode_count"),
            "power_fraction": read_last_snapshot(handle, "launch_power_fraction"),
            "topography_x_km": topography_x_km,
            "topography_y_km": topography_y_km,
            "full_topography_km": full_topography_km,
            "setup": setup,
        }


# Convert the fine-grid topography into CSV rows.
def build_terrain_rows(data):
    rows = []
    x_km = data["topography_x_km"]
    y_km = data["topography_y_km"]
    full_topography_km = data["full_topography_km"]
    for y_index, y_value in enumerate(y_km):
        for x_index, x_value in enumerate(x_km):
            rows.append({
                "x_index": x_index,
                "y_index": y_index,
                "x_km": format_float(x_value),
                "y_km": format_float(y_value),
                "full_topography_km": format_float(full_topography_km[y_index, x_index]),
            })
    return rows


# Convert final wind, retained count, and retained power into CSV rows.
def build_mode_rows(data):
    rows = []
    x_km = data["x_km"]
    y_km = data["y_km"]
    v_surface = data["v_surface"]
    mode_count = data["mode_count"]
    power_fraction = data["power_fraction"]
    power_loss_percent = np.where(power_fraction > 0.0, (1.0 - power_fraction) * 100.0, np.nan)
    for row_index, y_value in enumerate(y_km):
        for column_index, x_value in enumerate(x_km):
            rows.append({
                "column_index": column_index,
                "row_index": row_index,
                "x_km": format_float(x_value),
                "y_km": format_float(y_value),
                "v_surface_ms": format_float(v_surface[row_index, column_index]),
                "launch_mode_count": int(mode_count[row_index, column_index]),
                "launch_power_fraction": format_float(power_fraction[row_index, column_index]),
                "launch_power_loss_percent": format_float(power_loss_percent[row_index, column_index]),
            })
    return rows


# Build the scalar run-configuration and diagnostic summary row.
def build_summary_rows(data):
    setup = data["setup"]
    mode_count = data["mode_count"]
    power_fraction = data["power_fraction"]
    v_surface = data["v_surface"]
    active_power = power_fraction[power_fraction > 0.0]
    row = {name: format_float(value) for name, value in setup.items() if name != "wave_modes"}
    row["wave_modes"] = int(setup["wave_modes"])
    row["time_seconds"] = format_float(data["time_seconds"])
    row["time_minutes"] = format_float(data["time_seconds"] / 60.0)
    row["selected_cells"] = int(np.count_nonzero(mode_count > 0))
    row["mode_count_min"] = format_float(np.nanmin(mode_count))
    row["mode_count_max"] = format_float(np.nanmax(mode_count))
    row["power_fraction_min"] = format_float(np.nanmin(active_power)) if active_power.size else format_float(np.nan)
    row["power_fraction_max"] = format_float(np.nanmax(active_power)) if active_power.size else format_float(np.nan)
    row["power_loss_max_percent"] = format_float((1.0 - np.nanmin(active_power)) * 100.0) if active_power.size else format_float(np.nan)
    row["v_surface_min_ms"] = format_float(np.nanmin(v_surface))
    row["v_surface_max_ms"] = format_float(np.nanmax(v_surface))
    return [row]


# Extract the HDF5 run output and write all CSV files used by the figure.
def main():
    data = read_mountain_output(input_hdf)
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(build_terrain_rows(data), terrain_csv)
    write_dict_rows_csv(build_mode_rows(data), modes_csv)
    write_dict_rows_csv(build_summary_rows(data), summary_csv)
    active_power = data["power_fraction"][data["power_fraction"] > 0.0]
    print("Mountain-wave EMS extraction")
    print(f"input = {input_hdf}")
    print(f"final time = {data['time_seconds']:.0f} s")
    print(f"selected cells = {np.count_nonzero(data['mode_count'] > 0)}")
    print(f"K* range = {np.nanmin(data['mode_count']):g}, {np.nanmax(data['mode_count']):g}")
    print(f"max power loss = {(1.0 - np.nanmin(active_power)) * 100.0:.6g} %")
    print(f"Wrote {terrain_csv}")
    print(f"Wrote {modes_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
