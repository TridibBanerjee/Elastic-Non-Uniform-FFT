# Module_Alps.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the Alps DEM preprocessing, window geometry, and comparison helpers.

from pathlib import Path

import hashlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, cKDTree

from Module_Csa import (
    compute_csa_spectrum,
    compute_sorted_spectral_amplitudes,
    count_signed_nonzero_modes,
    count_unique_mode_pairs,
    find_dominant_mode_pair,
    reconstruct_at_points,
)
from Module_Ems import select_sparse_conjugate_modes
from Module_Helpers import polygon_box_intersection_area
from Module_Nufft import compute_nufft_coefficients


dem_dir = Path("./srtm_alps")
preprocessed_dem_name = "alps_dem_processed.npz"
preprocessed_dem = dem_dir / preprocessed_dem_name
earth_radius_km = 6371.0
alps_metric_version = "rel_rmse_sigma1m_v2"
relative_rmse_sigma_floor = 1.0
relative_rmse_variance_floor = relative_rmse_sigma_floor ** 2
alps_mesh_presets = {
    "r2b4": {
        "nominal_dx_km": 160.0,
        "n_modes": 16,
    },
    "r2b5": {
        "nominal_dx_km": 80.0,
        "n_modes": 32,
    },
}
# Recovered from preserved original Alps sweep centroids so tri_num remains a physical cell id.
legacy_cell_diagonal_patterns = {
    "r2b4": (
        "FBFBFB",
        "BFBFBF",
        "FBFBBF",
        "BFBFBB",
    ),
    "r2b5": (
        "FBFFBFBFBFF",
        "BFBFBFBBFBB",
        "BBFFBBFFBFB",
        "FBBBFFBBFBF",
        "BFBBFBFBFFB",
        "FBFBBFBFFBF",
        "FBBFBBFBBFB",
    ),
}
legacy_triangle_orders = {
    "r2b4": (
        2, 3, 12, 13, 0, 1, 39, 38, 25, 24, 37, 36,
        14, 15, 27, 26, 22, 23, 10, 11, 16, 17, 4, 5,
        19, 18, 21, 20, 6, 7, 8, 9, 47, 46, 34, 35,
        32, 33, 45, 44, 30, 31, 43, 42, 29, 28, 41, 40,
    ),
    "r2b5": (
        6, 7, 8, 9, 71, 70, 133, 132, 135, 134, 109, 108,
        130, 131, 106, 107, 14, 15, 16, 17, 10, 11, 12, 13,
        94, 95, 4, 5, 44, 45, 30, 31, 28, 29, 137, 136,
        116, 117, 92, 93, 115, 114, 113, 112, 141, 140, 139, 138,
        18, 19, 153, 152, 125, 124, 36, 37, 118, 119, 97, 96,
        77, 76, 98, 99, 79, 78, 101, 100, 104, 105, 103, 102,
        146, 147, 145, 144, 123, 122, 143, 142, 121, 120, 46, 47,
        49, 48, 26, 27, 75, 74, 52, 53, 72, 73, 50, 51,
        111, 110, 67, 66, 88, 89, 69, 68, 90, 91, 20, 21,
        43, 42, 85, 84, 86, 87, 65, 64, 38, 39, 40, 41,
        63, 62, 149, 148, 127, 126, 151, 150, 129, 128, 57, 56,
        34, 35, 55, 54, 32, 33, 58, 59, 80, 81, 60, 61,
        83, 82, 25, 24, 3, 2, 22, 23, 1, 0,
    ),
}
legacy_cell_tie_slots = {
    # One R2B4 DEM sample falls exactly on this split diagonal in the preserved run.
    "r2b4": {
        (0, 3): 1,
    },
}
window_strategies = [
    "square_centroid",
    "square_edge_aligned",
    "triangle_centroid",
    "circle_centroid",
    "triangle_edge_aligned",
    "circle_edge_aligned",
]
_worker_dem = None
_worker_mesh = None
_worker_case = None


# Return the local preprocessed DEM archive or a preprocessing instruction.
def require_preprocessed_dem():
    if preprocessed_dem.exists():
        return preprocessed_dem
    raise FileNotFoundError(
        "\n".join(
            [
                f"Preprocessed Alps DEM archive not found: {preprocessed_dem}",
                "",
                "Create it from local SRTM tiles with:",
                "  python3 Code_Alps_Preprocess.py",
                "",
                "If the raw SRTM tiles live outside the paper directory, pass only that raw tile directory:",
                "  python3 Code_Alps_Preprocess.py /path/to/tiles_hgt",
            ]
        )
    )


# Build one comparison case dictionary from the fixed Alps defaults.
def build_alps_case(updates=None):
    case = {
        "mesh_name": None,
        "cell_size_km": 80.0,
        "nominal_dx_km": 80.0,
        "n_modes": None,
        "oversample": 1.5,
        "kernel_half_width": 4,
        "kernel_beta": 2.34,
        "window_expansion": 2.0,
        "window_strategy": "square_centroid",
        "weight_type": "uniform",
        "min_points_per_triangle": 10,
        "enufft_k_min": 1,
        "enufft_k_max": None,
        "enufft_alpha_min": 0.0,
        "enufft_alpha_max": 0.70,
        "enufft_delta": 0.02,
        "enufft_w1": 0.5,
        "enufft_w2": 0.5,
        "csa_lambda_fa": 1e-1,
        "csa_lambda_sa": 1e-6,
        "csa_sparse_modes": None,
        "csa_chunk_size": 512,
        "max_triangles": None,
    }
    if updates is not None:
        case.update(updates)
    if case.get("mesh_name") is not None:
        mesh_name = normalize_mesh_name(case["mesh_name"])
        case["mesh_name"] = mesh_name
        if mesh_name in alps_mesh_presets:
            preset = alps_mesh_presets[mesh_name]
            case["cell_size_km"] = float(preset["nominal_dx_km"])
            case["nominal_dx_km"] = float(preset["nominal_dx_km"])
            if case["n_modes"] is None:
                case["n_modes"] = int(preset["n_modes"])
        else:
            case["nominal_dx_km"] = float(case["cell_size_km"])
    if case["n_modes"] is None:
        case["n_modes"] = 16 if case["cell_size_km"] >= 120.0 else 32
    if case.get("nominal_dx_km") is None:
        case["nominal_dx_km"] = float(case["cell_size_km"])
    return case


# Normalize a mesh preset name accepted on the command line.
def normalize_mesh_name(mesh_name):
    if mesh_name is None:
        return None
    text = str(mesh_name).strip().lower()
    text = text.replace("_", "").replace("-", "")
    if text in ("regular", "proxy", "regularproxy"):
        return "regular"
    if text in ("r02b04", "r2b4"):
        return "r2b4"
    if text in ("r02b05", "r2b5"):
        return "r2b5"
    raise ValueError(f"Unknown Alps mesh preset {mesh_name!r}; expected r2b4, r2b5, or regular")


# Convert explicit nodata and nonphysical SRTM values to NaN.
def mask_invalid_elevations(data, nodata=None):
    data = np.asarray(data, dtype=np.float64).copy()
    invalid = ~np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        invalid |= data == float(nodata)
    invalid |= data <= -32768.0
    data[invalid] = np.nan
    return data


# Parse the lower-left latitude and longitude from an SRTM tile name.
def parse_srtm_tile_name(path):
    name = Path(path).name
    stem = name.split(".")[0]
    lat_sign = -1 if stem[0].upper() == "S" else 1
    lon_sign = -1 if stem[3].upper() == "W" else 1
    return lat_sign * int(stem[1:3]), lon_sign * int(stem[4:7])


# Read one raw SRTM HGT tile as a north-up 3601 by 3601 grid.
def read_srtm_hgt(path):
    data = np.fromfile(path, dtype=">i2").reshape(3601, 3601)
    lat_value, lon_value = parse_srtm_tile_name(path)
    bounds = (lon_value, lat_value, lon_value + 1, lat_value + 1)
    return mask_invalid_elevations(data), bounds


# Read one SRTM tile from HGT or GeoTIFF storage.
def read_srtm_tile(path):
    path = Path(path)
    if path.suffix.lower() == ".hgt":
        return read_srtm_hgt(path)
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = mask_invalid_elevations(src.read(1), src.nodata)
            bounds = src.bounds
        return data, (bounds.left, bounds.bottom, bounds.right, bounds.top)
    except ImportError:
        pass
    try:
        from osgeo import gdal
        dataset = gdal.Open(str(path))
        band = dataset.GetRasterBand(1)
        data = mask_invalid_elevations(band.ReadAsArray(), band.GetNoDataValue())
        transform = dataset.GetGeoTransform()
        nx_value, ny_value = dataset.RasterXSize, dataset.RasterYSize
        left = transform[0]
        top = transform[3]
        right = left + nx_value * transform[1]
        bottom = top + ny_value * transform[5]
        dataset = None
        return data, (left, bottom, right, top)
    except ImportError:
        pass
    raise RuntimeError(f"Cannot read {path}; use HGT tiles or install rasterio/gdal")


# Find a local SRTM tile path for one latitude-longitude cell.
def resolve_srtm_tile(tile_dir, lat_value, lon_value):
    tile_dir = Path(tile_dir)
    stem = f"N{lat_value:02d}E{lon_value:03d}"
    for suffix in (".hgt", ".tif", ".tiff"):
        candidate = tile_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return tile_dir / f"{stem}.hgt"


# Load the requested SRTM tiles and assemble one north-up mosaic.
def load_and_mosaic_tiles(tile_dir, lat_range=(44, 49), lon_range=(5, 16)):
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    tile_size = 3601
    overlap = 1
    ny_total = (lat_max - lat_min) * (tile_size - overlap) + overlap
    nx_total = (lon_max - lon_min) * (tile_size - overlap) + overlap
    mosaic = np.full((ny_total, nx_total), np.nan, dtype=np.float64)
    tiles_loaded = 0
    tiles_missing = 0
    for lat_value in range(lat_min, lat_max):
        for lon_value in range(lon_min, lon_max):
            path = resolve_srtm_tile(tile_dir, lat_value, lon_value)
            if not path.exists():
                tiles_missing += 1
                continue
            data, _ = read_srtm_tile(path)
            if data.shape != (tile_size, tile_size):
                tiles_missing += 1
                continue
            row_offset = (lat_max - 1 - lat_value) * (tile_size - overlap)
            col_offset = (lon_value - lon_min) * (tile_size - overlap)
            mosaic[row_offset:row_offset + tile_size, col_offset:col_offset + tile_size] = data
            tiles_loaded += 1
    lat = np.linspace(lat_max, lat_min, ny_total)
    lon = np.linspace(lon_min, lon_max, nx_total)
    return mosaic, lat, lon, tiles_loaded, tiles_missing


# Average the DEM onto a coarser non-overlapping regular grid.
def coarse_grain(dem, lat, lon, block_size=30):
    ny_value, nx_value = dem.shape
    ny_trim = (ny_value // block_size) * block_size
    nx_trim = (nx_value // block_size) * block_size
    dem_trim = dem[:ny_trim, :nx_trim]
    lat_trim = lat[:ny_trim]
    lon_trim = lon[:nx_trim]
    dem_coarse = dem_trim.reshape(ny_trim // block_size, block_size, nx_trim // block_size, block_size).mean(axis=(1, 3))
    lat_coarse = lat_trim.reshape(-1, block_size).mean(axis=1)
    lon_coarse = lon_trim.reshape(-1, block_size).mean(axis=1)
    return dem_coarse, lat_coarse, lon_coarse


# Convert latitude and longitude arrays to local Plate Carree kilometres.
def latlon_to_xy_km(lat, lon, lat_ref, lon_ref):
    x_km = earth_radius_km * np.radians(lon - lon_ref) * np.cos(np.radians(lat_ref))
    y_km = earth_radius_km * np.radians(lat - lat_ref)
    return x_km, y_km


# Smooth terrain with the normalized five-kilometre spectral Gaussian.
def apply_5km_smoother(dem, dx_km, dy_km, smooth_scale_km=5.0):
    sigma_eq_km = smooth_scale_km / (np.sqrt(2.0) * np.pi)
    sigma_x = sigma_eq_km / dx_km
    sigma_y = sigma_eq_km / dy_km
    valid = np.isfinite(dem)
    dem_filled = np.where(valid, dem, 0.0)
    weight = valid.astype(np.float64)
    dem_smooth = gaussian_filter(dem_filled, sigma=[sigma_y, sigma_x], mode="constant", cval=0.0)
    weight_smooth = gaussian_filter(weight, sigma=[sigma_y, sigma_x], mode="constant", cval=0.0)
    result = dem_smooth / np.maximum(weight_smooth, 1e-10)
    result[~valid] = np.nan
    return result


# Clip implausible negative elevations before smoothing and export.
def clip_elevations(dem, min_elev=-500.0):
    return np.maximum(dem, min_elev)


# Build and save the local Alps DEM preprocessing archive.
def preprocess_alps_dem(tile_dir, output_path, block_size=30, smooth_km=5.0, min_elev=-500.0, lat_min=44, lat_max=49, lon_min=5, lon_max=16, smooth=True, hires_subsample=1, hires_dtype="float32"):
    mosaic, lat_raw, lon_raw, tiles_loaded, tiles_missing = load_and_mosaic_tiles(tile_dir, (lat_min, lat_max), (lon_min, lon_max))
    expected_tiles = (lat_max - lat_min) * (lon_max - lon_min)
    if tiles_loaded != expected_tiles or tiles_missing:
        raise FileNotFoundError(f"Missing local SRTM tiles in {tile_dir}: loaded {tiles_loaded} of {expected_tiles}")
    mosaic = clip_elevations(mosaic, min_elev)
    hstep = int(hires_subsample)
    storage_dtype = np.float32 if hires_dtype == "float32" else np.float64
    mosaic_hires = mosaic[::hstep, ::hstep].astype(storage_dtype, copy=False)
    lat_hires = lat_raw[::hstep]
    lon_hires = lon_raw[::hstep]
    dem_coarse, lat_coarse, lon_coarse = coarse_grain(mosaic, lat_raw, lon_raw, int(block_size))
    lat_ref = lat_coarse[-1]
    lon_ref = lon_coarse[0]
    x_km, _ = latlon_to_xy_km(lat_ref, lon_coarse, lat_ref, lon_ref)
    _, y_km = latlon_to_xy_km(lat_coarse, lon_ref, lat_ref, lon_ref)
    dx_km = np.abs(np.median(np.diff(x_km)))
    dy_km = np.abs(np.median(np.diff(y_km)))
    dem_block_avg = dem_coarse.copy()
    if smooth:
        dem_coarse = apply_5km_smoother(dem_coarse, dx_km, dy_km, smooth_km)
    x_km_hires, _ = latlon_to_xy_km(lat_ref, lon_hires, lat_ref, lon_ref)
    _, y_km_hires = latlon_to_xy_km(lat_hires, lon_ref, lat_ref, lon_ref)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, elev=dem_coarse, x_km=x_km, y_km=y_km, lon=lon_coarse, lat=lat_coarse, lat_ref=lat_ref, lon_ref=lon_ref, dx_km=dx_km, dy_km=dy_km, block_size=int(block_size), smooth_km=0.0 if not smooth else float(smooth_km), elev_block_avg=dem_block_avg, elev_hires=mosaic_hires, x_km_hires=x_km_hires, y_km_hires=y_km_hires, hires_subsample=hstep)
    return {"output_path": output_path, "tiles_loaded": tiles_loaded, "tiles_missing": tiles_missing, "raw_shape": mosaic.shape, "coarse_shape": dem_coarse.shape, "hires_shape": mosaic_hires.shape, "dx_km": float(dx_km), "dy_km": float(dy_km)}


# Orient gridded fields so y increases.
def orient_y_ascending(x_values, y_values, z_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    z_values = np.asarray(z_values, dtype=float)
    if y_values[0] > y_values[-1]:
        y_values = y_values[::-1]
        z_values = z_values[::-1, :]
    return x_values, y_values, z_values


# Load the processed Alps DEM and derived coordinate fields.
def load_alps_dem():
    data = np.load(require_preprocessed_dem(), allow_pickle=False)
    x_km, y_km, elev = orient_y_ascending(data["x_km"], data["y_km"], data["elev"])
    elev = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    lon_ref = float(data["lon_ref"])
    lat_ref = float(data["lat_ref"])
    interp = RegularGridInterpolator((y_km, x_km), elev, method="linear", bounds_error=False, fill_value=np.nan)
    x_grid, y_grid = np.meshgrid(x_km, y_km)
    dem_xy = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    dem_h = elev.ravel()
    return {"x_km": x_km, "y_km": y_km, "elev": elev, "dem_xy": dem_xy, "dem_h": dem_h, "interp": interp, "Lx": float(x_km[-1] - x_km[0]), "Ly": float(y_km[-1] - y_km[0]), "lon_ref": lon_ref, "lat_ref": lat_ref, "smooth_km": float(data["smooth_km"]) if "smooth_km" in data else np.nan}


# Return a deterministic cell diagonal code for one mesh cell.
def cell_diagonal_code(mesh_name, i_value, j_value, nx_cells, ny_cells):
    pattern = legacy_cell_diagonal_patterns.get(mesh_name)
    if pattern is not None and len(pattern) == ny_cells and all(len(row) == nx_cells for row in pattern):
        return pattern[j_value][i_value]
    return "F" if (i_value + j_value) % 2 == 0 else "B"


# Build a deterministic regular ICON-like triangular mesh over the Alps domain.
def regular_icon_like_triangles(x_values, y_values, target_dx_km, mesh_name="regular"):
    nx_value = int(np.ceil((x_values[-1] - x_values[0]) / target_dx_km)) + 1
    ny_value = int(np.ceil((y_values[-1] - y_values[0]) / target_dx_km)) + 1
    vx = np.linspace(x_values[0], x_values[-1], nx_value)
    vy = np.linspace(y_values[0], y_values[-1], ny_value)
    x_grid, y_grid = np.meshgrid(vx, vy)
    vertices = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    nx_cells = nx_value - 1
    ny_cells = ny_value - 1
    triangles = []
    cell_triangle_ids = np.full((ny_cells, nx_cells, 2), -1, dtype=int)
    cell_diagonal_codes = np.empty((ny_cells, nx_cells), dtype="U1")
    cell_tie_slots = np.full((ny_cells, nx_cells), -1, dtype=int)
    for (i_value, j_value), slot in legacy_cell_tie_slots.get(mesh_name, {}).items():
        if 0 <= i_value < nx_cells and 0 <= j_value < ny_cells:
            cell_tie_slots[j_value, i_value] = int(slot)

    def vertex_id(i_value, j_value):
        return j_value * nx_value + i_value

    for j_value in range(ny_cells):
        for i_value in range(nx_cells):
            bl = vertex_id(i_value, j_value)
            br = vertex_id(i_value + 1, j_value)
            tl = vertex_id(i_value, j_value + 1)
            tr = vertex_id(i_value + 1, j_value + 1)
            diagonal = cell_diagonal_code(mesh_name, i_value, j_value, nx_cells, ny_cells)
            cell_diagonal_codes[j_value, i_value] = diagonal
            if diagonal == "F":
                cell_triangles = ((bl, br, tr), (bl, tr, tl))
            else:
                cell_triangles = ((bl, br, tl), (br, tr, tl))
            for local_index, triangle in enumerate(cell_triangles):
                cell_triangle_ids[j_value, i_value, local_index] = len(triangles)
                triangles.append(triangle)

    triangles = np.asarray(triangles, dtype=int)
    triangle_order = legacy_triangle_orders.get(mesh_name)
    if triangle_order is not None:
        triangle_order = np.asarray(triangle_order, dtype=int)
        if triangle_order.size != len(triangles) or not np.array_equal(np.sort(triangle_order), np.arange(len(triangles))):
            raise ValueError(f"Legacy triangle order for {mesh_name} does not match generated mesh")
        inverse_order = np.empty_like(triangle_order)
        inverse_order[triangle_order] = np.arange(triangle_order.size)
        triangles = triangles[triangle_order]
        cell_triangle_ids = inverse_order[cell_triangle_ids]
    else:
        triangle_order = np.arange(len(triangles), dtype=int)
    mesh = {
        "mesh_name": mesh_name,
        "mesh_kind": "regular_icon_like",
        "mesh_version": "structured_legacy_v1",
        "nominal_dx_km": float(target_dx_km),
        "vertices": vertices,
        "triangles": triangles,
        "assignment": "structured_grid",
        "x_vertices": vx,
        "y_vertices": vy,
        "grid_shape": (ny_value, nx_value),
        "cell_triangle_ids": cell_triangle_ids,
        "cell_diagonal_codes": cell_diagonal_codes,
        "cell_tie_slots": cell_tie_slots,
        "triangle_order": triangle_order,
    }
    mesh["mesh_signature"] = mesh_geometry_signature(mesh)
    return mesh


# Return a stable fingerprint for one mesh geometry and assignment table.
def mesh_geometry_signature(mesh):
    digest = hashlib.sha256()
    for key in ("vertices", "triangles", "cell_triangle_ids", "cell_diagonal_codes", "cell_tie_slots"):
        if key not in mesh:
            continue
        values = np.ascontiguousarray(mesh[key])
        digest.update(str(key).encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


# Build the requested Alps analysis mesh from the original local proxy grids.
def build_alps_mesh(dem_data, mesh_name=None, cell_size_km=80.0):
    mesh_name = normalize_mesh_name(mesh_name) if mesh_name is not None else "regular"
    return regular_icon_like_triangles(dem_data["x_km"], dem_data["y_km"], float(cell_size_km), mesh_name=mesh_name)


# Assign points to the deterministic structured mesh triangles.
def find_mesh_triangle_ids(points, mesh):
    points = np.asarray(points, dtype=float)
    if mesh.get("assignment") != "structured_grid":
        triangulation = Delaunay(mesh["vertices"])
        simplex_raw = triangulation.find_simplex(points)
        source_to_mesh = np.full(len(triangulation.simplices), -1, dtype=int)
        triangle_lookup = {tuple(sorted(np.asarray(simplex, dtype=int))): index for index, simplex in enumerate(mesh["triangles"])}
        for source_index, simplex in enumerate(triangulation.simplices):
            source_to_mesh[source_index] = triangle_lookup.get(tuple(sorted(np.asarray(simplex, dtype=int))), -1)
        mesh_id = np.full(len(simplex_raw), -1, dtype=int)
        valid_source = simplex_raw >= 0
        mesh_id[valid_source] = source_to_mesh[simplex_raw[valid_source]]
        return mesh_id

    x_vertices = np.asarray(mesh["x_vertices"], dtype=float)
    y_vertices = np.asarray(mesh["y_vertices"], dtype=float)
    cell_triangle_ids = np.asarray(mesh["cell_triangle_ids"], dtype=int)
    cell_diagonal_codes = np.asarray(mesh["cell_diagonal_codes"])
    cell_tie_slots = np.asarray(mesh.get("cell_tie_slots", np.full(cell_diagonal_codes.shape, -1, dtype=int)), dtype=int)
    nx_cells = x_vertices.size - 1
    ny_cells = y_vertices.size - 1
    x_values = points[:, 0]
    y_values = points[:, 1]
    inside = (
        (x_values >= x_vertices[0]) & (x_values <= x_vertices[-1]) &
        (y_values >= y_vertices[0]) & (y_values <= y_vertices[-1])
    )
    mesh_id = np.full(points.shape[0], -1, dtype=int)
    if not np.any(inside):
        return mesh_id

    i_values = np.searchsorted(x_vertices, x_values[inside], side="right") - 1
    j_values = np.searchsorted(y_vertices, y_values[inside], side="right") - 1
    i_values = np.clip(i_values, 0, nx_cells - 1)
    j_values = np.clip(j_values, 0, ny_cells - 1)
    dx = x_vertices[i_values + 1] - x_vertices[i_values]
    dy = y_vertices[j_values + 1] - y_vertices[j_values]
    u_values = np.clip((x_values[inside] - x_vertices[i_values]) / dx, 0.0, 1.0)
    v_values = np.clip((y_values[inside] - y_vertices[j_values]) / dy, 0.0, 1.0)
    triangle_slot = np.zeros(i_values.shape, dtype=int)

    diagonal_is_forward = cell_diagonal_codes[j_values, i_values] == "F"
    triangle_slot[diagonal_is_forward] = (v_values[diagonal_is_forward] > u_values[diagonal_is_forward]).astype(int)
    triangle_slot[~diagonal_is_forward] = (v_values[~diagonal_is_forward] > (1.0 - u_values[~diagonal_is_forward])).astype(int)
    tie_slots = cell_tie_slots[j_values, i_values]
    has_tie_override = tie_slots >= 0
    if np.any(has_tie_override):
        diagonal_value = np.where(diagonal_is_forward, u_values, 1.0 - u_values)
        on_diagonal = np.abs(v_values - diagonal_value) <= 1e-12
        override = has_tie_override & on_diagonal
        triangle_slot[override] = tie_slots[override]

    inside_indices = np.flatnonzero(inside)
    mesh_id[inside_indices] = cell_triangle_ids[j_values, i_values, triangle_slot]
    return mesh_id


# Remove one mean height per mesh triangle.
def deplane_dem_on_mesh(dem_data, mesh):
    mesh_id = find_mesh_triangle_ids(dem_data["dem_xy"], mesh)
    heights = dem_data["dem_h"]
    valid = (mesh_id >= 0) & np.isfinite(heights)
    triangle_count = len(mesh["triangles"])
    counts = np.bincount(mesh_id[valid], minlength=triangle_count)
    sums = np.bincount(mesh_id[valid], weights=heights[valid], minlength=triangle_count)
    means = np.full(triangle_count, np.nan, dtype=float)
    has_points = counts > 0
    means[has_points] = sums[has_points] / counts[has_points]
    h_deplaned = np.full_like(heights, np.nan, dtype=float)
    h_deplaned[valid] = heights[valid] - means[mesh_id[valid]]
    updated = dict(dem_data)
    updated["dem_h_deplaned"] = h_deplaned
    updated["mesh_triangle_id"] = mesh_id
    updated["mesh_triangle_means"] = means
    updated["mesh_triangle_counts"] = counts
    return updated


# Test which points lie in one triangle with scale-aware tolerance.
def points_in_triangle_mask(points, triangle_vertices):
    v0, v1, v2 = triangle_vertices
    scale = max(float(np.ptp(triangle_vertices[:, 0])), float(np.ptp(triangle_vertices[:, 1])), 1.0)
    tol = 1e-10 * scale * scale
    d1 = (points[:, 0] - v1[0]) * (v0[1] - v1[1]) - (v0[0] - v1[0]) * (points[:, 1] - v1[1])
    d2 = (points[:, 0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (points[:, 1] - v2[1])
    d3 = (points[:, 0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (points[:, 1] - v0[1])
    has_negative = (d1 < -tol) | (d2 < -tol) | (d3 < -tol)
    has_positive = (d1 > tol) | (d2 > tol) | (d3 > tol)
    return ~(has_negative & has_positive)


# Split an Alps window strategy into alignment and support parts.
def split_window_strategy(strategy):
    if strategy == "square_centroid":
        return "centroid", "square"
    if strategy == "square_edge_aligned":
        return "edge_aligned", "square"
    if strategy == "triangle_centroid":
        return "centroid", "triangle"
    if strategy == "circle_centroid":
        return "centroid", "circle"
    if strategy == "triangle_edge_aligned":
        return "edge_aligned", "triangle"
    if strategy == "circle_edge_aligned":
        return "edge_aligned", "circle"
    raise ValueError(f"Unknown window strategy {strategy}")


# Return the square CSA reference for non-square ENUFFT supports.
def csa_window_strategy(strategy):
    alignment, support = split_window_strategy(strategy)
    if support == "square":
        return strategy
    if alignment == "centroid":
        return "square_centroid"
    if alignment == "edge_aligned":
        return "square_edge_aligned"
    return None


# Check whether CSA is directly available for one window strategy.
def is_csa_supported_window(strategy):
    _, support = split_window_strategy(strategy)
    return support == "square"


# Build the local analysis window for one triangle.
def get_analysis_window(dem_xy, triangle_vertices, case):
    alignment, support = split_window_strategy(case["window_strategy"])
    expansion = float(case["window_expansion"])
    if alignment == "edge_aligned":
        edges = [triangle_vertices[1] - triangle_vertices[0], triangle_vertices[2] - triangle_vertices[1], triangle_vertices[0] - triangle_vertices[2]]
        longest = max(edges, key=lambda edge: np.linalg.norm(edge))
        theta = float(np.arctan2(longest[1], longest[0]))
        cos_t = float(np.cos(-theta))
        sin_t = float(np.sin(-theta))
        dem_rot = np.column_stack([cos_t * dem_xy[:, 0] - sin_t * dem_xy[:, 1], sin_t * dem_xy[:, 0] + cos_t * dem_xy[:, 1]])
        tri_rot = np.column_stack([cos_t * triangle_vertices[:, 0] - sin_t * triangle_vertices[:, 1], sin_t * triangle_vertices[:, 0] + cos_t * triangle_vertices[:, 1]])
        window_center = np.mean(tri_rot, axis=0)
        extent = max(float(np.ptp(tri_rot[:, 0])), float(np.ptp(tri_rot[:, 1])))
        half = 0.5 * expansion * max(extent, 1e-8)
        lo_window = window_center - half
        hi_window = window_center + half
        source = dem_rot
        frame = "rect_rotated"
        window = {"theta": theta, "cos_t": cos_t, "sin_t": sin_t, "dem_local_source": dem_rot}
    else:
        window_center = np.mean(triangle_vertices, axis=0)
        extent = max(float(np.ptp(triangle_vertices[:, 0])), float(np.ptp(triangle_vertices[:, 1])))
        half = 0.5 * expansion * max(extent, 1e-8)
        lo_window = window_center - half
        hi_window = window_center + half
        source = dem_xy
        tri_rot = triangle_vertices
        frame = "rect"
        window = {}
    bounding_mask = (source[:, 0] >= lo_window[0]) & (source[:, 0] <= hi_window[0]) & (source[:, 1] >= lo_window[1]) & (source[:, 1] <= hi_window[1])
    tri_local = tri_rot - lo_window
    center_local = window_center - lo_window
    if support == "triangle":
        mask = bounding_mask & points_in_triangle_mask(source, tri_rot)
        support_area = polygon_box_intersection_area(tri_local, 0.0, 2.0 * half, 0.0, 2.0 * half)
    elif support == "circle":
        radius = half
        mask = bounding_mask & (np.linalg.norm(source - window_center, axis=1) <= radius)
        support_area = float(np.pi * radius ** 2)
    else:
        mask = bounding_mask
        support_area = float((2.0 * half) ** 2)
    window.update({"type": support, "frame": frame, "alignment": alignment, "x0": float(lo_window[0]), "y0": float(lo_window[1]), "Lx": float(2.0 * half), "Ly": float(2.0 * half), "center": window_center, "center_local": center_local, "tri_local": tri_local, "area": support_area})
    if support == "circle":
        window["radius"] = float(half)
    return mask, window


# Map physical points into local analysis-window coordinates.
def points_to_window_coordinates(points_xy, window):
    points_xy = np.asarray(points_xy, dtype=float)
    if window.get("frame") == "rect_rotated":
        cos_t = float(window["cos_t"])
        sin_t = float(window["sin_t"])
        src_x = cos_t * points_xy[:, 0] - sin_t * points_xy[:, 1]
        src_y = sin_t * points_xy[:, 0] + cos_t * points_xy[:, 1]
        x_local = src_x - window["x0"]
        y_local = src_y - window["y0"]
    else:
        x_local = points_xy[:, 0] - window["x0"]
        y_local = points_xy[:, 1] - window["y0"]
    return x_local, y_local


# Compute local coordinates for all masked DEM points.
def local_window_coordinates(dem_xy, mask, window):
    return points_to_window_coordinates(dem_xy[mask], window)


# Convert local kilometre coordinates to longitude and latitude.
def xy_to_lonlat(x_values, y_values, lon_ref, lat_ref):
    lat = lat_ref + np.rad2deg(y_values / earth_radius_km)
    lon = lon_ref + np.rad2deg(x_values / (earth_radius_km * np.cos(np.deg2rad(lat_ref))))
    return lon, lat


# Convert longitude and latitude to local kilometre coordinates.
def lonlat_to_xy(lon_value, lat_value, lon_ref, lat_ref):
    x_value = earth_radius_km * np.deg2rad(lon_value - lon_ref) * np.cos(np.deg2rad(lat_ref))
    y_value = earth_radius_km * np.deg2rad(lat_value - lat_ref)
    return float(x_value), float(y_value)


# Compute local Voronoi-style sample areas on the active support.
def compute_local_voronoi_weights(x_values, y_values, domain_length_x, domain_length_y, window=None, grid_res=96):
    point_count = len(x_values)
    if point_count < 4 or domain_length_x <= 0.0 or domain_length_y <= 0.0:
        return np.ones(point_count, dtype=float) * (domain_length_x * domain_length_y / max(point_count, 1))
    gx = np.linspace(0.0, domain_length_x, grid_res, endpoint=False) + domain_length_x / (2.0 * grid_res)
    gy = np.linspace(0.0, domain_length_y, grid_res, endpoint=False) + domain_length_y / (2.0 * grid_res)
    x_grid, y_grid = np.meshgrid(gx, gy, indexing="xy")
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    support_area = domain_length_x * domain_length_y
    if window is not None:
        support_area = float(window.get("area", support_area))
        if window.get("type") == "triangle":
            grid_points = grid_points[points_in_triangle_mask(grid_points, np.asarray(window["tri_local"]))]
        elif window.get("type") == "circle":
            center = np.asarray(window["center_local"], dtype=float)
            radius = float(window["radius"])
            grid_points = grid_points[np.linalg.norm(grid_points - center, axis=1) <= radius]
    if len(grid_points) == 0:
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        support_area = domain_length_x * domain_length_y
    _, owner = cKDTree(np.column_stack([x_values, y_values])).query(grid_points, k=1)
    counts = np.bincount(owner, minlength=point_count).astype(float)
    weights = counts * (support_area / len(grid_points))
    weights = np.maximum(weights, 1e-12 * support_area / point_count)
    return weights * (support_area / np.sum(weights))


# Convert a Fourier mode pair to a wave direction in degrees.
def compute_wave_direction(m_mode, n_mode, domain_length_x, domain_length_y):
    k_m = 2.0 * np.pi * m_mode / domain_length_x
    l_n = 2.0 * np.pi * n_mode / domain_length_y
    return float(np.degrees(np.arctan2(l_n, k_m)))


# Sum the non-DC Parseval variance proxy of one spectrum.
def spectral_variance(spectrum, m_values, n_values):
    spec = np.asarray(spectrum).copy()
    m_zero = int(np.where(np.asarray(m_values) == 0)[0][0])
    n_zero = int(np.where(np.asarray(n_values) == 0)[0][0])
    spec[m_zero, n_zero] = 0.0
    return float(np.sum(np.abs(spec) ** 2))


# Normalize RMSE by the target standard deviation when that reference is meaningful.
def relative_rmse(rmse, sigma):
    rmse = float(rmse)
    sigma = float(sigma)
    if not np.isfinite(rmse) or not np.isfinite(sigma) or sigma <= relative_rmse_sigma_floor:
        return np.nan
    return float(rmse / sigma)


# Return a finite median after dropping undefined values.
def finite_median(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


# Return a finite maximum after dropping undefined values.
def finite_max(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


# Divide only where the denominator has a meaningful reference.
def finite_ratio_median(numerator, denominator, floor):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    ratio = np.full_like(numerator, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > floor)
    ratio[valid] = numerator[valid] / denominator[valid]
    return finite_median(ratio)


# Initialize module globals for multiprocessing workers.
def init_alps_worker(dem_data, mesh, case):
    global _worker_dem, _worker_mesh, _worker_case
    _worker_dem = dem_data
    _worker_mesh = mesh
    _worker_case = case


# Run all spectral diagnostics for one Alps mesh triangle.
def process_alps_triangle(triangle_number):
    dem_xy = _worker_dem["dem_xy"]
    dem_h = _worker_dem["dem_h_deplaned"]
    mesh_triangle_id = _worker_dem["mesh_triangle_id"]
    mesh_triangle_means = _worker_dem["mesh_triangle_means"]
    vertices = _worker_mesh["vertices"]
    triangles = _worker_mesh["triangles"]
    case = _worker_case
    mode_limit = int(case["n_modes"])
    m_values = np.arange(-mode_limit, mode_limit + 1)
    n_values = np.arange(-mode_limit, mode_limit + 1)
    triangle_vertices = vertices[triangles[triangle_number]]
    result = {
        "tri_num": triangle_number,
        "mesh_name": case.get("mesh_name") or "regular",
        "mesh_kind": _worker_mesh.get("mesh_kind", "unknown"),
        "mesh_version": _worker_mesh.get("mesh_version", ""),
        "mesh_signature": _worker_mesh.get("mesh_signature", ""),
        "cell_size_km": float(case["cell_size_km"]),
        "nominal_dx_km": float(case.get("nominal_dx_km", case["cell_size_km"])),
        "n_modes": int(case["n_modes"]),
        "window": case["window_strategy"],
        "weight": case["weight_type"],
        "expansion": float(case["window_expansion"]),
        "oversample": float(case["oversample"]),
        "csa_signed_limit": int(case["csa_sparse_modes"] if case["csa_sparse_modes"] is not None else 2 * int(case["n_modes"])),
        "skip": False,
    }
    mask, window = get_analysis_window(dem_xy, triangle_vertices, case)
    mask = mask & np.isfinite(dem_h)
    n_window = int(np.sum(mask))
    if n_window < case["min_points_per_triangle"]:
        result["skip"] = True
        return result
    x_local, y_local = local_window_coordinates(dem_xy, mask, window)
    h_local = dem_h[mask].copy()
    mesh_ids_local = mesh_triangle_id[mask]
    h_mean = float(mesh_triangle_means[triangle_number])
    target_mask = (mesh_triangle_id == triangle_number) & np.isfinite(dem_h)
    n_triangle = int(np.sum(target_mask))
    if n_triangle < 3:
        result["skip"] = True
        return result
    x_tri, y_tri = points_to_window_coordinates(dem_xy[target_mask], window)
    h_tri = dem_h[target_mask].copy()
    true_var = float(np.var(h_tri))
    weights = compute_local_voronoi_weights(x_local, y_local, window["Lx"], window["Ly"], window=window) if case["weight_type"] == "voronoi" else None
    raw_spectrum = compute_nufft_coefficients(x_local, y_local, h_local, m_values, n_values, window["Lx"], window["Ly"], case["oversample"], case["kernel_half_width"], case["kernel_beta"], "optimized", weights)
    enufft = select_sparse_conjugate_modes(raw_spectrum, m_values, n_values, case["enufft_k_min"], case["enufft_k_max"] if case["enufft_k_max"] is not None else mode_limit, case["enufft_alpha_min"], case["enufft_alpha_max"], case["enufft_delta"], case["enufft_w1"], case["enufft_w2"])
    h_enufft = enufft["spectrum"]
    csa_reference_window = csa_window_strategy(case["window_strategy"])
    csa_valid = csa_reference_window is not None
    csa_window = window
    csa_mask = mask
    x_local_csa = x_local
    y_local_csa = y_local
    h_local_csa = h_local
    x_tri_csa = x_tri
    y_tri_csa = y_tri
    if csa_valid and csa_reference_window != case["window_strategy"]:
        csa_case = dict(case)
        csa_case["window_strategy"] = csa_reference_window
        csa_mask, csa_window = get_analysis_window(dem_xy, triangle_vertices, csa_case)
        csa_mask = csa_mask & np.isfinite(dem_h)
        if int(np.sum(csa_mask)) < case["min_points_per_triangle"]:
            csa_valid = False
        else:
            x_local_csa, y_local_csa = local_window_coordinates(dem_xy, csa_mask, csa_window)
            h_local_csa = dem_h[csa_mask].copy()
            x_tri_csa, y_tri_csa = points_to_window_coordinates(dem_xy[target_mask], csa_window)
    if csa_valid:
        csa = compute_csa_spectrum(x_local_csa, y_local_csa, h_local_csa, x_tri_csa, y_tri_csa, h_tri, m_values, n_values, csa_window["Lx"], csa_window["Ly"], case["csa_lambda_fa"], case["csa_lambda_sa"], case["csa_sparse_modes"], case["csa_chunk_size"])
        h_csa = csa["spectrum"]
    else:
        csa = {"spectrum": np.zeros((2 * mode_limit + 1, 2 * mode_limit + 1), dtype=complex), "selected_modes": np.zeros((0, 2), dtype=int), "energy_sorted": np.array([], dtype=float)}
        h_csa = csa["spectrum"]
    h_recon_en = reconstruct_at_points(h_enufft, x_tri, y_tri, m_values, n_values, window["Lx"], window["Ly"])
    rmse_en = float(np.sqrt(np.mean((h_recon_en - h_tri) ** 2)))
    sigma_tri = float(np.std(h_tri))
    rel_rmse_en = relative_rmse(rmse_en, sigma_tri)
    m_dom_en, n_dom_en, amp_dom_en = find_dominant_mode_pair(h_enufft, m_values, n_values)
    if csa_valid:
        h_recon_csa = reconstruct_at_points(h_csa, x_tri_csa, y_tri_csa, m_values, n_values, csa_window["Lx"], csa_window["Ly"])
        rmse_csa = float(np.sqrt(np.mean((h_recon_csa - h_tri) ** 2)))
        rel_rmse_csa = relative_rmse(rmse_csa, sigma_tri)
        m_dom_csa, n_dom_csa, amp_dom_csa = find_dominant_mode_pair(h_csa, m_values, n_values)
        dom_dir_csa = compute_wave_direction(m_dom_csa, n_dom_csa, csa_window["Lx"], csa_window["Ly"])
        var_csa = spectral_variance(h_csa, m_values, n_values)
        sorted_amplitudes_csa = compute_sorted_spectral_amplitudes(h_csa, m_values, n_values).tolist()
        csa_signed_modes_selected = int(len(csa["selected_modes"]))
        csa_signed_modes_used = count_signed_nonzero_modes(h_csa, m_values, n_values)
        csa_pairs_used = count_unique_mode_pairs(csa["selected_modes"], include_dc=False)
    else:
        rmse_csa = np.nan
        rel_rmse_csa = np.nan
        m_dom_csa = np.nan
        n_dom_csa = np.nan
        amp_dom_csa = np.nan
        dom_dir_csa = np.nan
        var_csa = np.nan
        sorted_amplitudes_csa = [np.nan] * ((2 * mode_limit + 1) ** 2)
        csa_signed_modes_selected = 0
        csa_signed_modes_used = 0
        csa_pairs_used = np.nan
    v0, v1, v2 = triangle_vertices
    triangle_area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]))
    centroid = np.mean(triangle_vertices, axis=0)
    n_window_target = int(np.sum(mesh_ids_local == triangle_number))
    n_encroached = int(n_window - n_window_target)
    n_neighbour_triangles = int(len(np.unique(mesh_ids_local[mesh_ids_local != triangle_number])))
    result.update({"metric_version": alps_metric_version, "rmse_en": float(rmse_en), "rel_rmse_en": float(rel_rmse_en), "rmse_csa": float(rmse_csa), "rel_rmse_csa": float(rel_rmse_csa), "K_star": int(enufft["k_star"]), "K_max": int(enufft["k_max"]), "Neff": float(enufft["n_eff"]), "S_enufft": float(enufft["s_delta"]), "C_enufft": float(enufft["c_measure"]), "alpha_enufft": float(enufft["alpha_c"]), "power_retained": float(enufft["power_retained"]), "csa_valid": bool(csa_valid), "csa_reference_window": csa_reference_window, "csa_Lx": float(csa_window["Lx"]) if csa_valid else np.nan, "csa_Ly": float(csa_window["Ly"]) if csa_valid else np.nan, "csa_window_points": int(np.sum(csa_mask)) if csa_valid else 0, "csa_signed_modes_selected": csa_signed_modes_selected, "csa_signed_modes_used": csa_signed_modes_used, "csa_pairs_used": float(csa_pairs_used), "dom_m_en": m_dom_en, "dom_n_en": n_dom_en, "dom_amp_en": amp_dom_en, "dom_dir_en": compute_wave_direction(m_dom_en, n_dom_en, window["Lx"], window["Ly"]), "dom_m_csa": m_dom_csa, "dom_n_csa": n_dom_csa, "dom_amp_csa": amp_dom_csa, "dom_dir_csa": dom_dir_csa, "true_var": true_var, "var_en": spectral_variance(h_enufft, m_values, n_values), "var_csa": var_csa, "sorted_amplitudes_en": compute_sorted_spectral_amplitudes(h_enufft, m_values, n_values).tolist(), "sorted_amplitudes_csa": sorted_amplitudes_csa, "spectrum_enufft": h_enufft, "spectrum_csa": h_csa, "n_window_points": n_window, "n_triangle_points": n_triangle, "n_encroached_points": n_encroached, "n_neighbour_triangles_in_window": n_neighbour_triangles, "triangle_area": float(triangle_area), "centroid": centroid.tolist(), "tri_verts": triangle_vertices.tolist(), "Lx_local": float(window["Lx"]), "Ly_local": float(window["Ly"]), "h_mean": h_mean, "h_peak": float(np.max(np.abs(h_tri)))})
    return result


# Summarise successful triangle diagnostics for one Alps run.
def summarise_alps_results(good, case):
    true_v = np.array([row["true_var"] for row in good], dtype=float)
    var_en = np.array([row["var_en"] for row in good], dtype=float)
    var_csa = np.array([row["var_csa"] for row in good], dtype=float)
    rel_en = np.array([row["rel_rmse_en"] for row in good], dtype=float)
    rel_csa = np.array([row["rel_rmse_csa"] for row in good], dtype=float)
    k_star = np.array([row["K_star"] for row in good], dtype=float)
    csa_pairs = np.array([row["csa_pairs_used"] for row in good], dtype=float)
    mesh_name = case.get("mesh_name") or "regular"
    return {"metric_version": alps_metric_version, "mesh_name": mesh_name, "mesh_kind": good[0].get("mesh_kind", "unknown"), "mesh_version": good[0].get("mesh_version", ""), "mesh_signature": good[0].get("mesh_signature", ""), "cell_size_km": float(case["cell_size_km"]), "nominal_dx_km": float(case.get("nominal_dx_km", case["cell_size_km"])), "n_triangles": len(good), "n_modes": int(case["n_modes"]), "window": case["window_strategy"], "weight": case["weight_type"], "expansion": float(case["window_expansion"]), "oversample": float(case["oversample"]), "csa_signed_limit": int(case["csa_sparse_modes"] if case["csa_sparse_modes"] is not None else 2 * int(case["n_modes"])), "csa_valid_triangles": int(np.sum([bool(row.get("csa_valid", False)) for row in good])), "rel_rmse_en_median": finite_median(rel_en), "rel_rmse_csa_median": finite_median(rel_csa), "rel_rmse_en_max": finite_max(rel_en), "rel_rmse_csa_max": finite_max(rel_csa), "K_star_median": finite_median(k_star), "K_star_max": finite_max(k_star), "csa_pairs_median": finite_median(csa_pairs), "true_var_median": finite_median(true_v), "var_en_median": finite_median(var_en), "var_csa_median": finite_median(var_csa), "var_ratio_en_median": finite_ratio_median(var_en, true_v, relative_rmse_variance_floor), "var_ratio_csa_median": finite_ratio_median(var_csa, true_v, relative_rmse_variance_floor), "n_window_points_median": float(np.nanmedian([row["n_window_points"] for row in good])), "n_triangle_points_median": float(np.nanmedian([row["n_triangle_points"] for row in good])), "n_encroached_points_median": float(np.nanmedian([row["n_encroached_points"] for row in good]))}


# Flatten one result row to scalar CSV fields.
def scalar_alps_result_row(row):
    skip_keys = {"spectrum_enufft", "spectrum_csa", "sorted_amplitudes_en", "sorted_amplitudes_csa"}
    out = {}
    for key, value in row.items():
        if key in skip_keys:
            continue
        if isinstance(value, list):
            if key == "centroid":
                out["centroid_x"] = value[0]
                out["centroid_y"] = value[1]
            continue
        out[key] = value
    return out


# Format a numeric value for a compact run tag.
def tag_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


# Build the compact tag for one Alps run configuration.
def make_alps_config_tag(case):
    sparse = case["csa_sparse_modes"] if case["csa_sparse_modes"] is not None else 2 * int(case["n_modes"])
    mesh_name = case.get("mesh_name")
    if mesh_name and normalize_mesh_name(mesh_name) != "regular":
        return f"_{normalize_mesh_name(mesh_name)}_N{case['n_modes']}_{case['window_strategy']}_{case['weight_type']}_eta{tag_number(case['window_expansion'])}_os{tag_number(case['oversample'])}_csa{sparse}_dx{tag_number(case['cell_size_km'])}"
    return f"_N{case['n_modes']}_{case['window_strategy']}_{case['weight_type']}_eta{tag_number(case['window_expansion'])}_os{tag_number(case['oversample'])}_csa{sparse}_dx{tag_number(case['cell_size_km'])}"


# Build the compact aggregate sweep tag for one Alps mesh and mode pair.
def make_alps_sweep_tag(case):
    mesh_name = case.get("mesh_name")
    if mesh_name and normalize_mesh_name(mesh_name) != "regular":
        return f"_{normalize_mesh_name(mesh_name)}_N{case['n_modes']}_dx{tag_number(case['cell_size_km'])}"
    return f"_N{case['n_modes']}_dx{tag_number(case['cell_size_km'])}"


# Build sparse coefficient rows from one stored spectrum.
def build_alps_modes_rows(case, row, method, spectrum):
    rows = []
    mode_limit = int(case["n_modes"])
    m_values = np.arange(-mode_limit, mode_limit + 1)
    n_values = np.arange(-mode_limit, mode_limit + 1)
    amplitude = np.abs(spectrum)
    keep = amplitude > 1e-15
    keep[mode_limit, mode_limit] = True
    for m_index, m_mode in enumerate(m_values):
        for n_index, n_mode in enumerate(n_values):
            if not keep[m_index, n_index]:
                continue
            value = spectrum[m_index, n_index]
            rows.append({"method": method, "tri_num": int(row["tri_num"]), "m_mode": int(m_mode), "n_mode": int(n_mode), "coeff_real": float(np.real(value)), "coeff_imag": float(np.imag(value)), "coeff_abs": float(np.abs(value)), "is_dc": bool(m_mode == 0 and n_mode == 0)})
    rows.sort(key=lambda item: -item["coeff_abs"])
    for rank, item in enumerate(rows, start=1):
        item["rank_abs"] = rank
    return rows


# Build sorted-amplitude rows for plot-ready CSV spectra.
def build_alps_spectra_rows(row, rank_count):
    rows = []
    for method, values in (("enufft", row["sorted_amplitudes_en"]), ("csa", row["sorted_amplitudes_csa"])):
        take = min(int(rank_count), len(values))
        for rank in range(take):
            value = float(values[rank])
            if not np.isfinite(value):
                continue
            rows.append({"tri_num": int(row["tri_num"]), "method": method, "rank": rank + 1, "amplitude": value})
    return rows
