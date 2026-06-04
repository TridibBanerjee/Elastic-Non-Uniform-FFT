# Code_Alps_Preprocess.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This script builds the local Alps DEM preprocessing archive from SRTM tiles.

import sys

from Module_Alps import dem_dir, preprocessed_dem, preprocess_alps_dem


tile_dir = dem_dir / "tiles_hgt"
block_size = 30
smooth_km = 5.0
min_elev = -500.0
hires_subsample = 1
hires_dtype = "float32"


# Print one progress line immediately.
def progress(message=""):
    print(message, flush=True)


# Select the raw SRTM tile directory.
def raw_tile_dir():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 Code_Alps_Preprocess.py [raw_tile_dir]")
    if len(sys.argv) == 2:
        return sys.argv[1]
    return tile_dir


# Build the local Alps preprocessing archive.
def main():
    source_dir = raw_tile_dir()
    progress("Alps SRTM preprocessing")
    progress(f"  Raw tiles {source_dir}")
    progress(f"  Output {preprocessed_dem}")
    result = preprocess_alps_dem(source_dir, preprocessed_dem, block_size, smooth_km, min_elev, smooth=True, hires_subsample=hires_subsample, hires_dtype=hires_dtype)
    progress(f"  Tiles loaded {result['tiles_loaded']}, missing {result['tiles_missing']}")
    progress(f"  Raw shape {result['raw_shape']}")
    progress(f"  Coarse shape {result['coarse_shape']}")
    progress(f"  High-resolution shape {result['hires_shape']}")
    progress(f"  Wrote {result['output_path']}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
