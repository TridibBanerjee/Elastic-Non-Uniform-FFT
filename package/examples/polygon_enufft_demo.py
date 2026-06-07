#!/usr/bin/env python3
"""Minimal polygon ENUFFT example."""

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

import numpy as np

from enufft import EMSConfig, WindowConfig, build_analysis_window, enufft_on_polygon


polygon = np.array([[0.10, 0.16], [1.35, 0.02], [1.72, 0.82], [1.05, 1.55], [0.16, 1.18]])
window_config = WindowConfig(support="square", alignment="centroid", expansion=1.42)
rng = np.random.default_rng(8)
points = rng.uniform([-0.45, -0.45], [2.05, 2.05], size=(3600, 2))
_, window = build_analysis_window(points, polygon, window_config)
local = window.to_local(points)
values = 900.0 * np.cos(2.0 * np.pi * (3 * local[:, 0] / window.Lx + 2 * local[:, 1] / window.Ly) + 0.35)

result = enufft_on_polygon(
    points,
    values,
    polygon,
    mode_limit=6,
    window_config=window_config,
    ems_config=EMSConfig(k_min=1, k_max=6, alpha_min=0.0, alpha_max=0.78),
    weight_type="voronoi",
)

print("Selected canonical mode pairs:")
print(result.selected_modes)
print(f"mode pairs retained: {result.mode_pair_count}")
print(f"signed modes retained: {result.signed_mode_count}")
print(f"power retained: {result.power_retained:.4f}")
