"""Polygonal analysis windows and support masks for ENUFFT."""

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree


SupportKind = Literal["square", "window", "box", "polygon", "triangle", "circle"]
AlignmentKind = Literal["centroid", "edge_aligned", "axis_aligned"]


@dataclass(frozen=True)
class WindowConfig:
    """Settings for building a square analysis domain from a polygon.

    The Fourier domain is always a square of side ``side_length``. If
    ``side_length`` is not supplied, it is computed as ``expansion`` times the
    larger axis extent of the polygon after the requested alignment transform.
    ``support`` decides which points inside that square are active samples.
    """

    support: SupportKind = "square"
    alignment: AlignmentKind = "centroid"
    expansion: float = 1.0
    side_length: float | None = None
    center: tuple[float, float] | None = None
    circle_radius: float | None = None

    def normalized_support(self) -> str:
        if self.support in {"square", "window", "box"}:
            return "square"
        if self.support in {"polygon", "triangle"}:
            return "polygon"
        if self.support == "circle":
            return "circle"
        raise ValueError(f"Unknown support {self.support!r}")

    def validate(self) -> None:
        self.normalized_support()
        if self.alignment not in {"centroid", "edge_aligned", "axis_aligned"}:
            raise ValueError("alignment must be 'centroid', 'edge_aligned', or 'axis_aligned'")
        if self.expansion <= 0.0:
            raise ValueError("expansion must be positive")
        if self.side_length is not None and self.side_length <= 0.0:
            raise ValueError("side_length must be positive when supplied")
        if self.circle_radius is not None and self.circle_radius <= 0.0:
            raise ValueError("circle_radius must be positive when supplied")


@dataclass(frozen=True)
class AnalysisWindow:
    """A square local coordinate system and its active support."""

    support: str
    alignment: str
    x0: float
    y0: float
    Lx: float
    Ly: float
    center: np.ndarray
    center_local: np.ndarray
    polygon_local: np.ndarray
    polygon_aligned: np.ndarray
    area: float
    theta: float = 0.0
    cos_t: float = 1.0
    sin_t: float = 0.0
    radius: float | None = None

    @property
    def side_length(self) -> float:
        return float(self.Lx)

    def to_local(self, points) -> np.ndarray:
        return np.column_stack(points_to_window_coordinates(points, self))

    def to_global(self, local_points) -> np.ndarray:
        local_points = np.asarray(local_points, dtype=float)
        source = np.column_stack([local_points[:, 0] + self.x0, local_points[:, 1] + self.y0])
        if self.alignment == "edge_aligned":
            x_global = self.cos_t * source[:, 0] + self.sin_t * source[:, 1]
            y_global = -self.sin_t * source[:, 0] + self.cos_t * source[:, 1]
            return np.column_stack([x_global, y_global])
        return source

    def support_mask(self, local_points) -> np.ndarray:
        local_points = np.asarray(local_points, dtype=float)
        in_square = (
            (local_points[:, 0] >= 0.0)
            & (local_points[:, 0] <= self.Lx)
            & (local_points[:, 1] >= 0.0)
            & (local_points[:, 1] <= self.Ly)
        )
        if self.support == "square":
            return in_square
        if self.support == "polygon":
            return in_square & points_in_polygon(local_points, self.polygon_local)
        if self.support == "circle":
            if self.radius is None:
                raise ValueError("circle support requires a radius")
            return in_square & (np.linalg.norm(local_points - self.center_local, axis=1) <= self.radius)
        raise ValueError(f"Unknown support {self.support!r}")


def polygon_area(vertices) -> float:
    """Compute the positive planar area of one polygon."""

    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2)")
    if len(vertices) < 3:
        return 0.0
    x_values = vertices[:, 0]
    y_values = vertices[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def _polygon_centroid(vertices) -> np.ndarray:
    area_signed = 0.5 * (
        np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
        - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
    )
    if abs(area_signed) <= 1e-15:
        return np.mean(vertices, axis=0)
    factor = (
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - np.roll(vertices[:, 0], -1) * vertices[:, 1]
    )
    cx = np.sum((vertices[:, 0] + np.roll(vertices[:, 0], -1)) * factor) / (6.0 * area_signed)
    cy = np.sum((vertices[:, 1] + np.roll(vertices[:, 1], -1)) * factor) / (6.0 * area_signed)
    return np.array([cx, cy], dtype=float)


def clip_polygon_to_box(vertices, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    """Clip one polygon to an axis-aligned rectangular box."""

    clipped = np.asarray(vertices, dtype=float)
    if len(clipped) == 0:
        return clipped
    planes = np.array(
        [
            [1.0, 0.0, -x_min],
            [-1.0, 0.0, x_max],
            [0.0, 1.0, -y_min],
            [0.0, -1.0, y_max],
        ],
        dtype=float,
    )
    for a_value, b_value, c_value in planes:
        output: list[list[float]] = []
        x_prev, y_prev = clipped[-1]
        g_prev = a_value * x_prev + b_value * y_prev + c_value
        for x_curr, y_curr in clipped:
            g_curr = a_value * x_curr + b_value * y_curr + c_value
            if g_prev * g_curr < 0.0:
                t_cross = g_prev / (g_prev - g_curr)
                output.append([x_prev + t_cross * (x_curr - x_prev), y_prev + t_cross * (y_curr - y_prev)])
            if g_curr >= 0.0:
                output.append([x_curr, y_curr])
            x_prev, y_prev, g_prev = x_curr, y_curr, g_curr
        clipped = np.asarray(output, dtype=float)
        if len(clipped) == 0:
            return clipped
    return clipped


def points_in_polygon(points, polygon, tolerance: float | None = None) -> np.ndarray:
    """Return a boolean mask for points inside or on the boundary of a polygon."""

    points = np.asarray(points, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must have shape (N, 2)")
    if len(polygon) < 3:
        return np.zeros(points.shape[0], dtype=bool)

    x = points[:, 0]
    y = points[:, 1]
    px = polygon[:, 0]
    py = polygon[:, 1]
    scale = max(float(np.ptp(px)), float(np.ptp(py)), 1.0)
    tol = 1e-12 * scale if tolerance is None else float(tolerance)

    inside = np.zeros(points.shape[0], dtype=bool)
    boundary = np.zeros(points.shape[0], dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= crosses

        edge_dx = xj - xi
        edge_dy = yj - yi
        cross = (x - xi) * edge_dy - (y - yi) * edge_dx
        dot = (x - xi) * edge_dx + (y - yi) * edge_dy
        edge_len2 = edge_dx**2 + edge_dy**2
        on_segment = (np.abs(cross) <= tol * max(np.sqrt(edge_len2), 1.0)) & (dot >= -tol) & (dot <= edge_len2 + tol)
        boundary |= on_segment
        j = i
    return inside | boundary


def _longest_edge_angle(vertices: np.ndarray) -> float:
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    return float(np.arctan2(edge[1], edge[0]))


def _rotate_points(points: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
    return np.column_stack([cos_t * points[:, 0] - sin_t * points[:, 1], sin_t * points[:, 0] + cos_t * points[:, 1]])


def build_analysis_window(points, polygon, config: WindowConfig | None = None) -> tuple[np.ndarray, AnalysisWindow]:
    """Build an analytic square window from a polygon and mask sample points."""

    config = WindowConfig() if config is None else config
    config.validate()
    points = np.asarray(points, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("polygon must have shape (N, 2) with at least three vertices")

    theta = _longest_edge_angle(polygon) if config.alignment == "edge_aligned" else 0.0
    cos_t = float(np.cos(-theta))
    sin_t = float(np.sin(-theta))
    points_aligned = _rotate_points(points, cos_t, sin_t) if config.alignment == "edge_aligned" else points.copy()
    polygon_aligned = _rotate_points(polygon, cos_t, sin_t) if config.alignment == "edge_aligned" else polygon.copy()

    if config.center is None:
        center = _polygon_centroid(polygon_aligned)
    else:
        center = np.asarray(config.center, dtype=float)
        if center.shape != (2,):
            raise ValueError("center must contain exactly two coordinates")
        if config.alignment == "edge_aligned":
            center = _rotate_points(center.reshape(1, 2), cos_t, sin_t)[0]

    extent = max(float(np.ptp(polygon_aligned[:, 0])), float(np.ptp(polygon_aligned[:, 1])), 1e-12)
    side = float(config.side_length) if config.side_length is not None else float(config.expansion * extent)
    half = 0.5 * side
    lo_window = center - half
    polygon_local = polygon_aligned - lo_window
    center_local = center - lo_window
    support = config.normalized_support()

    local_points = points_aligned - lo_window
    in_square = (
        (local_points[:, 0] >= 0.0)
        & (local_points[:, 0] <= side)
        & (local_points[:, 1] >= 0.0)
        & (local_points[:, 1] <= side)
    )
    radius = float(config.circle_radius) if config.circle_radius is not None else half
    if support == "square":
        mask = in_square
        support_area = side * side
    elif support == "polygon":
        mask = in_square & points_in_polygon(local_points, polygon_local)
        support_area = polygon_area(clip_polygon_to_box(polygon_local, 0.0, side, 0.0, side))
    elif support == "circle":
        mask = in_square & (np.linalg.norm(local_points - center_local, axis=1) <= radius)
        support_area = float(np.pi * radius**2)
    else:
        raise ValueError(f"Unknown support {support!r}")

    window = AnalysisWindow(
        support=support,
        alignment=config.alignment,
        x0=float(lo_window[0]),
        y0=float(lo_window[1]),
        Lx=side,
        Ly=side,
        center=center,
        center_local=center_local,
        polygon_local=polygon_local,
        polygon_aligned=polygon_aligned,
        area=float(support_area),
        theta=float(theta),
        cos_t=cos_t,
        sin_t=sin_t,
        radius=radius if support == "circle" else None,
    )
    return mask, window


def points_to_window_coordinates(points, window: AnalysisWindow) -> tuple[np.ndarray, np.ndarray]:
    """Map physical points into local analysis-window coordinates."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if window.alignment == "edge_aligned":
        aligned = _rotate_points(points, window.cos_t, window.sin_t)
    else:
        aligned = points
    local = aligned - np.array([window.x0, window.y0], dtype=float)
    return local[:, 0], local[:, 1]


def compute_local_voronoi_weights(
    x_values,
    y_values,
    domain_length_x: float,
    domain_length_y: float,
    window: AnalysisWindow | None = None,
    grid_res: int = 96,
) -> np.ndarray:
    """Approximate local sample areas by nearest-neighbour ownership."""

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    point_count = len(x_values)
    if point_count == 0:
        return np.array([], dtype=float)
    support_area = float(window.area) if window is not None else float(domain_length_x * domain_length_y)
    if point_count < 4 or domain_length_x <= 0.0 or domain_length_y <= 0.0:
        return np.ones(point_count, dtype=float) * (support_area / max(point_count, 1))

    gx = np.linspace(0.0, domain_length_x, grid_res, endpoint=False) + domain_length_x / (2.0 * grid_res)
    gy = np.linspace(0.0, domain_length_y, grid_res, endpoint=False) + domain_length_y / (2.0 * grid_res)
    x_grid, y_grid = np.meshgrid(gx, gy, indexing="xy")
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    if window is not None:
        grid_points = grid_points[window.support_mask(grid_points)]
    if len(grid_points) == 0:
        return np.ones(point_count, dtype=float) * (support_area / point_count)

    _, owner = cKDTree(np.column_stack([x_values, y_values])).query(grid_points, k=1)
    counts = np.bincount(owner, minlength=point_count).astype(float)
    weights = counts * (support_area / len(grid_points))
    weights = np.maximum(weights, 1e-12 * support_area / point_count)
    return weights * (support_area / np.sum(weights))
