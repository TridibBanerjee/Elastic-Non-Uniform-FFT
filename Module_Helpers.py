# Module_Helpers.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores small general helper functions shared by ENUFFT cases.

import numpy as np


# Compute the positive planar area of one polygon.
def polygon_area_2d(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 3:
        return 0.0
    x_values = vertices[:, 0]
    y_values = vertices[:, 1]
    return 0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))


# Clip one polygon to an axis-aligned rectangular box.
def clip_polygon_to_box(vertices, x_min, x_max, y_min, y_max):
    clipped = np.asarray(vertices, dtype=float)
    if len(clipped) == 0:
        return clipped
    planes = np.array([
        [1.0, 0.0, -x_min],
        [-1.0, 0.0, x_max],
        [0.0, 1.0, -y_min],
        [0.0, -1.0, y_max],
    ], dtype=float)
    for a_value, b_value, c_value in planes:
        output = []
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


# Compute the area of one polygon after clipping it to an axis-aligned box.
def polygon_box_intersection_area(vertices, x_min, x_max, y_min, y_max):
    clipped = clip_polygon_to_box(vertices, x_min, x_max, y_min, y_max)
    return polygon_area_2d(clipped)


# Test which points lie inside one triangle.
def points_in_triangle_mask(points, triangle_vertices):
    v0, v1, v2 = triangle_vertices
    d1 = (points[:, 0] - v1[0]) * (v0[1] - v1[1]) - (v0[0] - v1[0]) * (points[:, 1] - v1[1])
    d2 = (points[:, 0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (points[:, 1] - v2[1])
    d3 = (points[:, 0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (points[:, 1] - v0[1])
    has_negative = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
    has_positive = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
    return ~(has_negative & has_positive)


# Compute the planar area of one triangle.
def triangle_area_2d(v0, v1, v2):
    return 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]))


# Write enough digits that saved tables recover the same floating-point value.
def format_float(value):
    return f"{float(value):.18e}"
