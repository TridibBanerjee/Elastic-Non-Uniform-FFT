# ENUFFT Plot Documentation

[Back to documentation index](README.md)

## Plot Files

Case-level plotting drivers for figure construction from CSV outputs.

#### `Plot_Alps.py`

Plot driver for the Alps preprocessing composite and pooled parameter-sweep summaries.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $D_{\mathrm{csv}}$ | `csv_dir` | `./csv` | input directory for Alps CSV outputs |
| $D_{\mathrm{fig}}$ | `figure_dir` | `./figures` | output directory for Alps figures |
| $A_{\mathrm{DEM}}$ | `preprocessed_dem` | `./srtm_alps/alps_dem_processed.npz` | DEM archive used by maps and preprocessing composite |
| $N_x^{\mathrm{hires}}$ | `target_hires_nx` | `6601` | target high-resolution raster width for the composite |
| $N_x^{\mathrm{stage}}$ | `target_stage_nx` | `560` | target processed-surface raster width |
| $\Delta_{\mathrm{cell}}$ | `fallback_cell_size` | `80.0` km | fallback mesh spacing for map panels |
| $\mathcal T_{\mathrm{term}}$ | `terminal_only_requested` | boolean | branch selector for terminal-summary output |
| $\mathcal L_{\mathrm{term}}$ | `terminal_true_values`, `terminal_false_values` | accepted flag values | parser vocabulary for terminal-summary mode |
| $\mathcal S_{\mathrm{term}}$ | return of `make_terminal_store` | Physical, CSA, Square, Triangle, Circle, ENUFFT pooled | pooled terminal metric store |
| $C_{\mathrm{method}}$ | `method_color` | ENUFFT `#0072b2`, CSA `#4d4d4d` | method colors |
| $C_{\mathrm{summary}}$ | `summary_method_colors` | CSA, Square, Tri., Circle colors | pooled sweep colors |
| $R_\oplus$ | `earth_radius_km` | `6371.0` km | map projection radius |
| $\Delta_{\mathrm{R2B5}}$ | `icon_r2b5_effective_dx_km` | `78.90625` km | reference ICON-style mesh spacing |

##### `apply_alps_style`

The Alps figure font, tick, hatch, spine, and backend settings are applied by the function. `Agg` rendering and transparent export are kept consistent with the other plot scripts.

##### `parse_scalar`

The function converts one CSV field to the plotting scalar used downstream.

$$
\mathrm{""}\mapsto \mathrm{NaN},\quad
\mathrm{"true"}\mapsto \mathrm{True},\quad
\mathrm{"false"}\mapsto \mathrm{False},\quad
\mathrm{"1.25"}\mapsto 1.25.
$$

Finite floating-point values with integer value are converted to integers so triangle ids and ranks remain discrete.

##### `read_csv_rows`

For a CSV table with rows $r_i$ and columns $c_j$, the function returns dictionaries after applying `parse_scalar` to each entry.

$$
\{\{c_j:v_{ij}\}_j\}_i.
$$

It is the only scalar table reader used by the Alps plot paths.

##### `tag_from_per_triangle_path`

Given a filename of the form `Banerjee_2026_Enufft_Alps_PerTriangle<tau>.csv`, the function returns the configuration tag $\tau$. This tag is later reused to locate the matching `Summary` and `Spectra` CSV files.

##### `paths_for_tag`

For a tag $\tau$, the function returns the per-triangle, summary, and spectra CSV paths built from the shared CSV directory and the three filename prefixes.

##### `is_current_proxy_tag`

A discovered tag is accepted only when it has the current fixed proxy-mesh shape `_r2b[45]_..._dx<cell>` with an optional `_first<N>` suffix.

This excludes older actual-grid or pre-port outputs without a mesh prefix and `_dx...` suffix.

##### `discover_tags`

The function scans `./csv` for per-triangle files and keeps only tags where all three tables exist.

$$
\tau\in\mathcal T_{\mathrm{complete}}
\iff
O_1\tau,\ O_2\tau,\ O_3\tau\ \mathrm{exist}.
$$

It also applies `is_current_proxy_tag`, which excludes partially written outputs and stale no-`dx` outputs.

##### `load_case`

For one complete tag $\tau$, the function reads scalar rows, the one summary row, and sorted spectra. It attaches

$$
\{a_j^{\mathrm{ENUFFT}}\}_{j=1}^{R},
\qquad
\{a_j^{\mathrm{CSA}}\}_{j=1}^{R}
$$

to each triangle row and reconstructs the centroid vector from `centroid_x` and `centroid_y`.

##### `read_spectra_csv`

The spectra CSV stores rows $(r,\mathrm{method},j,a_j)$. The function reconstructs the dictionary

$$
(r,\mathrm{method})\mapsto
[a_1,a_2,\ldots,a_R],
$$

with rank $j$ stored at zero-based Python index $j-1$.

##### `alps_figure_path`

For a figure stem `s`, the output path is `./figures/<s>.png`.

The figure directory is created before the path is returned.

##### `combined_line_patch_handle`

The function builds a legend handle

$$
H=(P,L),
$$

where $P$ is a hatched patch encoding the spread style and $L$ is a line, optionally with a marker, encoding the median curve. This matches the visual grammar used by the histogram and spectral-decay panels.

##### `spread_hatch_color`

For RGB color $c=(r,g,b)$, the hatch color is blended toward white.

$$
c' = c + (1-c)b,\qquad b=0.55.
$$

The returned alpha is $1$ because PDF hatches do not reliably preserve translucent RGBA edge colors.

##### `plot_hatched_spread`

Given rank coordinates $x_j$ and percentile envelopes $(\ell_j,u_j)$, the following band is drawn:

$$
\{(x,y):x=x_j,\ \ell_j\le y\le u_j\}
$$

as a translucent band. If a hatch is supplied, a second transparent filled band overlays the hatch pattern.

##### `decay_markevery`

For $N$ plotted ranks, the marker cadence is

$$
m=\max(1,\lceil\frac{N}{10}\rceil).
$$

This keeps roughly ten visible markers on each median spectral curve.

##### `plot_hatched_histogram`

For values $z_i$ and bin edges $b_k$, the histogram counts are

$$
c_k=\lvert\{i:b_k\le z_i\lt b_{k+1}\}\rvert.
$$

The count staircase is drawn, and for hatched methods a transparent bar layer with the same bin widths is added.

##### `window_support_name`

The function maps `square_*` tags to `Square`, `triangle_*` tags to `Tri.`, and `circle_*` tags to `Circle`.

This label is used for pooling ENUFFT rows by support family.

##### `detect_support_from_filename`

The function searches a filename for any known Alps window strategy and returns the support label from `window_support_name`. If no known strategy appears, it returns no label and the file is excluded from pooled support plots.

##### `detect_strategy_from_filename`

The function extracts the full strategy $\mathcal W$ from a tagged filename by matching one of the six known values, for example `_triangle_centroid_` maps to `triangle_centroid`.

##### `csa_pooled_spectra_key_from_filename`

For a spectra filename, the CSA de-duplication key is

$$
\kappa_{\mathrm{CSA}}=
(r,\mathcal W_{\mathrm{CSA}},\Delta_{\mathrm{cell}},
N_{\max}^{\mathrm{mode}},S_{\mathrm{CSA}},\eta).
$$

The numeric values are parsed from the configuration tag, with `p` and `m` converted back to decimal points and minus signs.

##### `csa_pooled_key`

For scalar rows, the same CSA de-duplication key $\kappa_{\mathrm{CSA}}$ is built from either the filename or the row fields. This avoids counting the same CSA square-reference result multiple times when several non-square ENUFFT supports point to the same CSA reference.

##### `make_pooled_method_store`

The pooled store is initialized with the method keys `CSA`, `Square`, `Tri.`, `Circle`, `ENUFFT`, and `Physical`; each method has `rel_rmse`, `spectra`, `by_tri`, and `samples` buckets.

Each method bucket holds scalar distributions, sorted amplitude curves, per-triangle map values, and centroid-tagged samples for physical-cell remapping.

##### `add_tri_value`

For triangle $r$ and metric key $k$, the function appends one finite value.

$$
\mathrm{store}[r,k]\leftarrow \mathrm{store}[r,k]\cup\{v\}.
$$

When the scalar CSV row carries a centroid, the value is also stored as $(c_x,c_y,k,v)$. Non-finite values are skipped so medians are not polluted by invalid CSA rows.

##### `row_centroid`

For a scalar row containing `centroid_x` and `centroid_y`, the function returns

$$
c_r=(x_{c,r},y_{c,r})
$$

when both entries are finite. If either coordinate is missing or non-finite, it returns no centroid. The pooled map loaders use this to remap old rows by physical location rather than by stale triangle order.

##### `load_pooled_sweep_diagnostics`

For every selected tag, the function reads scalar rows and sorted spectra, then fills the pooled store. ENUFFT rows are grouped by support family. CSA rows use the key $\kappa_{\mathrm{CSA}}$ so one square-reference CSA result contributes once even if several ENUFFT supports reuse it.

##### `terminal_only_requested`

The optional terminal-summary flag is parsed from `sys.argv`. With no flag, false is returned. With a true flag in $\mathcal L_{\mathrm{term}}$, true is returned. With a false flag, false is returned. Any other argument list is rejected.

##### `append_metric`

For a terminal store bucket, metric name, and scalar value, the helper appends only finite numbers.

$$
\mathcal S_{\mathrm{term}}[\ell,k]
\leftarrow
\mathcal S_{\mathrm{term}}[\ell,k]\cup\{z\}
$$

when $z$ can be parsed as a finite float.

##### `make_terminal_store`

The terminal store uses the method keys `Physical`, `CSA`, `Square`, `Triangle`, `Circle`, and `ENUFFT pooled`; each method has `rel_rmse`, `mode_count`, and `variance` buckets.

Every leaf is initialized as an empty list.

##### `terminal_support_label`

The pooled plot label `Tri.` is expanded to `Triangle` for terminal output.

$$
\mathrm{Tri.}\mapsto\mathrm{Triangle}.
$$

All other support labels pass through unchanged.

##### `load_terminal_metrics`

For every complete selected tag, per-triangle scalar rows are read and $\mathcal S_{\mathrm{term}}$ is filled. ENUFFT values are stored both under their support family and under `ENUFFT pooled`. CSA values are de-duplicated with $\kappa_{\mathrm{CSA}}$ so a square-reference CSA result is counted once even when several ENUFFT supports reuse it. Physical variance values from `true_var` are stored once per triangle in the `Physical` variance bucket.

##### `terminal_metric_labels`

The terminal row order is metric-dependent. For relative RMSE and retained-mode count, rows are

$$
\mathrm{CSA},\quad \mathrm{Square},\quad
\mathrm{Triangle},\quad \mathrm{Circle},\quad
\mathrm{ENUFFT\ pooled}.
$$

For variance, the physical reference is prepended:

$$
\mathrm{Physical},\quad \mathrm{CSA},\quad \mathrm{Square},\quad
\mathrm{Triangle},\quad \mathrm{Circle},\quad
\mathrm{ENUFFT\ pooled}.
$$

##### `terminal_stats`

For a finite terminal metric sample $\{z_i\}$, the function returns

$$
(
\min z,\mathrm{percentile}_{10}z,
\mathrm{median}z,
\mathrm{percentile}_{90}z,
\max z,N
).
$$

If no finite sample exists, it returns no statistics.

##### `print_terminal_metric`

For one terminal metric key $k$, a compact table is printed over the labels returned by `terminal_metric_labels`. For relative RMSE and retained-mode count, the row order is

$$
\mathrm{CSA},\quad \mathrm{Square},\quad
\mathrm{Triangle},\quad \mathrm{Circle},\quad
\mathrm{ENUFFT\ pooled}.
$$

For variance, the same table is printed with the `Physical` reference row.

Each row contains minimum, 10th percentile, median, 90th percentile, maximum, and sample count.

##### `grouped_complete_tags`

The function discovers complete current proxy tags, reads each summary row, reconstructs the plotting case, and groups tags by

$$
(g,N_{\max}^{\mathrm{mode}},\Delta_{\mathrm{cell}}).
$$

This is the same grouping used for pooled sweep-summary figures.

##### `print_terminal_summary`

Grouped complete tags are traversed in terminal-summary mode. For each group, the sweep tag, configuration-tag count, and three metric tables for relative RMSE, retained mode count, and variance are printed. Relative RMSE rows are included only when `sqrt(true_var) > 1 m`, so old CSVs with small-denominator relative values are filtered at read time.

##### `pooled_decay_curve`

For curves $a_{i,j}$ with unequal lengths, the function pads to `NaN` and computes

$$
\tilde a_j=\mathrm{median}_i(a_{i,j}),\qquad
a_j^{10}=\mathrm{percentile}_{10,i}(a_{i,j}),\qquad
a_j^{90}=\mathrm{percentile}_{90,i}(a_{i,j}).
$$

Only ranks with at least one finite value are retained.

##### `circular_mean_degrees`

For directions $\theta_i$, the circular mean is

$$
\bar\theta=
\mathrm{atan2}(
\frac1N\sum_i\sin\theta_i,\,
\frac1N\sum_i\cos\theta_i
),
$$

returned in degrees on $[-180^\circ,180^\circ)$.

##### `median_map_values`

For current CSV rows with matching geometry, the map value is accumulated by `tri_num`. When mesh centroids are supplied, centroid-tagged samples are first remapped to the nearest current mesh triangle within a scale-aware tolerance. This makes old scalar CSVs robust to stale triangle ordering while preserving the physical triangle identity. For each triangle $r$, the map value is either

$$
\mathrm{median}\{z_{r,i}\}_i
$$

or, for circular directions, `circular_mean_degrees`. Missing triangles remain `NaN`.

##### `match_colorbar_height`

The colorbar axis height is set equal to the reference map axis height.

$$
y_0^{\mathrm{cbar}}\leftarrow y_0^{\mathrm{ref}},
\qquad
h^{\mathrm{cbar}}\leftarrow h^{\mathrm{ref}}.
$$

The x-position and width of the colorbar axis are left unchanged.

##### `cross2`

For two planar vectors $a$ and $b$, the function returns

$$
a\times b=a_xb_y-a_yb_x.
$$

This scalar cross product is used to clip hatch lines to triangle edges.

##### `hatch_segments_for_triangle`

For hatch angle $\theta$, the line direction and normal are

$$
u=(\cos\theta,\sin\theta),\qquad v=(-\sin\theta,\cos\theta).
$$

Candidate lines $p=p_c+s v+t u$ are intersected with each triangle edge. When two or more edge intersections exist, the segment between the smallest and largest $t$ values is retained.

##### `add_direction_hatching`

For each triangle polygon and direction angle, this function calls `hatch_segments_for_triangle` and adds the resulting line collection. Positive and negative directions use different hatch colors so the hatching remains visible on both light and dark colormap regions.

##### `set_lonlat_axes`

Given projected axis limits, longitude ticks $\lambda_k$ and latitude ticks $\phi_k$ are projected with `lonlat_to_xy`. Only ticks whose projected coordinates lie inside the map limits are displayed.

##### `full_metric`

For a per-triangle scalar row set, the function builds a full mesh vector

$$
z_r=
\begin{cases}
\mathrm{row}[k], & r\ \mathrm{appears\ in\ the\ CSV},\\
\mathrm{NaN}, & \mathrm{otherwise}.
\end{cases}
$$

This vector can be attached directly to a `PolyCollection`.

##### `plot_diagnostics`

For one already loaded tag $\tau$, this helper rebuilds the six-panel diagnostic figure.

$$
\text{relative RMSE histogram }(\sqrt{V_{\mathrm{true}}}>1\,\mathrm{m}),\quad
\text{sorted amplitudes},\quad
K^\star,\quad
\theta_{\mathrm{ENUFFT}},\quad
\theta_{\mathrm{CSA}},\quad
\sigma^2\text{ maps}.
$$

`Banerjee_2026_Enufft_Alps_Diagnostics*.png` and `.pdf` are written when the helper is called from code. The helper is no longer called by the fixed `main` driver.

The RMSE histogram uses raw scalar values, while the sorted-amplitude panel uses

$$
\tilde a_j,\quad a_j^{10},\quad a_j^{90}
$$

from the per-triangle curves. The map panels expand CSV scalars to full mesh arrays with `full_metric`.

##### `plot_pooled_sweep_summary`

This function pools all selected tags and rebuilds the parameter-sweep figure with pooled relative RMSE over target triangles satisfying `sqrt(true_var) > 1 m`, pooled sorted amplitudes, pooled ENUFFT $K^\star$, pooled ENUFFT/CSA direction maps, and physical/ENUFFT/CSA variance maps. It writes `Banerjee_2026_Enufft_Alps_SweepSummary*.png` and `.pdf`.

The pooled scalar panels use the union of selected tags. For each triangle, direction maps use circular means and variance maps use finite medians across repeated appearances.

##### `finite_field`

The function replaces non-finite raster values by the finite median.

$$
z'_{ij}=
\begin{cases}
\mathrm{median}(z), & z_{ij}\notin\mathbb R,\\
z_{ij}, & \mathrm{otherwise}.
\end{cases}
$$

This prevents rendering helpers from receiving `NaN` surfaces.

##### `orient_y_ascending`

If $y$ is descending, the function sets $\bar j=N_y-1-j$ and reverses both $y$ and the raster rows.

$$
y'_j=y_{\bar j},\qquad
z'_{j,i}=z_{\bar j,i}.
$$

It calls `finite_field` before orientation.

##### `thin_grid`

For target width $N_x^\ast$, the stride is

$$
s=\max(1,\lceil\frac{N_x}{N_x^\ast}\rceil).
$$

The returned raster is `(x[0:s:Nx], y[0:s:Ny], z[0:s:Ny, 0:s:Nx])` and the stride $s$.

##### `regular_icon_like_triangles_for_viz`

This visualization mesh uses the same vertex-count rule as the compute mesh.

$$
N_x^v=\lceil\frac{x_{\max}-x_{\min}}{\Delta_{\mathrm{R2B5}}}\rceil+1,
\qquad
N_y^v=\lceil\frac{y_{\max}-y_{\min}}{\Delta_{\mathrm{R2B5}}}\rceil+1.
$$

It returns a lightweight object with the same `points`, `simplices`, and `nsimplex` attributes expected by the plotting helpers, backed by the deterministic structured mesh.

##### `deplane_on_triangles`

For visualization raster cells assigned to simplex $r$, the local residual is

$$
z'_{q}=z_q-\bar z_r,\qquad
\bar z_r=\frac1{Q_r}\sum_{p:r(p)=r}z_p.
$$

The returned residual grid is reshaped to the input raster shape.

##### `surface_colors`

The base colormap value $C(z)$ is multiplied by a hillshade factor $S(z)$.

$$
\mathrm{rgb}=\mathrm{clip}(C(z)\,[0.72+0.35S(z)]+0.02,0,1).
$$

The vertical exaggeration in the hillshade is larger for residual panels than for elevation panels.

##### `edge_fade_mask`

For raster cell distance $d_{ij}$ to the nearest edge and margin $m$, the alpha multiplier is

$$
\alpha_{ij}=\mathrm{clip}(\frac{d_{ij}}{m},0,1)^{0.75}.
$$

This fades the hero terrain relief toward the raster boundary.

##### `map_colors`

The hero-map RGB field blends white background and shaded relief.

$$
\mathrm{RGB}_{ij}
=(1-\alpha_{ij})\,1+\alpha_{ij}\,\mathrm{relief}_{ij},
$$

where $\alpha_{ij}$ increases with positive elevation and is multiplied by `edge_fade_mask`.

##### `earth_curvature`

For the 3D terrain panels, the shallow sag is

$$
c(x,y)=-\frac{(x-x_c)^2+(y-y_c)^2}{2R_\oplus}.
$$

The plotted vertical coordinate is the scaled terrain plus this curvature term.

##### `mesh_segments_3d`

Each triangle edge is sampled at ten evenly spaced points. The terrain height is interpolated as $z(x,y)$, then plotted as

$$
Z=z(x,y)z_{\mathrm{scale}}+c(x,y)+0.8.
$$

Only edge segments with finite interpolated heights are retained.

##### `load_processing_stages`

The function loads $A_{\mathrm{DEM}}$ and prepares four stages.

$$
H^{1''},\qquad H^{30''},\qquad H^{\mathrm{smooth}},\qquad
H^{\mathrm{deplane}}.
$$

It thins only the plotted copies, not the stored archive, and records source resolution, plotted raster stride, mesh object, and geographic extent.

##### `geometry_rings`

The Natural Earth geometry is normalized as

$$
\mathrm{Polygon}\rightarrow\{\mathrm{rings}\},
\qquad
\mathrm{MultiPolygon}\rightarrow\bigcup\{\mathrm{polygon\ rings}\}.
$$

Unsupported geometry types return an empty ring list.

##### `ring_overlaps_extent`

For ring coordinates $(\lambda_i,\phi_i)$ and map extent $[\lambda_0,\lambda_1]\times[\phi_0,\phi_1]$, the overlap test is

$$
\max_i\lambda_i\ge\lambda_0,\quad
\min_i\lambda_i\le\lambda_1,\quad
\max_i\phi_i\ge\phi_0,\quad
\min_i\phi_i\le\phi_1.
$$

##### `load_country_boundaries`

The function caches the Natural Earth GeoJSON in `./figures` and retains rings for the Alpine-context country set. The mathematical output is a list of visible boundary polylines

$$
\{(\mathrm{name},\{(\lambda_i,\phi_i)\}_i)\}.
$$

##### `draw_country_boundaries`

For each retained ring overlapping the map extent, the following polyline is drawn:

$$
(\lambda_i,\phi_i)_{i=1}^{N_{\mathrm{ring}}}
$$

with a thin grey stroke.

##### `add_hero_graticule`

The graticule is the set of constant-longitude and constant-latitude guide lines

$$
\lambda\in\{6,8,10,12,14\},\qquad
\phi\in\{45,46,47,48\}.
$$

Labels are drawn only when the tick lies inside the displayed extent.

##### `add_scale_bar`

At the lower-right map latitude $\phi_b$, the conversion from kilometres to longitude degrees is

$$
\Delta\lambda_{\mathrm{deg}}=
\frac{180}{\pi}\frac{\Delta x_{\mathrm{km}}}{R_\oplus\cos\phi_b}.
$$

Ticks are drawn at $0$, $50$, and $100$ km.

##### `clean_3d_axis`

The 3D axes are made orthographic, transparent, and tick-free. The data limits are padded by $55$ km in $x$ and $0.65\cdot55$ km in $y$, and the box aspect is set near

$$
(x:y:z)=(1.85:1.0:0.34).
$$

##### `add_floor_contours`

For a terrain stage $z(x,y)$, contour bands are projected onto the constant base plane

$$
Z=z_{\mathrm{base}}.
$$

Residual stages use evenly spaced diverging levels, while elevation stages use fixed metre levels from $250$ to $4000$ m.

##### `add_latlon_graticule`

Geographic ticks are projected by the 3D panel graticule to local $(x,y)$ coordinates, then drawn at

$$
Z=z_{\mathrm{base}}+0.35.
$$

This helper is optional and is disabled in the final deplaned panel by default.

##### `draw_terrain_on_axis`

For a stage height $z(x,y)$ and scale $s_z$, the plotted surface is

$$
Z(x,y)=s_z z(x,y)+c(x,y),
$$

where $c(x,y)$ is the Earth-curvature sag. It adds surface colors, projected floor contours, and optional mesh or graticule overlays.

##### `render_terrain_image`

The function renders one 3D terrain stage to an off-screen transparent PNG, finds the nontransparent alpha bounding box, pads it by 18 pixels, and returns the cropped RGBA array normalized to $[0,1]$.

##### `add_plain_label`

For a panel rectangle $(x_0,y_0,w,h)$, the label y-position is

$$
y_{\mathrm{label}}=
\begin{cases}
y_0+h-0.020, & \mathrm{inside},\\
y_0+h+0.018, & \mathrm{outside}.
\end{cases}
$$

The text is drawn in figure coordinates.

##### `add_rendered_panel`

The function creates an axes at the supplied rectangle, displays the pre-rendered RGBA terrain image with Lanczos interpolation, and removes all axes decoration. It performs placement only and does not alter the image data.

##### `add_hero_map`

The high-resolution local projected raster corners are converted by the hero map back to longitude-latitude extent, then drawn as

$$
\mathrm{RGB}(\lambda,\phi)=C(H^{1''}).
$$

It overlays graticule lines, country boundaries, the scale bar, country labels, and the source-resolution title.

##### `add_colorbars`

The elevation colorbar maps

$$
h\in[-100,4600]\ \mathrm{m}
$$

through the terrain colormap. The residual colorbar maps

$$
h'\in[-h'_{\max},h'_{\max}]
$$

through a centred diverging normalization with tick labels at negative limit, zero, and positive limit.

##### `build_viz_figure`

The preprocessing composite places the hero map and three rendered terrain stages on a $170\times120$ mm canvas. The residual color scale uses

$$
h'_{\max}=\mathrm{clip}(
\mathrm{percentile}_{98.5}(|h'|),500,1900
).
$$

The exported files are `Banerjee_2026_Enufft_Alps_Viz.png` and `.pdf`.

##### `mesh_name_from_tag`

For current proxy tags, the function extracts the leading mesh prefix.

Tags such as `_r2b5_N32_...` map to the mesh name `r2b5`.

Tags without an `r2b4` or `r2b5` prefix return no mesh name.

##### `has_summary_value`

This predicate accepts a parsed summary value only when it is present, finite when numeric, and not an empty or textual `nan` field.

##### `mesh_name_for_summary`

The plotting mesh is inferred first from the `mesh_name` summary column and then from the output tag. If neither source identifies a mesh, the legacy fallback is `"regular"`.

##### `case_for_summary`

For one summary row and tag, the helper rebuilds a plotting case with the inferred mesh, summary `cell_size_km` when present, and summary `n_modes` when present. This keeps pooled maps aligned with the mesh that produced the CSVs.

##### `mesh_for_case`

The function caches meshes by `(mesh_name, cell_size)`, so multiple tags from the same mesh group reuse one `build_alps_mesh` result during plotting.

##### `main`

The Alps style is applied, the preprocessing composite is rendered, complete current proxy sweep tags are discovered, tags are grouped by `(mesh_name, n_modes, cell_size_km)`, and one pooled sweep-summary PNG/PDF pair is written per group. With a valid DEM archive, the preprocessing composite is always an output. Sweep summaries are skipped when no complete CSV groups are present.

#### `Plot_Nufft.py`

Case-level figure builder from the saved numerical tables.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $L_x,L_y$ | `lx`, `ly` | `10.0`, `10.0` km | physical domain lengths used on the terrain axes |
| $x_q,y_q$ | `x_values`, `y_values` | $x_q,y_q\in[0,10)$ km | scattered terrain coordinates |
| $h_q^{(j)}$ | `terrain_cases` | `3 x 2000` sampled heights, with $\min h_q^{(j)}=0$ | sampled terrain heights for case $j$ |
| $\lvert e_{m,n}^{\mathrm{opt}}\rvert,\lvert e_{m,n}^{\mathrm{base}}\rvert$ | `optimized_errors`, `baseline_errors` | `5043` pooled values per kernel | pooled coefficient errors |
| $e$ | `log_optimized`, `log_baseline` | $\log_{10}(\lvert\cdot\rvert+10^{-14})$ | plotted variable $e=\log_{10}\lvert\,\cdot\,\rvert$ |
| $z_{\min},z_{\max}$ | `z_min`, `z_max` | `-` | shared terrain color limits |
| $\mathcal S$ | `style_values` | baseline `#f06b4f`, optimized `#4aa3df`, baseline median `#c0392b`, optimized median `#1f78b4`, cmap `YlOrRd_r` | style dictionary from the template |
| $\mathcal F$ | `figure` | `170 x 128` mm canvas | Matplotlib figure object |

##### `read_terrain_data`

The terrain table is reconstructed as

$$
\{(x_q,y_q,h_q^{\mathrm{multi}},h_q^{\mathrm{ridge}},h_q^{\mathrm{basin}}):q=1,\ldots,Q\}.
$$

The function returns $\{x_q\}$, $\{y_q\}$, and the three sampled terrain arrays.

##### `read_error_arrays`

The modes table is reduced to the pooled sets

$$
\{|e_{m,n}^{\mathrm{opt}}|:m,n\},
\qquad
\{|e_{m,n}^{\mathrm{base}}|:m,n\},
$$

already aggregated over all terrain cases.

##### `build_histogram_panel`

The plotted variable is

$$
e=\log_{10}(|e_{m,n}|+\varepsilon),
$$

with $\varepsilon=10^{-14}$ for finite logarithms at zero error. The panel shows the empirical PDFs of $e$ for the optimized and baseline kernels, together with the two medians.

##### `build_terrain_panels`

The irregular cloud $\{x_q,y_q\}$ is triangulated, then each sampled field $h_q^{(j)}$ is rendered as a surface

$$
z=h^{(j)}(x,y).
$$

The same $(z_{\min},z_{\max})$ is used for all three panels, so the color scale is common across terrains.

##### `main`

The function builds

$$
\mathcal F=
[
\text{one histogram panel}
+
\text{three terrain panels}
],
$$

using the shared template style $\mathcal S$, then exports the PNG and PDF outputs.

#### `Plot_Ems.py`

Case-level figure builder for the EMS theory spectra.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $E_{(j)}$ | `e_sorted` | six `20`-point spectra | sorted EMS spectrum plotted in each panel |
| $K^{\star}$ | `k_star` | `1`, `3`, `7`, or `12` | retained EMS mode count |
| $K_{\mathrm{N}}$ | `n_eff_line` | $\lceil N_{\mathrm{eff}}^{\mathrm{clip}}\rceil$ | plotted participation-ratio line |
| $K_S$ | `s_delta_line` | $\lceil K_{\max}S_{\delta}\rceil$ | plotted similarity line |
| $\alpha_C,\alpha_C^{\mathrm{final}}$ | `alpha_c`, `alpha_c_final` | case dependent | annotated cumulative fractions |
| $\mathcal T_{\mathrm{spectra}}$ | `spectra_csv` | `Banerjee_2026_Enufft_Ems_Spectra.csv` | spectra input table used by the plot driver |
| $\mathcal T_{\mathrm{summary}}$ | `summary_csv` | `Banerjee_2026_Enufft_Ems_Summary.csv` | summary input table used by the plot driver |
| $\mathcal F$ | `figure` | `170 x 100` mm canvas | Matplotlib figure object |

##### `read_case_data`

The function reconstructs each EMS panel payload as

$$
\{E_{(j)},\delta,w_1,w_2,\alpha_{\min},\alpha_{\max},K_{\min},K_{\max},N_{\mathrm{eff}},S_{\delta},\mathcal C,\alpha_C,K^{\star},\alpha_C^{\mathrm{final}}\}
$$

from the saved spectra and summary tables.

##### `add_spectrum_panel`

For one case, the sorted samples $\{E_{(j)}\}$ are passed through a PCHIP interpolant to draw the monotone spectrum curve. The shaded window spans $j\le K_{\max}$, the hatched tail highlights $K^{\star}\le j\le K_{\max}$, and the three vertical guides mark

$$
K^{\star},
\qquad
K_{\mathrm{N}}=\lceil K_{\max}N_{\mathrm{eff}}^{\mathrm{norm}}\rceil,
\qquad
K_S=\lceil K_{\max}S_{\delta}\rceil.
$$

##### `main`

The driver rebuilds the full $2\times 3$ EMS figure from the CSV tables, applies the shared plot template, and exports `./figures/Banerjee_2026_Enufft_Ems.png` and `.pdf`.

#### `Plot_Mono.py`

Case-level figure builder for the monochromatic ENUFFT and CSA sweep.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $\mathcal T_{\mathrm{summary}}$ | `summary_csv` | `Banerjee_2026_Enufft_Mono_Summary.csv` | scalar input table |
| $L_x,L_y$ | `domain_length_km` through `representative_case` | `30.0`, `30.0` km | representative plot domain |
| $Q$ | `sample_count` through `representative_case` | `420` | representative DEM cloud size |
| $N_{\max}^{\mathrm{mode}}$ | `mode_limit` through `representative_case` | `8` | representative retained mode limit |
| $\theta_T$ | `triangle_orientation` through `representative_case` | `32.0` deg | representative triangle rotation |
| $\Delta y/L$ | `center_offset` through `representative_case` | `0.08` | representative centre offset |
| $u$ | `uniformity` through `representative_case` | `0.58` | representative triangle shape |
| $\mathcal W$ | `mask_condition` through `representative_case` | `circle` | representative mask used in the geometry panel |
| $w_q$ | `weight_type` through `representative_case` | `voronoi` | representative weighting rule used for DEM tiling |
| $\mathcal G_{\mathrm{region}}$ | `region_grid` | `240 x 240` in the piecewise panel | rasterized region map |
| $\mathcal V$ | `tile_values` | `135 x 135` in the Voronoi panel | nearest-sample area tile map |
| $C_{\mathrm{method}}$ | `method_colors` | Square, Tri., Circle, CSA colors | method styling |
| $D_\theta$ | `max_mode_direction_deviation_deg_*` | summary CSV columns | direction-error diagnostic |
| $D_A$ | `max_peak_amplitude_deviation_*` | summary CSV columns, m | amplitude-error diagnostic |
| $f_K$ | `k_star_enufft / mode_limit`, `csa_signed_modes_selected / (2 * mode_limit)` | summary-derived fractions | retained-mode fraction diagnostic |
| $p_{10},p_{90}$ | `p10`, `p90` | 10th and 90th percentiles | spread bands and whiskers |
| $\tilde r$ | `median_value` | case dependent | plotted method median |
| $r_{\max}$ | `r_max` | rounded metric-dependent upper limit | radar radial ceiling |
| $r_{\min}$ | `r_floor` | at least `1e-2` for log radar panels | positive lower radial floor |
| $\mathcal F$ | `figure` | `170 x 118` mm canvas | Matplotlib figure object |

##### `read_summary_rows`

The function reads the scalar Mono summary rows used by the diagnostic panels. These rows already contain the ENUFFT and CSA diagnostics, so the plot script does not recompute Fourier coefficients.

##### `representative_setup`

This deterministic setup rebuilds only the geometry and DEM cloud used by the top-row visual panels.

$$
N_{\max}^{\mathrm{mode}}=8,\quad
Q=420,\quad
\theta_T=32^\circ,\quad
\Delta y/L=0.08,\quad
u=0.58.
$$

It rebuilds geometry and sampled points, not the sweep Fourier results.

##### `compute_signal_angle_deg`

The plotted region stripe angle is

$$
\theta=\tan^{-1}(\frac{n}{m})\bmod 180^\circ.
$$

The modulo is used because $(m,n)$ and $(-m,-n)$ have the same line orientation.

##### `build_region_grid`

The function samples the region partition on a regular grid so the piecewise domain can be plotted with clean filled regions. The piecewise panel uses a $240\times240$ grid for the displayed $\mathcal G_{\mathrm{region}}$.

##### `ray_square_endpoint`

Each ray from the triangle centre through a vertex is extended to the boundary of $[0,L]\times[0,L]$.

##### `estimate_region_centroids`

For each region $r$, the label location is the mean of the raster-cell centres assigned to that region.

$$
\bar x_r=\frac{1}{\lvert\mathcal G_{\mathrm{region},r}\rvert}\sum_{q\in\mathcal G_{\mathrm{region},r}}x_q,
\qquad
\bar y_r=\frac{1}{\lvert\mathcal G_{\mathrm{region},r}\rvert}\sum_{q\in\mathcal G_{\mathrm{region},r}}y_q.
$$

##### `add_region_stripes`

A family of line segments is clipped to one region and oriented by the corresponding hidden mode direction. For a stripe direction

$$
d=(\cos\theta,\sin\theta),
\qquad
d_\perp=(-\sin\theta,\cos\theta),
$$

candidate lines are sampled as

$$
p(t,s)=p_c+s\,d_\perp+t\,d.
$$

The displayed stripes use `900` points along each candidate line and linewidth `0.9`.

##### `style_domain_axis`

The square-domain panels share the same $x/L$ and $y/L$ ticks, limits, aspect ratio, zero margins, and grid style.

##### `set_panel_title`

The helper places compact panel titles on a consistent axes-coordinate baseline.

##### `plot_piecewise_domain_map`

The panel shows the four-region partition, the triangle, sector rays, hidden mode labels $(m_r,n_r)$, and stripe directions. The filled colors come from `region_grid`. The stripes show the direction of the corresponding region mode rather than a fitted spectrum.

##### `plot_representative_setup`

The panel shows the square domain, triangle support, circular mask, centre offset, and triangle rotation. The circular mask radius is

$$
R_C=0.5L.
$$

##### `build_voronoi_tile_map`

The same nearest-sample area proxy used for quadrature weights is rasterized as a tile map. For each display cell, the nearest DEM point owns the cell, giving a relative area value

$$
\mathcal V=\frac{w_q}{\langle w\rangle}.
$$

The color limits are clipped to the 5th and 95th percentiles of this relative area field.

##### `plot_voronoi_panel`

The representative DEM points are overlaid on the tile map, and the triangle outline is drawn.

##### `collect_metric_by_method`

The summary table is grouped into Square, Tri., Circle, and CSA arrays for a requested metric pair. ENUFFT values are split by `mask`, while CSA values are pooled across all rows because the CSA reference is tied to the physical setup rather than to the ENUFFT support choice.

##### `compute_summary_band`

The function returns the finite-sample band

$$
(\tilde r,p_{10},p_{90})=
(
\mathrm{median}(r),
\mathrm{percentile}_{10}(r),
\mathrm{percentile}_{90}(r)
).
$$

##### `nice_upper_bound`

The helper rounds a positive maximum metric to a clean plotting limit by taking the first value in

$$
\{1,1.5,2,2.5,5,10\}\times10^p
$$

that exceeds the requested upper bound.

##### `build_radial_ticks`

For a linear radar scale, the function returns four positive tick radii between $0$ and $r_{\max}$. Integer panels round those ticks to integer labels.

##### `build_log_radial_ticks`

For a log radar scale, the function returns decade ticks between $r_{\min}$ and $r_{\max}$, adding $r_{\min}$ when the first decade would otherwise skip the lower ring.

##### `format_tick_label`

The function formats linear radial labels compactly, using integers when requested and `k` suffixes for values above $10^3$.

##### `format_log_tick_label`

The function formats logarithmic radial labels compactly across the positive range, including small values such as $10^{-2}$ and large values with `k` suffixes.

##### `format_median_annotation`

The function formats the direct text label placed at each radar median $\tilde r$, using the logarithmic formatter for log-scaled panels.

##### `plot_metric_boxplot`

The direction-error panel compares

$$
\lvert\theta_{\mathrm{true}}-\theta_{\mathrm{peak}}\rvert
$$

across the four methods using 10th-to-90th percentile whiskers, colored boxes, and direct median labels.

The y-axis limit is set from the maximum finite direction error after rounding with `nice_upper_bound`.

##### `plot_metric_radar`

The radar panel shows method-wise median and percentile bands for one scalar diagnostic. It closes the four method values into a polar polygon.

$$
[\tilde r_{\mathrm{Square}},
\tilde r_{\mathrm{Tri.}},
\tilde r_{\mathrm{Circle}},
\tilde r_{\mathrm{CSA}},
\tilde r_{\mathrm{Square}}].
$$

The median polygon $\tilde r$, filled $p_{10}$ to $p_{90}$ band, dotted percentile traces, direct median annotations, and low/high radial labels are drawn. The amplitude-error panel uses the log scale with positive floor $r_{\min}$.

##### `collect_kstar_fraction_by_method`

The retained-mode fractions are grouped as

$$
f_{\mathrm{ENUFFT}}=\frac{K^\star_{\mathrm{ENUFFT}}}{N_{\max}^{\mathrm{mode}}},
\qquad
f_{\mathrm{CSA}}=\frac{S_{\mathrm{CSA}}}{2N_{\max}^{\mathrm{mode}}},
$$

for Square, Tri., Circle, and CSA.

##### `plot_kstar_fraction`

The retained-fraction panel plots

$$
\frac{K^\star}{N_{\max}^{\mathrm{mode}}}
$$

for ENUFFT supports and

$$
\frac{S_{\mathrm{CSA}}}{2N_{\max}^{\mathrm{mode}}}
$$

for CSA, again using the median polygon, $p_{10}$ to $p_{90}$ band, dotted percentile traces, direct labels, and low/high radial annotations.

The fixed radial ticks are

$$
0.25,\quad0.50,\quad0.75,\quad1.00,
$$

with the radial ceiling set to `1.08` so labels can sit outside the unit ring.

##### `main`

The driver reads `Banerjee_2026_Enufft_Mono_Summary.csv`, applies the shared plotting style, builds a `2 x 3` canvas, and exports `./figures/Banerjee_2026_Enufft_Mono.png` and `.pdf`.

#### `Plot_Mountain.py`

Case-level figure builder for the mountain-wave EMS simulation output.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $\mathcal T_h$ | `terrain_csv` | `Banerjee_2026_Enufft_Mountain_Terrain.csv` | fine-grid topography input table |
| $\mathcal T_K$ | `modes_csv` | `Banerjee_2026_Enufft_Mountain_Modes.csv` | final surface wind, EMS count, and retained-power input table |
| $\mathcal T_s$ | `summary_csv` | `Banerjee_2026_Enufft_Mountain_Summary.csv` | scalar setup and diagnostic input table |
| $F_{\mathrm{png}}$ | `figure_output` | `Banerjee_2026_Enufft_Mountain.png` | PNG figure output |
| $F_{\mathrm{pdf}}$ | `figure_output` | `Banerjee_2026_Enufft_Mountain.pdf` | PDF figure output |
| $W,H$ | `figure_width_mm`, `figure_height_mm` | `170`, `128` mm | Matplotlib canvas size before tight export |
| $x_{\max},y_{\max}$ | `plot_limit_km` | `100.0` km | map-panel half-widths |
| $N_c$ | `color_levels` | `10` | number of plotted colorbar levels |
| $C_h$ | `topography_colors` | reddish sequential palette | orography color levels |
| $C_v$ | `wind_colors` | blue-white-red palette | signed meridional-wind color levels |
| $C_K$ | `mode_colors` | white-green palette | retained-mode count color levels |
| $C_A$ | `loss_colors` | reddish sequential palette | launch-power-loss color levels |

##### `apply_mountain_style`

The function applies the shared James-style Matplotlib settings, then sets the figure-specific legend, axis, hatch, and export defaults used for the mountain-wave EMS panels.

##### `linrange`

The helper returns $N_c$ evenly spaced levels between two scalar endpoints. These levels define the boundaries and labelled ticks for the continuous-looking binned color maps.

##### `symmetric_limits`

For the signed wind field, the helper returns $[-v_{\max},v_{\max}]$ with

$$
v_{\max}=\max_{i,j}\lvert v_{j,i}\rvert.
$$

If the input field is degenerate, it returns a nonzero fallback interval.

##### `finite_minmax`

The helper returns finite plotting limits for a scalar field after dropping missing values. It widens a zero-width range by one plotting unit so the color normalization remains defined.

##### `categorical_gradient`

The function converts endpoint colors and level boundaries into a discrete colormap and a `BoundaryNorm`. A missing field value is mapped to white.

##### `cell_edges`

For cell centers $x_i$, the helper reconstructs the cell edges by midpoint differencing.

$$
x_{i+1/2}=\frac{x_i+x_{i+1}}{2}.
$$

The first and last edges are extrapolated with the same nearest half-spacing.

##### `clip_line_to_box`

Given a diagonal line $y=mx+b$ and one cell box, the helper computes the two intersection points that form the visible segment inside that box.

##### `hatch_segments_for_mask`

The function loops over all cells where a boolean mask is true, clips a family of diagonal hatch lines to each selected cell, and returns the resulting line segments in km.

##### `add_signed_hatching`

The wind panel uses hatching to show the sign of $v$. Cells with $v$ above the first positive color level receive one diagonal direction, and cells with $v$ below the matching negative level receive the opposite diagonal direction.

##### `simple_labels`

The helper rounds colorbar tick values and formats them with compact decimal labels.

##### `configure_map_axis`

The helper applies the common map limits, tick marks, equal aspect ratio, transparent face, and axis labels for the $x$-$y$ panels.

##### `add_image_panel`

One gridded field is drawn with nearest-neighbour image cells, the map-axis settings are applied, and the corresponding compact vertical colorbar is attached.

##### `read_summary_row`

The function reads $\mathcal T_s$ and converts the scalar setup and diagnostic values into numeric Python values for the plotting calculations.

##### `read_terrain_data`

The function reads $\mathcal T_h$ and reconstructs the one-dimensional fine-grid coordinates and the two-dimensional height array $h_{j,i}$.

##### `read_mode_data`

The function reads $\mathcal T_K$ and reconstructs the final surface arrays $v_{j,i}$, $K^\star_{j,i}$, and $A_{j,i}$ on the coarse model grid.

##### `add_topography_panel`

The active part of the fine-grid topography is drawn as a 3D surface. The displayed height color levels extend from $0$ to the larger of the maximum stored height and $H_0$.

##### `save_figure`

The helper creates `./figures` and exports `./figures/Banerjee_2026_Enufft_Mountain.png` and `.pdf`.

##### `main`

The driver reads $\mathcal T_h$, $\mathcal T_K$, and $\mathcal T_s$, computes the active power loss

$$
100(1-A_{j,i}),
$$

builds the orography, wind, retained-mode-count, and launch-power-loss panels, and writes the PNG and PDF figure files.
