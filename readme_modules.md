# ENUFFT Module Documentation

[Back to documentation index](README.md)

## Module Files

Shared reusable modules used by one or more ENUFFT ports.

#### `Module_Helpers.py`

Generic helpers shared across ports.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $p_q=(x_q,y_q)$ | `points` | - | point coordinates tested by geometry helpers |
| $\Omega_P$ | `vertices` | - | ordered polygon vertices |
| $x_{\min},x_{\max},y_{\min},y_{\max}$ | `x_min`, `x_max`, `y_min`, `y_max` | - | rectangular clipping bounds |
| $\lvert\Omega_P\rvert$ | return of `polygon_area_2d` | - | positive planar polygon area |
| $\Omega_P\cap B$ | return of `clip_polygon_to_box` | - | polygon clipped to an axis-aligned box $B$ |
| $\lvert\Omega_P\cap B\rvert$ | return of `polygon_box_intersection_area` | - | clipped polygon area shared by Mono and Alps triangular supports |
| $v_0,v_1,v_2$ | `triangle_vertices`, `v0`, `v1`, `v2` | - | triangle vertices |
| $d_1(q),d_2(q),d_3(q)$ | `d1`, `d2`, `d3` | - | signed triangle-edge tests |
| $M_T(q)$ | return of `points_in_triangle_mask` | boolean mask | triangle membership for each point |
| $\lvert\Omega_T\rvert$ | return of `triangle_area_2d` | - | positive planar triangle area |
| $v$ | `value` | `-` | scalar written to disk |
| $s(v)$ | return value | `%.18e` | scientific-notation string used in CSV files |

##### `polygon_area_2d`

For ordered polygon vertices $p_i=(x_i,y_i)$, the function returns the positive shoelace area

$$
\lvert\Omega_P\rvert=\frac{1}{2}\lvert\sum_i x_i y_{i+1}-y_i x_{i+1}\rvert,
$$

with cyclic indexing $p_{N}=p_0$. If fewer than three vertices are supplied, the polygon is degenerate and the area is zero.

##### `clip_polygon_to_box`

The function starts with `clipped = vertices`. The rows of `planes` are the four box inequalities written as

$$
g(x,y)=a x+b y+c\ge 0.
$$

| `planes` row | $g(x,y)$ | inequality |
|---|---|---|
| `[1, 0, -x_min]` | $x-x_{\min}$ | $x\ge x_{\min}$ |
| `[-1, 0, x_max]` | $x_{\max}-x$ | $x\le x_{\max}$ |
| `[0, 1, -y_min]` | $y-y_{\min}$ | $y\ge y_{\min}$ |
| `[0, -1, y_max]` | $y_{\max}-y$ | $y\le y_{\max}$ |

For one row of `planes`, the code walks the closed polygon edge by edge. The first edge uses the last vertex as `prev`, so the closing edge is included.

$$
p_{\mathrm{prev}}=(x_{\mathrm{prev}},y_{\mathrm{prev}}),
\qquad
p_{\mathrm{curr}}=(x_{\mathrm{curr}},y_{\mathrm{curr}}).
$$

The edge segment is

$$
p(t)=p_{\mathrm{prev}}+t(p_{\mathrm{curr}}-p_{\mathrm{prev}}),
\qquad 0\le t\le 1.
$$

The code values

$$
g_{\mathrm{prev}}=a x_{\mathrm{prev}}+b y_{\mathrm{prev}}+c,
\qquad
g_{\mathrm{curr}}=a x_{\mathrm{curr}}+b y_{\mathrm{curr}}+c
$$

are the inequality values at the two edge endpoints. Along the edge,

$$
\begin{aligned}
g(p(t))
&=a[x_{\mathrm{prev}}+t(x_{\mathrm{curr}}-x_{\mathrm{prev}})]
 +b[y_{\mathrm{prev}}+t(y_{\mathrm{curr}}-y_{\mathrm{prev}})]+c\\
&=g_{\mathrm{prev}}+t(g_{\mathrm{curr}}-g_{\mathrm{prev}}).
\end{aligned}
$$

If

$$
g_{\mathrm{prev}}g_{\mathrm{curr}}<0,
$$

then the edge has one endpoint with $g<0$ and one endpoint with $g>0$. The new polygon must include the point where this edge meets the current box side, i.e. where $g(p(t))=0$.

$$
g_{\mathrm{prev}}+t(g_{\mathrm{curr}}-g_{\mathrm{prev}})=0,
$$

so

$$
t_{\mathrm{cross}}=\frac{g_{\mathrm{prev}}}{g_{\mathrm{prev}}-g_{\mathrm{curr}}}.
$$

That gives the appended boundary vertex

$$
p_{\mathrm{cross}}
=p_{\mathrm{prev}}
+t_{\mathrm{cross}}(p_{\mathrm{curr}}-p_{\mathrm{prev}}).
$$

The second append keeps the current endpoint exactly when it satisfies the active inequality.

$$
g_{\mathrm{curr}}\ge 0.
$$

Therefore

$$
\mathrm{clipped}
=\Omega_P\cap[x_{\min},x_{\max}]\times[y_{\min},y_{\max}],
$$

and `polygon_area_2d(clipped)` is the area of the polygon restricted to the analysis domain.

##### `polygon_box_intersection_area`

This helper combines the two previous operations:

$$
\mathrm{polygon\_box\_intersection\_area}(\Omega_P,B)=
\lvert\Omega_P\cap B\rvert.
$$

Both the ported mono triangular mask and the ported Alps triangular window use this helper for their rectangular-domain intersection areas, so the mask geometry and area normalization follow one shared convention.

##### `points_in_triangle_mask`

The function returns the triangle-membership mask for the supplied points. For a point $p_q=(x_q,y_q)$ and triangle vertices $v_0,v_1,v_2$, the three signed edge tests are

$$
d_1(q)=\det(p_q-v_1,\ v_0-v_1),
$$

$$
d_2(q)=\det(p_q-v_2,\ v_1-v_2),
$$

and

$$
d_3(q)=\det(p_q-v_0,\ v_2-v_0),
$$

where

$$
\det(a,b)=a_xb_y-a_yb_x.
$$

The returned mask is

$$
M_T(q)=
\begin{cases}
1, & d_1(q),d_2(q),d_3(q)\ge 0,\\
1, & d_1(q),d_2(q),d_3(q)\le 0,\\
0, & \mathrm{otherwise}.
\end{cases}
$$

Each determinant is the signed area test for one triangle edge. Points inside the triangle are on the same signed side of all three directed edges, so the three tests have one common sign. Boundary points are included because zero values are accepted.

##### `triangle_area_2d`

The function returns the positive planar area of the triangle.

$$
\lvert\Omega_T\rvert=\frac{1}{2}\lvert(x_1-x_0)(y_2-y_0)-(y_1-y_0)(x_2-x_0)\rvert.
$$

This is one half of the absolute signed area $\det(v_1-v_0,v_2-v_0)$.

##### `format_float`

$$
s(v)=\mathrm{\%.18e}(v).
$$

The map is purely for stable serialization. A value written to CSV and read back again stays numerically unchanged along this serialization path.

#### `Module_Nufft.py`

Shared ENUFFT pipeline for the irregular-sample Fourier calculation.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $x_q,y_q$ | `x_values`, `y_values` | $x_q,y_q\in[0,10)$ km | irregular DEM coordinates |
| $h_q$ | `h_values` | `-` | sampled terrain heights |
| $w_q$ | `sample_weights` | `None` | optional quadrature weights |
| $m,n$ | `m_values`, `n_values` | $m,n=-20,\ldots,20$ | retained Fourier indices, with $m\to x$ and $n\to y$ |
| $k_m,\ell_n$ | implicit | $2\pi m/10$, $2\pi n/10$ km$^{-1}$ | physical wavenumbers $2\pi m/L_x$ and $2\pi n/L_y$ |
| $L_x,L_y$ | `domain_length_x`, `domain_length_y` | `10.0`, `10.0` km | periodic domain lengths |
| $N_{\max,x}^{\mathrm{mode}},N_{\max,y}^{\mathrm{mode}}$ | `mode_limit_x`, `mode_limit_y` | `20`, `20` | retained mode limits |
| $\sigma$ | `oversample` | `1.5` | oversampling factor |
| $a$ | `kernel_half_width` | `4` | KB half-width in grid cells |
| $\beta$ | `kernel_beta` | `2.34` | KB denominator parameter |
| $\gamma$ | return of `kaiser_bessel_alpha` | optimized `18.5079`, baseline `2.34` | KB numerator parameter |
| $N_x^{\mathrm{aux}},N_y^{\mathrm{aux}}$ | `nx_aux`, `ny_aux` | `60`, `60` | auxiliary FFT grid sizes |
| $d_x^{\mathrm{aux}},d_y^{\mathrm{aux}}$ | `dx_aux`, `dy_aux` | `1/6`, `1/6` km | auxiliary grid spacings |
| $x_i^{\mathrm{aux}},y_j^{\mathrm{aux}}$ | `x_aux`, `y_aux` | $\{0,1/6,\ldots,59/6\}$ km | auxiliary-grid coordinates |
| $\varphi(\zeta)$ | return of `kaiser_bessel_kernel` | support on $\|\zeta\|\le 4$ | KB spreading kernel |
| $\Phi(\kappa)$ | return of `kb_fourier_transform` | $\kappa\in[-4\pi,4\pi]$ km$^{-1}$ on retained modes | kernel Fourier transform |
| $h^{\mathrm{aux}}_{i,j}$ | `auxiliary_grid` | `60 x 60` real grid | spread field on the auxiliary grid |
| $\hat h^{\mathrm{fft}}_{m,n}$ | `fft_coefficients` | `60 x 60` complex grid | raw auxiliary-grid FFT coefficients |
| $\hat h^{\mathrm{dft}}_{m,n}$ | `dft_coefficients` | `41 x 41` complex block per terrain | direct irregular-sample Fourier coefficients |
| $\hat h^{\mathrm{nufft}}_{m,n}$ | return of `compute_nufft_coefficients` | `41 x 41` complex block per terrain | NUFFT approximation on the retained mode block |

##### `build_auxiliary_grid`

$$
N_x^{\mathrm{aux}}=\max(\lceil 2\sigma N_{\max,x}^{\mathrm{mode}}\rceil,2N_{\max,x}^{\mathrm{mode}},2),
\qquad
N_y^{\mathrm{aux}}=\max(\lceil 2\sigma N_{\max,y}^{\mathrm{mode}}\rceil,2N_{\max,y}^{\mathrm{mode}},2),
$$

followed by even rounding. Then

$$
d_x^{\mathrm{aux}}=\frac{L_x}{N_x^{\mathrm{aux}}},
\qquad
d_y^{\mathrm{aux}}=\frac{L_y}{N_y^{\mathrm{aux}}},
\qquad
x_i^{\mathrm{aux}}=i\,d_x^{\mathrm{aux}},
\qquad
y_j^{\mathrm{aux}}=j\,d_y^{\mathrm{aux}}.
$$

##### `scale_sample_values`

If weights are present,

$$
h_q^\ast=h_q\frac{w_q}{\sum_r w_r}Q,
$$

so that

$$
\frac{1}{Q}\sum_q h_q^\ast=\frac{\sum_q w_q h_q}{\sum_q w_q}.
$$

If weights are absent, $h_q^\ast=h_q$.

##### `kaiser_bessel_alpha`

$$
\gamma=
\begin{cases}
\pi\sqrt{(2a)^2(\beta/\pi)^2-0.8}, & \text{optimized},\\
\beta, & \text{baseline}.
\end{cases}
$$

##### `kaiser_bessel_kernel`

For $\zeta=\mathrm{grid\_distance}$,

$$
\varphi(\zeta)=
\begin{cases}
\dfrac{1}{I_0(\beta)}I_0[\gamma\sqrt{1-(\zeta/a)^2}], & |\zeta|\le a,\\
0, & |\zeta|>a.
\end{cases}
$$

##### `kb_fourier_transform`

For $\kappa=\mathrm{wavenumber}$ and $d_\kappa=\mathrm{grid\_spacing}$,

$$
\Phi(\kappa)=\frac{2a}{I_0(\beta)}\frac{\sinh(\sqrt{\gamma^2-(a\kappa d_\kappa)^2})}{\sqrt{\gamma^2-(a\kappa d_\kappa)^2}},
$$

with the usual $\sin(s)/s$ continuation when $\gamma^2-(a\kappa d_\kappa)^2<0$.

##### `wrap_periodic_distances`

The displacement $\delta$ is mapped to its shortest periodic representative in

$$
[-\frac{L}{2},\frac{L}{2}].
$$

This is applied separately with $L=L_x$ and $L=L_y$.

##### `spread_samples_to_grid`

Each irregular sample contributes

$$
h_q^\ast\,
\varphi(\frac{x_q-x_i^{\mathrm{aux}}}{d_x^{\mathrm{aux}}})
\varphi(\frac{y_q-y_j^{\mathrm{aux}}}{d_y^{\mathrm{aux}}})
$$

to every stencil node $(i,j)$ in the support square $[-a,a]^2$. After accumulation,

$$
h^{\mathrm{aux}}_{i,j}\leftarrow h^{\mathrm{aux}}_{i,j}\frac{N_x^{\mathrm{aux}}N_y^{\mathrm{aux}}}{Q}.
$$

##### `extract_target_mode_block`

For each retained $(m,n)$,

$$
i_{\mathrm{wrap}}=m\bmod N_x^{\mathrm{aux}},
\qquad
j_{\mathrm{wrap}}=n\bmod N_y^{\mathrm{aux}},
$$

and

$$
\hat h_{m,n}^{\mathrm{deconv}}=\mathrm{fourier\_grid}[j_{\mathrm{wrap}},i_{\mathrm{wrap}}].
$$

##### `compute_nufft_coefficients`

The following coefficient pipeline is evaluated:

$$
h_q \rightarrow h_q^\ast \rightarrow h^{\mathrm{aux}}_{i,j}
\rightarrow \hat h^{\mathrm{fft}}_{m,n}
\rightarrow \hat h^{\mathrm{nufft}}_{m,n},
$$

with

$$
\hat h^{\mathrm{fft}}_{m,n}=\frac{1}{N_x^{\mathrm{aux}}N_y^{\mathrm{aux}}}\mathrm{FFT2}(h^{\mathrm{aux}}_{i,j}),
$$

and

$$
\hat h^{\mathrm{nufft}}_{m,n}=
\frac{\hat h^{\mathrm{fft}}_{m,n}}{\Phi(k_m)\Phi(\ell_n)}.
$$

The returned array keeps first index $m$ and second index $n$.

##### `compute_nufft_for_kernels`

At fixed $\{x_q,y_q,h_q\}$ and fixed retained mode block, the code evaluates

$$
\hat h_{m,n}^{\mathrm{nufft,opt}},
\qquad
\hat h_{m,n}^{\mathrm{nufft,base}}
$$

by repeating `compute_nufft_coefficients` over the kernel list.

##### `compute_direct_dft_coefficients`

For each retained $(m,n)$,

$$
k_m=\frac{2\pi m}{L_x},
\qquad
\ell_n=\frac{2\pi n}{L_y},
$$

and

$$
\hat h_{m,n}^{\mathrm{dft}}=\frac{1}{Q}\sum_{q=1}^{Q}h_q^\ast e^{-i(k_m x_q+\ell_n y_q)}.
$$

#### `Module_Csa.py`

Shared CSA pipeline for sparse Fourier ranking and sparse refitting.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $x_q^{\mathrm{FA}},y_q^{\mathrm{FA}},h_q^{\mathrm{FA}}$ | `x_fa`, `y_fa`, `h_fa` | full square DEM support in `Code_Mono.py` | full-analysis samples used to rank modes |
| $x_q^{\mathrm{SA}},y_q^{\mathrm{SA}},h_q^{\mathrm{SA}}$ | `x_sa`, `y_sa`, `h_sa` | clipped triangle DEM support in `Code_Mono.py` when enabled | sparse-analysis samples used to refit retained modes |
| $m,n$ | `m_values`, `n_values` | $-N_{\max}^{\mathrm{mode}},\ldots,N_{\max}^{\mathrm{mode}}$ | signed Fourier mode arrays |
| $k_m,\ell_n$ | `k_m`, `l_n` | $2\pi m/L_x$, $2\pi n/L_y$ | x- and y-direction wavenumbers |
| $\lambda_{\mathrm{FA}}$ | `lambda_fa` | `1e-1` in `Code_Mono.py` | Tikhonov parameter for mode ranking |
| $\lambda_{\mathrm{SA}}$ | `lambda_sa` | `1e-6` in `Code_Mono.py` | Tikhonov parameter for sparse refit |
| $S_{\mathrm{CSA}}$ | `sparse_modes` | `2 * mode_limit` in `Code_Mono.py` | number of signed modes retained after FA ranking |
| $Q_c$ | `chunk_size` | `None` in `Code_Mono.py` | optional number of sample rows accumulated per normal-equation chunk |
| $B_c$ | `basis` | at most `chunk_size x len(mode_list)` | chunk-local Fourier synthesis matrix |
| $\hat h_{m,n}^{\mathrm{csa}}$ | `spectrum` | `(2N+1) x (2N+1)` complex block | CSA sparse Fourier coefficients |
| $\iota_{\mathrm{dc}}$ | `include_dc` | `True` by default | whether the DC pair counts as a retained pair |

##### `build_dense_mode_list`

The dense CSA candidate set is

$$
\mathcal M=\{(m,n):m\in\mathrm{m\_values},\ n\in\mathrm{n\_values}\}.
$$

The returned array stores each candidate as `(m_mode, n_mode)`.

##### `build_normal_equations`

For a mode list $\mathcal M$, the synthesis matrix is

$$
B_{q,j}=e^{i(k_{m_j}x_q+\ell_{n_j}y_q)}.
$$

The function assembles

$$
A=B^\ast B,
\qquad
b=B^\ast h,
$$

as accumulated sums. If `chunk_size=None`, the whole sample cloud is one chunk. If `chunk_size=Q_c`, the samples are split into contiguous blocks and the code computes

$$
A=\sum_c B_c^\ast B_c,
\qquad
b=\sum_c B_c^\ast h_c.
$$

This changes the memory footprint of the basis construction, not the CSA approximation being solved. The mathematical least-squares system is the same. Only floating-point summation order can differ.

##### `solve_tikhonov_from_normal`

The regularized coefficient vector is

$$
c=(A+\lambda I)^{-1}b.
$$

If the direct solve is singular, the same system is solved by least squares.

##### `fit_fourier_modes`

This function combines the two previous steps and returns the Fourier coefficients $c_j$ for the supplied mode list.

$$
\mathcal M,\{x_q,y_q,h_q\},\lambda,Q_c
\rightarrow
A,b
\rightarrow
c.
$$

The `chunk_size` parameter is passed through only to the normal-equation assembly.

##### `compute_csa_spectrum`

CSA first fits all dense candidates on the FA support,

$$
c_j^{\mathrm{FA}}=
\arg\min_c
\lVert
\sum_j c_j e^{i(k_{m_j}x_q^{\mathrm{FA}}+\ell_{n_j}y_q^{\mathrm{FA}})}
-h_q^{\mathrm{FA}}
\rVert_2^2
+\lambda_{\mathrm{FA}}\lVert c\rVert_2^2,
$$

and ranks them by

$$
E_j^{\mathrm{FA}}=\lvert c_j^{\mathrm{FA}}\rvert^2.
$$

The largest $S_{\mathrm{CSA}}$ candidates form $\mathcal M_S$. The SA step then refits only $\mathcal M_S$ on the target support,

$$
c_j^{\mathrm{SA}}=
\arg\min_c
\lVert
\sum_{j\in\mathcal M_S} c_j e^{i(k_{m_j}x_q^{\mathrm{SA}}+\ell_{n_j}y_q^{\mathrm{SA}})}
-h_q^{\mathrm{SA}}
\rVert_2^2
+\lambda_{\mathrm{SA}}\lVert c\rVert_2^2,
$$

and stores the resulting coefficients as $\hat h_{m,n}^{\mathrm{csa}}$. If `sparse_modes=None`, the shared module sets

$$
S_{\mathrm{CSA}}=2\max(\max_m\lvert m\rvert,\max_n\lvert n\rvert).
$$

##### `reconstruct_at_points`

The forward-synthesis reconstruction is

$$
h^{\mathrm{rec}}(x_q,y_q)=\Re\sum_{m,n}\hat h_{m,n}^{\mathrm{csa}}e^{i(k_mx_q+\ell_ny_q)}.
$$

##### `compute_sorted_spectral_amplitudes`

The DC coefficient is set to zero, then all signed amplitudes $\lvert\hat h_{m,n}\rvert$ are sorted in descending order.

##### `find_dominant_mode_pair`

The function returns the signed non-DC pair $(m,n)$ with maximum $\lvert\hat h_{m,n}\rvert$. For real terrain, $(m,n)$ and $(-m,-n)$ are conjugate wave vectors with the same physical orientation but directions separated by $180^\circ$. The code preserves the original signed convention rather than canonicalizing that ambiguity.

##### `count_unique_mode_pairs`

Each signed pair is mapped to the canonical representative of its conjugate pair, so $(m,n)$ and $(-m,-n)$ count once. The optional `include_dc` switch controls whether $(0,0)$ contributes to the count.

##### `count_signed_nonzero_modes`

The function counts all signed non-DC coefficients satisfying $\lvert\hat h_{m,n}\rvert>\mathrm{tolerance}$.

#### `Module_Orography.py`

Shared DEM samplers and analytical terrain fields.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $x,y$ | `x_values`, `y_values` | $x,y\in[0,10)$ km | spatial coordinates |
| $L_x,L_y$ | `domain_length_x`, `domain_length_y` | `10.0`, `10.0` km | domain lengths |
| $Q$ | `sample_count` | `2000` | number of irregular DEM samples |
| $\rho_{\mathrm{case}}$ | `distribution` | `moderate` in `Code_Nufft.py`, `mono_ridge` in `Code_Mono.py` | selected DEM sampling density |
| $r_{\mathrm{seed}}$ | `random_seed` | `None` by default, `42` in `Code_Mono.py` | optional deterministic DEM seed |
| $x_{\mathrm{coast}}(y)$ | return of `coast_x_global` | $5.5+0.5\sin(2\pi y/10)$ km, so $[5.0,6.0]$ km | coastline used in the sampling density |
| $\rho(x,y)$ | implicit in `distribution` | `moderate`, so $\rho_{\mathrm{sea}}=0.35$ and $B_{\mathrm{coast}}=0.9$ | land-sea acceptance density |
| $\rho_{\mathrm{mono}}(x,y)$ | return of `mono_ridge_density` | used by `mono_ridge` | ridge-biased Mono DEM acceptance density |
| $h_0(x,y)$ | `heights` before roughness | depends on `kind` | base synthetic orography |
| $h(x,y)$ | return of `irregular_orography` | terrain kind in `multi_peak`, `ridge`, `basin`, sampled field shifted so $\min h_q=0$ | synthetic orography field |
| $r(x,y)$ | `roughness` | $[-80,80]$ m | deterministic roughness correction |

##### `coast_x_global`

$$
x_{\mathrm{coast}}(y)=0.55L_x+0.05L_x\sin(\frac{2\pi y}{L_y}).
$$

##### `mono_ridge_density`

The Mono DEM density is the sum of a floor, two ridge envelopes, and a waviness term.

$$
\rho_{\mathrm{mono}}(x,y)=0.20
+0.90e^{-[(x-x_{r1})/(0.10L_x)]^2}
+0.75e^{-[(x-x_{r2})/(0.12L_x)]^2}
+\omega(x,y),
$$

with

$$
x_{r1}(y)=0.30L_x+0.14L_x\sin(\frac{2\pi y}{L_y}+0.7),
$$

$$
x_{r2}(y)=0.72L_x+0.10L_x\cos(\frac{2\pi y}{L_y}-0.4),
$$

and

$$
\omega(x,y)=
0.55
+0.25\sin(\frac{2\pi x}{L_x}+0.3)
+0.20\cos(\frac{4\pi y}{L_y}-0.6)
+0.15\sin[2\pi(\frac{x}{L_x}+\frac{y}{L_y})].
$$

For the Mono square case, $L_x=L_y=L$, so this reduces to the density used by `Code_Mono.py`.

##### `generate_dem_points`

The function generates exactly $Q=\mathrm{sample\_count}$ DEM sample points inside the rectangle

$$
0\le x\le L_x,\qquad 0\le y\le L_y,
$$

It returns the generated coordinates as two arrays,

$$
(x_1,\dots,x_Q),
\qquad
(y_1,\dots,y_Q).
$$

If `random_seed` is supplied, the random draws are reproducible. For `distribution="uniform"`, the points are drawn directly from the uniform distribution,

$$
x_q\sim U(0,L_x),
\qquad
y_q\sim U(0,L_y),
\qquad q=1,\dots,Q.
$$

For all other distributions, the function uses accept-reject sampling. It first proposes trial points uniformly,

$$
(x_j^\ast,y_j^\ast)\sim U([0,L_x]\times[0,L_y]).
$$

Each trial point is assigned a raw sampling density $\rho_j^\ast=\rho(x_j^\ast,y_j^\ast)$. Within the current batch $\mathcal B$, this density is converted into an acceptance probability,

$$
P_j=
\frac{\rho_j^\ast}
{\max_{r\in\mathcal B}\rho_r^\ast+10^{-12}}.
$$

Uniform auxiliary variates are drawn as

$$
u_j\sim U(0,1)
$$

and keeps trial point $j$ if

$$
u_j<P_j.
$$

Thus, trial points with larger $\rho_j^\ast$ are more likely to become final DEM points. For `distribution="mono_ridge"`, the density is simply

$$
\rho(x,y)=\rho_{\mathrm{mono}}(x,y),
$$

where $\rho_{\mathrm{mono}}$ is computed by `mono_ridge_density`. For `distribution="mild"`, `"moderate"`, or `"strong"`, the density is a land-sea-coast density. First, the parameters are

$$
(\rho_{\mathrm{sea}},B_{\mathrm{coast}})=
\begin{cases}
(0.70,0.40), & \text{mild},\\
(0.35,0.90), & \text{moderate},\\
(0.12,1.40), & \text{strong}.
\end{cases}
$$

For a trial point $(x_j^\ast,y_j^\ast)$, the coastline location is $x_{\mathrm{coast}}(y_j^\ast)$. The base density is

$$
\rho_{\mathrm{base},j}=
\begin{cases}
1, & x_j^\ast\ge x_{\mathrm{coast}}(y_j^\ast),\\
\rho_{\mathrm{sea}}, & x_j^\ast\lt x_{\mathrm{coast}}(y_j^\ast).
\end{cases}
$$

The scaled horizontal distance from the coastline is

$$
d_j=
\frac{|x_j^\ast-x_{\mathrm{coast}}(y_j^\ast)|}
{0.08L_x}.
$$

The final raw density is

$$
\rho_j^\ast=
\rho_{\mathrm{base},j}
(1+B_{\mathrm{coast}}e^{-d_j^2}).
$$

So land points are favored through $\rho_{\mathrm{base},j}$, and points close to the coastline are favored through the exponential factor. The loop continues until exactly $Q$ final DEM points have been generated. If the trial limit is reached first, the function raises an error.

##### `irregular_orography`

The function first builds a base terrain $h_0(x,y)$ from the selected `kind`. For `multi_peak`,

$$
\begin{aligned}
h_0^{\mathrm{multi}}(x,y)
&=980e^{-[(x-4.9)^2/3.8+(y-5.2)^2/1.3]}\\
&\quad+430e^{-[(x-2.4)^2/0.9+(y-7.2)^2/1.8]}\\
&\quad+310e^{-[(x-7.7)^2/1.2+(y-2.7)^2/0.8]}\\
&\quad+170\sin(2.8x+0.7\sin(1.2y))\cos(1.4y).
\end{aligned}
$$

For `ridge`, the ridge envelope is

$$
R(x,y)=e^{-[y-(4.9+0.9\sin(0.75x))]^2/0.55},
$$

and

$$
\begin{aligned}
h_0^{\mathrm{ridge}}(x,y)
&=760R(x,y)
+260\sin(2.4x+0.8y)\\
&\quad+180\cos(3.3y-0.5x)
-250e^{-[(x-6.4)^2/1.5+(y-6.1)^2/0.6]}.
\end{aligned}
$$

For `basin`,

$$
\begin{aligned}
h_0^{\mathrm{basin}}(x,y)
&=620e^{-[(x-2.8)^2/1.0+(y-3.1)^2/1.7]}\\
&\quad+520e^{-[(x-7.2)^2/1.3+(y-7.4)^2/1.0]}\\
&\quad-370e^{-[(x-5.1)^2/4.8+(y-5.1)^2/2.6]}\\
&\quad+145\sin(3.4x)\sin(2.1y+0.6\cos x).
\end{aligned}
$$

After the base field is formed, the same deterministic roughness is added.

$$
r(x,y)=45\sin(7.2x+1.8\cos y)+35\cos(6.5y-0.7\sin(1.5x)),
$$

and the sampled heights are shifted so the minimum sampled value is zero.

$$
h_q=h_0(x_q,y_q)+r(x_q,y_q)-\min_s[h_0(x_s,y_s)+r(x_s,y_s)].
$$

#### `Module_Csv.py`

Writers for the plot-ready numerical tables.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $q$ | row index | $q=0,\ldots,1999$ | sample index in terrain tables |
| $m,n$ | `m_values`, `n_values` | $m,n=-20,\ldots,20$ | retained Fourier indices |
| $j$ | `j_mode` | $j=1,\ldots,20$ | sorted EMS spectrum index |
| $x_q,y_q$ | `x_dem`, `y_dem` | $x_q,y_q\in[0,10)$ km | sampled coordinates |
| $h_q^{(j)}$ | `h_dem` | `3 x 2000` heights, with $\min h_q^{(j)}=0$ for each terrain | sampled terrain for case $j$ |
| $\hat h_{m,n}^{\mathrm{dft}}$ | `h_dft` | `41 x 41` complex block per terrain, `5043` mode rows total | direct DFT coefficient block |
| $\hat h_{m,n}^{\mathrm{nufft,opt}}$ | `h_nufft_optimized` | `41 x 41` complex block per terrain, `5043` mode rows total | optimized NUFFT block |
| $\hat h_{m,n}^{\mathrm{nufft,base}}$ | `h_nufft_baseline` | `41 x 41` complex block per terrain, `5043` mode rows total | baseline NUFFT block |
| $e_{m,n}^{\mathrm{opt}},e_{m,n}^{\mathrm{base}}$ | `error_optimized`, `error_baseline` | `41 x 41` error block per terrain, `5043` absolute errors per kernel | coefficient errors |
| $E_{(j)}$ | `energy` | six `20`-point EMS spectra | sorted EMS energy values |
| $\vartheta(j)$ | `cumulative_fraction` | six `20`-point EMS arrays | cumulative EMS power fraction |
| $G_j$ | `gap_ratio` | first `J_{\mathrm{window}}-1` rows finite, final row `nan` | local EMS gap ratio $E_{(j)}/E_{(j+1)}$ |
| $K^{\star}$ | `k_star` | `1`, `3`, `7`, or `12` in the EMS theory cases | retained EMS mode count |
| $\mathcal T_{\mathrm{terrain}}$ | `terrain_csv` | case dependent | terrain output table |
| $\mathcal T_{\mathrm{spectra}}$ | `spectra_csv` | case dependent | spectra output table |
| $\mathcal T_{\mathrm{mode}}$ | `modes_csv` | case dependent | mode output table |
| $\mathcal T_{\mathrm{summary}}$ | `summary_csv` | case dependent | summary output table |
| $\mathcal R$ | `rows` | case dependent | generic list of output rows |

##### `write_dict_rows_csv`

For a list of dictionaries $\mathcal R$, the function preserves the first-seen key order and writes one CSV row per element of $\mathcal R$. It is used by `Code_Mono.py` because that case has wide sweep tables whose columns are already assembled mathematically before serialization.

##### `write_terrain_csv`

The table stores one row per sample point.

$$
(q,x_q,y_q,h_q^{\mathrm{multi}},h_q^{\mathrm{ridge}},h_q^{\mathrm{basin}}).
$$

##### `write_modes_csv`

For each terrain and each retained $(m,n)$, the table stores

$$
\hat h_{m,n}^{\mathrm{dft}},
\qquad
\hat h_{m,n}^{\mathrm{nufft,opt}},
\qquad
\hat h_{m,n}^{\mathrm{nufft,base}},
$$

and

$$
e_{m,n}^{\mathrm{opt}}=\hat h_{m,n}^{\mathrm{nufft,opt}}-\hat h_{m,n}^{\mathrm{dft}},
\qquad
e_{m,n}^{\mathrm{base}}=\hat h_{m,n}^{\mathrm{nufft,base}}-\hat h_{m,n}^{\mathrm{dft}}.
$$

Real part, imaginary part, and absolute value are all written explicitly.

##### `write_summary_csv`

The summary table stores

$$
\mathrm{median}(|e_{m,n}^{\mathrm{opt}}|),\quad
\mathrm{mean}(|e_{m,n}^{\mathrm{opt}}|),\quad
\mathrm{median}(|e_{m,n}^{\mathrm{base}}|),\quad
\mathrm{mean}(|e_{m,n}^{\mathrm{base}}|),
$$

pooled over all retained modes and all terrain cases.

##### `write_ems_spectra_csv`

For the EMS theory case, the file named `spectra_csv` stores the input energy spectra passed into EMS. There is one row for each sorted spectrum position $j$ in each theory example.

The columns have the following meanings.

- `spectrum_index` identifies which of the six theory spectra this row belongs to
- `spectrum_title` gives the plain-language spectrum name such as `Uniform` or `Geometric`
- `spectrum_label` gives the mathematical label used in the figure
- `j_mode` gives the sorted position $j=1,\ldots,J^{\star}$
- `energy` gives the energy value $E_{(j)}$ at that sorted position

So each row stores

$$
(\iota,\text{title},\text{label},j,E_{(j)}),
$$

with $\iota$ selecting the theory spectrum and $j$ selecting the row within that sorted spectrum.


##### `write_ems_modes_csv`

For each EMS spectrum, the table has one row for each sorted spectrum position $j$. Each row stores

$$
\vartheta(j)=\frac{\sum_{r=1}^{j}E_{(r)}}{\sum_{r=1}^{J^{\star}}E_{(r)}},
\qquad
G_j=\frac{E_{(j)}}{E_{(j+1)}},
$$

plus two simple yes/no markers.

$$
\mathrm{in\_window}=
\begin{cases}
1, & \text{this row lies in the comparison window } j=1,\ldots,J_{\mathrm{window}},\\
0, & \text{this row lies outside that window},
\end{cases}
$$

$$
\mathrm{retained\_mode}=
\begin{cases}
1, & \text{this mode is kept by EMS because } j\le K^{\star},\\
0, & \text{this mode is discarded because } j>K^{\star}.
\end{cases}
$$

##### `write_ems_summary_csv`

The EMS summary table stores the fixed control parameters

$$
\delta,\quad w_1,\quad w_2,\quad \alpha_{\min},\quad \alpha_{\max},\quad K_{\min},\quad K_{\max},
$$

and the spectrum-dependent diagnostics

$$
J^{\star},\quad J_{\mathrm{window}},\quad \sum_j E_{(j)},\quad N_{\mathrm{eff}},\quad S_{\delta},\quad \mathcal C,\quad \alpha_C,\quad K^{\star},\quad \alpha_C^{\mathrm{final}}.
$$

#### `Module_Ems.py`

Shared, case-agnostic elastic-mode-selection logic.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $\delta$ | `delta` | `0.02` | exponential gap-similarity scale |
| $w_1,w_2$ | `w1`, `w2` | `0.5`, `0.5` | EMS blend weights |
| $\alpha_{\min},\alpha_{\max}$ | `alpha_min`, `alpha_max` | `0.0`, `0.7` | lower and upper cumulative-power bounds |
| $K_{\min},K_{\max}$ | `k_min`, `k_max` | `1`, `12` | admissible retained-count bounds |
| $E_{(j)}$ | `e_values`, `e_sorted` | `-` | unordered input spectrum and its descending reorder |
| $J^{\star}$ | `j_star` | `-` | total number of available spectral entries |
| $J_{\mathrm{window}}$ | `j_window` | `-` | comparison window $\min(J^{\star},K_{\max})$ |
| $G_j$ | `g` | `-` | local gap ratio $E_{(j)}/E_{(j+1)}$ over the comparison window |
| $\vartheta(K)$ | `f_k` | `-` | cumulative power fraction |
| $N_{\mathrm{eff}}$ | `n_eff` | `-` | participation-ratio effective mode count |
| $N_{\mathrm{eff}}^{\mathrm{clip}}$ | `n_eff_clip` | `-` | participation-ratio count clipped at $K_{\max}$ |
| $S_{\delta}$ | `s_delta` | `-` | neighbor-similarity score |
| $\mathcal C$ | `c` | `-` | blended EMS control variable |
| $\alpha_C$ | `alpha_c` | `-` | target cumulative power fraction |
| $K^{\star}$ | `k_star` | `-` | retained EMS mode count |
| $\alpha_C^{\mathrm{final}}$ | `alpha_c_final` | `-` | realized cumulative power fraction at $K^{\star}$ |
| $\mathcal P_{\pm}$ | `mode_pairs` | non-DC conjugate representatives | paired signed modes used by the Fourier selector |
| $E_{m,n}^{\pm}$ | `energy` | - | paired energy $\lvert\hat h_{m,n}\rvert^2+\lvert\hat h_{-m,-n}\rvert^2$ |

##### `elastic_mode_selection`

Given any nonnegative spectrum $\{E_j\}_{j=1}^{J^{\star}}$, the function first sorts it in descending order as $E_{(j)}$ and computes

$$
N_{\mathrm{eff}}=\frac{(\sum_{j=1}^{J^{\star}}E_{(j)})^2}{\sum_{j=1}^{J^{\star}}E_{(j)}^2},
\qquad
N_{\mathrm{eff}}^{\mathrm{clip}}=\min(N_{\mathrm{eff}},K_{\max}).
$$

It then restricts the local gap analysis to

$$
J_{\mathrm{window}}=\min(J^{\star},K_{\max})
$$

and forms

$$
G_j=\frac{E_{(j)}}{E_{(j+1)}},
\qquad
j=1,\ldots,J_{\mathrm{window}}-1,
$$

together with

$$
S_{\delta}=
\begin{cases}
\dfrac{1}{J_{\mathrm{window}}-1}\sum_{j=1}^{J_{\mathrm{window}}-1}\exp[-\dfrac{G_j-1}{\delta}], & J_{\mathrm{window}}>1,\\
1, & J_{\mathrm{window}}=1.
\end{cases}
$$

The blended control variable is

$$
\mathcal C=w_1\frac{N_{\mathrm{eff}}^{\mathrm{clip}}}{K_{\max}}+w_2S_{\delta},
$$

which sets the target cumulative power fraction

$$
\alpha_C=\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})\mathcal C.
$$

Next, the cumulative fraction is

$$
\vartheta(K)=\frac{\sum_{j=1}^{K}E_{(j)}}{\sum_{j=1}^{J^{\star}}E_{(j)}},
\qquad
K=1,\ldots,J^{\star}.
$$

The retained count is chosen from the admissible set

$$
\mathcal K=\{K\in\{K_{\min},\ldots,K_{\max}\}:\vartheta(K)\ge \alpha_C\}.
$$

If $\mathcal K$ is nonempty, then

$$
K^{\star}=\min \mathcal K.
$$

If no admissible $k$ satisfies the target, the function falls back to

$$
K^{\star}=K_{\max}.
$$

Finally, the realized cumulative fraction is

$$
\alpha_C^{\mathrm{final}}=\vartheta(K^{\star}).
$$

##### `select_sparse_conjugate_modes`

For a two-dimensional Fourier block $\hat h_{m,n}$, the function first keeps one non-DC representative from each conjugate pair.

$$
\mathcal P_{\pm}=\{(m,n):m>0\ \mathrm{or}\ (m=0\ \mathrm{and}\ n>0)\}.
$$

The Parseval-pair energy is

$$
E_{m,n}^{\pm}=
\lvert\hat h_{m,n}\rvert^2
+\lvert\hat h_{-m,-n}\rvert^2.
$$

The same EMS rule used by `elastic_mode_selection` chooses $K^\star$ from the sorted pair energies. The returned sparse spectrum writes both $\hat h_{m,n}$ and $\hat h_{-m,-n}$ for each retained representative, so the reconstructed field remains real-valued and the retained energy follows the two-sided Parseval accounting. If the total pair energy is zero, the function returns an all-zero sparse block and $K^\star=0$.

#### `Module_Plot_Template.py`

Reusable plotting-format helper.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $\mathcal S$ | `style_values` | baseline `#f06b4f`, optimized `#4aa3df`, baseline median `#c0392b`, optimized median `#1f78b4`, cmap `YlOrRd_r` | recurring plot style dictionary |
| $b_{\mathrm{mpl}}$ | `matplotlib.use("Agg")` | `Agg` | deterministic non-interactive rendering backend |
| $c_{\mathrm{grid}}$ | `grid.color` | `#6f6f6f` | shared grid-line color |
| $f_{\mathrm{stem}}$ | `file_stem` | `-` | figure filename stem |
| $p_{\mathrm{out}}$ | `output_path` | `-` | output path for the saved figure |
| $\mathcal F$ | `figure` | `-` | Matplotlib figure object |
| $\mathrm{mm}$ | `mm` | $1/25.4$ | millimetre-to-inch conversion factor |

##### `apply_james_style`

The function fixes the global plotting parameters and returns

$$
\mathcal S=
\{
\text{baseline color},
\text{optimized color},
\text{median colors},
\text{terrain colormap}
\}.
$$

The backend is set to `Agg`, and the grid, spine, tick, font, save, and hatch parameters are centralized here so all plotting scripts render from the same style state.

##### `figure_output_path`

For a stem $f_{\mathrm{stem}}$,

$$
p_{\mathrm{out}}=\mathrm{./figures/}f_{\mathrm{stem}}\mathrm{.png}.
$$

##### `save_png_and_pdf`

Given $\mathcal F$ and $p_{\mathrm{out}}$, the function writes

$$
\mathcal F \rightarrow \{p_{\mathrm{out}},\,p_{\mathrm{out}}|_{\mathrm{.png}\rightarrow \mathrm{.pdf}}\}.
$$

#### `Module_Alps.py`

Shared Alps DEM preprocessing, local-window geometry, and per-triangle comparison helpers.

##### Index

| Symbol | Code variable | Value | Purpose |
|---|---|---|---|
| $\mathcal D_{\mathrm{Alps}}$ | `dem_dir` | `./srtm_alps` | local Alps DEM directory copied into the project |
| $A_{\mathrm{DEM}}$ | `preprocessed_dem` | `./srtm_alps/alps_dem_processed.npz` | processed DEM archive used by the compute and plot scripts |
| $\phi,\lambda$ | `lat`, `lon` | $\phi\in[44,49]$, $\lambda\in[5,16]$ | SRTM tile latitude and longitude coverage |
| $B$ | `block_size` | `30` | non-overlapping block average from 1 arc-sec to 30 arc-sec |
| $\lambda_0$ | `smooth_km` | `5.0` km | spectral e-fold wavelength for the Gaussian low-pass filter |
| $h_{\min}$ | `min_elev` | `-500.0` m | lower elevation clip before smoothing |
| $s_{\mathrm{hi}}$ | `hires_subsample` | `1` | high-resolution figure-raster stride stored in the archive |
| $d_{\mathrm{hi}}$ | `hires_dtype` | `float32` | high-resolution plotting-raster storage dtype |
| $R_\oplus$ | `earth_radius_km` | `6371.0` km | Plate Carrée projection radius |
| $\Delta_{\mathrm{cell}}$ | `cell_size_km` | `80.0` km | default triangular analysis mesh spacing |
| $g$ | `mesh_name` | `r2b4`, `r2b5`, or `regular` | deterministic Alps proxy mesh label |
| $\Gamma_g$ | `mesh_signature` | 16 hex characters | stable fingerprint of mesh geometry and assignment tables |
| $r(q)$ | return of `find_mesh_triangle_ids` | `-1` or triangle id | mesh triangle assigned to DEM point $q$ |
| $N_{\max}^{\mathrm{mode}}$ | `n_modes` | `32` for `80.0` km cells | maximum signed Fourier mode index |
| $\eta$ | `window_expansion` | `2.0` | square/circle/triangle support expansion factor |
| $\sigma$ | `oversample` | `1.5` | ENUFFT auxiliary-grid oversampling |
| $a$ | `kernel_half_width` | `4` | Kaiser-Bessel half-width in auxiliary grid cells |
| $\beta$ | `kernel_beta` | `2.34` | Kaiser-Bessel denominator parameter |
| $S_{\mathrm{CSA}}$ | `csa_sparse_modes` | `2 * n_modes` when omitted | CSA signed sparse-mode limit |

##### `require_preprocessed_dem`

The function returns $A_{\mathrm{DEM}}$ when the processed archive exists. If it does not exist, the function raises a `FileNotFoundError` with the two supported preprocessing commands.

$$
A_{\mathrm{DEM}}=
\mathrm{./srtm\_alps/alps\_dem\_processed.npz}.
$$

This makes every compute and plot path fail with the same recovery instruction.

##### `build_alps_case`

The function returns the Alps parameter dictionary

$$
\mathcal P_{\mathrm{Alps}}=\{ \Delta_{\mathrm{cell}},N_{\max}^{\mathrm{mode}},\sigma,a,\beta,\eta,\mathcal W,\ldots \}.
$$

User-supplied updates replace defaults. If `mesh_name` is one of the fixed Alps presets, it is normalized and the preset overrides the local proxy spacing.

$$
\mathrm{r2b4}\mapsto(\Delta_{\mathrm{cell}},N_{\max}^{\mathrm{mode}})=(160\ \mathrm{km},16),
\qquad
\mathrm{r2b5}\mapsto(\Delta_{\mathrm{cell}},N_{\max}^{\mathrm{mode}})=(80\ \mathrm{km},32),
$$

where the mode value is filled only when `n_modes` was omitted. For non-preset regular proxy meshes, if $N_{\max}^{\mathrm{mode}}$ is omitted, the fallback rule is

$$
N_{\max}^{\mathrm{mode}}=
\begin{cases}
16, & \Delta_{\mathrm{cell}}\ge120\ \mathrm{km},\\
32, & \Delta_{\mathrm{cell}}<120\ \mathrm{km}.
\end{cases}
$$

##### `normalize_mesh_name`

The function maps accepted spellings to the canonical mesh labels

$$
\mathrm{regular},\qquad \mathrm{r2b4},\qquad \mathrm{r2b5}.
$$

It removes underscores and hyphens before matching. The accepted aliases are

$$
\{\mathrm{regular},\mathrm{proxy},\mathrm{regularproxy}\}\mapsto\mathrm{regular},
$$

$$
\{\mathrm{r02b04},\mathrm{r2b4}\}\mapsto\mathrm{r2b4},
\qquad
\{\mathrm{r02b05},\mathrm{r2b5}\}\mapsto\mathrm{r2b5}.
$$

Unknown labels raise an error before any mesh-dependent file is written.

##### `mask_invalid_elevations`

For each DEM sample $h_q$, the function builds the invalid mask

$$
M_q=
\neg\mathrm{finite}(h_q)
\ \lor\ 
(h_q=h_{\mathrm{nodata}})
\ \lor\
(h_q\le -32768),
$$

where the explicit nodata test is used only when a finite nodata value is supplied. The returned field is

$$
h_q^{\mathrm{clean}}=
\begin{cases}
\mathrm{NaN}, & M_q=1,\\
h_q, & M_q=0.
\end{cases}
$$

##### `parse_srtm_tile_name`

The SRTM tile name `N47E011.hgt` is parsed as

$$
\phi_0=47,\qquad \lambda_0=11.
$$

South and west prefixes apply negative signs. The returned pair is the lower-left integer latitude-longitude corner $(\phi_0,\lambda_0)$ of the one-degree tile.

##### `read_srtm_hgt`

The raw HGT file is interpreted as a big-endian int16 array

$$
H\in\mathbb Z^{3601\times3601}.
$$

The tile bounds are

$$
(\lambda_0,\phi_0,\lambda_0+1,\phi_0+1),
$$

and the height field is passed through `mask_invalid_elevations`, so all returned elevations are float values in metres with invalid points set to `NaN`.

##### `read_srtm_tile`

This function implements the reader cascade

$$
\mathrm{HGT}\rightarrow\mathrm{rasterio}\rightarrow\mathrm{GDAL}.
$$

All branches return the same mathematical object, a north-up DEM array $H_{ij}$ and geographic bounds $(\lambda_{\min},\phi_{\min},\lambda_{\max},\phi_{\max})$. The GeoTIFF readers additionally use file nodata metadata in the invalid-value mask.

##### `resolve_srtm_tile`

For integer tile coordinates $(\phi_0,\lambda_0)$, the tile stem is

$$
\mathrm{N}\phi_0\mathrm{E}\lambda_0,
$$

with zero padding. The function searches local suffixes `.hgt`, `.tif`, and `.tiff` and returns the first path that exists. The default returned path is the HGT path, which makes missing-tile accounting deterministic.

##### `load_and_mosaic_tiles`

For latitude cells $\phi=\phi_{\min},\ldots,\phi_{\max}-1$ and longitude cells $\lambda=\lambda_{\min},\ldots,\lambda_{\max}-1$, each SRTM tile has $3601$ samples per side. Adjacent tiles share one border row and column, so the mosaic shape is

$$
N_y=(\phi_{\max}-\phi_{\min})(3601-1)+1,\qquad
N_x=(\lambda_{\max}-\lambda_{\min})(3601-1)+1.
$$

The north-up row offset is

$$
i_0=(\phi_{\max}-1-\phi)(3601-1),
$$

and the column offset is

$$
j_0=(\lambda-\lambda_{\min})(3601-1).
$$

##### `coarse_grain`

The coarse grid is the non-overlapping block mean

$$
\bar h_{ij}=\frac{1}{B^2}\sum_{p=0}^{B-1}\sum_{q=0}^{B-1}h_{Bi+p,Bj+q}.
$$

The latitude and longitude vectors are coarse-grained with the same block-centre average,

$$
\bar\phi_i=\frac1B\sum_{p=0}^{B-1}\phi_{Bi+p},
\qquad
\bar\lambda_j=\frac1B\sum_{q=0}^{B-1}\lambda_{Bj+q}.
$$

##### `latlon_to_xy_km`

The local Plate Carrée projection is

$$
x=R_\oplus(\lambda-\lambda_0)\cos\phi_0,\qquad
y=R_\oplus(\phi-\phi_0),
$$

with angular differences in radians and distances in kilometres. The code applies this formula elementwise to scalar or array latitude-longitude inputs.

##### `apply_5km_smoother`

The Gaussian smoothing scale is

$$
\sigma_{\mathrm{eq}}=\frac{\lambda_0}{\sqrt{2}\pi}.
$$

The code convolves both the filled terrain and the finite-value mask, then divides the smoothed fields so missing values and finite-domain truncation keep unit local gain.

##### `clip_elevations`

The clipped terrain is

$$
h_q^{\mathrm{clip}}=\max(h_q,h_{\min}),
$$

with the default $h_{\min}=-500\ \mathrm{m}$. This is applied after invalid SRTM sentinels are converted to `NaN`, so nodata values are not turned into valid terrain.

##### `preprocess_alps_dem`

The preprocessing chain is

$$
H^{1''}\rightarrow H^{\mathrm{clip}}\rightarrow
H^{30''}\rightarrow H^{\mathrm{smooth}}\rightarrow A_{\mathrm{DEM}}.
$$

It also stores a high-resolution clipped raster for plotting.

$$
H^{\mathrm{hires}}=H^{\mathrm{clip}}_{::s_{\mathrm{hi}},\,::s_{\mathrm{hi}}},
$$

with `float32` storage by default. The saved archive contains projected coordinates $(x_i,y_j)$, geographic coordinates $(\lambda_i,\phi_j)$, the smoothed elevation `elev`, the block-averaged elevation before smoothing `elev_block_avg`, the high-resolution raster `elev_hires`, high-resolution coordinates `x_km_hires` and `y_km_hires`, and the stored `hires_subsample`.

##### `orient_y_ascending`

If the coordinate vector has $y_0>y_{N_y-1}$, the function returns

$$
y'_j=y_{N_y-1-j},\qquad
z'_{j,i}=z_{N_y-1-j,i}.
$$

If $y$ is already ascending, the arrays are returned unchanged.

##### `load_alps_dem`

The function opens $A_{\mathrm{DEM}}$, orients $y$ upward, replaces remaining non-finite heights by the median finite height, and flattens the grid as

$$
(x_q,y_q,h_q)=(X_{ji},Y_{ji},H_{ji}),
\qquad q=jN_x+i.
$$

It also builds a regular-grid interpolator on $(y_j,x_i)$ and stores

$$
L_x=x_{N_x-1}-x_0,\qquad L_y=y_{N_y-1}-y_0.
$$

##### `cell_diagonal_code`

For a structured mesh cell $(i,j)$, the function returns the diagonal code

$$
d_{ij}\in\{\mathrm F,\mathrm B\}.
$$

When a legacy diagonal table exists for the requested mesh and has the expected shape, the stored value is used. Otherwise the fallback checkerboard is

$$
d_{ij}=
\begin{cases}
\mathrm F, & i+j\ \mathrm{even},\\
\mathrm B, & i+j\ \mathrm{odd}.
\end{cases}
$$

The code `F` means lower-left to upper-right. The code `B` means lower-right to upper-left.

##### `regular_icon_like_triangles`

The mesh vertices are a regular grid with

$$
N_x^{v}=\lceil\frac{x_{\max}-x_{\min}}{\Delta_{\mathrm{cell}}}\rceil+1,
\qquad
N_y^{v}=\lceil\frac{y_{\max}-y_{\min}}{\Delta_{\mathrm{cell}}}\rceil+1.
$$

Each cell is split by a stored deterministic diagonal code. For code `F`, the cell uses the lower-left to upper-right diagonal,

$$
(v_{BL},v_{BR},v_{TR}),\qquad (v_{BL},v_{TR},v_{TL}).
$$

for code `B`, it uses the lower-right to upper-left diagonal,

$$
(v_{BL},v_{BR},v_{TL}),\qquad (v_{BR},v_{TR},v_{TL}).
$$

The internal closure `vertex_id` maps a structured vertex coordinate to its row-major vertex index.

$$
\nu(i,j)=jN_x^v+i.
$$

The R2B4 and R2B5 diagonal tables are fixed in `legacy_cell_diagonal_patterns`, then reordered by `legacy_triangle_orders` so the final `tri_num` order matches the original Alps sweep. `legacy_cell_tie_slots` records exact diagonal ownership for DEM samples that fall exactly on a split edge. The mesh record stores `mesh_version = "structured_legacy_v1"`.

##### `vertex_id`

This nested helper is defined inside `regular_icon_like_triangles` and is not exported. It provides the row-major map

$$
\nu(i,j)=jN_x^v+i
$$

used to build the two triangle vertex triples in each structured cell.

##### `mesh_geometry_signature`

The signature is a SHA-256 digest truncated to 16 hexadecimal characters. The digest is built from the mesh arrays

$$
\{\mathrm{vertices},\mathrm{triangles},
\mathrm{cell\_triangle\_ids},
\mathrm{cell\_diagonal\_codes},
\mathrm{cell\_tie\_slots}\}.
$$

For each stored array, the key name, dtype, shape, and raw bytes enter the hash. The reducer uses this value to reject rows produced by a different physical mesh or triangle ordering.

##### `build_alps_mesh`

The function normalizes the requested mesh label and dispatches to `regular_icon_like_triangles`.

$$
\mathcal M_g=
\mathrm{regular\_icon\_like\_triangles}
(x_i,y_j,\Delta_{\mathrm{cell}},g).
$$

If no mesh label is supplied, the fallback is `regular`. The returned mesh contains the vertices, triangles, assignment tables, legacy ordering, and $\Gamma_g$.

##### `find_mesh_triangle_ids`

For structured Alps meshes, the function assigns each point $p_q=(x_q,y_q)$ analytically. It first finds the structured cell

$$
i=\mathrm{search}(x_q,\{x_i^v\})-1,
\qquad
j=\mathrm{search}(y_q,\{y_j^v\})-1,
$$

then converts to local cell coordinates

$$
u_q=\frac{x_q-x_i^v}{x_{i+1}^v-x_i^v},
\qquad
v_q=\frac{y_q-y_j^v}{y_{j+1}^v-y_j^v}.
$$

For an `F` diagonal, the upper slot is selected when $v_q>u_q$. For a `B` diagonal, the upper slot is selected when $v_q>1-u_q$. Exact diagonal ties use `cell_tie_slots` when a legacy override is stored. Points outside the mesh receive $r(q)=-1$.

If the mesh does not use structured assignment, the fallback builds a Delaunay triangulation and maps each simplex back to the stored triangle list by sorted vertex ids.

##### `deplane_dem_on_mesh`

For each DEM point, `find_mesh_triangle_ids` computes the rectangular cell and then applies that cell's diagonal test to assign a mesh triangle id $r(q)$. For triangle $r$, the mean height is

$$
\bar h_r=\frac{1}{Q_r}\sum_{q:r(q)=r}h_q,
$$

and the deplaned sample is

$$
h'_q=h_q-\bar h_{r(q)}.
$$

The function stores $h'_q$, triangle ids, triangle means, and triangle sample counts.

##### `points_in_triangle_mask`

For triangle vertices $v_0,v_1,v_2$ and point $p_q$, the signed edge tests are

$$
d_1(q)=\det(p_q-v_1,v_0-v_1),\quad
d_2(q)=\det(p_q-v_2,v_1-v_2),\quad
d_3(q)=\det(p_q-v_0,v_2-v_0).
$$

With tolerance $\epsilon=10^{-10}\max(\Delta x_T,\Delta y_T,1)^2$, the point is inside if the tests do not contain both a value below $-\epsilon$ and a value above $\epsilon$.

##### `split_window_strategy`

The function maps a name such as `circle_edge_aligned` to

$$
(\mathrm{alignment},\mathrm{support})=(\mathrm{edge\_aligned},\mathrm{circle}).
$$

The six supported combinations are square, triangle, and circle supports with centroid or edge-aligned frames.

##### `csa_window_strategy`

For CSA, the support must be square. Therefore

$$
\mathrm{CSA}(\mathrm{triangle\_centroid})=\mathrm{square\_centroid},
\qquad
\mathrm{CSA}(\mathrm{circle\_edge\_aligned})=\mathrm{square\_edge\_aligned},
$$

and square supports map to themselves.

##### `is_csa_supported_window`

This predicate returns true exactly when

$$
\mathrm{support}(\mathcal W)=\mathrm{square}.
$$

It is used to decide whether the current ENUFFT support can reuse its CSA window or must build the square reference window.

##### `get_analysis_window`

For one triangle, `get_analysis_window` builds the local support. The square support has side lengths $L_x^{\mathrm{loc}}=L_y^{\mathrm{loc}}=\eta E$, where $E$ is the larger triangle extent in the chosen local frame. The returned local coordinates are the inputs to the ENUFFT and CSA Fourier fits.

For edge-aligned windows, points are rotated by the longest-edge angle $\theta$.

$$
\begin{bmatrix}x'\\y'\end{bmatrix}=
\begin{bmatrix}\cos(-\theta)&-\sin(-\theta)\\ \sin(-\theta)&\cos(-\theta)\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}.
$$

Let

$$
B_{\mathrm{loc}}=[0,L_x^{\mathrm{loc}}]\times[0,L_y^{\mathrm{loc}}],
\qquad
T_{\mathrm{loc}}=T'- (x_0,y_0),
$$

where $T'$ is the triangle in the chosen local frame. The support area stored with the window is

$$
\lvert D\rvert=
\begin{cases}
(2h)^2, & \mathrm{square},\\
\lvert T_{\mathrm{loc}}\cap B_{\mathrm{loc}}\rvert, & \mathrm{triangle},\\
\pi h^2, & \mathrm{circle},
\end{cases}
$$

where $h=\eta E/2$. Thus a triangular Alps support uses the same clipped-support convention as the ported mono triangular mask: samples are selected by the local square window intersected with the triangle, and `polygon_box_intersection_area` provides the matching intersection area for Voronoi weighting and reported support area. The Fourier basis remains rectangular on $L_x^{\mathrm{loc}}\times L_y^{\mathrm{loc}}$.

##### `points_to_window_coordinates`

For unrotated windows, the local coordinates are

$$
x_q^{\mathrm{loc}}=x_q-x_0,\qquad y_q^{\mathrm{loc}}=y_q-y_0.
$$

For edge-aligned windows, the same translation is applied after the rotation stored in the window dictionary.

##### `local_window_coordinates`

This is the masked form of `points_to_window_coordinates`.

$$
\{(x_q^{\mathrm{loc}},y_q^{\mathrm{loc}}):M_q=1\}
\leftarrow
\{(x_q,y_q):M_q=1\}.
$$

It keeps the local coordinates aligned with the filtered height array used by ENUFFT and CSA.

##### `xy_to_lonlat`

The inverse projection for arrays is

$$
\phi=\phi_0+\frac{y}{R_\oplus},\qquad
\lambda=\lambda_0+\frac{x}{R_\oplus\cos\phi_0},
$$

with radians converted back to degrees.

##### `lonlat_to_xy`

The scalar forward projection is the same Plate Carrée map used in preprocessing,

$$
x=R_\oplus(\lambda-\lambda_0)\cos\phi_0,\qquad
y=R_\oplus(\phi-\phi_0),
$$

returned as Python floats for tick and label placement.

##### `compute_local_voronoi_weights`

The quadrature weights assign a regular support grid to the nearest DEM sample. If $c_q$ grid cells belong to sample $q$, the weight is normalized as

$$
w_q=c_q\frac{\lvert D\rvert}{N_{\mathrm{grid}}},
\qquad
\sum_q w_q=\lvert D\rvert.
$$

For triangle and circle supports, the regular grid is clipped to the active support before nearest-neighbour ownership is counted. For triangle supports, the normalization area is the clipped polygon area $\lvert T_{\mathrm{loc}}\cap B_{\mathrm{loc}}\rvert$, so the grid mask and the area normalization describe the same geometric support.

##### `compute_wave_direction`

The dominant wave direction is

$$
\theta_{m,n}=\tan^{-1}(\frac{\ell_n}{k_m}),
\qquad
k_m=\frac{2\pi m}{L_x},\quad \ell_n=\frac{2\pi n}{L_y}.
$$

The returned angle is in degrees.

##### `spectral_variance`

The non-DC Parseval-style variance proxy is

$$
V_{\mathrm{spec}}
=\sum_{(m,n)\ne(0,0)}\lvert \hat h_{m,n}\rvert^2.
$$

The DC coefficient is set to zero before summation.

##### `init_alps_worker`

The function stores the shared DEM, mesh, and case objects in worker globals.

$$
(\mathcal D,\mathcal M,\mathcal P_{\mathrm{Alps}})\rightarrow
(\_worker\_dem,\_worker\_mesh,\_worker\_case).
$$

This avoids pickling the full DEM for every triangle task when the forked pool starts.

##### `process_alps_triangle`

For triangle $T_r$, the function builds the local window samples

$$
\{x_q^{\mathrm{loc}},y_q^{\mathrm{loc}},h_q'\}_{q\in D_r}.
$$

It computes the raw ENUFFT block $\hat h_{m,n}^{\mathrm{raw}}$, selects the EMS sparse ENUFFT block $\hat h_{m,n}^{\mathrm{ENUFFT}}$, and computes the CSA block $\hat h_{m,n}^{\mathrm{CSA}}$ on the square reference support. Reconstruction diagnostics on the target triangle use

$$
\mathrm{RMSE}=
\sqrt{\frac1{Q_T}\sum_{q\in T_r}
(h_q^{\mathrm{rec}}-h_q')^2},
\qquad
\mathrm{relRMSE}=\frac{\mathrm{RMSE}}{\mathrm{std}_{q\in T_r}(h_q')}.
$$

When the target triangle has reference standard deviation at or below `1 m`, `rel_rmse_en` and `rel_rmse_csa` are written as `NaN` because the relative normalization is not meaningful for flat or nearly flat terrain. The absolute `rmse_en` and `rmse_csa` values are still stored, so these cells remain diagnosable without creating artificial small-denominator outliers in pooled relative-RMSE summaries. It also stores $K^\star$, dominant mode direction, non-DC spectral variance, sorted amplitudes, point counts, triangle geometry, and `metric_version` so old summary CSVs are not reused after metric semantics change.

##### `summarise_alps_results`

For a scalar diagnostic $z_r$ over successful triangles, finite medians and maxima are computed as

$$
\tilde z=\mathrm{median}\{z_r:\mathrm{finite}(z_r)\},
\qquad
z_{\max}=\max\{z_r:\mathrm{finite}(z_r)\}.
$$

The variance-ratio medians use only triangles whose true reference standard deviation is greater than `1 m`:

$$
\mathrm{median}(
\frac{V_{\mathrm{spec},r}}{V_{\mathrm{true},r}}
).
$$

##### `scalar_alps_result_row`

The in-memory row contains arrays and geometry lists. This function removes dense spectra and sorted amplitude arrays, and converts the centroid list

$$
(x_c,y_c)\rightarrow
\{\mathrm{centroid\_x}=x_c,\mathrm{centroid\_y}=y_c\}.
$$

##### `tag_number`

The function converts numeric tag values by

$$
1.25\mapsto\mathrm{1p25},\qquad
-0.5\mapsto\mathrm{m0p5}.
$$

This keeps file tags shell-safe while preserving the parameter value.

##### `make_alps_config_tag`

For one case dictionary, the tag is

$$
\tau=
[\mathrm{\_}\mathrm{mesh}]
\mathrm{\_N}N_{\max}^{\mathrm{mode}}
\mathrm{\_}\mathcal W
\mathrm{\_}w
\mathrm{\_eta}\eta
\mathrm{\_os}\sigma
\mathrm{\_csa}S_{\mathrm{CSA}}
\mathrm{\_dx}\Delta_{\mathrm{cell}}.
$$

All numeric pieces pass through `tag_number`.

##### `make_alps_sweep_tag`

For one mesh-level sweep summary, the aggregate tag is

$$
\tau_{\mathrm{sweep}}=
[\mathrm{\_}\mathrm{mesh}]
\mathrm{\_N}N_{\max}^{\mathrm{mode}}
\mathrm{\_dx}\Delta_{\mathrm{cell}}.
$$

The mesh prefix is included for `r2b4` and `r2b5`, giving the fixed outputs `_r2b4_N16_dx160` and `_r2b5_N32_dx80`.

##### `build_alps_modes_rows`

The sparse mode CSV includes the DC coefficient and all coefficients satisfying

$$
\lvert\hat h_{m,n}\rvert>10^{-15}.
$$

Each row stores $(m,n)$, real part, imaginary part, amplitude, method, triangle id, and a descending amplitude rank.

##### `build_alps_spectra_rows`

For a sorted amplitude sequence $a_1\ge a_2\ge\cdots$, the function writes

$$
\{(r,a_r):r=1,\ldots,R_{\mathrm{spec}}\}
$$

for ENUFFT and CSA separately. Non-finite CSA amplitudes are skipped, so invalid CSA cases do not enter pooled spectral plots.
