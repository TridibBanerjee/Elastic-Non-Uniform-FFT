# ENUFFT API Notes

## Coefficient Convention

The package follows the upstream ENUFFT coefficient convention.

```text
h_hat[m, n] = mean_q h_q exp[-i(k_m x_q + l_n y_q)]
k_m = 2 pi m / Lx
l_n = 2 pi n / Ly
```

Optional quadrature weights are rescaled into the same sample mean
normalization before direct DFT or NUFFT evaluation.

## NUFFT

Use `compute_nufft_coefficients` for an explicit sample set and mode block.

```python
from enufft import compute_nufft_coefficients, mode_values

modes = mode_values(8)
spectrum = compute_nufft_coefficients(x, y, h, modes, modes, Lx, Ly)
```

Use `compute_direct_dft_coefficients` for exact validation on smaller inputs.

## Polygon Windows

`nufft_on_polygon` and `enufft_on_polygon` construct an analytic square from a
supplied polygon.

```python
from enufft import WindowConfig, enufft_on_polygon

result = enufft_on_polygon(
    points,
    values,
    polygon,
    mode_limit=8,
    window_config=WindowConfig(
        support="polygon",
        alignment="edge_aligned",
        expansion=1.5,
    ),
    weight_type="voronoi",
)
```

The Fourier basis is evaluated on the square window. The support controls which
samples are used inside that square.

## EMS Counts

For a one dimensional nonnegative energy spectrum.

```python
from enufft import EMSConfig, elastic_mode_selection

ems = elastic_mode_selection(energy, EMSConfig(k_min=1, k_max=12))
ems.retain_count("pairs")
ems.retain_count("signed")
```

For a Fourier block, use `select_sparse_conjugate_modes`. It groups non DC
signed modes into canonical conjugate pairs `(m, n)` and `(-m, -n)`, applies EMS
to pair energies, and returns a sparse conjugate symmetric spectrum.

```python
from enufft import select_sparse_conjugate_modes

selection = select_sparse_conjugate_modes(spectrum, m_values, n_values)
selection.mode_pair_count
selection.signed_mode_count
selection.selected_modes
```

## Reconstruction

`reconstruct_at_points` evaluates a coefficient block on local window
coordinates. `ENUFFTResult.reconstruct()` reconstructs on the active sample
coordinates by default.
