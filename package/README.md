# ENUFFT

[![PyPI](https://img.shields.io/pypi/v/enufft.svg)](https://pypi.org/project/enufft/)
[![Python](https://img.shields.io/pypi/pyversions/enufft.svg)](https://pypi.org/project/enufft/)
[![License](https://img.shields.io/badge/License-Apache--2.0-red.svg)](https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT/blob/main/package/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20544458-blue.svg)](https://doi.org/10.5281/zenodo.20544458)

**ENUFFT** is a Python package for local Fourier analysis on irregularly sampled
terrain, polygonal cells, and other scattered two-dimensional fields.

It combines two pieces of the Elastic Non-Uniform FFT framework:

- A Kaiser-Bessel non-uniform FFT path for estimating local Fourier
  coefficients directly from scattered samples.
- Elastic Mode Selection (EMS), which compresses the spectrum to an adaptive
  number of physically paired retained modes.

The canonical research repository, derivation, case studies, and citation
metadata live at
[TridibBanerjee/Elastic-Non-Uniform-FFT](https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT).

## Install

```bash
pip install enufft
```

Optional extras:

```bash
pip install "enufft[plots]"
pip install "enufft[test]"
```

Install the package directly from the repository:

```bash
pip install "git+https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT.git#subdirectory=package"
```

ENUFFT requires Python 3.10 or newer, NumPy, and SciPy.

## Quick Start

```python
import numpy as np
from enufft import EMSConfig, WindowConfig, enufft_on_polygon

polygon = np.array([
    [0.10, 0.16],
    [1.35, 0.02],
    [1.72, 0.82],
    [1.05, 1.55],
    [0.16, 1.18],
])

rng = np.random.default_rng(8)
points = rng.uniform([-0.45, -0.45], [2.05, 2.05], size=(3600, 2))
values = np.cos(2.0 * np.pi * (3.0 * points[:, 0] + 2.0 * points[:, 1]))

result = enufft_on_polygon(
    points,
    values,
    polygon,
    mode_limit=6,
    window_config=WindowConfig(
        support="square",
        alignment="centroid",
        expansion=1.42,
    ),
    ems_config=EMSConfig(k_min=1, k_max=6, alpha_min=0.0, alpha_max=0.78),
    weight_type="voronoi",
)

print(result.selected_modes)
print(result.mode_pair_count, result.signed_mode_count)
print(result.power_retained)
```

## What You Get

- Exact scattered-point DFT coefficients for reference calculations.
- Kaiser-Bessel NUFFT coefficient blocks for faster local spectral analysis.
- Polygon-derived square analysis windows with square, polygon, or circular
  sample supports.
- Uniform or local Voronoi-style quadrature weights.
- EMS diagnostics for any nonnegative spectrum.
- Pair-aware EMS for signed Fourier coefficient blocks.
- Sparse inverse reconstruction at arbitrary local coordinates.

## Public API

The most useful entry points are:

| Function or class | Purpose |
| --- | --- |
| `enufft_on_polygon` | Compute polygon-windowed NUFFT coefficients and apply EMS. |
| `nufft_on_polygon` | Compute the raw polygon-windowed NUFFT coefficient block. |
| `compute_nufft_coefficients` | Compute a Kaiser-Bessel NUFFT block from explicit samples and modes. |
| `compute_direct_dft_coefficients` | Compute the exact scattered direct DFT block. |
| `elastic_mode_selection` | Run EMS on a one-dimensional nonnegative spectrum. |
| `select_sparse_conjugate_modes` | Apply EMS to signed Fourier coefficients by conjugate mode pair. |
| `reconstruct_at_points` | Evaluate an inverse Fourier series at local coordinates. |
| `WindowConfig`, `NUFFTConfig`, `EMSConfig` | Configure windowing, NUFFT, and EMS behavior. |

See the
[API notes](https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT/blob/main/package/docs/API.md)
for additional examples.

## EMS On A Supplied Spectrum

```python
import numpy as np
from enufft import EMSConfig, elastic_mode_selection

energy = np.array([10.0, 3.0, 1.0, 0.4, 0.1])
diagnostics = elastic_mode_selection(
    energy,
    EMSConfig(k_min=1, k_max=5, alpha_min=0.0, alpha_max=0.7),
)

print(diagnostics.retain_count("pairs"))
print(diagnostics.retain_count("signed"))
```

## Coefficient Convention

ENUFFT follows the sample-mean Fourier convention used by the research code:

```text
h_hat[m, n] = mean_q h_q exp[-i(k_m x_q + l_n y_q)]
k_m = 2 pi m / Lx
l_n = 2 pi n / Ly
```

Optional sample weights are normalized into the same sample-mean convention
before direct DFT or NUFFT evaluation.

## Method

The NUFFT path spreads irregular samples to an oversampled auxiliary grid with a
compact Kaiser-Bessel kernel, applies `fft2`, deconvolves the kernel transform,
and extracts the requested signed mode block.

EMS sorts nonnegative mode or pair energies, computes a participation-ratio
effective count, measures local spectral smoothness from adjacent energy gaps,
and selects the smallest admissible retained count that satisfies the resulting
adaptive retained-power target.

For signed Fourier coefficient blocks, non-DC modes are grouped as conjugate
pairs `(m, n)` and `(-m, -n)`. Retaining both signs keeps inverse
reconstructions real-valued and keeps energy accounting consistent.

## Validation Figures

The repository includes proof plots generated from synthetic fields and
polygonal cells.

The NUFFT proof compares accelerated coefficients against exact direct DFT
coefficients across three synthetic terrain fields.

![NUFFT proof](https://raw.githubusercontent.com/TridibBanerjee/Elastic-Non-Uniform-FFT/main/package/proof/figures/enufft_proof_nufft.png)

The EMS proof evaluates the retained-count rule across analytical spectra. The
red dashed line marks the selected retained count.

![EMS proof](https://raw.githubusercontent.com/TridibBanerjee/Elastic-Non-Uniform-FFT/main/package/proof/figures/enufft_proof_ems.png)

The polygon proof shows the supplied samples, raw Fourier block, EMS-selected
pair, sparse reconstruction, and sorted pair-energy decay for one polygonal
cell.

![Polygon proof](https://raw.githubusercontent.com/TridibBanerjee/Elastic-Non-Uniform-FFT/main/package/proof/figures/enufft_proof_polygon.png)

Regenerate the proof artifacts from a source checkout with:

```bash
python package/scripts/make_proof_plots.py
```

## Citation

If you use ENUFFT in scholarly, published, or publicly distributed work, cite
the framework metadata in
[CITATION.cff](https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT/blob/main/package/CITATION.cff):

```text
Banerjee, Tridib. Elastic Non-Uniform FFT (ENUFFT).
https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT
DOI: 10.5281/zenodo.20544458
```

## License

Released under the Apache License 2.0. See
[LICENSE](https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT/blob/main/package/LICENSE)
and
[NOTICE](https://github.com/TridibBanerjee/Elastic-Non-Uniform-FFT/blob/main/package/NOTICE).
