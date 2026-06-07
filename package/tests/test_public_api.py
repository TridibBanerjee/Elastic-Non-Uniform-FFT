import enufft

# Adapted from Elastic Non Uniform FFT by Tridib Banerjee
# Original research code by Dr T Banerjee
# Apache 2.0 license terms are provided in LICENSE


def test_public_api_exports_expected_entry_points():
    expected = {
        "EMSConfig",
        "NUFFTConfig",
        "WindowConfig",
        "compute_direct_dft_coefficients",
        "compute_nufft_coefficients",
        "count_signed_nonzero_modes",
        "count_unique_mode_pairs",
        "elastic_mode_selection",
        "ems_retain_count",
        "enufft_on_polygon",
        "nufft_on_polygon",
        "reconstruct_at_points",
        "select_sparse_conjugate_modes",
    }

    assert expected.issubset(set(enufft.__all__))
