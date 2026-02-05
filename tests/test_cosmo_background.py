import numpy as np

from sixbirds_cosmo.cosmo_background import (
    E_lcdm,
    H_lcdm,
    angular_diameter_distance,
    comoving_distance,
    luminosity_distance,
)


def test_normalization_z0():
    H0 = 70.0
    om = 0.3
    ol = 0.7
    assert abs(E_lcdm(0.0, om, ol) - 1.0) < 1e-12
    assert abs(H_lcdm(0.0, H0, om, ol) - H0) < 1e-12
    assert abs(comoving_distance(0.0, H0, om, ol) - 0.0) < 1e-12


def test_eds_comoving_distance_regression():
    H0 = 70.0
    c = 299792.458
    om = 1.0
    ol = 0.0
    z_values = np.array([0.5, 1.0, 2.0])
    analytic = 2.0 * c / H0 * (1.0 - 1.0 / np.sqrt(1.0 + z_values))

    dc_quad = comoving_distance(z_values, H0, om, ol, method="quad")
    dc_trapz = comoving_distance(z_values, H0, om, ol, method="trapz")

    rel_err_quad = np.max(np.abs(dc_quad - analytic) / analytic)
    rel_err_trapz = np.max(np.abs(dc_trapz - analytic) / analytic)

    assert rel_err_quad < 1e-6
    assert rel_err_trapz < 1e-4


def test_comoving_distance_monotonicity():
    H0 = 70.0
    om = 0.3
    ol = 0.7
    z_grid = np.linspace(0.0, 3.0, 50)
    dc = comoving_distance(z_grid, H0, om, ol, method="trapz")
    diffs = np.diff(dc)
    assert np.all(diffs >= -1e-10)


def test_trapz_vs_quad_lcdm():
    H0 = 70.0
    om = 0.3
    ol = 0.7
    z_values = np.array([0.2, 1.0, 2.0])
    dc_trapz = comoving_distance(z_values, H0, om, ol, method="trapz", n_grid=8192)
    dc_quad = comoving_distance(z_values, H0, om, ol, method="quad")
    rel_err = np.max(np.abs(dc_trapz - dc_quad) / dc_quad)
    assert rel_err < 5e-4


def test_distance_relations():
    H0 = 70.0
    om = 0.3
    ol = 0.7
    z = np.array([0.5, 1.0])
    dm = comoving_distance(z, H0, om, ol, method="trapz")
    da = angular_diameter_distance(z, H0, om, ol, method="trapz")
    dl = luminosity_distance(z, H0, om, ol, method="trapz")
    assert np.allclose(da, dm / (1.0 + z))
    assert np.allclose(dl, dm * (1.0 + z))
