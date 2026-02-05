import numpy as np

from sixbirds_cosmo.cosmo_background import luminosity_distance, luminosity_distance_wcdm


def test_wcdm_reduces_to_lcdm():
    z = np.array([0.1, 0.5, 1.0])
    H0 = 70.0
    om = 0.3
    dl_lcdm = luminosity_distance(z, H0, om, 1.0 - om, method="trapz")
    dl_wcdm = luminosity_distance_wcdm(z, H0, om, -1.0, method="trapz")
    rel_err = np.max(np.abs(dl_lcdm - dl_wcdm) / dl_lcdm)
    assert rel_err < 1e-6
