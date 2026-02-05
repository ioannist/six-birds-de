import numpy as np

from sixbirds_cosmo.cosmo_background import distance_modulus
from sixbirds_cosmo.rewrite_background import distance_modulus_rewrite


def test_rewrite_reduces_to_lcdm_when_m_zero():
    z = np.array([0.1, 0.5, 1.0])
    H0 = 70.0
    om = 0.3
    A = 0.7
    mu_lcdm = distance_modulus(z, H0, om, A, method="trapz")
    mu_rw = distance_modulus_rewrite(z, H0, om, A, m=0.0, method="trapz")
    assert np.max(np.abs(mu_lcdm - mu_rw)) < 1e-6
