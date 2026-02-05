import numpy as np

from sixbirds_cosmo.likelihoods.gaussian_vector import chi2_gaussian


def test_chi2_gaussian_matches_solve():
    cov = np.array([[2.0, 0.3], [0.3, 1.5]])
    resid = np.array([0.4, -0.2])
    chi2_expected = float(resid @ np.linalg.solve(cov, resid))
    chi2, jitter = chi2_gaussian(resid, cov)
    assert abs(chi2 - chi2_expected) < 1e-12
    assert jitter == 0.0
