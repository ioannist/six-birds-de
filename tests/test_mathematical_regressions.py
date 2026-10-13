"""Regression checks for mathematical identities, domains, and evidence semantics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sixbirds_cosmo.core import Completion, Lens, PackagingOperator, default_distance, route_mismatch
from sixbirds_cosmo.cosmo_background import E_lcdm, H_lcdm, distance_modulus
from sixbirds_cosmo.datasets import des_sn5yr as sn
from sixbirds_cosmo.datasets.des_y6_bao import evaluate_alpha_chi2
from sixbirds_cosmo.infer_distance import distance_from_Hz, make_redshift_from_a
from sixbirds_cosmo.likelihoods.gaussian_vector import apply_mask, chi2_gaussian
from sixbirds_cosmo.lss.ppd_probe_split import build_basis, build_surrogate_theory, fit_linear_correction, eval_metrics
from sixbirds_cosmo.lss.theory.reference import ReferenceVectorBackend
from sixbirds_cosmo.lss.theory.stub import StubBackend
from sixbirds_cosmo.numerics import generalized_least_squares
from sixbirds_cosmo.infer.fit import fit_map
from sixbirds_cosmo.likelihoods.base import Dataset
from sixbirds_cosmo.rewrite import H_rewrite, D_rewrite, positivity_ratio_max, fit_rewrite
from sixbirds_cosmo.rewrite_effective_w import fit_effective_w0wa
from sixbirds_cosmo.toy1 import simulate_micro
from sixbirds_cosmo.toy2_patch import simulate_patches


@pytest.mark.parametrize("beta", [0., -.5, .5, 2.])
def test_quadratic_mean_mismatch_exact_identity(beta):
    params = dict(alpha=.7, beta=beta, dt=.001, x0_low=.2, x0_high=1.)
    history = simulate_micro(31, 12, params, 42)
    np.testing.assert_allclose(history["rm"], abs(beta) * params["dt"] * history["x_var"][:-1], atol=2e-15)


def test_noncommutation_can_be_invisible_to_the_lens():
    E = PackagingOperator(Lens(lambda x: x[0]), Completion(lambda y: np.array([y, 0.])))
    T = lambda x: np.array([x[0], 1.])
    assert route_mismatch(T, E, np.array([2., 3.]), space="x") == 1.
    assert route_mismatch(T, E, np.array([2., 3.]), space="y") == 0.


def patch_params(dt=.02, delta=.9):
    return dict(dt=dt, g4pi=1., rho_mean=1., delta_rho=delta, f_void=.3, kappa_scale=5.)


def test_patch_dust_solution_and_weighted_acceleration_identity():
    homogeneous = simulate_patches(10, 100, patch_params(delta=0.), 1)
    H_initial = np.sqrt(2 / 3)
    a_exact = (1 + 1.5 * H_initial * homogeneous["t"])**(2 / 3)
    np.testing.assert_allclose(homogeneous["aD"], a_exact, rtol=1e-8)
    np.testing.assert_allclose(homogeneous["HD"], H_initial / a_exact**1.5, rtol=1e-8)
    np.testing.assert_allclose(homogeneous["Q"], 0, atol=1e-28)
    hetero = simulate_patches(10, 100, patch_params(), 1)
    np.testing.assert_allclose(hetero["Q"], 6 * hetero["var_H"], atol=0)
    np.testing.assert_allclose(hetero["Q"], 3 * hetero["ddot_aD"] / hetero["aD"] + hetero["rhoD"], atol=1e-14)
    assert np.min(hetero["var_H"]) >= 0


def test_patch_integrator_converges_and_seeds_only_permute():
    coarse = simulate_patches(10, 50, patch_params(dt=.04), 1)
    fine = simulate_patches(10, 100, patch_params(dt=.02), 2)
    assert np.max(fine["energy_relative_error"]) < np.max(coarse["energy_relative_error"]) / 10
    permuted = simulate_patches(10, 100, patch_params(dt=.02), 99)
    np.testing.assert_allclose(fine["aD"], permuted["aD"], rtol=1e-14)
    np.testing.assert_allclose(fine["var_H"], permuted["var_H"], rtol=1e-13)


@pytest.mark.parametrize("kind", ["stat", "stat+sys"])
def test_sn_triplet_total_covariance_retains_statistical_diagonal(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(sn, "extract_distances_covmat", lambda _: tmp_path)
    filename = "STATONLY.txt" if kind == "stat" else "STAT+SYS.txt"
    values = "1 1 0\n2 2 0\n" if kind == "stat" else "1 1 1\n1 2 .1\n2 2 1\n"
    (tmp_path / filename).write_text(values)
    sigma = np.array([.2, .3])
    raw = sn.load_des_sn5yr_covariance(tmp_path, kind=kind)["cov"]
    total = sn.load_des_sn5yr_covariance(tmp_path, kind=kind, mu_err=sigma)["cov"]
    np.testing.assert_allclose(total, raw + np.diag(sigma**2))


def test_covariance_duplicate_zero_and_negative_indices_rejected(tmp_path):
    p = tmp_path / "cov.txt"
    p.write_text("0 0 0\n0 0 1\n")
    with pytest.raises(ValueError, match="inconsistent"):
        sn.parse_triplet_cov(p)
    p.write_text("-1 0 1\n")
    with pytest.raises(ValueError, match="nonnegative"):
        sn.parse_triplet_cov(p)


@pytest.mark.parametrize("cov", [np.array([[1., 2.], [2., 1.]]), np.array([[1., 0.], [2., 1.]])])
def test_invalid_covariance_is_not_repaired_silently(cov):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        sn.prepare_cov_solve(cov)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        chi2_gaussian(np.ones(2), cov)


def test_profiled_intercept_is_offset_invariant_without_cancellation():
    cho = sn.prepare_cov_solve(np.eye(3))["cho"]
    chi2, intercept = sn.chi2_profiled_over_M(1e8 + np.array([-1., 0., 1.]), cho)
    assert chi2 == pytest.approx(2., abs=1e-12)
    assert intercept == pytest.approx(1e8)
    delta = sn.heliocentric_mu_correction(np.array([.1]), np.array([.101]))
    assert delta[0] == pytest.approx(5 * np.log10(1.101 / 1.1))


def test_boolean_mask_selects_rows_instead_of_casting_to_zero_one():
    y, c = apply_mask(np.array([1., 2., 3.]), np.eye(3), np.array([False, True, True]))
    np.testing.assert_array_equal(y, [2., 3.])
    assert c.shape == (2, 2)


def test_bao_curve_has_no_flat_extrapolated_tails():
    grid = np.array([.9, 1., 1.1]); chi2 = np.array([1., 0., 1.])
    assert evaluate_alpha_chi2(.95, grid, chi2) == pytest.approx(.5)
    assert np.isinf(evaluate_alpha_chi2(.89, grid, chi2))
    assert np.isinf(evaluate_alpha_chi2(float("nan"), grid, chi2))


def test_scalar_fit_counts_profiled_nuisance_and_checks_boundary():
    class ShapeDataset(Dataset):
        def loglike(self, theta):
            return -.5 * (theta[0] + 1)**2
    fit = fit_map(ShapeDataset("shape", n_data=10, n_profiled_parameters=1), np.array([.5]), bounds=[(0., 1.)])
    assert fit["success"]
    assert fit["theta_hat"] == [0.]
    assert fit["k"] == 2
    assert fit["k_profiled"] == 1
    assert fit["aic"] == pytest.approx(5.)


def test_continuum_positivity_detects_an_interior_maximum():
    # g=4z has max g/(1+z)^3 at z=1/2, missed by endpoints 0 and 1.
    zp = np.array([0., 1.]); gp = np.array([0., 4.])
    assert positivity_ratio_max(1., zp, gp) == pytest.approx(16 / 27)
    with pytest.raises(ValueError, match="entire"):
        D_rewrite(np.array([1.]), 1., 1.8, zp, gp, n_grid=2)
    with pytest.raises(ValueError, match="Non-positive"):
        H_rewrite(np.array([.5]), 1., 1.8, zp, gp)


def test_rewrite_recovers_physical_noiseless_parameters():
    z = np.linspace(.05, 1., 20)
    zp = np.array([0., 1.]); gp = np.array([1., .5])
    data = D_rewrite(z, .8, .2, zp, gp)
    fit = fit_rewrite(z, data, np.full_like(z, .001), dict(z_sorted=zp, g_sorted=gp))
    assert fit["success"]
    assert fit["params"]["H0"] == pytest.approx(.8, rel=1e-6)
    assert fit["params"]["A"] == pytest.approx(.2, rel=1e-6)
    assert fit["H_present"] == pytest.approx(np.sqrt(.8**2 - .2))


def test_distance_and_background_domains_are_not_clipped():
    for scale in ([1., 0.], [1., float("nan")]):
        with pytest.raises(ValueError):
            make_redshift_from_a(np.array(scale))
    with pytest.raises(ValueError, match="cover"):
        distance_from_Hz(np.array([0., 1.]), np.ones(2), np.array([2.]))
    with pytest.raises(ValueError, match="positive"):
        distance_from_Hz(np.array([0., 1.]), np.array([1., 0.]), np.array([.5]))
    np.testing.assert_allclose(distance_from_Hz(np.array([0., 1.]), np.full(2, 2.), np.array([.2, .7])), [.1, .35])
    with pytest.raises(ValueError): H_lcdm(.5, 0., .3, .7)
    with pytest.raises(ValueError): E_lcdm(1., 0., 2.)
    with pytest.raises(ValueError): distance_modulus(0., 70., .3, .7)
    with pytest.raises(ValueError, match="H0"):
        fit_effective_w0wa(np.array([.1, .2]), np.array([70., 72.]), om=.3)


def test_generalized_least_squares_identifiability():
    x = np.linspace(0, 1, 20)
    B = np.column_stack([np.ones(20), 1 + 1e-8 * x])
    expected = np.array([.2, -.1])
    p, chi2 = generalized_least_squares(B @ expected, B, np.eye(20))
    np.testing.assert_allclose(B @ p, B @ expected, atol=1e-14)
    assert chi2 < 1e-26
    with pytest.raises(ValueError, match="rank deficient"):
        generalized_least_squares(x, np.ones((20, 2)), np.eye(20))


def test_unconstrained_gls_predictions_do_not_depend_on_basis_units():
    x = np.linspace(-1, 1, 10)
    B = np.column_stack([np.ones(10), x])
    res = .4 + .2 * x + .05 * x**2
    p, chi2 = generalized_least_squares(res, B, np.eye(10))
    scaled = B * np.array([10., .01])
    p_scaled, chi2_scaled = generalized_least_squares(res, scaled, np.eye(10))
    np.testing.assert_allclose(B @ p, scaled @ p_scaled, atol=1e-14)
    assert chi2 == pytest.approx(chi2_scaled, abs=1e-15)


def test_injected_lss_scores_are_independent_of_measured_central_values():
    cov = np.eye(8)
    block = pd.DataFrame(dict(i=np.arange(8), probe=["shear"]*4+["ggl"]*4, x=np.arange(8)))
    idx = np.arange(4)
    results = []
    for y in [np.zeros(8), np.arange(8)*100.]:
        t0 = build_surrogate_theory(y, cov, block, frac_sigma=.1, beta_x=.2, kappa_probe={})
        B = build_basis(y, cov, block, model="rewrite_like")
        p, _ = fit_linear_correction(y, cov, t0, B, idx)
        results.append(eval_metrics(y, cov, t0 + B @ p, np.arange(4, 8))["chi2"])
    assert results[0] == pytest.approx(results[1], abs=1e-24)


def test_backends_respect_indices_even_when_lengths_match(tmp_path):
    vec = np.array([1., 2., 3.]); idx = pd.DataFrame(dict(i=[2, 0, 1]))
    p = tmp_path / "ref.npy"; np.save(p, vec)
    np.testing.assert_array_equal(ReferenceVectorBackend(p).predict(np.array([]), idx), vec[[2, 0, 1]])
    np.testing.assert_array_equal(StubBackend(vec, np.eye(3)).predict(np.array([]), idx), (vec+.1)[[2, 0, 1]])
    with pytest.raises(ValueError, match="same shape"):
        default_distance(np.ones((2, 1)), np.ones(2))
