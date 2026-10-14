"""Adversarial checks for format, domain and optimizer gaps in the final pass."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from experiments import run_y3_shear_physical_smoke as shear
from sixbirds_cosmo.cosmo_background import comoving_distance, E_lcdm, E_wcdm
from sixbirds_cosmo.datasets.des_y3_2pt import load_des_y3_2pt_fits, build_y3_block_index
from sixbirds_cosmo.datasets.des_y3_chain_meta import infer_loglike_columns
from sixbirds_cosmo.datasets.des_y6_bao import BAODataset, DESY6BAOAlphaDataset
from sixbirds_cosmo.datasets.des_sn5yr import DESSN5YRDistanceDataset
from sixbirds_cosmo.infer.fit import fit_map_multistart
from sixbirds_cosmo.likelihoods.base import Dataset
from sixbirds_cosmo.lss.config import resolve_lss_config
from sixbirds_cosmo.numerics import validated_indices
from sixbirds_cosmo.ppd.ppd_gaussian import ppd_fit_predict
from sixbirds_cosmo.rewrite_background import E_rewrite, comoving_distance_rewrite, rewrite_interval_minimum
from sixbirds_cosmo.audit_ppd import interp_at, rmse, predict_varH_rewrite_from_template
from sixbirds_cosmo.toy1 import simulate_micro
from sixbirds_cosmo import manifest
from experiments import run_y3_ppd_robustness as robustness
import json


def _spectrum(name, values):
    n = len(values)
    return fits.BinTableHDU.from_columns([
        fits.Column(name="BIN1", format="J", array=np.ones(n, dtype=int)),
        fits.Column(name="BIN2", format="J", array=np.ones(n, dtype=int)),
        fits.Column(name="ANG", format="D", unit="arcmin", array=np.arange(n)+1.),
        fits.Column(name="VALUE", format="D", array=values),
    ], name=name)


@pytest.mark.parametrize("ordering", ["covariance_header", "spectra_table"])
def test_declared_covariance_order_overrides_conventional_stat_order(tmp_path, ordering):
    cov = fits.ImageHDU(np.diag([1., 2., 3.]), name="COVMAT")
    hdus = [fits.PrimaryHDU(), cov, _spectrum("xip", [7., 8.]), _spectrum("xim", [9.])]
    if ordering == "covariance_header":
        cov.header["NAME_0"] = "xim"; cov.header["STRT_0"] = 0
        cov.header["NAME_1"] = "xip"; cov.header["STRT_1"] = 1
    else:
        hdus.append(fits.BinTableHDU.from_columns([
            fits.Column(name="NAME", format="16A", array=["xim", "xip"])
        ], name="SPECTRA"))
    p = tmp_path / "vector.fits"; fits.HDUList(hdus).writeto(p)
    data = load_des_y3_2pt_fits(p); blocks = build_y3_block_index(p)
    np.testing.assert_array_equal(data["data_vector"], [9., 7., 8.])
    np.testing.assert_array_equal(data["data_vector"], blocks["value"])
    assert list(blocks["stat"]) == ["xim", "xip", "xip"]


def test_inconsistent_covariance_block_starts_are_rejected(tmp_path):
    cov = fits.ImageHDU(np.eye(3), name="COVMAT")
    cov.header["NAME_0"] = "xip"; cov.header["STRT_0"] = 0
    cov.header["NAME_1"] = "xim"; cov.header["STRT_1"] = 1
    p = tmp_path / "bad.fits"
    fits.HDUList([fits.PrimaryHDU(), cov, _spectrum("xip", [7., 8.]), _spectrum("xim", [9.])]).writeto(p)
    with pytest.raises(ValueError, match="covariance start"):
        load_des_y3_2pt_fits(p)


def _nz_hdu(name, bin_names, values):
    z = np.array([.1, .2, .3])
    cols = [fits.Column(name=key, format="D", array=z+offset)
            for key, offset in [("Z_LOW", -.05), ("Z_MID", 0), ("Z_HIGH", .05)]]
    cols += [fits.Column(name=key, format="D", array=val) for key, val in zip(bin_names, values)]
    return fits.BinTableHDU.from_columns(cols, name=name)


def test_source_kernel_reader_uses_fiducial_source_and_preserves_signed_tails(tmp_path):
    source = _nz_hdu("nz_source", ["BIN5", "BIN2"], [[1., 3., 1.], [-.01, 2., .1]])
    lens = _nz_hdu("nz_lens", ["BIN1"], [[100., 100., 100.]])
    p = tmp_path / "nz.fits"; fits.HDUList([fits.PrimaryHDU(), lens, source]).writeto(p)
    z, kernels, ids = shear._load_nz_from_fits(p)
    np.testing.assert_array_equal(z, [.1, .2, .3])
    assert ids == [2, 5]
    np.testing.assert_array_equal(kernels[0], [-.01, 2., .1])
    np.testing.assert_array_equal(kernels[1], [1., 3., 1.])


def test_finite_shear_transform_matches_analytic_gaussian_and_bin_labels(monkeypatch):
    # Integral ell exp(-ell^2/2) J0(ell*theta)/(2*pi) = exp(-theta^2/2)/(2*pi).
    seen = []
    def angular_cl(cosmo, first, second, ell):
        seen.append((first, second))
        return np.exp(-ell**2/2)
    monkeypatch.setitem(sys.modules, "pyccl", SimpleNamespace(angular_cl=angular_cl))
    block = pd.DataFrame(dict(x=[.5, 30.], x_unit=["rad", "deg"],
                              bin1=[5, 2], bin2=[2, 5], stat=["xip", "xip"]))
    result = shear._build_theory_vector(block, None, ["bin2", "bin5"], np.linspace(1e-6, 10, 30000), tracer_bin_ids=[2, 5])
    expected = np.exp(-np.array([.5, np.pi/6])**2/2)/(2*np.pi)
    np.testing.assert_allclose(result, expected, rtol=2e-8)
    assert set(seen) == {("bin5", "bin2"), ("bin2", "bin5")}
    block.loc[0, "bin1"] = 9
    with pytest.raises(ValueError, match="no corresponding"):
        shear._build_theory_vector(block, None, ["bin2", "bin5"], np.array([1., 2.]), tracer_bin_ids=[2, 5])


def test_unknown_angle_units_and_non_shear_statistics_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="unsupported"):
        shear._theta_to_rad(np.array([1.]), "parsec")
    np.testing.assert_allclose(shear._theta_to_rad(np.array([3600.]), "arcsec"), [np.pi/180])
    monkeypatch.setitem(sys.modules, "pyccl", SimpleNamespace(angular_cl=lambda *args: np.ones(2)))
    block = pd.DataFrame(dict(x=[1.], x_unit=["rad"], bin1=[1], bin2=[1], stat=["gammat"]))
    with pytest.raises(ValueError, match="statistic"):
        shear._build_theory_vector(block, None, ["source"], np.array([1., 2.]), tracer_bin_ids=[1])


@pytest.mark.parametrize("idx", [[0., 1.], [True, False], [0, 0], [-1, 0], [0, 3]])
def test_indices_are_neither_truncated_nor_duplicated(idx):
    with pytest.raises(ValueError): validated_indices(idx, 3)


def test_probe_audit_rejects_overlap_instead_of_scoring_training_points_as_heldout():
    with pytest.raises(ValueError, match="disjoint"):
        ppd_fit_predict(np.zeros(3), np.eye(3), np.array([0, 1]), np.array([1, 2]), "lcdm_stub")


@pytest.mark.parametrize("extra", [
    {"theory_stub": {"frac_sigma": float("inf")}},
    {"mask": {"kind": "indices", "indices": [0, 0]}},
    {"split": {"kind": "custom", "A": [-1], "B": [0]}},
])
def test_invalid_lss_configuration_is_rejected_before_data_access(extra):
    cfg = {"dataset": {"key": "smoke_small", "file": "TEMPLATE.md"}, **extra}
    with pytest.raises(ValueError): resolve_lss_config(cfg)


def test_bao_vector_types_cannot_silently_truncate_and_broadcast():
    dataset = BAODataset("bao", z=np.array([.1, .2]), data=np.array([1., 2.]),
                         obs_type=["DM_over_rd"], cov=np.eye(2))
    with pytest.raises(ValueError, match="observable type"):
        dataset.loglike(np.array([70., .3]))
    dataset.obs_type = ["DM_over_rd", "DM_over_rd"]; dataset.rd_mpc = 0.
    with pytest.raises(ValueError, match="sound horizon"):
        dataset.loglike(np.array([70., .3]))


@pytest.mark.parametrize("dataset", [DESSN5YRDistanceDataset("SN"), DESY6BAOAlphaDataset("alpha")])
def test_likelihood_cannot_ignore_an_extra_fitted_parameter(dataset):
    with pytest.raises(ValueError, match="parameter"):
        dataset.loglike(np.array([.3, .5]))


def test_matter_equation_of_state_is_not_inferred_as_chain_weight():
    assert "weight" not in infer_loglike_columns(["w0", "wa", "loglike"])


def test_multistart_records_the_lower_minimum_without_claiming_global_optimality():
    class TwoWells(Dataset):
        def loglike(self, theta):
            x = theta[0]
            return -((x*x-1)**2 + .1*x)
    fit = fit_map_multistart(TwoWells("two_wells"), [[1.], [-1.]], bounds=[(-2., 2.)], method="Powell")
    assert fit["success"] and fit["theta_hat"][0] < -1
    assert fit["optimization_scope"] == "best_of_declared_starts"
    assert len(fit["starts"]) == 2


def test_background_distance_checks_continuum_instead_of_coarse_endpoints():
    # The endpoints z=0 and z=25 both have E^2>0, but the interior has E^2<0.
    assert np.all(E_lcdm(np.array([0., 25.]), .05, 2.) > 0)
    with pytest.raises(ValueError): comoving_distance(25., 70., .05, 2., n_grid=2)
    assert np.all(E_rewrite(np.array([0., 25.]), .05, 2., 0.) > 0)
    minimum_z, minimum = rewrite_interval_minimum(25., .05, 2., 0.)
    assert minimum_z == pytest.approx(13.) and minimum < 0
    with pytest.raises(ValueError, match="entire"):
        comoving_distance_rewrite(25., 70., .05, 2., 0., n_grid=2)


@pytest.mark.parametrize("m", [0., 1., 3., 6.])
def test_flat_public_rewrite_is_exact_constant_w_family(m):
    z = np.linspace(0, 2, 11)
    np.testing.assert_allclose(E_rewrite(z, .3, .7, m), E_wcdm(z, .3, -1-m/3), rtol=2e-15)


def test_proxy_score_does_not_broadcast_or_extrapolate_missing_observations():
    with pytest.raises(ValueError, match="same shape"):
        rmse(np.ones((2, 1)), np.ones(2))
    with pytest.raises(ValueError, match="cover"):
        interp_at(np.array([0., 1.]), np.array([1., 2.]), np.array([2.]))
    with pytest.raises(ValueError, match="reference amplitude"):
        predict_varH_rewrite_from_template(np.array([.5]), A_fit=1., A_ref=0., z_ref=np.array([0., 1.]), varH_ref=np.ones(2))


def test_toy_simulation_rejects_fractional_population_and_nonfinite_parameters():
    params = dict(alpha=0., beta=1., dt=.1, x0_low=0., x0_high=1.)
    with pytest.raises(ValueError): simulate_micro(.5, 2, params, 1)
    with pytest.raises(ValueError): simulate_micro(2, 2, {**params, "beta": float("nan")}, 1)


def test_consecutive_run_bundles_do_not_overwrite_same_second(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(manifest, "_timestamp_utc", lambda: "same_second")
    monkeypatch.setattr(manifest, "_git_info", lambda _: ("same_commit", False))
    first = manifest.create_run_dir("audit")
    (first / "metrics.json").write_text("original")
    second = manifest.create_run_dir("audit")
    assert first != second
    assert (first / "metrics.json").read_text() == "original"


def test_failed_robustness_scenario_never_leaves_partial_green_rows(tmp_path, monkeypatch):
    cfg = dict(robustness=dict(splits=["shear_vs_rest"], n_boot=1, block_keep_frac=1.,
                               min_points_train=2, min_points_test=2, max_scenarios=1),
               surrogate_theory=dict(frac_sigma=.1, beta_x=.1),
               models=[dict(name="lcdm_like", k=1), dict(name="rewrite_like", k=2)],
               ppd=dict(directions=["AtoB", "BtoA"]))
    resolved = dict(dataset=dict(resolved_path="unused"), run=dict(exp_name="failure_control"), seed=0)
    block = pd.DataFrame(dict(i=np.arange(8), probe=["shear"]*4+["ggl"]*4,
                               stat=["xip"]*4+["gammat"]*4, bin1=[1]*8, bin2=[1]*8, x=np.arange(8)))
    monkeypatch.setattr(sys, "argv", ["robustness"])
    monkeypatch.setattr(robustness, "load_yaml_config", lambda _: cfg)
    monkeypatch.setattr(robustness, "load_and_resolve_lss_config", lambda _: resolved)
    monkeypatch.setattr(robustness, "load_des_y3_2pt_fits", lambda _: dict(data_vector=np.zeros(8), cov=np.eye(8)))
    monkeypatch.setattr(robustness, "build_y3_block_index", lambda _: block)
    monkeypatch.setattr(manifest, "create_run_dir", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(manifest, "_repo_root", lambda: tmp_path)
    original = robustness.fit_linear_correction
    def fail_second_model(y, cov, t0, basis, idx):
        if basis.shape[1] == 2:
            raise ValueError("adversarial second model failure")
        return original(y, cov, t0, basis, idx)
    monkeypatch.setattr(robustness, "fit_linear_correction", fail_second_model)
    robustness.main()
    rows = pd.read_csv(tmp_path / "robustness_runs.csv")
    assert len(rows) == 4 and set(rows["status"]) == {"failed"}
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_ok"] == 0 and metrics["n_failed"] == 1


def test_bao_alpha_reference_is_release_template():
    """alpha is defined relative to the DES Y6 BAO Planck template (Omega_m=0.31, h=0.676)."""
    from sixbirds_cosmo.datasets.des_y6_bao import DES_Y6_BAO_TEMPLATE, DESY6BAOAlphaDataset

    assert DES_Y6_BAO_TEMPLATE == {"om": 0.31, "ol": 0.69, "H0": 67.6}
    ds = DESY6BAOAlphaDataset("alpha", model_kind="flat_lcdm")
    assert ds.alpha_pred(0.31) == pytest.approx(1.0, abs=1e-12)
