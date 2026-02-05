import numpy as np
import pandas as pd

from sixbirds_cosmo.lss.ppd_probe_split import (
    build_basis,
    build_surrogate_theory,
    eval_metrics,
    fit_linear_correction,
    get_probe_split_indices,
)


def test_get_probe_split_indices():
    df = pd.DataFrame(
        {
            "i": [0, 1, 2, 3],
            "probe": ["shear", "clustering", "ggl", "shear"],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    idx_shear, idx_other = get_probe_split_indices(df)
    assert np.array_equal(idx_shear, np.array([0, 3]))
    assert np.array_equal(idx_other, np.array([1, 2]))


def test_fit_linear_correction_reduces_chi2():
    rng = np.random.default_rng(0)
    n = 20
    y = rng.normal(size=n)
    cov = np.eye(n)
    block = pd.DataFrame({"i": np.arange(n), "probe": ["shear"] * n, "x": rng.normal(size=n)})
    t0 = build_surrogate_theory(y, cov, block, frac_sigma=0.1, beta_x=0.2, kappa_probe={"shear": 0.0})
    B = build_basis(y, cov, block, model="rewrite_like")
    idx = np.arange(n // 2)
    p_hat, chi2_train = fit_linear_correction(y, cov, t0, B, idx)
    assert np.all(np.isfinite(p_hat))
    # baseline chi2 with p=0
    resid0 = (y - t0)[idx]
    chi2_0 = float(resid0 @ resid0)
    assert chi2_train <= chi2_0


def test_eval_metrics_finite():
    rng = np.random.default_rng(1)
    n = 10
    y = rng.normal(size=n)
    cov = np.eye(n)
    t = y + 0.1
    metrics = eval_metrics(y, cov, t, np.arange(n))
    assert np.isfinite(metrics["chi2"])
    assert np.isfinite(metrics["rmse"])
