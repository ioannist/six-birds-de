# Experiment Log
## Objective
Run a probe-split PPD check on DES Y3 maglim (shear vs clustering+ggl) in surrogate mode, comparing lcdm_like vs rewrite_like held-out performance.

## Model & lens/completion definition
Surrogate LSS theory: deterministic r0 injection scaled by probe/x structure; two linear correction models (k=1 lcdm_like, k=2 rewrite_like). No physical 3×2pt backend.

## Parameters & seeds
Run folder: results/y3_ppd_probe_split/20260204T070918Z_nogit

## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| n_data | 1000 |
| n_A | 400 |
| n_B | 600 |
| A→B lcdm_like chi2_test | 0.1164976921453756 |
| A→B rewrite_like chi2_test | 0.08851930993536573 |
| B→A lcdm_like chi2_test | 0.06737338520662844 |
| B→A rewrite_like chi2_test | 0.030537522884920083 |
| A→B lcdm_like rmse_test | 1.9189478053160963e-05 |

Plots:
- [ppd_residuals_testB.png](../../results/y3_ppd_probe_split/20260204T070918Z_nogit/ppd_residuals_testB.png)
- [ppd_residuals_testA.png](../../results/y3_ppd_probe_split/20260204T070918Z_nogit/ppd_residuals_testA.png)

## Key observations
- Surrogate mode only; results validate the split/fit/predict plumbing.
- Rewrite_like improves held-out χ² in both directions for this run.
- χ² magnitudes are small by construction of the surrogate residual injection.

## Failure modes / limitations
- Not a physical 3×2pt prediction; depends on surrogate structure.
- Backend availability (pyccl) is still required for a physical validation.
