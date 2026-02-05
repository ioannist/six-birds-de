# Experiment Log
## Objective
Sweep minimum scale cuts (x_min) and measure how held‑out PPD performance changes with staging for shear↔(clustering+ggl).

## Model & lens/completion definition
Same surrogate LSS setup as y3_ppd_split; scale cuts applied uniformly to xip/xim/wtheta/gammat before splitting.

## Parameters & seeds
Run folder: results/y3_scale_cut_sweep/20260204T073454Z_nogit

## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| x_min_values_used | [0.0, 2.8277, 3.5240, 4.3914, 6.8164, 10.5744, 16.3962, 25.4100] |
| xmin_min_chi2_AtoB_lcdm_like | 25.409992709359 |
| xmin_min_chi2_AtoB_rewrite_like | 25.409992709359 |
| xmin_min_chi2_BtoA_lcdm_like | 25.409992709359 |
| xmin_min_chi2_BtoA_rewrite_like | 25.409992709359 |
| nonflat_check_passed | true |

Plots:
- [chi2_test_vs_xmin_AtoB.png](../../results/y3_scale_cut_sweep/20260204T073454Z_nogit/chi2_test_vs_xmin_AtoB.png)
- [chi2_test_vs_xmin_BtoA.png](../../results/y3_scale_cut_sweep/20260204T073454Z_nogit/chi2_test_vs_xmin_BtoA.png)
- [params_vs_xmin.png](../../results/y3_scale_cut_sweep/20260204T073454Z_nogit/params_vs_xmin.png)

## Key observations
- Held‑out χ² varies with x_min (non‑flat), indicating staging dependence.
- Best x_min is the same for both models in this run.
- This provides the “staging fingerprint” on public Y3 2pt data.

## Failure modes / limitations
- Surrogate mode only; absolute χ² values are not physically meaningful.
- Scale‑cut rules are uniform across probes and may be too coarse for real analyses.
