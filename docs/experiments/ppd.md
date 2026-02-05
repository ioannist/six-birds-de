# Experiment Log
## Objective
Review run outputs for ppd and summarize core signals.
## Model & lens/completion definition
See experiment script and run configuration for model details; lens/completion are as defined in the corresponding toy or inference script.
## Parameters & seeds
Run directory: `/home/repos/six-birds-de/results/ppd_synthetic/20260202T201136Z_nogit`
## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| hetero_rmse_ratio_median | 0.008774173708296642 |
| hetero_bias_lcdm_mean | -4.1882841185905216e-05 |
| hetero_bias_rewrite_mean | -3.272188755465493e-07 |
| homo_rmse_lcdm | 2.4851412774416836e-18 |
| homo_rmse_rewrite | 2.4851412774416836e-18 |
Plots:
- [ppd_varH_hetero_seed123.png](../../results/ppd_synthetic/20260202T201136Z_nogit/ppd_varH_hetero_seed123.png)
- [ppd_rmse_across_seeds.png](../../results/ppd_synthetic/20260202T201136Z_nogit/ppd_rmse_across_seeds.png)
## Key observations
- Under heterogeneity, ΛCDM predicts var_H ≈ 0 by construction, producing systematic underprediction (bias) on Probe 2.
- The rewrite-template predictor substantially reduces RMSE on var_H across multiple hetero seeds (median RMSE ratio ≪ 1).
- In the homogeneous case, both predictors yield ~0 variance and RMSE is near numerical precision.
- Multi-seed evaluation reduces the chance of seed cherry-picking.
## Failure modes / limitations
- Template-based prediction depends on the chosen calibration run; stronger robustness would use cross-validation or multi-template aggregation.
- var_H is a synthetic internal proxy, not a direct observable.
