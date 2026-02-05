# Experiment Log
## Objective
Review run outputs for rewrite_vs_lambda and summarize core signals.
## Model & lens/completion definition
See experiment script and run configuration for model details; lens/completion are as defined in the corresponding toy or inference script.
## Parameters & seeds
Run directory: `/home/repos/six-birds-de/results/rewrite_vs_lambda/20260202T194715Z_nogit`
## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| chi2_lcdm | 379.6089507955338 |
| chi2_rewrite | 378.59252843705406 |
| aic_lcdm | 383.6089507955338 |
| aic_rewrite | 382.59252843705406 |
| rewrite_A | 1509.9065707193965 |
| spearman_rho | 1.0 |
| A_monotone_increasing | True |
Plots:
- [rewrite_distance_fit.png](../../results/rewrite_vs_lambda/20260202T194715Z_nogit/rewrite_distance_fit.png)
- [A_vs_delta_rho.png](../../results/rewrite_vs_lambda/20260202T194715Z_nogit/A_vs_delta_rho.png)
## Key observations
- The rewrite model achieves χ² comparable to ΛCDM with the same parameter count (k=2), using a mismatch-derived proxy shape.
- Learned rewrite amplitude A increases strongly with heterogeneity amplitude (delta sweep is monotone; Spearman correlation ~1 in this run).
- The rewrite term uses a normalized proxy shape (from var_H), so A’s absolute magnitude is scale-dependent and requires calibration for interpretation.
- Seed sensitivity was observed during development: some seeds can violate “rewrite comparable to ΛCDM” unless proxy/penalty settings are tuned.
## Failure modes / limitations
- Proxy normalization can produce very large/small A; compare across runs using proxy scale metadata.
- Clipping/penalties for negative H² are pragmatic; they can mask model invalidity if abused.
