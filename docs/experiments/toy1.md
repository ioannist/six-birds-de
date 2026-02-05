# Experiment Log
## Objective
Review run outputs for toy1 and summarize core signals.
## Model & lens/completion definition
See experiment script and run configuration for model details; lens/completion are as defined in the corresponding toy or inference script.
## Parameters & seeds
Run directory: `/home/repos/six-birds-de/results/toy1/20260202T175625Z_nogit`
## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| beta_linear_case_rm_max | 0.0 |
| beta_small_rm_max | 0.020259883177980997 |
| beta_large_rm_max | 0.08103953271192499 |
| delta_max_overall | 7.021666937153402e-15 |
Plots:
- [rm_vs_time_by_n.png](../../results/toy1/20260202T175625Z_nogit/rm_vs_time_by_n.png)
- [rm_summary_vs_beta.png](../../results/toy1/20260202T175625Z_nogit/rm_summary_vs_beta.png)
## Key observations
- RM(t) is exactly ~0 in the linear case (β=0), consistent with commutation under this lens.
- RM increases with β; mismatch is driven by the nonlinearity and the variance collapsed by completion.
- Idempotence defect δ is near machine precision because packaging is a projection (mean + constant completion).
- RM(t) tracks the heterogeneity that the mean lens discards (variance term).
## Failure modes / limitations
- If `dt` or `beta` is too large, the quadratic update can become unstable or overflow.
- RM is measured in macro-space here; micro-level discrepancies may be larger than reported.
