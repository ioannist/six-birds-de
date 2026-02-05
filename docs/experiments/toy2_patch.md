# Experiment Log
## Objective
Review run outputs for toy2_patch and summarize core signals.
## Model & lens/completion definition
See experiment script and run configuration for model details; lens/completion are as defined in the corresponding toy or inference script.
## Parameters & seeds
Run directory: `/home/repos/six-birds-de/results/toy2_patch/20260202T182630Z_nogit`
## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| homogeneous_Q_max_abs | 3.1086244689504383e-15 |
| hetero_Q_peak | 1.7420340032028825 |
| hetero_Q_mean_late | 3.2553925131061715e-05 |
| hetero_frac_ddot_aD_pos_late | 0.9875518672199171 |
Plots:
- [aD_vs_time.png](../../results/toy2_patch/20260202T182630Z_nogit/aD_vs_time.png)
- [ddot_aD_vs_time.png](../../results/toy2_patch/20260202T182630Z_nogit/ddot_aD_vs_time.png)
- [Q_vs_time.png](../../results/toy2_patch/20260202T182630Z_nogit/Q_vs_time.png)
## Key observations
- Homogeneous initial conditions yield Q(t) ≈ 0 (numerical noise only).
- Strong heterogeneity produces positive late-time Q(t) and largely positive late-time (\ddot a_D), i.e., a backreaction-like acceleration proxy at the domain level.
- The sign/magnitude of Q(t) is sensitive to heterogeneity amplitude and the void/wall split (volume weighting dominates late times).
- The analytic (\ddot a_D/a_D) computation via V, (\dot V), (\ddot V) avoids finite-difference noise.
## Failure modes / limitations
- RK4 can fail if any `a_i` approaches 0; clipping/guards are required.
- This is not full GR; Q is a closure proxy meant to illustrate noncommutation/backreaction, not a direct observable.
