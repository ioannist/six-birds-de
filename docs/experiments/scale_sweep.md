# Experiment Log
## Objective
Review run outputs for scale_sweep and summarize core signals.
## Model & lens/completion definition
See experiment script and run configuration for model details; lens/completion are as defined in the corresponding toy or inference script.
## Parameters & seeds
Run directory: `/home/repos/six-birds-de/results/scale_sweep/20260202T204839Z_nogit`
## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| L_min_delta | 1 |
| RM_at_L_min_delta | 0.0 |
| rm_L1 | 0.0 |
| rm_Lmax | 0.00012971642967604183 |
Plots:
- [rm_mean_late_vs_L.png](../../results/scale_sweep/20260202T204839Z_nogit/rm_mean_late_vs_L.png)
- [delta_max_vs_L.png](../../results/scale_sweep/20260202T204839Z_nogit/delta_max_vs_L.png)
## Key observations
- RM(L) is non-flat and increases with coarser grouping (L), demonstrating staging dependence of mismatch under lens scale.
- Idempotence defect δ(L) remains near numerical noise across L because the chosen completion is a projection/replication operator.
- Sorting by initial density and grouping provides a smooth one-parameter family of lenses (f_L).
- The sweep supports the “bounded interface” intuition: coarser packaging discards more heterogeneity and induces larger mismatch.
## Failure modes / limitations
- With deterministic projection completion, δ(L) is not very informative; alternative completions could yield a richer coherence diagnostic.
- Only L values dividing N are supported in this grouping scheme.
