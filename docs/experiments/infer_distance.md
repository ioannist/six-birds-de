# Experiment Log
## Objective
Review run outputs for infer_distance and summarize core signals.
## Model & lens/completion definition
See experiment script and run configuration for model details; lens/completion are as defined in the corresponding toy or inference script.
## Parameters & seeds
Run directory: `/home/repos/six-birds-de/results/infer_distance/20260202T190430Z_nogit`
## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| lcdm_omega_lambda | 0.6015003326974425 |
| delta_aic | 8217.98409867457 |
| delta_chi2 | 8219.98409867457 |
| matter_only_H0 | 0.034859962571874833 |
| lcdm_H0 | 0.04230387990606948 |
Plots:
- [distance_fit.png](../../results/infer_distance/20260202T190430Z_nogit/distance_fit.png)
## Key observations
- ΛCDM fits the synthetic distance–redshift data vastly better than matter-only (ΔAIC strongly favors ΛCDM).
- Best-fit ΩΛ emerges > 0 even though the toy generator contains no fundamental Λ term (an “inference illusion” from coarse-grained dynamics).
- The fitted H0 here is a scale parameter in toy units; absolute magnitude is not physically meaningful without calibration.
- The t→z mapping is normalized so the final time corresponds to z=0, ensuring consistent distance integration.
## Failure modes / limitations
- If z(t) is non-monotone in pathological runs, interpolation/integration can become ill-defined.
- Noise model is simplistic (Gaussian, diagonal).
