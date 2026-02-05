# Experiment Log
## Objective
Probe whether a physical LSS backend (pyccl) is available and record success/failure without blocking progress.

## Model & lens/completion definition
Backend probe only; no lens/completion or LSS theory evaluation if backend unavailable.

## Parameters & seeds
Run folder: results/lss_backend_probe/20260204T061434Z_nogit

## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| backend_attempted | pyccl |
| backend_available | false |
| success_stage | none |
| note | No module named 'pyccl' |

Plots:
- (none; backend unavailable)

## Key observations
- Baseline environment does not have pyccl installed.
- Probe is non-blocking and records availability explicitly.
- Enables clean switch to physical mode when extras are installed.

## Failure modes / limitations
- Physical backend requires: `python -m pip install -e '.[dev,lss]'` then rerun.
- Without pyccl, no physical P(k) or Cℓ can be computed.
