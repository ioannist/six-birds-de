# Experiment Log
## Objective
Build a deterministic block map for DES Y3 maglim 2pt data so each element is labeled by probe/stat/bin/x, enabling meaningful probe splits and scale cuts.

## Model & lens/completion definition
Data-vector bookkeeping only; no lens/completion or theory model applied.

## Parameters & seeds
Run folder: results/des_y3_blockmap/20260204T045551Z_nogit

## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| n_data | 1000 |
| n_shear | 400 |
| n_clust | 120 |
| n_ggl | 480 |
| n_unknown | 0 |
| unique_stats | ["gammat", "wtheta", "xim", "xip"] |

Plots:
- [block_boundaries.png](../../results/des_y3_blockmap/20260204T045551Z_nogit/block_boundaries.png)

## Key observations
- Probe counts match the expected Y3 2pt decomposition with zero unknown blocks.
- Block ordering consistency checks passed (block_index matches data vector ordering).
- This block map enables clean shear vs (clustering+ggl) splits and scale-cut staging.

## Failure modes / limitations
- If FITS HDU naming changes, block extraction could fail or reorder rows.
- This is purely structural; no physical predictions are made here.
