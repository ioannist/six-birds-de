# Experiment Log
## Objective
Re-run the Y3 probe split PPD with geometry anchors (SN+BAO) and compare held‑out performance with and without anchors.

## Model & lens/completion definition
Surrogate LSS objective with cosmology‑dependent scaling; geometry anchors use DES SN5YR distances (profiled intercept) and DES Y6 BAO alpha likelihood.

## Parameters & seeds
Run folder: results/y3_ppd_with_geometry/20260204T080238Z_nogit

## Outputs (metrics + plots)
| Metric | Value |
|---|---|
| n_A | 400 |
| n_B | 600 |
| chi2_B_lcdm_noanchors | 0.6749610323126477 |
| chi2_B_lcdm_anchors | 1.0797680999142443 |
| chi2_B_rewrite_noanchors | 0.996693926532935 |
| chi2_B_rewrite_anchors | 1.082006080932547 |
| chi2_A_lcdm_noanchors | 0.718572456107841 |
| chi2_A_lcdm_anchors | 1.2329320485895274 |
| chi2_A_rewrite_noanchors | 0.4984658184372126 |
| chi2_A_rewrite_anchors | 1.2458174627084326 |

Plots:
- [heldout_chi2_AtoB.png](../../results/y3_ppd_with_geometry/20260204T080238Z_nogit/heldout_chi2_AtoB.png)
- [heldout_chi2_BtoA.png](../../results/y3_ppd_with_geometry/20260204T080238Z_nogit/heldout_chi2_BtoA.png)

## Key observations
- Geometry anchors shift held‑out χ² upward in this surrogate mapping.
- Rewrite vs lcdm ranking changes by direction in the no‑anchors case.
- Anchors are real (SN+BAO); LSS portion remains surrogate.

## Failure modes / limitations
- Without a physical 3×2pt backend, results are methodological rather than physical.
- Anchor influence is sensitive to the surrogate scaling design.
