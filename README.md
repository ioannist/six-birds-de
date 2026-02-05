# A Six Birds' Eye View of Dark Energy

This repository contains the **cosmology instantiation** for the paper:

> **A Six Birds' Eye View of Dark Energy: Closure, Route Mismatch, and Audits for Apparent Acceleration**
>
> Archived at: https://zenodo.org/records/18494480
>
> DOI: https://doi.org/10.5281/zenodo.18494480

This paper applies Six Birds Theory (SBT) to cosmology by treating standard inference as a closure package (lens + completion + audits). It demonstrates mechanism and audit signatures in toy models and synthetic data, and builds reproducible pipelines for public background probes and public large-scale-structure vectors.

## What this repository provides

- **Toy mechanism demonstrations**: route mismatch under coarse-graining; patch-expansion proxy with an effective acceleration diagnostic.
- **Synthetic inference tests**: null-\Lambda heterogeneous generator fit by homogeneous macro models, plus a rewrite family that matches \LambdaCDM on the synthetic distances.
- **Public background probes**: DES SN5YR distance-modulus likelihood with correct covariance handling and DES Y6 BAO \alpha-likelihood ingestion.
- **LSS data-layer and audit protocol**: DES Y3 2pt block map, probe splits, staging sweeps, robustness suite, and KiDS-450 replication (surrogate backend).
- **Evidence artifacts**: manifest run bundles, vendored figures/tables, provenance maps, and a single-command evidence suite runner.

## Scope and limitations

- This paper is **not** a full physical 3x2pt likelihood reproduction; LSS audits use a surrogate backend and are presented as protocol demonstrations.
- Background-probe fits are **compressed likelihood** uses (SN distance vector, BAO \alpha-curve) with explicit assumptions.
- The rewrite term is **phenomenological** on public data; the synthetic experiments demonstrate generator-tied proxy rewrites.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

## Run public-data evidence suites

```bash
make exp-public-evidence-background
make exp-public-evidence-lss
```

Aggregate metrics across runs:

```bash
make collect-metrics
```

## Build the paper

```bash
make paper
```

Output:

- `docs/paper/build/sixbirds_dark_energy.pdf`

## Vendor paper artifacts (tracked figures/tables)

```bash
python scripts/vendor_figures.py
python scripts/vendor_tables.py
```

These scripts copy from `results/**` into tracked paths under:

- `docs/paper/figures/`
- `docs/paper/tables/`

and write provenance YAML files alongside the artifacts.
