# A Six Birds' Eye View of Dark Energy — paper build

## Build the PDF

From the repo root:

```bash
make paper
```

This produces:

* `docs/paper/build/sixbirds_dark_energy.pdf`

## Reproduce evidence runs (public-data suites)

Assuming datasets are already fetched (or will be cache hits):

```bash
make exp-public-evidence-background
make exp-public-evidence-lss
```

Aggregate metrics across all runs:

```bash
make collect-metrics
```

Outputs:

* `results/_aggregate/metrics_table.csv`

## Build paper artifacts (tracked figures/tables)

Figures and quantitative tables used by the manuscript are tracked under:

* `docs/paper/figures/` (vector PDFs)
* `docs/paper/tables/`

Both are generated from the audited run bundles in `results/**` (clean runs at
commit `bfa9bde`; the exact folders are listed at the top of each script):

```bash
PYTHONPATH=src python scripts/make_paper_figures.py   # needs matplotlib
PYTHONPATH=src python scripts/vendor_tables.py
```

The figure script recomputes any plotted series not stored in a run bundle from
that bundle's configuration and stops if the result does not reproduce the
recorded metrics.

Provenance files:

* `docs/paper/figures/provenance_composed.yaml`
* `docs/paper/tables/provenance.yaml`

## Create arXiv bundle zip

```bash
bash scripts/make_arxiv_bundle.sh
```

This writes a self-contained zip under `docs/paper/build/` that should compile on arXiv without referencing the repository layout.
