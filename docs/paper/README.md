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

## Vendor paper artifacts (tracked figures/tables)

Figures and quantitative tables used by the manuscript are tracked under:

* `docs/paper/figures/`
* `docs/paper/tables/`

Vendoring scripts copy from `results/**` into tracked paths and write provenance:

```bash
python scripts/vendor_figures.py
python scripts/vendor_tables.py
```

Provenance files:

* `docs/paper/figures/provenance.yaml`
* `docs/paper/tables/provenance.yaml`

## Create arXiv bundle zip

```bash
bash scripts/make_arxiv_bundle.sh
```

This writes a self-contained zip under `docs/paper/build/` that should compile on arXiv without referencing the repository layout.
