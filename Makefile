.PHONY: install test lint paper exp-toy1 exp-toy2 exp-all exp-smoke-manifest exp-infer-distance exp-rewrite-vs-lambda exp-ppd-synthetic exp-scale-sweep exp-smoke-fit exp-fit-des-y6-bao exp-fit-des-sn5yr exp-sn5yr-cov-sanity exp-ppd-background exp-rewrite-background-vs-lcdm exp-des-y3-vector-sanity exp-3x2pt-smoke-like exp-rewrite-effective-w-validate exp-3x2pt-ppd-split exp-des-y6-3x2pt-smoke check-y6-3x2pt-availability exp-lss-config-smoke exp-des-y3-blockmap exp-des-y3-chain-sanity exp-lss-backend-probe exp-y3-shear-physical-smoke exp-y3-ppd-probe-split exp-y3-scale-cut-sweep exp-y3-ppd-with-geometry exp-kids450-ppd-probe-split exp-y3-ppd-robustness exp-public-evidence-background exp-public-evidence-lss fetch-public-evidence-background fetch-public-evidence-lss collect-metrics data-fetch-smoke-small data-fetch-smoke-small-force

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	python -m compileall src tests

paper:
	bash scripts/build_paper.sh

exp-toy1:
	python experiments/run_toy1.py

exp-toy2:
	python experiments/run_toy2_patch.py

exp-all: exp-toy1 exp-toy2

exp-smoke-manifest:
	python experiments/_smoke_manifest.py

exp-infer-distance:
	python experiments/run_infer_distance.py

exp-rewrite-vs-lambda:
	python experiments/run_rewrite_vs_lambda.py

exp-ppd-synthetic:
	python experiments/run_ppd_synthetic.py

exp-scale-sweep:
	python experiments/run_scale_sweep.py

exp-smoke-fit:
	python experiments/_smoke_fit.py

exp-fit-des-y6-bao:
	python experiments/run_fit_des_y6_bao.py

exp-fit-des-sn5yr:
	python experiments/run_fit_des_sn5yr.py

exp-sn5yr-cov-sanity:
	python experiments/run_des_sn5yr_cov_sanity.py

exp-ppd-background:
	python experiments/run_ppd_background.py

exp-rewrite-background-vs-lcdm:
	python experiments/run_rewrite_background_vs_lcdm.py

exp-des-y3-vector-sanity:
	python experiments/run_des_y3_vector_sanity.py

exp-3x2pt-smoke-like:
	python experiments/run_3x2pt_smoke_like.py

exp-rewrite-effective-w-validate:
	python experiments/run_rewrite_effective_w_validate.py

exp-3x2pt-ppd-split:
	python experiments/run_3x2pt_ppd_split.py

exp-des-y6-3x2pt-smoke:
	python experiments/run_des_y6_3x2pt_smoke.py

exp-lss-config-smoke:
	python experiments/_smoke_lss_config.py

exp-des-y3-blockmap:
	python experiments/run_des_y3_blockmap.py

exp-des-y3-chain-sanity:
	python experiments/run_des_y3_chain_sanity.py

exp-lss-backend-probe:
	python experiments/run_lss_backend_probe.py

exp-y3-shear-physical-smoke:
	python experiments/run_y3_shear_physical_smoke.py --config configs/lss/y3_shear_physical_smoke.yaml

exp-y3-ppd-probe-split:
	python experiments/run_y3_ppd_probe_split.py --config configs/lss/y3_ppd_probe_split.yaml

exp-y3-scale-cut-sweep:
	python experiments/run_y3_scale_cut_sweep.py --config configs/lss/y3_scale_cut_sweep.yaml

exp-y3-ppd-with-geometry:
	python experiments/run_y3_ppd_with_geometry.py --config configs/lss/y3_ppd_with_geometry.yaml

exp-kids450-ppd-probe-split:
	python experiments/run_kids450_ppd_probe_split.py --config configs/lss/kids450_ppd_probe_split.yaml

exp-y3-ppd-robustness:
	python experiments/run_y3_ppd_robustness.py --config configs/lss/y3_ppd_robustness.yaml

fetch-public-evidence-background:
	python scripts/fetch_data.py --dataset des_sn5yr_distances
	python scripts/fetch_data.py --dataset des_y6_bao_release

fetch-public-evidence-lss:
	python scripts/fetch_data.py --dataset des_y3a2_datavectors_2pt

collect-metrics:
	python scripts/collect_metrics.py

exp-public-evidence-background: fetch-public-evidence-background exp-fit-des-y6-bao exp-fit-des-sn5yr exp-ppd-background exp-rewrite-background-vs-lcdm exp-rewrite-effective-w-validate collect-metrics

exp-public-evidence-lss: fetch-public-evidence-lss exp-des-y3-blockmap exp-y3-ppd-probe-split exp-y3-scale-cut-sweep exp-y3-ppd-with-geometry exp-y3-ppd-robustness collect-metrics

check-y6-3x2pt-availability:
	python scripts/check_des_y6_3x2pt_availability.py

data-fetch-smoke-small:
	python scripts/fetch_data.py --dataset smoke_small

data-fetch-smoke-small-force:
	python scripts/fetch_data.py --dataset smoke_small --force
