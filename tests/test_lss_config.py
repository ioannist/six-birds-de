from pathlib import Path

import pytest

from sixbirds_cosmo.lss.config import load_and_resolve_lss_config, resolve_lss_config


def test_resolve_defaults_applied():
    cfg = {
        "dataset": {"key": "smoke_small", "file": "TEMPLATE.md", "kind": "generic"},
    }
    resolved = resolve_lss_config(cfg)
    assert resolved["schema_version"] == 0
    assert resolved["seed"] == 0
    assert resolved["mask"]["kind"] == "all"
    assert resolved["split"]["kind"] == "none"
    assert resolved["theory_stub"]["frac_sigma"] == 0.1
    assert resolved["run"]["exp_name"] == "lss_config_smoke"


def test_invalid_mask_kind_raises():
    cfg = {
        "dataset": {"key": "smoke_small", "file": "TEMPLATE.md"},
        "mask": {"kind": "bad"},
    }
    with pytest.raises(ValueError):
        resolve_lss_config(cfg)


def test_custom_split_requires_A_B_disjoint():
    cfg = {
        "dataset": {"key": "smoke_small", "file": "TEMPLATE.md"},
        "split": {"kind": "custom", "A": [0, 1], "B": [1, 2]},
    }
    with pytest.raises(ValueError):
        resolve_lss_config(cfg)


def test_dataset_path_resolves_smoke_small():
    cfg_path = Path("configs/lss/smoke_small.yaml")
    resolved = load_and_resolve_lss_config(cfg_path)
    resolved_path = Path(resolved["dataset"]["resolved_path"])
    assert resolved_path.exists()
