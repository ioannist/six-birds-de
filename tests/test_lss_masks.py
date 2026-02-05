import numpy as np
import pandas as pd

from sixbirds_cosmo.lss.masks import (
    mask_clustering,
    mask_cosmic_shear,
    mask_ggl,
    mask_intersection,
    mask_scale_cut,
    mask_union,
)


def make_block_index():
    return pd.DataFrame(
        {
            "i": [0, 1, 2, 3, 4, 5],
            "probe": ["shear", "shear", "clustering", "ggl", "ggl", "unknown"],
            "stat": ["xip", "xim", "wtheta", "gammat", "gammat", "unknown"],
            "x": [10.0, 300.0, 20.0, 4.0, 50.0, np.nan],
        }
    )


def test_probe_masks():
    bi = make_block_index()
    assert np.array_equal(mask_cosmic_shear(bi), np.array([0, 1]))
    assert np.array_equal(mask_clustering(bi), np.array([2]))
    assert np.array_equal(mask_ggl(bi), np.array([3, 4]))


def test_scale_cut_rules():
    bi = make_block_index()
    rules = {
        ("shear", "xip"): {"x_min": 5.0, "x_max": 50.0},
        ("shear", "xim"): {"x_min": 5.0, "x_max": 50.0},
        ("ggl", "gammat"): {"x_min": 5.0, "x_max": 100.0},
    }
    mask = mask_scale_cut(bi, rules=rules)
    assert np.array_equal(mask, np.array([0, 2, 4, 5]))


def test_union_intersection():
    a = np.array([0, 1, 2])
    b = np.array([2, 3])
    assert np.array_equal(mask_union(a, b), np.array([0, 1, 2, 3]))
    assert np.array_equal(mask_intersection(a, b), np.array([2]))
