"""Mask constructors for LSS block indices."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _mask_by_probe(block_index: pd.DataFrame, probe: str) -> np.ndarray:
    idx = block_index.loc[block_index["probe"] == probe, "i"].to_numpy(dtype=int)
    return np.unique(np.sort(idx))


def mask_cosmic_shear(block_index: pd.DataFrame) -> np.ndarray:
    return _mask_by_probe(block_index, "shear")


def mask_clustering(block_index: pd.DataFrame) -> np.ndarray:
    return _mask_by_probe(block_index, "clustering")


def mask_ggl(block_index: pd.DataFrame) -> np.ndarray:
    return _mask_by_probe(block_index, "ggl")


def mask_scale_cut(block_index: pd.DataFrame, *, rules: Dict[Tuple[str, str], Dict[str, float]]) -> np.ndarray:
    keep = np.ones(len(block_index), dtype=bool)
    for (probe, stat), rule in rules.items():
        x_min = rule.get("x_min", -np.inf)
        x_max = rule.get("x_max", np.inf)
        mask_rows = (block_index["probe"] == probe) & (block_index["stat"] == stat)
        if not mask_rows.any():
            continue
        x = block_index.loc[mask_rows, "x"]
        x_ok = x.notna() & (x >= x_min) & (x <= x_max)
        keep[mask_rows.to_numpy()] = x_ok.to_numpy()
    idx = block_index.loc[keep, "i"].to_numpy(dtype=int)
    return np.unique(np.sort(idx))


def mask_union(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        return np.array([], dtype=int)
    return np.unique(np.concatenate([np.asarray(m, dtype=int) for m in masks]))


def mask_intersection(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        return np.array([], dtype=int)
    inter = np.asarray(masks[0], dtype=int)
    for m in masks[1:]:
        inter = np.intersect1d(inter, np.asarray(m, dtype=int))
    return inter
