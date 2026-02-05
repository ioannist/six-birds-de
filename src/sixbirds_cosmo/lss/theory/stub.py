"""Stub theory backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np
import pandas as pd


@dataclass
class StubBackend:
    """Stub backend: t = y_obs + frac_sigma * sigma."""

    y_obs: np.ndarray
    cov: np.ndarray
    frac_sigma: float = 0.1
    name: str = "stub"
    supports_probes: Set[str] | None = None
    supports_w0wa: bool = False

    def predict(self, theta: np.ndarray, block_index: pd.DataFrame) -> np.ndarray:
        sigma = np.sqrt(np.diag(self.cov))
        t = np.asarray(self.y_obs, dtype=float) + self.frac_sigma * sigma
        if len(t) != len(block_index):
            raise ValueError("StubBackend prediction length does not match block_index.")
        return t
