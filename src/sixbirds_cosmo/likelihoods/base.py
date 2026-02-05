"""Base classes for likelihood datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def as_theta_dict(theta: np.ndarray, param_names: List[str] | None) -> Dict[str, float]:
    """Convert a parameter vector into a dict with optional names."""
    values = np.asarray(theta, dtype=float).tolist()
    if not param_names:
        return {f"theta_{i}": float(val) for i, val in enumerate(values)}
    return {name: float(val) for name, val in zip(param_names, values)}


@dataclass
class Dataset:
    """Base dataset interface for likelihood evaluation."""

    name: str
    n_data: int | None = None
    param_names: List[str] | None = None
    meta: Dict[str, object] = field(default_factory=dict)
    loglike_is_minus_half_chi2: bool = True

    def loglike(self, theta: np.ndarray) -> float:
        """Return log-likelihood at parameters theta."""
        raise NotImplementedError("loglike must be implemented by subclasses.")
