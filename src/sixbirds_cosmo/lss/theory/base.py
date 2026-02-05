"""Theory backend interface."""

from __future__ import annotations

from typing import Protocol, Set

import numpy as np
import pandas as pd


class TheoryBackend(Protocol):
    """Protocol for theory backends."""

    name: str
    supports_probes: Set[str] | None
    supports_w0wa: bool

    def predict(self, theta: np.ndarray, block_index: pd.DataFrame) -> np.ndarray:
        """Return theory vector aligned to block_index rows."""
        ...
