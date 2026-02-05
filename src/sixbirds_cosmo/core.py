"""Core abstractions for lensing, completion, and packaging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import numbers

import numpy as np

X = TypeVar("X")
Y = TypeVar("Y")


def default_distance(a: object, b: object) -> float:
    """Return a distance between numbers or array-like inputs.

    Supported types:
    - Numbers (int/float): absolute difference
    - NumPy arrays or list/tuple of numbers: L2 norm of the difference
    """
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return float(abs(a - b))

    if _is_array_like(a) and _is_array_like(b):
        try:
            diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                "default_distance supports numbers and array-like inputs"
            ) from exc
        return float(np.linalg.norm(diff))

    raise TypeError("default_distance supports numbers and array-like inputs")


def _is_array_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, list, tuple))


@dataclass(frozen=True)
class Lens(Generic[X, Y]):
    """Lens mapping micro-state X to macro-state Y."""

    f: Callable[[X], Y]
    distance_y: Callable[[Y, Y], float] | None = None

    def __call__(self, x: X) -> Y:
        return self.f(x)

    def dist_y(self, y1: Y, y2: Y) -> float:
        if self.distance_y is not None:
            return self.distance_y(y1, y2)
        return default_distance(y1, y2)


@dataclass(frozen=True)
class Completion(Generic[X, Y]):
    """Deterministic completion operator U mapping Y to X."""

    f: Callable[[Y], X]

    def __call__(self, y: Y) -> X:
        return self.f(y)


@dataclass(frozen=True)
class PackagingOperator(Generic[X, Y]):
    """Packaging operator E(x) = U(f(x))."""

    lens: Lens[X, Y]
    completion: Completion[X, Y]
    distance_x: Callable[[X, X], float] | None = None

    def __call__(self, x: X) -> X:
        return self.completion(self.lens(x))

    def macro(self, x: X) -> Y:
        return self.lens(x)

    def dist_x(self, x1: X, x2: X) -> float:
        if self.distance_x is not None:
            return self.distance_x(x1, x2)
        return default_distance(x1, x2)


def E_tau(x: X, T_tau: Callable[[X], X], E: PackagingOperator[X, Y]) -> X:
    """Apply transport T_tau followed by packaging E."""
    return E(T_tau(x))


def idempotence_defect(
    E: PackagingOperator[X, Y], x: X, *, space: str = "auto"
) -> float:
    """Compute the distance between E(E(x)) and E(x)."""
    space = _resolve_space(space, E)
    ex = E(x)
    eex = E(ex)
    if space == "x":
        return E.dist_x(eex, ex)
    return E.lens.dist_y(E.lens(eex), E.lens(ex))


def route_mismatch(
    T: Callable[[X], X], E: PackagingOperator[X, Y], x: X, *, space: str = "auto"
) -> float:
    """Compute the distance between E(T(x)) and T(E(x))."""
    space = _resolve_space(space, E)
    left = E(T(x))
    right = T(E(x))
    if space == "x":
        return E.dist_x(left, right)
    return E.lens.dist_y(E.lens(left), E.lens(right))


def _resolve_space(space: str, E: PackagingOperator[X, Y]) -> str:
    if space == "auto":
        if E.distance_x is not None:
            return "x"
        if E.lens.distance_y is not None:
            return "y"
        return "x"
    if space in {"x", "y"}:
        return space
    raise ValueError("space must be one of: 'auto', 'x', 'y'")
