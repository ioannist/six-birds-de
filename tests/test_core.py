import math

import numpy as np
import pytest

from sixbirds_cosmo.core import (
    Completion,
    E_tau,
    Lens,
    PackagingOperator,
    default_distance,
    idempotence_defect,
    route_mismatch,
)


def test_idempotence_defect_floor_idempotent() -> None:
    lens = Lens(lambda x: math.floor(x))
    completion = Completion(lambda y: float(y))
    E = PackagingOperator(lens, completion)
    assert idempotence_defect(E, 3.7) == 0.0


def test_route_mismatch_commuting_projection() -> None:
    lens = Lens(lambda x: float(x[0]))
    completion = Completion(lambda y: np.array([y, 0.0]))
    E = PackagingOperator(lens, completion)

    def T(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * x[0], 3.0 * x[1]])

    x = np.array([1.5, -2.0])
    assert route_mismatch(T, E, x) == 0.0


def test_route_mismatch_noncommuting_positive() -> None:
    lens = Lens(lambda x: float(x[0]))
    completion = Completion(lambda y: np.array([y, 0.0]))
    E = PackagingOperator(lens, completion)

    def T(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + x[1], x[1]])

    x = np.array([1.0, 2.0])
    assert route_mismatch(T, E, x) > 0.0


def test_default_distance_numbers() -> None:
    assert default_distance(3, 5) == 2.0


def test_default_distance_arrays() -> None:
    a = np.array([3.0, 4.0])
    b = np.array([0.0, 0.0])
    assert default_distance(a, b) == pytest.approx(5.0)


def test_default_distance_invalid_type() -> None:
    with pytest.raises(TypeError):
        default_distance({"a": 1}, {"a": 2})


def test_lens_custom_distance_y_used() -> None:
    lens = Lens(lambda x: x, distance_y=lambda a, b: 7.0)
    completion = Completion(lambda y: y)
    E = PackagingOperator(lens, completion)
    assert idempotence_defect(E, 1.23, space="y") == 7.0


def test_lens_dist_y_default_distance() -> None:
    lens = Lens(lambda x: x)
    assert lens.dist_y(1.0, 4.0) == 3.0


def test_packaging_macro_and_custom_distance_x() -> None:
    lens = Lens(lambda x: x + 1)
    completion = Completion(lambda y: y - 1)
    E = PackagingOperator(lens, completion, distance_x=lambda a, b: 5.0)
    assert E.macro(2.0) == 3.0
    assert E.dist_x(0.0, 1.0) == 5.0


def test_auto_space_prefers_distance_x() -> None:
    lens = Lens(lambda x: x, distance_y=lambda a, b: 1.0)
    completion = Completion(lambda y: y)
    E = PackagingOperator(lens, completion, distance_x=lambda a, b: 9.0)
    assert idempotence_defect(E, 2.0, space="auto") == 9.0


def test_auto_space_prefers_distance_y() -> None:
    lens = Lens(lambda x: x, distance_y=lambda a, b: 4.0)
    completion = Completion(lambda y: y)
    E = PackagingOperator(lens, completion)
    assert idempotence_defect(E, 2.0, space="auto") == 4.0


def test_route_mismatch_space_y() -> None:
    lens = Lens(lambda x: float(x[0]))
    completion = Completion(lambda y: np.array([y, 0.0]))
    E = PackagingOperator(lens, completion)

    def T(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + x[1], x[1]])

    x = np.array([1.0, 2.0])
    assert route_mismatch(T, E, x, space="y") > 0.0


def test_E_tau_matches_composition() -> None:
    lens = Lens(lambda x: x + 1)
    completion = Completion(lambda y: y - 1)
    E = PackagingOperator(lens, completion)

    def T(x: float) -> float:
        return x * 2

    assert E_tau(3.0, T, E) == E(T(3.0))


def test_invalid_space_raises() -> None:
    lens = Lens(lambda x: x)
    completion = Completion(lambda y: y)
    E = PackagingOperator(lens, completion)
    with pytest.raises(ValueError):
        idempotence_defect(E, 1.0, space="bad")
