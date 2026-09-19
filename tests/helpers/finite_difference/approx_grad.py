from collections.abc import Callable, Sequence
from math import isclose
from typing import TypeAlias, overload

from src.tensor import Tensor

ScalarFunction: TypeAlias = Callable[..., "float | Tensor[float]"]


@overload
def approx_equals(
    a: float, b: float, *, rel_tol: float = ..., abs_tol: float = ...
) -> bool: ...


@overload
def approx_equals(
    a: Sequence[float],
    b: Sequence[float],
    *,
    rel_tol: float = ...,
    abs_tol: float = ...,
) -> bool: ...


def approx_equals(
    a: float | Sequence[float],
    b: float | Sequence[float],
    *,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8,
) -> bool:
    """Compare two numbers, or two equally long sequences element by element.

    The relative tolerance keeps large gradients from failing on rounding error;
    the absolute tolerance handles gradients that should be zero.
    """
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        return len(a) == len(b) and all(
            isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b)
        )

    if isinstance(a, Sequence) or isinstance(b, Sequence):
        raise TypeError("Cannot compare a number with a sequence.")

    return isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def approx_grad(
    f: ScalarFunction, inputs: Sequence[float], h: float = 1e-6
) -> list[float]:
    """Central-difference estimate of the gradient of f at inputs."""
    return [_approx_partial(f, inputs, i, h) for i in range(len(inputs))]


def _approx_partial(
    f: ScalarFunction, inputs: Sequence[float], i: int, h: float
) -> float:
    right = _evaluate(f, _nudge(inputs, i, h))
    left = _evaluate(f, _nudge(inputs, i, -h))

    return (right - left) / (2 * h)


def _nudge(inputs: Sequence[float], i: int, delta: float) -> list[float]:
    nudged = list(inputs)
    nudged[i] += delta

    return nudged


def _evaluate(f: ScalarFunction, inputs: Sequence[float]) -> float:
    result = f(*inputs)

    return result.value if isinstance(result, Tensor) else result
