import numpy as np

from src.custom_types import Array

from .operator import Operator


class MatMul(Operator):
    """Matrix multiplication over the last two axes; earlier axes are batch."""

    def forward(self, x: Array, y: Array) -> Array:  # type: ignore[override]
        _require_matrices(x, y)

        return x @ y

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array, y: Array
    ) -> tuple[Array, Array]:
        return (adjoint @ _transpose(y), _transpose(x) @ adjoint)


def _transpose(value: Array) -> Array:
    # Not .T, which reverses every axis and would scramble the batch dimensions.
    return np.swapaxes(value, -1, -2)


def _require_matrices(x: Array, y: Array) -> None:
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError(
            "MatMul requires at least 2 dimensions, "
            f"got {x.ndim} and {y.ndim}. "
            "Reshape a vector to (1, n) or (n, 1) to make the intent explicit."
        )
