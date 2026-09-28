import numpy as np

from src.custom_types import Array

from .operator import Operator


class Pow(Operator):
    """x ** y where the exponent is itself part of the graph."""

    def forward(self, x: Array, y: Array) -> Array:  # type: ignore[override]
        return x**y

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array, y: Array
    ) -> tuple[Array, Array]:
        # log(x) only exists for x > 0; off there x**y has no derivative in y.
        return (
            adjoint * y * x ** (y - 1),
            adjoint * x**y * np.log(np.where(x > 0, x, 1.0)),
        )


class UnaryPow(Operator):
    """x ** n for a constant n, kept off the graph so no log is ever needed."""

    exponent: Array

    def __init__(self, exponent: Array) -> None:
        self.exponent = exponent

    def forward(self, x: Array) -> Array:  # type: ignore[override]
        return x**self.exponent

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        return (adjoint * self.exponent * x ** (self.exponent - 1),)
