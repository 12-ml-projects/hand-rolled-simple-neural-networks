import numpy as np

from src.custom_types import Value

from .operator import Operator


class Pow(Operator):
    """x ** y where the exponent is itself part of the graph."""

    def forward(self, x: Value, y: Value) -> Value:  # type: ignore[override]
        return x**y

    def backward(  # type: ignore[override]
        self, adjoint: Value, x: Value, y: Value
    ) -> tuple[Value, Value]:
        # log(x) only exists for x > 0; off there x**y has no derivative in y.
        return (
            adjoint * y * x ** (y - 1),
            adjoint * x**y * np.log(np.where(x > 0, x, 1.0)),
        )


class UnaryPow(Operator):
    """x ** n for a constant n, kept off the graph so no log is ever needed."""

    exponent: Value

    def __init__(self, exponent: Value) -> None:
        self.exponent = exponent

    def forward(self, x: Value) -> Value:  # type: ignore[override]
        return x**self.exponent

    def backward(  # type: ignore[override]
        self, adjoint: Value, x: Value
    ) -> tuple[Value]:
        return (adjoint * self.exponent * x ** (self.exponent - 1),)
