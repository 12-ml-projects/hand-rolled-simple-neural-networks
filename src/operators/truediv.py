from src.custom_types import Array

from .operator import Operator


class TrueDiv(Operator):
    def forward(self, x: Array, y: Array) -> Array:  # type: ignore[override]
        return x / y

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array, y: Array
    ) -> tuple[Array, Array]:
        return (adjoint / y, -adjoint * x / (y * y))
