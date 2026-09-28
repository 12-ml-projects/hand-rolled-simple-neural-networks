from src.custom_types import Array

from .operator import Operator


class Neg(Operator):
    def forward(self, x: Array) -> Array:  # type: ignore[override]
        return -x

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        return (-adjoint,)
