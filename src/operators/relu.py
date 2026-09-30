import numpy as np

from src.custom_types import Array
from src.operators.operator import Operator


class ReLU(Operator):
    def forward(self, x: Array) -> Array:  # type: ignore[override]
        return np.maximum(0, x)

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        return (adjoint * (x > 0),)
