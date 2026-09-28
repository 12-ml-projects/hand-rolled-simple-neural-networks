import numpy as np

from src.custom_types import Array

from .operator import Operator


class Abs(Operator):
    def forward(self, x: Array) -> Array:  # type: ignore[override]
        return np.abs(x)

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        # np.sign is 0 at the kink, as in PyTorch.
        return (adjoint * np.sign(x),)
