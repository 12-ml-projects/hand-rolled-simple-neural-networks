import numpy as np

from src.custom_types import Value

from .operator import Operator


class Abs(Operator):
    def forward(self, x: Value) -> Value:  # type: ignore[override]
        return np.abs(x)

    def backward(  # type: ignore[override]
        self, adjoint: Value, x: Value
    ) -> tuple[Value]:
        # np.sign is 0 at the kink, as in PyTorch.
        return (adjoint * np.sign(x),)
