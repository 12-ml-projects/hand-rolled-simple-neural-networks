import numpy as np

from src.custom_types import Array, as_array

from .operator import Operator

Axis = int | tuple[int, ...] | None


class Sum(Operator):
    axis: Axis
    keepdims: bool

    def __init__(self, axis: Axis = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: Array) -> Array:  # type: ignore[override]
        # numpy overloads keepdims on Literal[True]/[False], so a bool fits none.
        total = np.sum(  # type: ignore[call-overload]
            x, axis=self.axis, keepdims=self.keepdims
        )

        return as_array(total)

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        # copy() because broadcast_to returns a read-only view
        return (np.broadcast_to(self._restore_axes(adjoint), x.shape).copy(),)

    def _restore_axes(self, adjoint: Array) -> Array:
        if self.keepdims or self.axis is None:
            return adjoint

        return np.expand_dims(adjoint, self.axis)
