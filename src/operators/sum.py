import numpy as np

from src.custom_types import Array, as_array

from .operator import Operator

Axis = int | tuple[int, ...] | None


class Sum(Operator):
    _axis: Axis
    _keepdims: bool

    def __init__(self, axis: Axis = None, keepdims: bool = False) -> None:
        self._axis = axis
        self._keepdims = keepdims

    def forward(self, x: Array) -> Array:  # type: ignore[override]
        # numpy overloads keepdims on Literal[True]/[False], so a bool fits none.
        total = np.sum(  # type: ignore[call-overload]
            x, axis=self._axis, keepdims=self._keepdims
        )

        return as_array(total)

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        # copy() because broadcast_to returns a read-only view
        return (np.broadcast_to(self._restore_axes(adjoint), x.shape).copy(),)

    def _restore_axes(self, adjoint: Array) -> Array:
        if self._keepdims or self._axis is None:
            return adjoint

        return np.expand_dims(adjoint, self._axis)
