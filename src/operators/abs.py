from typing import TypeVar

from src.custom_types import ValueLike

from .operator import Operator

T = TypeVar("T", bound=ValueLike)


class Abs(Operator[T]):
    def forward(self, x: T) -> T:  # type: ignore[override]
        return abs(x)  # type: ignore

    def backward(self, adjoint: T, x: T) -> tuple[T]:  # type: ignore[override]
        if x > 0:  # type: ignore
            return (adjoint,)
        elif x < 0:  # type: ignore
            return (-adjoint,)
        else:
            return (0,)  # type: ignore
