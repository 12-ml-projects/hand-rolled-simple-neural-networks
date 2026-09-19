from typing import TypeVar

from src.custom_types import ValueLike

from .operator import Operator

T = TypeVar("T", bound=ValueLike)


class Neg(Operator[T]):
    def forward(self, x: T) -> T:  # type: ignore[override]
        return -x

    def backward(self, adjoint: T, x: T) -> tuple[T]:  # type: ignore[override]
        return (-adjoint,)
