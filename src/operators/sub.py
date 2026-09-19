from typing import TypeVar

from src.custom_types import ValueLike

from .operator import Operator

T = TypeVar("T", bound=ValueLike)


class Sub(Operator[T]):
    def forward(self, x: T, y: T) -> T:  # type: ignore[override]
        return x - y

    def backward(self, adjoint: T, x: T, y: T) -> tuple[T, T]:  # type: ignore[override]
        return (adjoint, -adjoint)
