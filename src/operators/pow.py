from math import log
from typing import TypeVar

from src.custom_types import ValueLike

from .operator import Operator

T = TypeVar("T", bound=ValueLike)


class Pow(Operator[T]):
    def forward(self, x: T, y: T) -> T:  # type: ignore[override]
        return x**y

    def backward(self, adjoint: T, x: T, y: T) -> tuple[T, T]:  # type: ignore[override]
        return (adjoint * y * x ** (y - 1), adjoint * x**y * log(x))  # type: ignore


class UnaryPow(Operator[T]):
    exponent: T

    def __init__(self, exponent: T) -> None:
        self.exponent = exponent

    def forward(self, x: T) -> T:  # type: ignore[override]
        return x**self.exponent

    def backward(self, adjoint: T, x: T) -> tuple[T]:  # type: ignore[override]
        return (adjoint * self.exponent * x ** (self.exponent - 1),)  # type: ignore
