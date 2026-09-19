from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.custom_types import ValueLike

T = TypeVar("T", bound=ValueLike)


class Operator(ABC, Generic[T]):
    @abstractmethod
    def forward(self, *args: T, **kwargs: T) -> T:
        pass

    @abstractmethod
    def backward(self, adjoint: T, *args: T) -> tuple[T, ...]:
        pass
