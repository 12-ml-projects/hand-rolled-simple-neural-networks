from abc import ABC, abstractmethod

from src.custom_types import Array


class Operator(ABC):
    @abstractmethod
    def forward(self, *args: Array) -> Array:
        pass

    @abstractmethod
    def backward(self, adjoint: Array, *args: Array) -> tuple[Array, ...]:
        pass
