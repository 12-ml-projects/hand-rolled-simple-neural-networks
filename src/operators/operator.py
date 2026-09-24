from abc import ABC, abstractmethod

from src.custom_types import Value


class Operator(ABC):
    @abstractmethod
    def forward(self, *args: Value) -> Value:
        pass

    @abstractmethod
    def backward(self, adjoint: Value, *args: Value) -> tuple[Value, ...]:
        pass
