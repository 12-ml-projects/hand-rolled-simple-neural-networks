from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Operator(ABC, Generic[T]):
    @abstractmethod
    def forward(self, *args, **kwargs) -> T:
        pass
