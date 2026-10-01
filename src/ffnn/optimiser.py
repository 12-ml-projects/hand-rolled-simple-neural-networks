from abc import ABC, abstractmethod

from src.tensor import Tensor


class Optimiser(ABC):
    _params: list[Tensor]
    _lr: float

    def __init__(self, params: list[Tensor], lr: float):
        self._params = params
        self._lr = lr

    def zero_grad(self) -> None:
        for param in self._params:
            param.zero_grad()

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimiser):
    def step(self) -> None:
        for param in self._params:
            if param.grad is None:
                continue

            param.value -= self._lr * param.grad
