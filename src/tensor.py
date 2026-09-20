from typing import Generic, TypeVar

from src.custom_types import ValueLike
from src.dag import DAG
from src.operators import (
    Abs,
    Add,
    Mul,
    Neg,
    Operator,
    Pow,
    Source,
    Sub,
    TrueDiv,
    UnaryPow,
)

T = TypeVar("T", bound=ValueLike)


class Tensor(Generic[T]):
    _dag: DAG[T]

    def __init__(self, value: T, requires_grad: bool = True):
        self._dag = DAG(Source(), value=value, _requires_grad=requires_grad)

    @property
    def value(self) -> T:
        return self._dag.value

    @value.setter
    def value(self, value: T) -> None:
        if self._dag.dependencies:
            raise ValueError("Cannot set the value of a non-leaf Tensor.")

        self._dag.value = value

    @property
    def grad(self) -> T | None:
        return self._dag.adjoint

    @property
    def requires_grad(self) -> bool:
        return self._dag.requires_grad

    @classmethod
    def _from_dag(cls, dag: DAG[T]) -> "Tensor[T]":
        # Bypass __init__, which would build a leaf DAG only to throw it away.
        tensor = cls.__new__(cls)
        tensor._dag = dag

        return tensor

    @staticmethod
    def _as_dag(operand: "Tensor[T] | T") -> DAG[T]:
        """Plain numbers enter the graph as constant leaves."""
        if isinstance(operand, Tensor):
            return operand._dag

        return DAG(Source(), value=operand)

    @classmethod
    def _apply(cls, operator: Operator[T], *operands: "Tensor[T] | T") -> "Tensor[T]":
        dependencies = [cls._as_dag(operand) for operand in operands]

        return cls._from_dag(DAG(operator, dependencies))

    def equals(self, other: "Tensor[T] | T") -> bool:
        """Exact value equality. `==` is left as identity, as in PyTorch."""
        other_value = other.value if isinstance(other, Tensor) else other

        return bool(self.value == other_value)

    def backward(self, adjoint: T = 1.0) -> None:  # type: ignore
        self._dag.backward(adjoint)

    def reset(self) -> None:
        self._dag.reset()

    def zero_grad(self) -> None:
        self._dag.zero_grad()

    def __add__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Add(), self, other)

    def __radd__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Add(), other, self)

    def __sub__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Sub(), self, other)

    def __rsub__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Sub(), other, self)

    def __neg__(self) -> "Tensor[T]":
        return self._apply(Neg(), self)

    def __mul__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Mul(), self, other)

    def __rmul__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Mul(), other, self)

    def __truediv__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(TrueDiv(), self, other)

    def __rtruediv__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(TrueDiv(), other, self)

    def __pow__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        if isinstance(other, Tensor) and not other.requires_grad:
            return self._apply(UnaryPow(other.value), self)

        if not isinstance(other, Tensor):
            return self._apply(UnaryPow(other), self)

        return self._apply(Pow(), self, other)

    def __rpow__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        # NOTE: this is only reached if `other` is not a tensor
        if self.requires_grad:
            return self._apply(Pow(), other, self)

        return self._apply(UnaryPow(self.value), other)

    def __abs__(self) -> "Tensor[T]":
        return self._apply(Abs(), self)
