from typing import TypeAlias

import numpy as np

from src.custom_types import Value, ValueLike, as_value
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

Operand: TypeAlias = "Tensor | ValueLike"


class Tensor:
    _dag: DAG

    def __init__(self, value: ValueLike, requires_grad: bool = True):
        self._dag = DAG(Source(), value=as_value(value), _requires_grad=requires_grad)

    @property
    def value(self) -> Value:
        return self._dag.value

    @value.setter
    def value(self, value: ValueLike) -> None:
        if self._dag.dependencies:
            raise ValueError("Cannot set the value of a non-leaf Tensor.")

        self._dag.value = as_value(value)

    @property
    def grad(self) -> Value | None:
        return self._dag.adjoint

    @property
    def requires_grad(self) -> bool:
        return self._dag.requires_grad

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dag.value.shape

    @classmethod
    def _from_dag(cls, dag: DAG) -> "Tensor":
        # Bypass __init__, which would build a leaf DAG only to throw it away.
        tensor = cls.__new__(cls)
        tensor._dag = dag

        return tensor

    @staticmethod
    def _as_dag(operand: Operand) -> DAG:
        """Plain values enter the graph as constant leaves."""
        if isinstance(operand, Tensor):
            return operand._dag

        return DAG(Source(), value=as_value(operand))

    @classmethod
    def _apply(cls, operator: Operator, *operands: Operand) -> "Tensor":
        dependencies = [cls._as_dag(operand) for operand in operands]

        return cls._from_dag(DAG(operator, dependencies))

    def equals(self, other: Operand) -> bool:
        """Exact value equality. `==` is left as identity, as in PyTorch."""
        other_value = other.value if isinstance(other, Tensor) else as_value(other)

        return bool(np.array_equal(self.value, other_value))

    def backward(self, adjoint: ValueLike | None = None) -> None:
        self._dag.backward(self._seed(adjoint))

    def _seed(self, adjoint: ValueLike | None) -> Value:
        if adjoint is not None:
            return as_value(adjoint)

        if self.value.shape != ():
            raise ValueError(
                "backward() needs an explicit adjoint for a non-scalar tensor."
            )

        return as_value(1.0)

    def reset(self) -> None:
        self._dag.reset()

    def zero_grad(self) -> None:
        self._dag.zero_grad()

    def __add__(self, other: Operand) -> "Tensor":
        return self._apply(Add(), self, other)

    def __radd__(self, other: Operand) -> "Tensor":
        return self._apply(Add(), other, self)

    def __sub__(self, other: Operand) -> "Tensor":
        return self._apply(Sub(), self, other)

    def __rsub__(self, other: Operand) -> "Tensor":
        return self._apply(Sub(), other, self)

    def __neg__(self) -> "Tensor":
        return self._apply(Neg(), self)

    def __mul__(self, other: Operand) -> "Tensor":
        return self._apply(Mul(), self, other)

    def __rmul__(self, other: Operand) -> "Tensor":
        return self._apply(Mul(), other, self)

    def __truediv__(self, other: Operand) -> "Tensor":
        return self._apply(TrueDiv(), self, other)

    def __rtruediv__(self, other: Operand) -> "Tensor":
        return self._apply(TrueDiv(), other, self)

    def __pow__(self, other: Operand) -> "Tensor":
        if isinstance(other, Tensor) and not other.requires_grad:
            return self._apply(UnaryPow(other.value), self)

        if not isinstance(other, Tensor):
            return self._apply(UnaryPow(as_value(other)), self)

        return self._apply(Pow(), self, other)

    def __rpow__(self, other: Operand) -> "Tensor":
        if self.requires_grad:
            return self._apply(Pow(), other, self)

        return self._apply(UnaryPow(self.value), other)

    def __abs__(self) -> "Tensor":
        return self._apply(Abs(), self)
