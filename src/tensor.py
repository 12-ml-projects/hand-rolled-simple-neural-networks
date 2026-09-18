from typing import Generic, TypeVar

from src.custom_types import ValueLike
from src.dag import DAG
from src.operators import Add, Mul, Neg, Operator, Source, Sub, TrueDiv

T = TypeVar("T", bound=ValueLike)


class Tensor(ValueLike, Generic[T]):
    _dag: DAG[T]

    def __init__(self, value: T):
        self._dag = DAG(Source(), value=value)

    @property
    def value(self) -> T:
        return self._dag.value

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

    def __add__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Add(), self, other)

    def __radd__(self, other: T) -> "Tensor[T]":
        return self._apply(Add(), other, self)

    def __sub__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Sub(), self, other)

    def __rsub__(self, other: T) -> "Tensor[T]":
        return self._apply(Sub(), other, self)

    def __neg__(self) -> "Tensor[T]":
        return self._apply(Neg(), self)

    def __mul__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(Mul(), self, other)

    def __rmul__(self, other: T) -> "Tensor[T]":
        return self._apply(Mul(), other, self)

    def __truediv__(self, other: "Tensor[T] | T") -> "Tensor[T]":
        return self._apply(TrueDiv(), self, other)

    def __rtruediv__(self, other: T) -> "Tensor[T]":
        return self._apply(TrueDiv(), other, self)
