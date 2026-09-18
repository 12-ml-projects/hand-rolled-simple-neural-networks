from typing import Generic, Optional, TypeAlias, TypeVar, overload

from src.custom_types import ValueLike
from src.operators import Operator, Source

T = TypeVar("T", bound=ValueLike)


Dependencies: TypeAlias = list["DAG[T]"]


class DAG(Generic[T]):
    """Computational DAG for representing dependencies between tensors."""

    dependencies: Dependencies
    operator: Operator[T]
    value: T
    adjoint: T | None

    @overload
    def __init__(self, operator: Source, *, value: T) -> None: ...

    @overload
    def __init__(self, operator: Operator[T], dependencies: Dependencies) -> None: ...

    def __init__(
        self,
        operator: Operator[T],
        dependencies: Optional[Dependencies] = None,
        *,
        value: Optional[T] = None,
    ) -> None:
        self.operator = operator
        self.adjoint = None

        if isinstance(operator, Source):
            if value is None:
                raise ValueError("A source node needs a value.")
            self.dependencies = []
            self.value = value
        else:
            self.dependencies = dependencies or []
            self.value = operator.forward(*(dep.value for dep in self.dependencies))

    @property
    def arity(self) -> int:
        return len(self.dependencies)

    def is_leaf(self) -> bool:
        return self.arity == 0
