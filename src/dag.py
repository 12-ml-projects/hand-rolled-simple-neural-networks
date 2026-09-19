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

    def backward(self, adjoint: T) -> None:
        self.adjoint = adjoint

        for node in reversed(self._topological_order()):
            if not node.dependencies:
                continue

            for dep, contribution in zip(
                node.dependencies,
                node.operator.backward(
                    node.adjoint, *(d.value for d in node.dependencies)  # type: ignore
                ),
            ):
                dep.accumulate(contribution)

    def accumulate(self, contribution: T) -> None:
        if self.adjoint is None:
            self.adjoint = 0.0  # type: ignore

        self.adjoint += contribution  # type: ignore

    def _topological_order(self) -> list["DAG[T]"]:
        return self.visit(self)

    @classmethod
    def visit(
        cls,
        node: "DAG[T]",
        seen: Optional[set["DAG[T]"]] = None,
        order: Optional[list["DAG[T]"]] = None,
    ) -> list["DAG[T]"]:
        # NOTE for very large networks, or for e.g., summing over losses,
        # this recursive approach may hit the recursion limit. But fails loudly.
        # We will cross that bridge when we get there.
        if seen is None:
            seen = set()
        if order is None:
            order = []

        if node in seen:
            return order

        seen.add(node)

        for dep in node.dependencies:
            order = cls.visit(dep, seen, order)

        order.append(node)

        return order
