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
    requires_grad: bool

    @overload
    def __init__(
        self, operator: Source, *, value: T, _requires_grad: Optional[bool] = None
    ) -> None: ...

    @overload
    def __init__(
        self,
        operator: Operator[T],
        dependencies: Dependencies,
        *,
        _requires_grad: Optional[bool] = None,
    ) -> None: ...

    def __init__(
        self,
        operator: Operator[T],
        dependencies: Optional[Dependencies] = None,
        *,
        value: Optional[T] = None,
        _requires_grad: Optional[bool] = None,
    ) -> None:
        self.operator = operator
        self.adjoint = None
        self.requires_grad = (
            _requires_grad
            if _requires_grad is not None
            else any(dep.requires_grad for dep in dependencies or [])
        )

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
        if not self.requires_grad:
            return

        order = self._topological_order()

        # An intermediate adjoint belongs to one pass and one seed, so clearing
        # them here keeps a second backward() from double-counting. Leaves are
        # left alone: they accumulate until an explicit zero_grad(), which is
        # what lets gradients pile up across a minibatch.
        self._reset_intermediates(order)

        self.adjoint = adjoint

        for node in reversed(order):
            if node.is_leaf() or not node.requires_grad:
                continue

            for dep, contribution in zip(
                node.dependencies,
                node.operator.backward(
                    node.adjoint, *(d.value for d in node.dependencies)  # type: ignore
                ),
            ):
                if not dep.requires_grad:
                    continue

                dep.accumulate(contribution)

    def reset(self) -> None:
        """Clear every intermediate adjoint reachable from here, leaves aside."""
        self._reset_intermediates(self._topological_order())

    def _reset_intermediates(self, order: list["DAG[T]"]) -> None:
        # The topological order already holds every reachable node exactly once,
        # so this is a flat pass: recursing into dependencies here would re-walk
        # shared subgraphs once per path reaching them.
        for node in order:
            if not node.is_leaf():
                node.adjoint = None

    def zero_grad(self) -> None:
        self.adjoint = None

    def accumulate(self, contribution: T) -> None:
        if self.adjoint is None:
            self.adjoint = contribution
        else:
            self.adjoint += contribution

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
