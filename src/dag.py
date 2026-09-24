from typing import Optional, TypeAlias, overload

from src.custom_types import Value
from src.operators import Operator, Source

Dependencies: TypeAlias = list["DAG"]


class DAG:
    """Computational DAG for representing dependencies between tensors."""

    dependencies: Dependencies
    operator: Operator
    value: Value
    adjoint: Value | None
    requires_grad: bool

    @overload
    def __init__(
        self, operator: Source, *, value: Value, _requires_grad: Optional[bool] = None
    ) -> None: ...

    @overload
    def __init__(
        self,
        operator: Operator,
        dependencies: Dependencies,
        *,
        _requires_grad: Optional[bool] = None,
    ) -> None: ...

    def __init__(
        self,
        operator: Operator,
        dependencies: Optional[Dependencies] = None,
        *,
        value: Optional[Value] = None,
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

    def backward(self, adjoint: Value) -> None:
        if not self.requires_grad:
            return

        order = self._topological_order()

        # Leaves are left alone; they accumulate until an explicit zero_grad().
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

    def _reset_intermediates(self, order: list["DAG"]) -> None:
        for node in order:
            if not node.is_leaf():
                node.adjoint = None

    def zero_grad(self) -> None:
        self.adjoint = None

    def accumulate(self, contribution: Value) -> None:
        if self.adjoint is None:
            # Copy: an operator may hand the same array to several dependencies
            # (Add returns (adjoint, adjoint)) and `+=` below mutates in place.
            self.adjoint = contribution.copy()
        else:
            self.adjoint += contribution

    def _topological_order(self) -> list["DAG"]:
        return self.visit(self)

    @classmethod
    def visit(
        cls,
        node: "DAG",
        seen: Optional[set["DAG"]] = None,
        order: Optional[list["DAG"]] = None,
    ) -> list["DAG"]:
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
