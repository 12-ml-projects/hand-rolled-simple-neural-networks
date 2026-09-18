import operator
from typing import Callable

import pytest

from src.operators import Add, Mul, Neg, Source, Sub, TrueDiv
from src.tensor import Tensor

BinaryOp = Callable[[object, object], object]


BINARY_CASES = [
    (operator.add, Add, 15.0),
    (operator.sub, Sub, -5.0),
    (operator.mul, Mul, 50.0),
    (operator.truediv, TrueDiv, 0.5),
]


class TestForward:
    @pytest.mark.parametrize(("op", "_", "expected"), BINARY_CASES)
    def test_binary_operations(self, op: BinaryOp, _: type, expected: float) -> None:
        result = op(Tensor(5.0), Tensor(10.0))

        assert isinstance(result, Tensor)
        assert result.value == pytest.approx(expected)

    def test_negation(self) -> None:
        assert (-Tensor(5.0)).value == -5.0

    def test_chained_expression(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)

        assert ((x + y) * x - y / x).value == pytest.approx(8.5)

    def test_operations_leave_tensors_unchanged(self) -> None:
        x = Tensor(5.0)
        y = Tensor(10.0)

        _ = x * y

        assert x.value == 5.0
        assert y.value == 10.0


class TestPlainNumbers:
    @pytest.mark.parametrize(("op", "_", "expected"), BINARY_CASES)
    def test_number_on_the_right(self, op: BinaryOp, _: type, expected: float) -> None:
        result = op(Tensor(5.0), 10.0)

        assert isinstance(result, Tensor)
        assert result.value == pytest.approx(expected)

    @pytest.mark.parametrize(("op", "_", "expected"), BINARY_CASES)
    def test_number_on_the_left(self, op: BinaryOp, _: type, expected: float) -> None:
        result = op(5.0, Tensor(10.0))

        assert isinstance(result, Tensor)
        assert result.value == pytest.approx(expected)

    def test_integers_are_accepted(self) -> None:
        assert (2 * Tensor(3.0) + 1).value == 7.0

    def test_number_becomes_a_leaf_dependency(self) -> None:
        x = Tensor(5.0)

        result = 1.0 - x

        constant, operand = result._dag.dependencies
        assert operand is x._dag
        assert constant.is_leaf()
        assert isinstance(constant.operator, Source)
        assert constant.value == 1.0


class TestGraph:
    def test_new_tensor_is_a_leaf(self) -> None:
        x = Tensor(5.0)

        assert x._dag.is_leaf()
        assert x._dag.dependencies == []
        assert isinstance(x._dag.operator, Source)

    def test_adjoint_starts_empty(self) -> None:
        assert Tensor(5.0)._dag.adjoint is None
        assert (Tensor(5.0) * Tensor(2.0))._dag.adjoint is None

    @pytest.mark.parametrize(("op", "operator_type", "_"), BINARY_CASES)
    def test_binary_operation_records_its_operator(
        self, op: BinaryOp, operator_type: type, _: float
    ) -> None:
        result = op(Tensor(5.0), Tensor(10.0))

        assert isinstance(result, Tensor)
        assert isinstance(result._dag.operator, operator_type)
        assert result._dag.arity == 2

    def test_negation_records_its_operator(self) -> None:
        result = -Tensor(5.0)

        assert isinstance(result._dag.operator, Neg)
        assert result._dag.arity == 1

    def test_dependencies_are_shared_not_copied(self) -> None:
        x = Tensor(5.0)
        y = Tensor(10.0)

        result = x * y

        assert result._dag.dependencies[0] is x._dag
        assert result._dag.dependencies[1] is y._dag

    def test_dependencies_keep_operand_order(self) -> None:
        x = Tensor(5.0)
        y = Tensor(10.0)

        result = y - x

        assert result._dag.dependencies == [y._dag, x._dag]

    def test_reused_tensor_is_one_shared_node(self) -> None:
        x = Tensor(5.0)

        result = x * x

        first, second = result._dag.dependencies
        assert first is second is x._dag

    def test_intermediate_nodes_are_shared(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)
        s = x + y

        result = s * x

        assert result._dag.dependencies[0] is s._dag
        assert s._dag.dependencies[0] is result._dag.dependencies[1] is x._dag


class TestEquality:
    def test_equals_compares_values(self) -> None:
        assert Tensor(5.0).equals(Tensor(5.0))
        assert not Tensor(5.0).equals(Tensor(6.0))

    def test_equals_accepts_plain_numbers(self) -> None:
        assert Tensor(5.0).equals(5.0)
        assert Tensor(5.0).equals(5)
        assert not Tensor(5.0).equals(6.0)

    def test_double_equals_is_identity(self) -> None:
        x = Tensor(5.0)

        assert x == x
        assert x != Tensor(5.0)

    def test_tensors_are_hashable(self) -> None:
        x = Tensor(5.0)
        y = Tensor(5.0)

        assert len({x, y, x}) == 2
