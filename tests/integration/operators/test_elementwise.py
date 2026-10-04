"""The elementwise operators.

Add, Sub, Mul, TrueDiv, Pow, Neg, Abs and ReLU are all stateless and all
shape-preserving, so there is no per-operator behaviour to pin -- only that the
dunder builds the operator it claims to, and that the arithmetic is numpy's.
Hence one parameterised table rather than eight near-identical files.

MatMul and Sum get files of their own because they change tensor shapes.
"""

import operator
from typing import Callable

import numpy as np
import pytest

from src.operators import Abs, Add, Mul, Neg, Pow, Source, Sub, TrueDiv
from src.tensor import Tensor

BinaryOp = Callable[[object, object], object]
UnaryOp = Callable[[object], object]


# Applied to (5.0, 10.0).
BINARY_CASES = [
    (operator.add, Add, 15.0),
    (operator.sub, Sub, -5.0),
    (operator.mul, Mul, 50.0),
    (operator.truediv, TrueDiv, 0.5),
    (operator.pow, Pow, 9765625.0),
]

# Applied to -5.0.
UNARY_CASES = [
    (operator.neg, Neg, 5.0),
    (operator.abs, Abs, 5.0),
]


class TestForward:
    @pytest.mark.parametrize(("op", "_", "expected"), BINARY_CASES)
    def test_binary_operations(self, op: BinaryOp, _: type, expected: float) -> None:
        result = op(Tensor(5.0), Tensor(10.0))

        assert isinstance(result, Tensor)
        assert result.value == pytest.approx(expected)

    @pytest.mark.parametrize(("op", "_", "expected"), UNARY_CASES)
    def test_unary_operations(self, op: UnaryOp, _: type, expected: float) -> None:
        result = op(Tensor(-5.0))

        assert isinstance(result, Tensor)
        assert result.value == pytest.approx(expected)

    def test_pow_with_fractional_exponent(self) -> None:
        assert (Tensor(9.0) ** 0.5).value == pytest.approx(3.0)

    def test_chained_expression(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)

        assert ((x + y) * x - y / x).value == pytest.approx(8.5)

    def test_operations_leave_operands_unchanged(self) -> None:
        # Only meaningful on arrays: `+=` rebinds a float but mutates an
        # ndarray, so an operator writing to its own input would corrupt the
        # graph in place and silently. Checks the backward pass too, which is
        # where the adjoint arrays get accumulated.
        x = Tensor(np.array([1.0, 2.0]))
        y = Tensor(np.array([3.0, 4.0]))

        for combine in (
            lambda a, b: a * b,
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a / b,
        ):
            combine(x, y).backward(np.ones(2))

        assert x.value == pytest.approx([1.0, 2.0])
        assert y.value == pytest.approx([3.0, 4.0])


class TestOperatorMapping:
    """Which operator each dunder builds -- the table above read backwards."""

    @pytest.mark.parametrize(("op", "operator_type", "_"), BINARY_CASES)
    def test_binary_operation_records_its_operator(
        self, op: BinaryOp, operator_type: type, _: float
    ) -> None:
        result = op(Tensor(5.0), Tensor(10.0))

        assert isinstance(result, Tensor)
        assert isinstance(result._dag.operator, operator_type)
        assert result._dag.arity == 2

    @pytest.mark.parametrize(("op", "operator_type", "_"), UNARY_CASES)
    def test_unary_operation_records_its_operator(
        self, op: UnaryOp, operator_type: type, _: float
    ) -> None:
        result = op(Tensor(-5.0))

        assert isinstance(result, Tensor)
        assert isinstance(result._dag.operator, operator_type)
        assert result._dag.arity == 1


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


class TestMethods:
    def test_relu_gates_each_entry(self) -> None:
        x = Tensor(np.array([-2.0, -0.5, 0.5, 2.0]))

        activated = x.relu()
        activated.backward(np.ones(4))

        assert activated.value == pytest.approx([0.0, 0.0, 0.5, 2.0])
        assert x.grad == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_relu_at_zero_is_closed_on_the_left(self) -> None:
        x = Tensor(0.0)

        x.relu().backward()

        assert x.grad == pytest.approx(0.0)

    def test_abs_at_zero_is_zero(self) -> None:
        x = Tensor(0.0)

        abs(x).backward()

        assert x.grad == pytest.approx(0.0)

    def test_sqrt(self) -> None:
        x = Tensor(9.0)

        root = x.sqrt()
        root.backward()

        assert root.value == pytest.approx(3.0)
        assert x.grad == pytest.approx(1.0 / 6.0)
