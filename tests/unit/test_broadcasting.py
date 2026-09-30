"""Broadcasting is the one thing numpy does silently, that's not in the graph.

A stretched operand is copied in the forward pass, so its adjoint must be summed
back down in the backward pass. The operators cannot do that themselves: numpy
fuses the broadcast into the arithmetic, so by the time `Add.backward` runs, the
original operand shapes are gone. Hence the invariant these tests pin:

    an adjoint always has the shape of the value it belongs to

(because the adjoint is partial L / partial a_ij...z)

Shape alone is not enough to check, though. Summing over the wrong axis of a
square operand yields the right shape and the wrong numbers, so every case is
also compared against a finite-difference estimate using the Jacobian.
"""

import operator
from typing import Any, Callable

import numpy as np
import pytest

from src.custom_types import Array
from src.tensor import Tensor
from tests.helpers.finite_difference.approx_grad import approx_vjp

Op = Callable[..., Any]

BINARY_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
]

SHAPE_PAIRS = [
    ((3, 4), (3, 4)),  # control: nothing is broadcast
    ((3, 1), (1, 4)),  # both operands stretched
    ((8, 3), (3,)),  # rank mismatch, as a bias against a batch
    ((), (2, 3)),  # 0-d against an array
    ((2, 3), ()),  # and the other way round
    ((1, 3), (4, 3)),  # one axis stretched
    ((5, 1, 3), (2, 3)),  # rank mismatch and a stretch together
    ((2, 1, 4), (3, 4)),  # stretched on both sides, rank 3
    ((1, 3), (5, 2, 3)),  # needs BOTH steps: drop an axis, then un-stretch
]

UNARY_OPS = [
    operator.neg,
    operator.abs,
    lambda x: x**2.0,
]

UNARY_SHAPES = [(), (4,), (2, 3), (5, 1, 3)]


def positive(*shapes: tuple[int, ...]) -> list[Array]:
    """Arrays drawn from [0.5, 2.0].

    Strictly positive and away from zero, so one set of data suits every
    operator at once: no zero denominators for TrueDiv, and no non-positive
    bases for Pow, whose backward needs log(x).
    """
    rng = np.random.default_rng(abs(hash(shapes)) % (2**32))

    return [rng.uniform(0.5, 2.0, size=shape) for shape in shapes]


def adjoint_for(shape: tuple[int, ...]) -> Array:
    rng = np.random.default_rng(abs(hash(("adjoint", shape))) % (2**32))

    return rng.normal(size=shape)


class TestBinaryBroadcasting:
    @pytest.mark.parametrize("op", BINARY_OPS, ids=lambda op: op.__name__)
    @pytest.mark.parametrize(("left", "right"), SHAPE_PAIRS, ids=str)
    def test_gradients_keep_the_operand_shapes(
        self, op: Op, left: tuple[int, ...], right: tuple[int, ...]
    ) -> None:
        a, b = positive(left, right)
        x, y = Tensor(a), Tensor(b)

        out = op(x, y)
        out.backward(adjoint_for(out.shape))

        assert x.grad is not None and y.grad is not None
        assert x.grad.shape == left
        assert y.grad.shape == right

    @pytest.mark.parametrize("op", BINARY_OPS, ids=lambda op: op.__name__)
    @pytest.mark.parametrize(("left", "right"), SHAPE_PAIRS, ids=str)
    def test_gradients_match_finite_differences(
        self, op: Op, left: tuple[int, ...], right: tuple[int, ...]
    ) -> None:
        a, b = positive(left, right)
        x, y = Tensor(a), Tensor(b)

        out = op(x, y)
        adjoint = adjoint_for(out.shape)
        out.backward(adjoint)

        expected = approx_vjp(op, [a, b], adjoint)

        assert x.grad == pytest.approx(expected[0], rel=1e-4, abs=1e-6)
        assert y.grad == pytest.approx(expected[1], rel=1e-4, abs=1e-6)

    def test_a_bias_collects_one_gradient_per_batch_row(self) -> None:
        rows = 8
        batch = Tensor(np.ones((rows, 3)))
        bias = Tensor(np.zeros(3))

        total = batch + bias
        total.backward(np.ones((rows, 3)))

        assert bias.grad is not None
        assert bias.grad.shape == (3,)
        assert bias.grad == pytest.approx(np.full(3, float(rows)))

    def test_a_scalar_collects_every_gradient(self) -> None:
        scale = Tensor(2.0)
        values = Tensor(np.ones((4, 5)))

        product = scale * values
        product.backward(np.ones((4, 5)))

        assert scale.grad is not None
        assert scale.grad.shape == ()
        assert scale.grad == pytest.approx(20.0)


class TestUnaryShapes:
    @pytest.mark.parametrize("op", UNARY_OPS, ids=["neg", "abs", "pow"])
    @pytest.mark.parametrize("shape", UNARY_SHAPES, ids=str)
    def test_shape_is_preserved(self, op: Op, shape: tuple[int, ...]) -> None:
        # A unary operator has one operand, so its output shape is its input
        # shape and there is never anything to unbroadcast. Pinned rather than
        # assumed.
        (a,) = positive(shape)
        x = Tensor(a)

        out = op(x)
        out.backward(adjoint_for(shape))

        assert x.grad is not None
        assert x.grad.shape == shape

    @pytest.mark.parametrize("op", UNARY_OPS, ids=["neg", "abs", "pow"])
    @pytest.mark.parametrize("shape", UNARY_SHAPES, ids=str)
    def test_gradients_match_finite_differences(
        self, op: Op, shape: tuple[int, ...]
    ) -> None:
        (a,) = positive(shape)
        x = Tensor(a)

        out = op(x)
        adjoint = adjoint_for(shape)
        out.backward(adjoint)

        (expected,) = approx_vjp(op, [a], adjoint)

        assert x.grad == pytest.approx(expected, rel=1e-4, abs=1e-6)


class TestMatMulBroadcasting:
    def test_batch_axes_are_reduced(self) -> None:
        # MatMul broadcasts its leading axes but not the trailing two, so it is
        # the one non-elementwise operator that needs unbroadcasting.
        a, b = positive((1, 2, 3), (7, 3, 4))
        x, y = Tensor(a), Tensor(b)

        product = x @ y
        assert product.shape == (7, 2, 4)

        product.backward(adjoint_for((7, 2, 4)))

        assert x.grad is not None and y.grad is not None
        assert x.grad.shape == (1, 2, 3)
        assert y.grad.shape == (7, 3, 4)

    def test_batch_gradients_match_finite_differences(self) -> None:
        a, b = positive((1, 2, 3), (7, 3, 4))
        x, y = Tensor(a), Tensor(b)

        product = x @ y
        adjoint = adjoint_for((7, 2, 4))
        product.backward(adjoint)

        expected = approx_vjp(operator.matmul, [a, b], adjoint)

        assert x.grad == pytest.approx(expected[0], rel=1e-4, abs=1e-6)
        assert y.grad == pytest.approx(expected[1], rel=1e-4, abs=1e-6)
