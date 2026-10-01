import numpy as np
import pytest

from src.operators.sum import Axis
from src.tensor import Tensor
from tests.helpers.finite_difference.approx_grad import approx_vjp

CASES: list[tuple[tuple[int, ...], Axis, bool]] = [
    ((4,), None, False),
    ((2, 3), None, False),
    ((2, 3), 0, False),
    ((2, 3), 1, False),
    ((2, 3), 0, True),
    ((2, 3), 1, True),
    ((2, 3), (0, 1), False),
    ((2, 3, 4), 1, False),
    ((2, 3, 4), (0, 2), False),
    ((2, 3, 4), (0, 2), True),
    ((), None, False),
]


def values(shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng([len(shape), *shape]).normal(size=shape)


def numpy_sum(a: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    # numpy overloads keepdims on Literal[True]/[False], so a bool fits none.
    return np.sum(a, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]


def numpy_mean(a: np.ndarray, axis: Axis, keepdims: bool) -> np.ndarray:
    return np.mean(a, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]


class TestSumForward:
    @pytest.mark.parametrize(("shape", "axis", "keepdims"), CASES, ids=str)
    def test_matches_numpy(
        self, shape: tuple[int, ...], axis: Axis, keepdims: bool
    ) -> None:
        a = values(shape)

        total = Tensor(a).sum(axis, keepdims)

        assert total.value == pytest.approx(numpy_sum(a, axis, keepdims))
        assert total.shape == numpy_sum(a, axis, keepdims).shape


class TestSumBackward:
    @pytest.mark.parametrize(("shape", "axis", "keepdims"), CASES, ids=str)
    def test_matches_finite_differences(
        self, shape: tuple[int, ...], axis: Axis, keepdims: bool
    ) -> None:
        a = values(shape)
        x = Tensor(a)

        total = x.sum(axis, keepdims)
        adjoint = values(total.shape)
        total.backward(adjoint)

        (expected,) = approx_vjp(lambda v: numpy_sum(v, axis, keepdims), [a], adjoint)

        assert x.grad == pytest.approx(expected, rel=1e-4, abs=1e-6)

    def test_a_full_reduction_sends_the_adjoint_everywhere(self) -> None:
        x = Tensor(np.zeros((2, 3)))

        x.sum().backward(2.0)

        assert x.grad == pytest.approx(np.full((2, 3), 2.0))

    def test_the_axis_decides_how_the_adjoint_is_tiled(self) -> None:
        # (3, 3) reduced over either axis gives (3,), so the adjoint alone is
        # ambiguous. This is why Sum stores the axis.
        adjoint = np.array([1.0, 2.0, 3.0])

        down = Tensor(np.zeros((3, 3)))
        down.sum(axis=0).backward(adjoint)

        across = Tensor(np.zeros((3, 3)))
        across.sum(axis=1).backward(adjoint)

        assert down.grad == pytest.approx(np.tile(adjoint, (3, 1)))
        assert across.grad == pytest.approx(np.tile(adjoint[:, None], (1, 3)))

    def test_the_gradient_is_writable(self) -> None:
        # broadcast_to yields a read-only view; the operator must not leak one.
        x = Tensor(np.zeros((2, 2)))

        x.sum().backward()

        assert x.grad is not None and x.grad.flags.writeable


class TestMean:
    @pytest.mark.parametrize(("shape", "axis", "keepdims"), CASES, ids=str)
    def test_matches_numpy(
        self, shape: tuple[int, ...], axis: Axis, keepdims: bool
    ) -> None:
        a = values(shape)

        average = Tensor(a).mean(axis, keepdims)

        assert average.value == pytest.approx(numpy_mean(a, axis, keepdims))

    @pytest.mark.parametrize(("shape", "axis", "keepdims"), CASES, ids=str)
    def test_matches_finite_differences(
        self, shape: tuple[int, ...], axis: Axis, keepdims: bool
    ) -> None:
        a = values(shape)
        x = Tensor(a)

        average = x.mean(axis, keepdims)
        adjoint = values(average.shape)
        average.backward(adjoint)

        (expected,) = approx_vjp(lambda v: numpy_mean(v, axis, keepdims), [a], adjoint)

        assert x.grad == pytest.approx(expected, rel=1e-4, abs=1e-6)

    def test_the_gradient_is_the_reciprocal_of_the_count(self) -> None:
        x = Tensor(np.zeros((2, 5)))

        x.mean().backward()

        assert x.grad == pytest.approx(np.full((2, 5), 1.0 / 10.0))
