import numpy as np
import pytest

from src.custom_types import Array
from src.tensor import Tensor
from tests.helpers.finite_difference.approx_grad import approx_vjp

SHAPES = [
    ((2, 3), (3, 4)),
    ((1, 5), (5, 1)),
    ((4, 4), (4, 4)),
    ((5, 2, 3), (5, 3, 4)),
    ((2, 5, 2, 3), (2, 5, 3, 4)),
]


def random_matrix(*shapes: tuple[int, ...]) -> list[Array]:
    # Keyed on the shapes themselves. Summing their dimensions would give
    # ((2,3),(3,4)) and ((1,5),(5,1)) the same seed.
    key: list[int] = []
    for shape in shapes:
        key.append(len(shape))
        key.extend(shape)

    rng = np.random.default_rng(key)

    return [rng.normal(size=shape) for shape in shapes]


class TestMatMulForward:
    @pytest.mark.parametrize(("left", "right"), SHAPES)
    def test_matches_numpy(self, left: tuple[int, ...], right: tuple[int, ...]) -> None:
        a, b = random_matrix(left, right)

        assert (Tensor(a) @ Tensor(b)).value == pytest.approx(a @ b)

    def test_plain_arrays_are_accepted(self) -> None:
        a, b = random_matrix((2, 3), (3, 4))

        assert (Tensor(a) @ b).value == pytest.approx(a @ b)
        assert (a @ Tensor(b)).value == pytest.approx(a @ b)


class TestMatMulBackward:
    @pytest.mark.parametrize(("left", "right"), SHAPES)
    def test_matches_finite_differences(
        self, left: tuple[int, ...], right: tuple[int, ...]
    ) -> None:
        a, b = random_matrix(left, right)
        x, y = Tensor(a), Tensor(b)

        product = x @ y
        adjoint = random_matrix(product.shape)[0]
        product.backward(adjoint)

        expected = approx_vjp(lambda p, q: p @ q, [a, b], adjoint)

        assert x.grad == pytest.approx(expected[0], rel=1e-4, abs=1e-6)
        assert y.grad == pytest.approx(expected[1], rel=1e-4, abs=1e-6)

    def test_batch_axes_are_not_transposed(self) -> None:
        # `.T` reverses every axis, which happens to agree with swapaxes on 2-d
        # input. A non-square batch of non-square matrices tells them apart.
        a, b = random_matrix((5, 2, 3), (5, 3, 4))
        x, y = Tensor(a), Tensor(b)

        product = x @ y
        adjoint = random_matrix((5, 2, 4))[0]
        product.backward(adjoint)

        per_batch = np.stack([adjoint[i] @ b[i].T for i in range(5)])
        assert x.grad == pytest.approx(per_batch)


class TestMatMulRank:
    @pytest.mark.parametrize(
        ("left", "right"),
        [(Tensor(2.0), Tensor(3.0)), (Tensor(np.ones(3)), Tensor(np.ones((3, 4))))],
    )
    def test_rejects_fewer_than_two_dimensions(
        self, left: Tensor, right: Tensor
    ) -> None:
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            left @ right
