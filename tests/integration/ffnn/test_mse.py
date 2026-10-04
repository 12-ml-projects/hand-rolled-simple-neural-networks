import numpy as np
import pytest

from src.ffnn.mse import mse
from src.tensor import Tensor
from tests.helpers.finite_difference.approx_grad import approx_vjp


def numpy_mse(pred: np.ndarray, true: np.ndarray) -> np.floating:
    return np.mean((pred - true) ** 2)


class TestMse:
    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        pred, true = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))

        loss = mse(Tensor(pred), Tensor(true, requires_grad=False))

        assert loss.value == pytest.approx(numpy_mse(pred, true))
        assert loss.shape == ()

    def test_its_root_is_in_the_units_of_the_target(self) -> None:
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 5.0])

        loss = mse(Tensor(pred), Tensor(true, requires_grad=False))

        assert loss.sqrt().value == pytest.approx(np.sqrt(numpy_mse(pred, true)))

    def test_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(1)
        pred, true = rng.normal(size=(5,)), rng.normal(size=(5,))
        x = Tensor(pred)

        mse(x, Tensor(true, requires_grad=False)).backward()

        (expected,) = approx_vjp(lambda p: numpy_mse(p, true), [pred])

        assert x.grad == pytest.approx(expected, rel=1e-4, abs=1e-6)
