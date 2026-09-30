import numpy as np
import pytest

from src.custom_types import Array
from src.operators import SoftmaxCrossEntropyLoss
from src.tensor import Tensor
from tests.helpers.finite_difference.approx_grad import approx_vjp


def one_hot(labels: list[int], classes: int) -> Array:
    return np.eye(classes)[labels]


def loss_of(logits: Array, targets: Array) -> Tensor:
    return Tensor._apply(SoftmaxCrossEntropyLoss(targets), Tensor(logits))


class TestForward:
    @pytest.mark.parametrize("classes", [2, 3, 10])
    def test_uniform_logits_cost_log_of_the_class_count(self, classes: int) -> None:
        logits = np.zeros((4, classes))
        targets = one_hot([0, 1, 0, 1], classes)

        assert loss_of(logits, targets).value == pytest.approx(np.log(classes))

    def test_a_confident_correct_prediction_costs_almost_nothing(self) -> None:
        logits = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        targets = one_hot([0, 1], 3)

        assert loss_of(logits, targets).value == pytest.approx(0.0, abs=1e-8)

    def test_a_confident_wrong_prediction_is_expensive(self) -> None:
        logits = np.array([[20.0, 0.0, 0.0]])
        targets = one_hot([1], 3)

        assert loss_of(logits, targets).value == pytest.approx(20.0, rel=1e-6)

    def test_it_is_a_mean_so_batch_size_does_not_change_it(self) -> None:
        small = loss_of(np.zeros((2, 3)), one_hot([0, 1], 3))
        large = loss_of(np.zeros((200, 3)), one_hot([0, 1] * 100, 3))

        assert small.value == pytest.approx(large.value)

    @pytest.mark.parametrize("magnitude", [1e2, 1e3, 1e4, 1e8])
    def test_enormous_logits_stay_finite(self, magnitude: float) -> None:
        # Without subtracting the row maximum, exp() overflows to inf here
        logits = np.array([[magnitude, 0.0, -magnitude]])

        assert np.isfinite(loss_of(logits, one_hot([0], 3)).value)


class TestBackward:
    def test_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(5, 4))
        targets = one_hot([0, 3, 1, 2, 0], 4)

        x = Tensor(logits)
        Tensor._apply(SoftmaxCrossEntropyLoss(targets), x).backward()

        operation = SoftmaxCrossEntropyLoss(targets)
        (expected,) = approx_vjp(operation.forward, [logits])

        assert x.grad == pytest.approx(expected, rel=1e-4, abs=1e-7)

    def test_each_row_of_the_gradient_sums_to_zero(self) -> None:
        # Both the softmax and the one-hot row sum to 1, so their difference
        # sums to 0. Catches a reduction over the wrong axis, or a bad one-hot.
        rng = np.random.default_rng(1)
        x = Tensor(rng.normal(size=(6, 5)))

        Tensor._apply(
            SoftmaxCrossEntropyLoss(one_hot([0, 1, 2, 3, 4, 0], 5)), x
        ).backward()

        assert x.grad is not None
        assert x.grad.sum(axis=-1) == pytest.approx(np.zeros(6), abs=1e-12)

    def test_it_points_away_from_the_correct_class(self) -> None:
        # The gradient is (p - y), so the true class is the only entry pulled
        # down; descending it raises that logit and lowers the others.
        x = Tensor(np.zeros((1, 3)))

        Tensor._apply(SoftmaxCrossEntropyLoss(one_hot([1], 3)), x).backward()

        assert x.grad is not None
        assert x.grad[0, 1] < 0.0
        assert x.grad[0, 0] > 0.0
        assert x.grad[0, 2] > 0.0


class TestRows:
    def test_a_single_unbatched_sample_is_one_row(self) -> None:
        # x.size // x.shape[-1] rather than x.shape[0], which would divide a
        # (classes,) input by the class count instead of by one.
        logits = np.array([0.0, 0.0, 0.0])

        assert loss_of(logits, one_hot([0], 3)[0]).value == pytest.approx(np.log(3))

    def test_leading_axes_all_count_as_batch(self) -> None:
        flat = loss_of(np.zeros((6, 3)), one_hot([0] * 6, 3))
        nested = loss_of(np.zeros((2, 3, 3)), one_hot([0] * 6, 3).reshape(2, 3, 3))

        assert flat.value == pytest.approx(nested.value)

    def test_scalar_logits_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one axis"):
            loss_of(np.asarray(1.0), np.asarray(1.0))
