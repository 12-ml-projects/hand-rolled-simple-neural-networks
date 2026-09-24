import random
from typing import TypeAlias

import pytest

from src.tensor import Tensor
from tests.helpers.least_squares.fit import least_squares_fit

Parameters: TypeAlias = tuple[Tensor, Tensor]
Sample: TypeAlias = tuple[float, float]


TRUE_SLOPE = 2.0
TRUE_INTERCEPT = -1.0


def linear_model(slope: Tensor, intercept: Tensor, x: float) -> Tensor:
    return slope * x + intercept


def squared_error(prediction: Tensor, y: float) -> Tensor:
    return (prediction - y) ** 2


def sample_line(rng: random.Random, count: int, *, noise: float = 0.0) -> list[Sample]:
    xs = [rng.uniform(-1.0, 1.0) for _ in range(count)]

    return [
        (x, TRUE_SLOPE * x + TRUE_INTERCEPT + (rng.gauss(0.0, noise) if noise else 0.0))
        for x in xs
    ]


def train(
    data: list[Sample],
    rng: random.Random,
    *,
    learning_rate: float,
    epochs: int,
    start: Parameters | None = None,
) -> Parameters:
    slope, intercept = start if start is not None else (Tensor(0.0), Tensor(0.0))

    shuffled = list(data)

    for _ in range(epochs):
        rng.shuffle(shuffled)

        for x, y in shuffled:
            loss = squared_error(linear_model(slope, intercept, x), y)
            loss.backward()

            slope_grad, intercept_grad = slope.grad, intercept.grad
            assert slope_grad is not None and intercept_grad is not None

            # NOTE: assign through `value` keeps the update out of the graph. Writing
            # `slope = slope - lr * grad` would expand the graph
            slope.value -= learning_rate * slope_grad
            intercept.value -= learning_rate * intercept_grad

            slope.zero_grad()
            intercept.zero_grad()

    return slope, intercept


class TestLinearRegression:
    def test_recovers_the_generating_line(self) -> None:
        rng = random.Random(0)
        data = sample_line(rng, 50)

        slope, intercept = train(data, rng, learning_rate=0.05, epochs=20)

        assert slope.value == pytest.approx(TRUE_SLOPE, abs=1e-6)
        assert intercept.value == pytest.approx(TRUE_INTERCEPT, abs=1e-6)

    def test_matches_the_closed_form_on_noisy_data(self) -> None:
        rng = random.Random(1)
        data = sample_line(rng, 200, noise=0.3)
        xs, ys = [x for x, _ in data], [y for _, y in data]

        best_slope, best_intercept = least_squares_fit(xs, ys)

        # A decaying rate settles into the optimum; at a fixed rate the per-sample
        # gradients stay non-zero there and the parameters rattle around it.
        parameters: Parameters | None = None
        for rate, epochs in ((0.05, 20), (0.01, 20), (0.002, 20), (0.0004, 20)):
            parameters = train(
                data, rng, learning_rate=rate, epochs=epochs, start=parameters
            )

        assert parameters is not None
        slope, intercept = parameters

        assert slope.value == pytest.approx(best_slope, abs=1e-3)
        assert intercept.value == pytest.approx(best_intercept, abs=1e-3)

    def test_parameters_stay_leaves(self) -> None:
        rng = random.Random(0)
        data = sample_line(rng, 10)

        slope, intercept = train(data, rng, learning_rate=0.05, epochs=20)

        # If the update were recorded in the graph these would be interior nodes
        # of an ever-growing graph
        assert slope._dag.is_leaf()
        assert intercept._dag.is_leaf()
