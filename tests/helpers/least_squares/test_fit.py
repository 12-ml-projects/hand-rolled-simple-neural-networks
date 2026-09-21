import random

import pytest

from .fit import least_squares_fit


class TestLeastSquaresFit:
    def test_recovers_an_exact_line(self) -> None:
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        ys = [3.0 * x - 4.0 for x in xs]

        slope, intercept = least_squares_fit(xs, ys)

        assert slope == pytest.approx(3.0)
        assert intercept == pytest.approx(-4.0)

    def test_solution_satisfies_the_normal_equations(self) -> None:
        rng = random.Random(0)
        xs = [rng.uniform(-5.0, 5.0) for _ in range(40)]
        ys = [2.0 * x - 1.0 + rng.gauss(0.0, 1.0) for x in xs]

        slope, intercept = least_squares_fit(xs, ys)
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]

        assert sum(residuals) == pytest.approx(0.0, abs=1e-9)
        assert sum(r * x for r, x in zip(residuals, xs)) == pytest.approx(0.0, abs=1e-9)

    def test_rejects_degenerate_input(self) -> None:
        with pytest.raises(ValueError, match="at least two points"):
            least_squares_fit([1.0], [1.0])

        with pytest.raises(ValueError, match="one y for every x"):
            least_squares_fit([1.0, 2.0], [1.0])

        with pytest.raises(ValueError, match="single x"):
            least_squares_fit([1.0, 1.0], [1.0, 2.0])
