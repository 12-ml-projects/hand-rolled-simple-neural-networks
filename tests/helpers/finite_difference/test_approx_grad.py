import numpy as np
import pytest

from .approx_grad import approx_grad, approx_vjp


class TestApproxGrad:
    @pytest.mark.parametrize(("a", "b"), [(1.5, 3.0), (-2.0, 0.5), (10.0, -7.0)])
    def test_matches_analytical_partials(self, a: float, b: float) -> None:
        def f(a: float, b: float) -> float:
            c = a + b
            return c * c * (a - b)

        c = a + b
        partial_a = 2 * c * (a - b) + c * c
        partial_b = 2 * c * (a - b) - c * c

        assert approx_grad(f, [a, b]) == pytest.approx([partial_a, partial_b])


class TestApproxVjp:
    def test_scalar_case_matches_approx_grad(self) -> None:
        def f(x: float, y: float) -> float:
            return x * y + x

        assert approx_vjp(f, [2.0, 3.0])[0] == pytest.approx(4.0)
        assert approx_vjp(f, [2.0, 3.0])[1] == pytest.approx(2.0)

    def test_returns_one_array_per_input_shaped_like_it(self) -> None:
        a = np.ones((2, 3))
        b = np.ones((3, 4))

        partials = approx_vjp(lambda x, y: x @ y, [a, b], np.ones((2, 4)))

        assert partials[0].shape == a.shape
        assert partials[1].shape == b.shape

    def test_matches_the_analytic_vjp_of_matmul(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(2, 3))
        b = rng.normal(size=(3, 4))
        adjoint = rng.normal(size=(2, 4))

        partials = approx_vjp(lambda x, y: x @ y, [a, b], adjoint)

        assert partials[0] == pytest.approx(adjoint @ b.T, rel=1e-4, abs=1e-6)
        assert partials[1] == pytest.approx(a.T @ adjoint, rel=1e-4, abs=1e-6)

    def test_adjoint_selects_a_row_of_the_jacobian(self) -> None:
        # With a one-hot adjoint the VJP is one row of J, so this pins the whole
        # Jacobian of a 2-element output one row at a time.
        def f(x: np.ndarray) -> np.ndarray:
            return np.stack([x[0] * x[1], x[0] + x[1]])

        first = approx_vjp(f, [np.array([3.0, 5.0])], np.array([1.0, 0.0]))[0]
        second = approx_vjp(f, [np.array([3.0, 5.0])], np.array([0.0, 1.0]))[0]

        assert first == pytest.approx([5.0, 3.0], rel=1e-4, abs=1e-6)
        assert second == pytest.approx([1.0, 1.0], rel=1e-4, abs=1e-6)

    def test_non_scalar_output_needs_an_adjoint(self) -> None:
        with pytest.raises(ValueError, match="explicit adjoint"):
            approx_vjp(lambda x: x * 2.0, [np.ones(3)])

    def test_inputs_are_left_unchanged(self) -> None:
        a = np.ones((2, 2))
        original = a.copy()

        approx_vjp(lambda x: x * x, [a], np.ones((2, 2)))

        assert np.array_equal(a, original)
