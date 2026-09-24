import pytest

from src.tensor import Tensor

from .approx_grad import approx_equals, approx_grad


class TestApproxGrad:
    @pytest.mark.parametrize(("a", "b"), [(1.5, 3.0), (-2.0, 0.5), (10.0, -7.0)])
    def test_matches_analytical_partials(self, a: float, b: float) -> None:
        def f(a: float, b: float) -> float:
            c = a + b
            return c * c * (a - b)

        c = a + b
        partial_a = 2 * c * (a - b) + c * c
        partial_b = 2 * c * (a - b) - c * c

        assert approx_equals(approx_grad(f, [a, b]), [partial_a, partial_b])

    def test_single_input(self) -> None:
        assert approx_equals(approx_grad(lambda x: x**3, [2.0]), [12.0])

    def test_unused_input_has_zero_partial(self) -> None:
        assert approx_equals(approx_grad(lambda x, y: x, [1.0, 2.0]), [1.0, 0.0])

    def test_function_may_return_a_tensor(self) -> None:
        def f(x: float) -> Tensor:
            return Tensor(x) * x

        assert approx_equals(approx_grad(f, [3.0]), [6.0])

    def test_inputs_are_left_unchanged(self) -> None:
        inputs = [1.0, 2.0]

        approx_grad(lambda x, y: x * y, inputs)

        assert inputs == [1.0, 2.0]


class TestApproxEquals:
    def test_numbers(self) -> None:
        assert approx_equals(1.0, 1.0 + 1e-9)
        assert not approx_equals(1.0, 1.001)

    def test_tolerance_is_relative_for_large_values(self) -> None:
        assert approx_equals(1e6, 1e6 + 1.0)

    def test_tolerance_is_absolute_near_zero(self) -> None:
        assert approx_equals(0.0, 1e-10)
        assert not approx_equals(0.0, 1e-6)

    def test_sequences(self) -> None:
        assert approx_equals([1.0, 2.0], [1.0, 2.0 + 1e-9])
        assert not approx_equals([1.0, 2.0], [1.0, 2.1])

    def test_sequences_of_different_length_differ(self) -> None:
        assert not approx_equals([1.0, 2.0], [1.0])

    def test_number_and_sequence_cannot_be_compared(self) -> None:
        with pytest.raises(TypeError):
            approx_equals(1.0, [1.0])  # type: ignore[call-overload]
