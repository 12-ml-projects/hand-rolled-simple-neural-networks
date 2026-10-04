import pytest

from src.operators import Source
from src.tensor import Tensor


class TestGraph:
    def test_new_tensor_is_a_leaf(self) -> None:
        x = Tensor(5.0)

        assert x._dag.is_leaf()
        assert x._dag.dependencies == []
        assert isinstance(x._dag.operator, Source)

    def test_adjoint_starts_empty(self) -> None:
        assert Tensor(5.0)._dag.adjoint is None
        assert (Tensor(5.0) * Tensor(2.0))._dag.adjoint is None

    def test_reflected_pow_keeps_operand_order(self) -> None:
        x = Tensor(3.0)

        result = 2.0**x

        base, exponent = result._dag.dependencies
        assert base.value == 2.0
        assert exponent is x._dag
        assert result.value == pytest.approx(8.0)

    def test_dependencies_are_shared_not_copied(self) -> None:
        x = Tensor(5.0)
        y = Tensor(10.0)

        result = x * y

        assert result._dag.dependencies[0] is x._dag
        assert result._dag.dependencies[1] is y._dag

    def test_dependencies_keep_operand_order(self) -> None:
        x = Tensor(5.0)
        y = Tensor(10.0)

        result = y - x

        assert result._dag.dependencies == [y._dag, x._dag]

    def test_reused_tensor_is_one_shared_node(self) -> None:
        x = Tensor(5.0)

        result = x * x

        first, second = result._dag.dependencies
        assert first is second is x._dag

    def test_intermediate_nodes_are_shared(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)
        s = x + y

        result = s * x

        assert result._dag.dependencies[0] is s._dag
        assert s._dag.dependencies[0] is result._dag.dependencies[1] is x._dag


class TestEquality:
    def test_equals_compares_values(self) -> None:
        assert Tensor(5.0).equals(Tensor(5.0))
        assert not Tensor(5.0).equals(Tensor(6.0))

    def test_equals_accepts_plain_numbers(self) -> None:
        assert Tensor(5.0).equals(5.0)
        assert Tensor(5.0).equals(5)
        assert not Tensor(5.0).equals(6.0)

    def test_double_equals_is_identity(self) -> None:
        x = Tensor(5.0)

        assert x == x
        assert x != Tensor(5.0)

    def test_tensors_are_hashable(self) -> None:
        # Not trivia: DAG.visit dedupes with a set, so identity hashing is what
        # makes the topological sort terminate on a shared node. Defining
        # __eq__ without __hash__ would set __hash__ to None and break it.
        x = Tensor(5.0)
        y = Tensor(5.0)

        assert len({x, y, x}) == 2


class TestBackward:
    def test_backward_propagates_adjoint(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)

        z = x * y * x + x * x
        z.backward()

        assert z._dag.adjoint == pytest.approx(1.0)
        assert x._dag.adjoint == pytest.approx(16.0)
        assert y._dag.adjoint == pytest.approx(4.0)

    def test_backward_on_a_leaf_root(self) -> None:
        # No operator involved: the root is the leaf, so seeding it is the whole
        # backward pass.
        x = Tensor(2.0)

        x.backward()

        assert x.grad == pytest.approx(1.0)


class TestGradients:
    def test_requires_grad_flag(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0, requires_grad=False)

        z = x * y
        z.backward()

        assert x._dag.adjoint == pytest.approx(3.0)
        assert y._dag.adjoint is None

    def test_pow_with_false_requires_grad(self) -> None:
        x = Tensor(2.0)
        n = Tensor(3.0, requires_grad=False)

        z = x**n
        z.backward()

        assert x._dag.adjoint == pytest.approx(12.0)
        assert n._dag.adjoint is None

    def test_reset_clears_intermediates_but_keeps_leaves(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)

        inner = x * y
        z = inner + x
        z.backward()

        assert inner._dag.adjoint == pytest.approx(1.0)

        z._dag.reset()

        assert inner._dag.adjoint is None
        assert z._dag.adjoint is None

        # Leaves keep their gradients; only zero_grad() clears those.
        assert x._dag.adjoint == pytest.approx(4.0)
        assert y._dag.adjoint == pytest.approx(2.0)

    def test_backward_twice_leaves_intermediates_unchanged(self) -> None:
        x = Tensor(2.0)
        y = Tensor(3.0)

        # A chain deep enough that every node between the leaves and the root
        # carries an adjoint of its own.
        inner = x * y
        outer = inner * x
        z = outer + y

        z.backward()
        first = (z._dag.adjoint, outer._dag.adjoint, inner._dag.adjoint)

        z.backward()
        second = (z._dag.adjoint, outer._dag.adjoint, inner._dag.adjoint)

        assert first == (1.0, 1.0, pytest.approx(2.0))
        assert second == first

    def test_backward_twice_accumulates_on_leaves(self) -> None:
        x = Tensor(2.0)

        z = x * x
        z.backward()
        assert x._dag.adjoint == pytest.approx(4.0)

        # Leaf gradients pile up by design, so a minibatch can sum its steps.
        z.backward()
        assert x._dag.adjoint == pytest.approx(8.0)

        x.zero_grad()
        z.backward()
        assert x._dag.adjoint == pytest.approx(4.0)
