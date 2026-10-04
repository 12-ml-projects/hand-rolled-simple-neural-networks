import numpy as np
import pytest

from src.broadcasting import unbroadcast


class TestUnbroadcast:
    def test_a_matching_shape_is_returned_untouched(self) -> None:
        adjoint = np.ones((3, 4))

        # Identity, not just equality: this is the fast path that keeps the
        # helper free for operators that never broadcast.
        assert unbroadcast(adjoint, (3, 4)) is adjoint

    def test_prepended_axes_are_summed_away(self) -> None:
        adjoint = np.arange(6.0).reshape(2, 3)

        reduced = unbroadcast(adjoint, (3,))

        assert reduced.shape == (3,)
        assert reduced == pytest.approx([3.0, 5.0, 7.0])

    def test_stretched_axes_are_summed_with_the_rank_kept(self) -> None:
        adjoint = np.arange(6.0).reshape(2, 3)

        assert unbroadcast(adjoint, (2, 1)) == pytest.approx(np.array([[3.0], [12.0]]))
        assert unbroadcast(adjoint, (1, 3)) == pytest.approx(
            np.array([[3.0, 5.0, 7.0]])
        )

    def test_both_steps_apply_together(self) -> None:
        # (1,3) broadcast to (5,2,3) gains a leading axis AND has axis 0
        # stretched, so reducing back needs a drop and an un-stretch.
        adjoint = np.ones((5, 2, 3))

        reduced = unbroadcast(adjoint, (1, 3))

        assert reduced.shape == (1, 3)
        assert reduced == pytest.approx(np.full((1, 3), 10.0))

    def test_reduces_all_the_way_to_a_scalar(self) -> None:
        adjoint = np.arange(6.0).reshape(2, 3)

        reduced = unbroadcast(adjoint, ())

        assert reduced.shape == ()
        assert reduced == pytest.approx(15.0)

    def test_totals_are_preserved(self) -> None:
        # Summing rearranges the adjoint but never loses any of it.
        rng = np.random.default_rng(0)
        adjoint = rng.normal(size=(4, 1, 5))

        for shape in [(4, 1, 5), (1, 1, 5), (4, 1, 1), (5,), (1,), ()]:
            assert np.sum(unbroadcast(adjoint, shape)) == pytest.approx(np.sum(adjoint))

    @pytest.mark.parametrize(
        ("adjoint_shape", "shape"),
        [
            ((3, 4), (5,)),  # 4 and 5 disagree, and neither is 1
            ((3, 4), (2, 4)),  # 3 and 2 disagree
            ((4,), (3, 4)),  # target outranks the adjoint
            ((), (2,)),  # nothing can broadcast to a scalar
        ],
    )
    def test_a_genuine_mismatch_raises(
        self, adjoint_shape: tuple[int, ...], shape: tuple[int, ...]
    ) -> None:
        # Not a broadcast to undo -- it means the operator's backward is wrong,
        # so this doubles as a shape check on every operator.
        with pytest.raises(ValueError, match="does not broadcast"):
            unbroadcast(np.ones(adjoint_shape), shape)
