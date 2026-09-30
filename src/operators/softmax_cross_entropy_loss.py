import numpy as np

from src.custom_types import Array, ArrayLike, as_array

from .operator import Operator


class SoftmaxCrossEntropyLoss(Operator):
    """Softmax and cross-entropy as one node, reduced to a mean over the batch.

    Fused rather than composed because exp overflows on large logits unless the
    row maximum is subtracted first, and because the gradient then collapses to
    (softmax(x) - targets) instead of four chained VJPs.
    """

    _targets: Array

    def __init__(self, one_hot_targets: ArrayLike) -> None:
        self._targets = as_array(one_hot_targets)

    def forward(self, x: Array) -> Array:  # type: ignore[override]
        # as_array because np.sum returns a numpy scalar, not the 0-d array the
        # graph expects everywhere else.
        return as_array(-np.sum(self._targets * self._log_softmax(x)) / self._rows(x))

    def backward(  # type: ignore[override]
        self, adjoint: Array, x: Array
    ) -> tuple[Array]:
        probabilities = np.exp(self._log_softmax(x))

        return (adjoint * (probabilities - self._targets) / self._rows(x),)

    @staticmethod
    def _log_softmax(x: Array) -> Array:
        # Shift to avoid overflow. Standard trick
        shifted = x - x.max(axis=-1, keepdims=True)

        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    @staticmethod
    def _rows(x: Array) -> int:
        """How many predictions the mean is over: everything but the classes."""
        if x.ndim == 0:
            raise ValueError("Logits need at least one axis, for the classes.")

        return x.size // x.shape[-1]
