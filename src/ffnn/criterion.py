import numpy as np

from src.custom_types import Array
from src.operators import SoftmaxCrossEntropyLoss
from src.tensor import Tensor


class CrossEntropyLoss:
    """Takes integer class labels and expands them to one-hot internally.

    Softmax lives inside the operator, fused with the loss, so this is handed
    raw logits -- never a normalised distribution.
    """

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def __call__(self, logits: Tensor, labels: Array) -> Tensor:
        one_hot = np.eye(self._n_classes)[labels]

        # _apply is private; a public entry point for operators is still owed.
        return Tensor._apply(SoftmaxCrossEntropyLoss(one_hot), logits)
