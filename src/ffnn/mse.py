from src.operators.sum import Axis
from src.tensor import Tensor


def mse(y_pred: Tensor, y_true: Tensor, axis: Axis = None) -> Tensor:
    error = y_pred - y_true

    return (error * error).mean(axis)
