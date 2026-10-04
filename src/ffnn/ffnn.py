import numpy as np

from src.ffnn.model import Model
from src.tensor import Tensor


class Mlp(Model):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_classes: int,
        rng: np.random.Generator,
    ) -> None:
        self._weights_in = _he_normal(rng, input_size, hidden_size)
        self._bias_in = Tensor(np.zeros(hidden_size))
        self._weights_out = _he_normal(rng, hidden_size, n_classes)
        self._bias_out = Tensor(np.zeros(n_classes))

    def parameters(self) -> list[Tensor]:
        return [self._weights_in, self._bias_in, self._weights_out, self._bias_out]

    def __call__(self, features: Tensor) -> Tensor:
        hidden = (features @ self._weights_in + self._bias_in).relu()

        return hidden @ self._weights_out + self._bias_out


def _he_normal(rng: np.random.Generator, fan_in: int, fan_out: int) -> Tensor:
    """He initialisation."""
    return Tensor(rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
