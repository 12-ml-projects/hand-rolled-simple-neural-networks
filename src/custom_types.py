from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

Value: TypeAlias = NDArray[np.float64]

ValueLike: TypeAlias = ArrayLike


def as_value(value: ValueLike) -> Value:
    """Coerce anything array-shaped into the graph's representation."""
    return np.asarray(value, dtype=np.float64)
