from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Scalars are 0-d arrays, so no operator has to ask which it is holding.
Array: TypeAlias = npt.NDArray[np.float64]

ArrayLike: TypeAlias = npt.ArrayLike


def as_array(value: ArrayLike) -> Array:
    """Coerce anything array-shaped into the graph's representation."""
    return np.asarray(value, dtype=np.float64)
