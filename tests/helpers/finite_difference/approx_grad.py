"""Numerical gradients, for checking the analytic ones.

For f: R^n -> R^m at a point x, the Jacobian is J_ij = d f_i / d x_j, of shape
(m, n). Reverse-mode AD never computes J. It computes the vector-Jacobian
product J^T v, for a covector v shaped like f's output -- the adjoint that
`Tensor.backward` takes.

Everything here rests on one identity. Seed the output into a scalar:

    L(x) = <v, f(x)> = sum_i v_i f_i(x)

    dL/dx_j = sum_i v_i df_i/dx_j = sum_i v_i J_ij = (J^T v)_j

So the gradient of L *is* the VJP. That turns an m-output problem, which finite
differences cannot address, into a scalar one, which they can: perturb one
coordinate of x at a time and central-difference L. The cost is linear in the
size of x, and J is never formed.

`approx_vjp` takes f's arguments separately, so the single J^T v vector comes
back partitioned by argument, each piece shaped like the argument it belongs to.

Choose the adjoint at random. With v = ones, L is the plain sum of the outputs, so an
error of +d in one position cancels -d in another; a generic v makes that
cancellation measure-zero. A one-hot v yields one row of J, if J is what you
want.
"""

from collections.abc import Callable, Sequence
from math import isclose
from typing import TypeAlias, overload

import numpy as np

from src.custom_types import Array, ArrayLike, as_array
from src.tensor import Tensor

# f is called with plain arrays, so the same lambda serves both the numerical
# path here and the analytic one built from Tensors. It may return either.
Function: TypeAlias = Callable[..., "ArrayLike | Tensor"]


@overload
def approx_equals(
    a: float, b: float, *, rel_tol: float = ..., abs_tol: float = ...
) -> bool: ...


@overload
def approx_equals(
    a: Sequence[float],
    b: Sequence[float],
    *,
    rel_tol: float = ...,
    abs_tol: float = ...,
) -> bool: ...


def approx_equals(
    a: float | Sequence[float],
    b: float | Sequence[float],
    *,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8,
) -> bool:
    """Compare two numbers, or two equally long sequences element by element."""
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        return len(a) == len(b) and all(
            isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b)
        )

    if isinstance(a, Sequence) or isinstance(b, Sequence):
        raise TypeError("Cannot compare a number with a sequence.")

    return isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def approx_vjp(
    f: Function,
    inputs: Sequence[ArrayLike],
    adjoint: ArrayLike | None = None,
    *,
    h: float = 1e-6,
) -> list[Array]:
    """Central-difference estimate of J.T @ v, one array per argument of f."""
    args = [as_array(value) for value in inputs]

    def output_of(perturbed: Sequence[Array]) -> Array:
        result = f(*perturbed)

        return result.value if isinstance(result, Tensor) else as_array(result)

    if adjoint is None:
        shape = output_of(args).shape
        if shape != ():
            raise ValueError(
                "approx_vjp needs an explicit adjoint for a non-scalar output, "
                f"which here has shape {shape}."
            )
        adjoint = 1.0

    v = as_array(adjoint)
    gradients = []

    for argument, original in enumerate(args):
        gradient = np.zeros_like(original)

        # One scratch copy per argument; the loop below writes into it and puts
        # each entry back, so the caller's arrays are never touched.
        perturbed = list(args)
        scratch = original.copy()
        perturbed[argument] = scratch

        for index in np.ndindex(original.shape):
            entry = float(original[index])
            # Scale the step with the entry, or it vanishes into rounding error
            # for large values and swamps the signal for tiny ones.
            step = h * max(1.0, abs(entry))

            scratch[index] = entry + step
            right = float(np.sum(v * output_of(perturbed)))

            scratch[index] = entry - step
            left = float(np.sum(v * output_of(perturbed)))

            scratch[index] = entry

            gradient[index] = (right - left) / (2 * step)

        gradients.append(gradient)

    return gradients


def approx_grad(f: Function, inputs: Sequence[float], h: float = 1e-6) -> list[float]:
    """Central-difference gradient of a scalar function of scalars."""
    return [float(gradient) for gradient in approx_vjp(f, inputs, h=h)]
